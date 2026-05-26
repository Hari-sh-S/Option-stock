"""
core/scanner.py
Scanner — runs entry/exit SignalRules across a universe of stocks,
computes scoring metrics per stock, and returns a ranked DataFrame.

Works in both Equity and Options (Nifty Index) modes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.indicators import IndicatorLibrary
from core.signal_builder import SignalBuilder, SignalRule
from core.universe import UniverseManager


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _percentile_rank(series: pd.Series) -> pd.Series:
    """Rank a Series as percentile (0–100), higher = better rank."""
    return series.rank(pct=True) * 100


def _percentile_rank_inverse(series: pd.Series) -> pd.Series:
    """Inverse-rank: lower value = better (e.g. volatility, drawdown)."""
    return (1 - series.rank(pct=True)) * 100


# ─── Default scoring weight config ───────────────────────────────────────────

DEFAULT_SCORING_WEIGHTS: Dict[str, Tuple[float, str]] = {
    # (weight, 'asc' = higher is better | 'desc' = lower is better)
    "3 Month Performance":       (0.25, "asc"),
    "6 Month Performance":       (0.20, "asc"),
    "1 Year Sharpe":             (0.20, "asc"),
    "3 Month Volatility":        (0.10, "desc"),
    "3 Month Max Drawdown":      (0.10, "desc"),
    "Entry_Signal_Strength":     (0.15, "asc"),
}


# ─── Scanner ─────────────────────────────────────────────────────────────────

class Scanner:
    """
    Scans a universe of stocks/index for entry signals and scores them.

    Usage:
        scanner = Scanner(
            universe=['RELIANCE', 'TCS', 'INFY'],
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            start_date='2023-01-01',
            end_date='2024-12-31',
            scoring_weights=DEFAULT_SCORING_WEIGHTS,
        )
        results = scanner.run()  # → pd.DataFrame ranked by score
    """

    def __init__(
        self,
        universe: List[str],
        entry_rule: SignalRule,
        exit_rule: SignalRule,
        start_date: str,
        end_date: str,
        scoring_weights: Optional[Dict[str, Tuple[float, str]]] = None,
        on_progress: Optional[callable] = None,
    ):
        self.universe = universe
        self.entry_rule = entry_rule
        self.exit_rule = exit_rule
        self.start_date = start_date
        self.end_date = end_date
        self.scoring_weights = scoring_weights or DEFAULT_SCORING_WEIGHTS
        self.on_progress = on_progress  # callback(ticker, i, total)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Scan all tickers. Returns a ranked DataFrame with columns:
          Ticker, Last_Close, Entry_Signal, Exit_Signal, Days_Since_Entry,
          <scoring metrics...>, Score, Rank
        """
        rows = []
        total = len(self.universe)

        for i, ticker in enumerate(self.universe):
            if self.on_progress:
                self.on_progress(ticker, i, total)

            try:
                row = self._scan_ticker(ticker)
                if row:
                    rows.append(row)
            except Exception as e:
                # Skip broken tickers silently
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = self._compute_scores(df)
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        return df

    # ── Per-ticker scan ───────────────────────────────────────────────────────

    def _scan_ticker(self, ticker: str) -> Optional[dict]:
        """Fetch data, compute indicators, evaluate signals, compute raw metrics."""
        symbol = UniverseManager.ticker_to_yfinance(ticker)

        # ── Fetch OHLCV ──────────────────────────────────────────────────────
        df = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60:
            return None

        # ── Indicators ───────────────────────────────────────────────────────
        df = IndicatorLibrary.add_all_for_signal_builder(df)
        df = IndicatorLibrary.add_momentum_volatility_metrics(df)

        # ── Evaluate signals on latest bar ───────────────────────────────────
        sb = SignalBuilder(df)
        entry_series = sb.evaluate(self.entry_rule)
        exit_series  = sb.evaluate(self.exit_rule)

        latest_entry = bool(entry_series.iloc[-1])
        latest_exit  = bool(exit_series.iloc[-1])

        # Days since last entry signal
        entry_dates = entry_series[entry_series].index
        days_since_entry = (pd.Timestamp(self.end_date) - entry_dates[-1]).days if len(entry_dates) > 0 else None

        # Signal strength: fraction of last 20 bars where entry was True
        signal_strength = float(entry_series.tail(20).mean()) * 100

        # ── Latest metrics ───────────────────────────────────────────────────
        last = df.iloc[-1]
        row: dict = {
            "Ticker":               ticker,
            "Last_Close":           round(float(last["Close"]), 2),
            "Entry_Signal":         latest_entry,
            "Exit_Signal":          latest_exit,
            "Days_Since_Entry":     days_since_entry,
            "Entry_Signal_Strength": signal_strength,
        }

        # Pull scoring metric columns from the latest bar
        for metric in self.scoring_weights:
            if metric in df.columns:
                row[metric] = float(last[metric]) if not pd.isna(last[metric]) else 0.0
            elif metric == "Entry_Signal_Strength":
                row[metric] = signal_strength

        return row

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a weighted percentile Score across all tickers.
        Score = Σ (weight_i × percentile_rank_i)
        Range: 0–100 (higher = better)
        """
        score_col = pd.Series(0.0, index=df.index)
        total_weight = sum(w for w, _ in self.scoring_weights.values())

        for metric, (weight, direction) in self.scoring_weights.items():
            if metric not in df.columns:
                continue
            raw = df[metric].fillna(0)
            if direction == "asc":
                ranked = _percentile_rank(raw)
            else:
                ranked = _percentile_rank_inverse(raw)
            score_col += (weight / total_weight) * ranked

        df["Score"] = score_col.round(1)
        return df

    # ── Nifty Options Signal Scan (new) ───────────────────────────────────────

    @classmethod
    def run_nifty_options_scan(
        cls,
        entry_rule: SignalRule,
        exit_rule: SignalRule,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Evaluate entry/exit rules on the NIFTY underlying (^NSEI).
        Returns a summary dict with:
          - signal_dates_df: DataFrame of entry/exit dates
          - latest_entry: bool
          - latest_exit: bool
          - underlying_df: full OHLCV+indicator DataFrame for charting
        """
        um = UniverseManager()
        df = um.get_nifty_underlying_data(start_date, end_date)
        if df.empty:
            return {}

        df = IndicatorLibrary.add_all_for_signal_builder(df)

        sb = SignalBuilder(df)
        signals = sb.get_entry_exit_series(entry_rule, exit_rule)

        return {
            "underlying_df":   df,
            "signal_df":       signals,
            "latest_entry":    bool(signals["Entry_Signal"].iloc[-1]),
            "latest_exit":     bool(signals["Exit_Signal"].iloc[-1]),
            "entry_dates":     signals[signals["Entry_Signal"] == 1].index.tolist(),
            "exit_dates":      signals[signals["Exit_Signal"] == 1].index.tolist(),
        }
