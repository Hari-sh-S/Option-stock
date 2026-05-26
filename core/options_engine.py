"""
core/options_engine.py
OptionsBacktestEngine — backtests Nifty Index option strategies
triggered by entry/exit signals from SignalBuilder.

Supports:
  Single-leg : Buy ATM Call, Buy ATM Put
  Multi-leg  : Bull Call Spread, Bear Put Spread, Long Straddle, Custom legs

Data source: Dhan Rolling Options API (primary) + Black-Scholes fallback.

P&L model:
  - Entry: buy option at ask (or mid) on signal day → deduct premium
  - Exit:  sell option at bid (or mid) on exit day  → receive premium
  - Net P&L per trade = (exit_premium - entry_premium) × lot_size × lots
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
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


# ─── Constants ────────────────────────────────────────────────────────────────

NIFTY_LOT_SIZE = 75          # Current NIFTY lot size
RISK_FREE_RATE  = 0.065       # 6.5% p.a. (RBI repo rate approx)


# ─── Option Leg spec ──────────────────────────────────────────────────────────

@dataclass
class OptionLeg:
    """Defines one leg of an options strategy."""
    option_type:  str        # 'CE' or 'PE'
    direction:    str        # 'BUY' or 'SELL'
    strike_offset: int = 0   # 0=ATM, +1=+1 OTM step, -1=-1 OTM step (each step=50pts for NIFTY)
    lots:          int = 1
    expiry_type:  str = 'weekly'   # 'weekly' | 'monthly' | 'dte30'


# ─── Pre-built strategy definitions ──────────────────────────────────────────

OPTION_STRATEGY_CONFIGS: Dict[str, List[OptionLeg]] = {
    "Buy ATM Call": [
        OptionLeg("CE", "BUY", strike_offset=0, lots=1),
    ],
    "Buy ATM Put": [
        OptionLeg("PE", "BUY", strike_offset=0, lots=1),
    ],
    "Bull Call Spread": [
        OptionLeg("CE", "BUY",  strike_offset=0, lots=1),
        OptionLeg("CE", "SELL", strike_offset=2, lots=1),  # +100 pts OTM
    ],
    "Bear Put Spread": [
        OptionLeg("PE", "BUY",  strike_offset=0, lots=1),
        OptionLeg("PE", "SELL", strike_offset=-2, lots=1), # -100 pts OTM
    ],
    "Long Straddle": [
        OptionLeg("CE", "BUY", strike_offset=0, lots=1),
        OptionLeg("PE", "BUY", strike_offset=0, lots=1),
    ],
    "Long Strangle": [
        OptionLeg("CE", "BUY", strike_offset=2, lots=1),   # +100 pts OTM
        OptionLeg("PE", "BUY", strike_offset=-2, lots=1),  # -100 pts OTM
    ],
}


# ─── Black-Scholes pricer (fallback) ─────────────────────────────────────────

def _bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Standard Black-Scholes option price. T in years, sigma annualised."""
    if T <= 0 or sigma <= 0:
        # Intrinsic value only
        if option_type == 'CE':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    from scipy.stats import norm
    N = norm.cdf

    if option_type == 'CE':
        price = S * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        price = K * math.exp(-r * T) * N(-d2) - S * N(-d1)

    return max(price, 0.0)


def _bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict:
    """Compute Delta, Gamma, Theta, Vega via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return {"delta": 1.0 if (option_type == 'CE' and S > K) else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    from scipy.stats import norm
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    n_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == 'CE' else norm.cdf(d1) - 1
    gamma = n_d1 / (S * sigma * math.sqrt(T))
    theta = (-(S * n_d1 * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm.cdf(d2 if option_type == 'CE' else -d2)) / 365
    vega  = S * n_d1 * math.sqrt(T) / 100  # per 1% IV move

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ─── ATM strike selection ─────────────────────────────────────────────────────

def _get_atm_strike(spot: float, step: int = 50) -> float:
    """Round spot price to nearest ATM strike (default step=50 for NIFTY)."""
    return round(spot / step) * step


def _get_strike_for_leg(spot: float, leg: OptionLeg, step: int = 50) -> float:
    """Compute strike price for a leg given ATM + offset."""
    atm = _get_atm_strike(spot, step)
    return atm + leg.strike_offset * step


# ─── Options Backtest Engine ──────────────────────────────────────────────────

class OptionsBacktestEngine:
    """
    Backtests options strategies on the NIFTY Index using entry/exit signals.

    Args:
        entry_rule       : SignalRule that triggers option entry
        exit_rule        : SignalRule that triggers option exit
        strategy_name    : Key in OPTION_STRATEGY_CONFIGS, or 'Custom'
        custom_legs      : List[OptionLeg] (used when strategy_name == 'Custom')
        start_date       : 'YYYY-MM-DD'
        end_date         : 'YYYY-MM-DD'
        initial_capital  : Starting capital in INR
        dhan_client      : authenticated dhanhq.DhanHQ instance (None = BS fallback)
        lot_size         : NIFTY lot size (default 75)
        strike_step      : Strike interval in points (default 50 for NIFTY)
        use_vix          : If True, use India VIX for IV estimate in BS fallback
    """

    def __init__(
        self,
        entry_rule: SignalRule,
        exit_rule: SignalRule,
        strategy_name: str = "Buy ATM Call",
        custom_legs: Optional[List[OptionLeg]] = None,
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = 500_000,
        dhan_client=None,
        lot_size: int = NIFTY_LOT_SIZE,
        strike_step: int = 50,
        use_vix: bool = True,
    ):
        self.entry_rule     = entry_rule
        self.exit_rule      = exit_rule
        self.strategy_name  = strategy_name
        self.legs           = custom_legs if strategy_name == "Custom" else OPTION_STRATEGY_CONFIGS.get(strategy_name, [])
        self.start_date     = start_date
        self.end_date       = end_date
        self.initial_capital = initial_capital
        self.dhan_client    = dhan_client
        self.lot_size       = lot_size
        self.strike_step    = strike_step
        self.use_vix        = use_vix

        self.trades: List[dict]    = []
        self.equity_curve: List[dict] = []
        self.underlying_df: pd.DataFrame = pd.DataFrame()
        self.signals_df: pd.DataFrame    = pd.DataFrame()

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the options backtest.
        Returns dict with keys: trades_df, equity_df, metrics, underlying_df, signals_df
        """
        # 1. Fetch NIFTY underlying
        um = UniverseManager()
        df = um.get_nifty_underlying_data(self.start_date, self.end_date)
        if df.empty:
            raise RuntimeError("Failed to fetch NIFTY data from yfinance.")

        # 2. Compute indicators on underlying
        df = IndicatorLibrary.add_all_for_signal_builder(df)
        df = IndicatorLibrary.add_momentum_volatility_metrics(df)

        # Add India VIX if requested
        if self.use_vix:
            df = self._add_vix(df)

        self.underlying_df = df

        # 3. Evaluate entry/exit signals
        sb = SignalBuilder(df)
        self.signals_df = sb.get_entry_exit_series(self.entry_rule, self.exit_rule)

        # 4. Simulate trades
        self._simulate_trades()

        # 5. Build outputs
        trades_df  = pd.DataFrame(self.trades)
        equity_df  = pd.DataFrame(self.equity_curve).set_index("Date") if self.equity_curve else pd.DataFrame()
        metrics    = self._compute_metrics(trades_df, equity_df)

        return {
            "trades_df":     trades_df,
            "equity_df":     equity_df,
            "metrics":       metrics,
            "underlying_df": self.underlying_df,
            "signals_df":    self.signals_df,
        }

    # ── Trade simulation ──────────────────────────────────────────────────────

    def _simulate_trades(self):
        """Walk through signal series, open/close option positions."""
        capital     = self.initial_capital
        in_trade    = False
        entry_info  = {}

        for date, row in self.signals_df.iterrows():
            spot = float(self.underlying_df.loc[date, "Close"])

            # Mark-to-Market equity
            if in_trade:
                mtm_pnl = self._price_legs(entry_info["legs_opened"], spot, date)
                equity = capital + mtm_pnl
            else:
                equity = capital
            self.equity_curve.append({"Date": date, "Equity": equity})

            # Exit condition
            if in_trade and row["Exit_Signal"] == 1:
                exit_pnl = self._price_legs(entry_info["legs_opened"], spot, date)
                capital  += exit_pnl
                trade = {
                    **entry_info,
                    "exit_date":     date,
                    "exit_spot":     spot,
                    "exit_pnl_inr":  round(exit_pnl, 2),
                    "total_pnl_inr": round(exit_pnl, 2),
                    "pnl_pct":       round(exit_pnl / (entry_info["premium_paid"] + 1e-9) * 100, 2),
                    "exit_reason":   "Signal",
                }
                self.trades.append(trade)
                in_trade = False

            # Entry condition
            elif not in_trade and row["Entry_Signal"] == 1:
                legs_opened, premium_paid, legs_detail = self._open_legs(spot, date)
                if legs_opened is not None:
                    in_trade = True
                    entry_info = {
                        "entry_date":    date,
                        "entry_spot":    spot,
                        "strategy":      self.strategy_name,
                        "legs_detail":   legs_detail,
                        "legs_opened":   legs_opened,
                        "premium_paid":  premium_paid,
                        "lots":          self.legs[0].lots if self.legs else 1,
                    }

        # Close any open position at end
        if in_trade and self.signals_df is not None and len(self.signals_df) > 0:
            last_date = self.signals_df.index[-1]
            last_spot = float(self.underlying_df.iloc[-1]["Close"])
            exit_pnl  = self._price_legs(entry_info["legs_opened"], last_spot, last_date)
            capital   += exit_pnl
            trade = {
                **entry_info,
                "exit_date":     last_date,
                "exit_spot":     last_spot,
                "exit_pnl_inr":  round(exit_pnl, 2),
                "total_pnl_inr": round(exit_pnl, 2),
                "pnl_pct":       round(exit_pnl / (entry_info["premium_paid"] + 1e-9) * 100, 2),
                "exit_reason":   "End of Period",
            }
            self.trades.append(trade)

    def _open_legs(self, spot: float, date: pd.Timestamp) -> Tuple:
        """
        Price all legs at entry using Dhan API or BS fallback.
        Returns (legs_opened_list, total_net_premium, legs_detail_str)
        legs_opened_list: list of dicts with entry premium per leg
        """
        legs_opened = []
        net_premium = 0.0
        detail_parts = []

        for leg in self.legs:
            strike = _get_strike_for_leg(spot, leg, self.strike_step)
            premium = self._get_option_price(spot, strike, leg.option_type, date)
            if premium is None:
                continue
            signed_premium = premium if leg.direction == "BUY" else -premium
            leg_pnl_entry = signed_premium * self.lot_size * leg.lots
            net_premium  += leg_pnl_entry
            legs_opened.append({
                "leg": leg,
                "strike": strike,
                "entry_premium": premium,
                "direction": leg.direction,
            })
            detail_parts.append(
                f"{leg.direction} {leg.option_type} {int(strike)} @ ₹{premium:.1f}"
            )

        if not legs_opened:
            return None, 0, ""

        return legs_opened, net_premium, " | ".join(detail_parts)

    def _price_legs(self, legs_opened: list, spot: float, date: pd.Timestamp) -> float:
        """
        Price all open legs at current spot. Returns net P&L in INR.
        For BUY legs:  P&L = (current_premium - entry_premium) × lot_size × lots
        For SELL legs: P&L = (entry_premium - current_premium) × lot_size × lots
        """
        total = 0.0
        for info in legs_opened:
            leg    = info["leg"]
            strike = info["strike"]
            ep     = info["entry_premium"]
            cp     = self._get_option_price(spot, strike, leg.option_type, date) or ep

            if info["direction"] == "BUY":
                pnl = (cp - ep) * self.lot_size * leg.lots
            else:
                pnl = (ep - cp) * self.lot_size * leg.lots
            total += pnl
        return total

    # ── Option pricing ────────────────────────────────────────────────────────

    def _get_option_price(
        self, spot: float, strike: float, option_type: str, date: pd.Timestamp
    ) -> Optional[float]:
        """
        Try Dhan API first; fall back to Black-Scholes with VIX-derived IV.
        """
        if self.dhan_client is not None:
            try:
                price = self._dhan_option_price(spot, strike, option_type, date)
                if price and price > 0:
                    return price
            except Exception:
                pass

        # Black-Scholes fallback
        return self._bs_option_price(spot, strike, option_type, date)

    def _dhan_option_price(
        self, spot: float, strike: float, option_type: str, date: pd.Timestamp
    ) -> Optional[float]:
        """Fetch historical option price from Dhan Rolling Options API."""
        try:
            from dhan_data_fetcher import DhanDataFetcher
            fetcher = DhanDataFetcher(self.dhan_client)
            price = fetcher.get_historical_option_price(
                "NIFTY", strike=strike, option_type=option_type,
                date=date.strftime("%Y-%m-%d"),
            )
            return price
        except Exception:
            return None

    def _bs_option_price(
        self, spot: float, strike: float, option_type: str, date: pd.Timestamp
    ) -> float:
        """Black-Scholes price using VIX as IV estimate."""
        T = self._dte(date) / 365.0
        iv = self._get_iv(date)
        return _bs_price(spot, strike, T, RISK_FREE_RATE, iv, option_type)

    def _dte(self, date: pd.Timestamp) -> int:
        """Days to nearest weekly expiry (Thursday for NIFTY)."""
        days_to_thu = (3 - date.weekday()) % 7
        return max(days_to_thu, 1)

    def _get_iv(self, date: pd.Timestamp) -> float:
        """Get implied volatility for date. Uses VIX/100 if available, else 20% default."""
        if "VIX" in self.underlying_df.columns:
            try:
                vix = self.underlying_df.loc[date, "VIX"]
                if pd.notna(vix) and vix > 0:
                    return float(vix) / 100.0
            except Exception:
                pass
        return 0.20   # 20% default IV

    # ── India VIX ─────────────────────────────────────────────────────────────

    def _add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch India VIX (^INDIAVIX) from yfinance and merge into df."""
        try:
            vix = yf.download("^INDIAVIX", start=self.start_date, end=self.end_date, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            if not vix.empty:
                df["VIX"] = vix["Close"].reindex(df.index, method="ffill")
        except Exception:
            pass
        return df

    # ── Performance Metrics ───────────────────────────────────────────────────

    def _compute_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
        if trades_df.empty:
            return {
                "Total Trades": 0, "Win Rate (%)": 0, "Total P&L (INR)": 0,
                "Total Return (%)": 0, "Max Drawdown (%)": 0,
                "Sharpe Ratio": 0, "Avg P&L per Trade (INR)": 0,
                "Best Trade (INR)": 0, "Worst Trade (INR)": 0,
            }

        total_pnl = trades_df["total_pnl_inr"].sum()
        wins      = trades_df[trades_df["total_pnl_inr"] > 0]
        win_rate  = len(wins) / len(trades_df) * 100

        metrics = {
            "Total Trades":            len(trades_df),
            "Win Rate (%)":            round(win_rate, 1),
            "Total P&L (INR)":         round(total_pnl, 2),
            "Total Return (%)":        round(total_pnl / self.initial_capital * 100, 2),
            "Avg P&L per Trade (INR)": round(trades_df["total_pnl_inr"].mean(), 2),
            "Best Trade (INR)":        round(trades_df["total_pnl_inr"].max(), 2),
            "Worst Trade (INR)":       round(trades_df["total_pnl_inr"].min(), 2),
        }

        if not equity_df.empty:
            eq = equity_df["Equity"]
            running_max = eq.cummax()
            drawdown    = (eq - running_max) / running_max
            metrics["Max Drawdown (%)"] = round(float(drawdown.min() * 100), 2)

            daily_ret = eq.pct_change().dropna()
            if daily_ret.std() > 0:
                metrics["Sharpe Ratio"] = round(
                    float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)), 2
                )
            else:
                metrics["Sharpe Ratio"] = 0.0

        return metrics
