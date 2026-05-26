"""
core/universe.py
UniverseManager — handles Equity and Nifty Index Options universes.

Equity mode   : returns list of NSE stock tickers from predefined index or custom input
Options mode  : returns NIFTY index options chain metadata from Dhan API
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ── path shim so this works both when run from root and from core/ ─────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nifty_universe import INDEX_NAMES, DISPLAY_NAMES, NIFTY_50
from nse_fetcher import get_universe as _nse_get_universe, get_all_universe_names as _nse_get_names


class UniverseManager:
    """
    Provides stock/options universes for the scanner and backtest engines.

    Equity mode
    -----------
    Sources (in priority):
      1. Custom ticker list (user-provided)
      2. Predefined NSE index via NSEFetcher (live + cached)
      3. Fallback hardcoded NIFTY_50 list

    Options mode (Nifty Index)
    --------------------------
    Underlying  : NIFTY50 (^NSEI / NIFTY.NS)
    Chain source: Dhan Rolling Options API via DhanDataFetcher
    """

    MODES = ["Equity", "Options (Nifty Index)"]

    # Index display name -> nse_fetcher API key
    INDEX_DISPLAY_TO_API = {v: k for k, v in DISPLAY_NAMES.items()}

    # Nifty Index underlying symbols
    NIFTY_UNDERLYING_YFINANCE = "^NSEI"
    NIFTY_UNDERLYING_NSE      = "NIFTY"

    def __init__(self):
        pass  # nse_fetcher uses module-level functions, no class instance needed

    # ── Equity Universe ────────────────────────────────────────────────────────

    def get_equity_universe(
        self,
        index_display_name: str = "Nifty 50",
        custom_tickers: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Returns a list of NSE stock ticker symbols (without .NS suffix).

        Args:
            index_display_name : One of DISPLAY_NAMES values (e.g. "Nifty 50")
            custom_tickers     : If provided, returns this list directly (overrides index)

        Returns:
            List of ticker strings e.g. ['RELIANCE', 'TCS', ...]
        """
        if custom_tickers:
            return self._clean_tickers(custom_tickers)

        api_name = self.INDEX_DISPLAY_TO_API.get(index_display_name)
        if not api_name:
            return list(NIFTY_50)

        try:
            tickers = _nse_get_universe(api_name)
            if tickers:
                return tickers
        except Exception:
            pass

        # Final fallback
        return list(NIFTY_50)

    def parse_custom_tickers(self, raw_text: str) -> List[str]:
        """
        Parse a comma/newline/space separated string of tickers.
        Strips .NS / .BO suffixes, uppercases, deduplicates.

        Example input:
            "RELIANCE, TCS\nINFY HDFCBANK"
        """
        import re
        tokens = re.split(r'[\s,;]+', raw_text.strip())
        tickers = []
        for t in tokens:
            t = t.strip().upper()
            for suffix in ('.NS', '.BO', '.NSE', '.BSE'):
                if t.endswith(suffix):
                    t = t[:-len(suffix)]
            if t:
                tickers.append(t)
        return list(dict.fromkeys(tickers))  # deduplicate preserving order

    def get_index_display_names(self) -> List[str]:
        """All supported index display names for the UI dropdown."""
        try:
            names = _nse_get_names()
            return sorted(names) if names else list(DISPLAY_NAMES.values())
        except Exception:
            return list(DISPLAY_NAMES.values())

    # ── Options Universe (Nifty Index) ─────────────────────────────────────────

    def get_nifty_options_chain(
        self,
        dhan_client,
        expiry: Optional[str] = None,
        option_type: str = "both",
    ) -> pd.DataFrame:
        """
        Fetches NIFTY options chain from Dhan Rolling Options API.

        Args:
            dhan_client : authenticated dhanhq.DhanHQ instance
            expiry      : 'YYYY-MM-DD' string; None = nearest weekly expiry
            option_type : 'CE', 'PE', or 'both'

        Returns:
            DataFrame with columns:
              ['strike', 'option_type', 'expiry', 'ltp', 'iv', 'delta',
               'gamma', 'theta', 'vega', 'oi', 'volume', 'bid', 'ask']
        """
        try:
            from dhan_data_fetcher import DhanDataFetcher
            fetcher = DhanDataFetcher(dhan_client)
            chain = fetcher.get_option_chain("NIFTY", expiry=expiry)
            if chain is None or chain.empty:
                return pd.DataFrame()

            if option_type == 'CE':
                chain = chain[chain['option_type'] == 'CE']
            elif option_type == 'PE':
                chain = chain[chain['option_type'] == 'PE']

            return chain

        except Exception as e:
            return pd.DataFrame()

    def get_nifty_expiries(self, dhan_client) -> List[str]:
        """Returns list of available NIFTY expiry dates (YYYY-MM-DD strings)."""
        try:
            from dhan_data_fetcher import DhanDataFetcher
            fetcher = DhanDataFetcher(dhan_client)
            return fetcher.get_expiry_dates("NIFTY")
        except Exception:
            return []

    def get_nifty_underlying_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches NIFTY50 index OHLCV data via yfinance for indicator calculation.
        Returns standard OHLCV DataFrame indexed by Date.
        """
        import yfinance as yf
        try:
            df = yf.download(
                self.NIFTY_UNDERLYING_YFINANCE,
                start=start_date,
                end=end_date,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            return pd.DataFrame()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_tickers(tickers: List[str]) -> List[str]:
        """Strip .NS / .BO suffixes and deduplicate."""
        result = []
        seen = set()
        for t in tickers:
            t = t.strip().upper()
            for suffix in ('.NS', '.BO', '.NSE', '.BSE'):
                if t.endswith(suffix):
                    t = t[:-len(suffix)]
            if t and t not in seen:
                result.append(t)
                seen.add(t)
        return result

    @staticmethod
    def ticker_to_yfinance(ticker: str) -> str:
        """Convert bare NSE ticker to yfinance symbol (e.g. RELIANCE → RELIANCE.NS)."""
        if not ticker.endswith(('.NS', '.BO')):
            return f"{ticker}.NS"
        return ticker
