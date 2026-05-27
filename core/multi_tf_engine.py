"""
core/multi_tf_engine.py
Multi-Timeframe Signal Engine for NIFTY Options Backtest.

Handles:
  1. Fetching NIFTY index intraday data from Dhan API (/charts/intraday)
     - intervals: 1, 5, 15, 25, 60 minutes (up to 90 days per request, 5 year history)
     - fallback: yfinance for daily/weekly/monthly
  2. Resampling to any timeframe: 1m, 5m, 15m, 25m, 1h, 1D, 1W, 1M
  3. Computing indicators on any timeframe's OHLCV data
  4. Evaluating multi-condition signal rules where each condition can use
     a different timeframe (checks previous closed bar)
  5. Returning a daily boolean Series: True = signal fired on that day

Dhan intraday API details:
  POST https://api.dhan.co/v2/charts/intraday
  securityId: "13"  (NIFTY 50 index)
  exchangeSegment: "IDX_I"
  instrument: "INDEX"
  interval: "1" | "5" | "15" | "25" | "60"
  fromDate: "YYYY-MM-DD HH:MM:SS"
  toDate:   "YYYY-MM-DD HH:MM:SS"
  Max 90 days per request; data available for last 5 years.
"""
from __future__ import annotations

import time
import sys
import math
import requests
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

NIFTY_SECURITY_ID     = "13"
NIFTY_EXCHANGE_SEG    = "IDX_I"
NIFTY_INSTRUMENT      = "INDEX"
NIFTY_YFINANCE_SYMBOL = "^NSEI"
DHAN_INTRADAY_URL     = "https://api.dhan.co/v2/charts/intraday"
DHAN_DAILY_URL        = "https://api.dhan.co/v2/charts/historical"
DHAN_CHUNK_DAYS       = 85   # stay under 90-day API limit
INTRADAY_INTERVALS    = {"1m": 1, "5m": 5, "15m": 15, "25m": 25, "1h": 60}
DAILY_INTERVALS       = {"1D", "1W", "1M"}

# Resample rules for pandas (applied to 1-min or daily base data)
RESAMPLE_RULES = {
    "1m":  None,    # base data
    "5m":  "5min",
    "15m": "15min",
    "25m": "25min",
    "1h":  "60min",
    "1D":  "D",
    "1W":  "W-THU",    # week ending Thursday (NSE)
    "1M":  "ME",
}

# Price field aliases (what user picks in UI)
PRICE_FIELDS = [
    "Close", "Open", "High", "Low", "Volume",
    "OHLC4", "HL2", "OC2",   # derived
]

# Indicator definitions: name -> (params_schema)
INDICATOR_DEFS = {
    "EMA":         {"source": True,  "period": True,  "period2": False, "period3": False},
    "SMA":         {"source": True,  "period": True,  "period2": False, "period3": False},
    "RSI":         {"source": True,  "period": True,  "period2": False, "period3": False},
    "ATR":         {"source": False, "period": True,  "period2": False, "period3": False},
    "MACD":        {"source": True,  "period": True,  "period2": True,  "period3": True},   # fast, slow, signal
    "MACD_Signal": {"source": True,  "period": True,  "period2": True,  "period3": True},
    "MACD_Hist":   {"source": True,  "period": True,  "period2": True,  "period3": True},
    "SuperTrend":  {"source": False, "period": True,  "period2": True,  "period3": False},  # period, multiplier
    "BB_Upper":    {"source": True,  "period": True,  "period2": True,  "period3": False},  # period, stddev
    "BB_Lower":    {"source": True,  "period": True,  "period2": True,  "period3": False},
    "BB_Mid":      {"source": True,  "period": True,  "period2": True,  "period3": False},
    "Stoch_K":     {"source": False, "period": True,  "period2": True,  "period3": False},  # period, smooth
    "Stoch_D":     {"source": False, "period": True,  "period2": True,  "period3": False},
    "ADX":         {"source": False, "period": True,  "period2": False, "period3": False},
    "DI_Plus":     {"source": False, "period": True,  "period2": False, "period3": False},
    "DI_Minus":    {"source": False, "period": True,  "period2": False, "period3": False},
    "VWAP":        {"source": False, "period": False, "period2": False, "period3": False},
    "OBV":         {"source": False, "period": False, "period2": False, "period3": False},
    "Volume_MA":   {"source": False, "period": True,  "period2": False, "period3": False},
}

OPERATORS = [">" , "<", ">=", "<=", "==", "crosses_above", "crosses_below"]


# ── Operand & Condition dataclasses ──────────────────────────────────────────

@dataclass
class Operand:
    """One side of a condition (indicator or price field or fixed value)."""
    kind: str = "price"        # 'price' | 'indicator' | 'value'
    price_field: str = "Close" # used when kind='price' or as indicator source
    indicator: str = "EMA"     # used when kind='indicator'
    source: str = "Close"      # source OHLCV for indicator
    period: int = 20           # main period
    period2: float = 2.0       # secondary (mult, slow, stddev, smooth, signal)
    period3: int = 9           # tertiary (MACD signal)
    timeframe: str = "1D"      # one of RESAMPLE_RULES keys
    value: float = 0.0         # used when kind='value'

    def label(self) -> str:
        """Human-readable label for display."""
        if self.kind == "value":
            return str(self.value)
        if self.kind == "price":
            return f"{self.price_field}[{self.timeframe}]"
        # indicator
        p = self.indicator
        schema = INDICATOR_DEFS.get(self.indicator, {})
        if schema.get("source"):
            p += f"({self.source},{self.period}"
        elif schema.get("period"):
            p += f"({self.period}"
        else:
            p += "("
        if schema.get("period2"):
            p += f",{self.period2}"
        if schema.get("period3"):
            p += f",{self.period3}"
        p += f")[{self.timeframe}]"
        return p

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d: dict) -> "Operand":
        o = Operand()
        for k, v in d.items():
            if hasattr(o, k):
                setattr(o, k, v)
        return o


@dataclass
class MultiTFCondition:
    """One condition: left OP right, where each side may have its own timeframe."""
    left: Operand = field(default_factory=Operand)
    operator: str = ">"
    right: Operand = field(default_factory=lambda: Operand(kind="value", value=0.0))

    def label(self) -> str:
        return f"{self.left.label()} {self.operator} {self.right.label()}"

    def to_dict(self) -> dict:
        return {"left": self.left.to_dict(), "operator": self.operator, "right": self.right.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "MultiTFCondition":
        return MultiTFCondition(
            left=Operand.from_dict(d.get("left", {})),
            operator=d.get("operator", ">"),
            right=Operand.from_dict(d.get("right", {})),
        )


@dataclass
class MultiTFRule:
    """A complete entry or exit rule: list of conditions joined by AND/OR."""
    conditions: List[MultiTFCondition] = field(default_factory=list)
    logic: str = "AND"   # 'AND' | 'OR'
    name: str = ""

    def is_empty(self) -> bool:
        return len(self.conditions) == 0

    def to_dict(self) -> dict:
        return {
            "conditions": [c.to_dict() for c in self.conditions],
            "logic": self.logic,
            "name": self.name,
        }

    @staticmethod
    def from_dict(d: dict) -> "MultiTFRule":
        return MultiTFRule(
            conditions=[MultiTFCondition.from_dict(c) for c in d.get("conditions", [])],
            logic=d.get("logic", "AND"),
            name=d.get("name", ""),
        )


# ── Data Fetching ─────────────────────────────────────────────────────────────

def _parse_dhan_response(resp_data: dict) -> pd.DataFrame:
    """Parse Dhan historical/intraday API response dict into OHLCV DataFrame."""
    timestamps = resp_data.get("timestamp", [])
    if not timestamps:
        return pd.DataFrame()
    records = []
    for i, ts in enumerate(timestamps):
        try:
            dt = datetime.fromtimestamp(int(ts))
        except Exception:
            continue

        def _g(k, default=0.0):
            arr = resp_data.get(k, [])
            try:
                return float(arr[i]) if i < len(arr) else default
            except (TypeError, ValueError):
                return default

        records.append({
            "Datetime": dt,
            "Open":   _g("open"),
            "High":   _g("high"),
            "Low":    _g("low"),
            "Close":  _g("close"),
            "Volume": int(_g("volume")),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _fetch_dhan_intraday_chunk(
    from_dt: datetime, to_dt: datetime, interval_min: int, dhan_client=None
) -> pd.DataFrame:
    """Fetch one chunk of Dhan intraday data (≤90 days)."""
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str   = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "securityId":      NIFTY_SECURITY_ID,
        "exchangeSegment": NIFTY_EXCHANGE_SEG,
        "instrument":      NIFTY_INSTRUMENT,
        "interval":        str(interval_min),
        "oi":              False,
        "fromDate":        from_str,
        "toDate":          to_str,
    }

    # Try SDK first
    if dhan_client is not None:
        try:
            resp = dhan_client.intraday_minute_data(
                security_id=NIFTY_SECURITY_ID,
                exchange_segment=NIFTY_EXCHANGE_SEG,
                instrument_type=NIFTY_INSTRUMENT,
                interval=str(interval_min),
                from_date=from_str,
                to_date=to_str,
            )
            if isinstance(resp, dict) and resp.get("status") == "success":
                return _parse_dhan_response(resp.get("data", {}))
        except Exception as e:
            print(f"[MultiTF] SDK intraday call failed: {e}, trying REST...")

    # Direct REST
    try:
        from config import get_saved_credentials
        creds = get_saved_credentials()
        headers = {
            "access-token": creds.get("access_token", ""),
            "client-id":    creds.get("client_id", ""),
            "Content-Type": "application/json",
        }
        r = requests.post(DHAN_INTRADAY_URL, json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            body = r.json()
            data = body if "timestamp" in body else body.get("data", {})
            return _parse_dhan_response(data)
        else:
            print(f"[MultiTF] REST intraday {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[MultiTF] REST intraday failed: {e}")

    return pd.DataFrame()


def fetch_nifty_intraday(
    start_date: str,
    end_date: str,
    interval_min: int = 5,
    dhan_client=None,
    delay_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch NIFTY intraday data from Dhan API in 85-day chunks.
    Falls back to yfinance if Dhan fails.

    Returns DataFrame with DatetimeIndex and Open/High/Low/Close/Volume.
    """
    from_dt = datetime.strptime(start_date, "%Y-%m-%d")
    to_dt   = datetime.strptime(end_date,   "%Y-%m-%d").replace(hour=23, minute=59)

    chunks = []
    cur = from_dt
    while cur < to_dt:
        end = min(cur + timedelta(days=DHAN_CHUNK_DAYS), to_dt)
        chunks.append((cur, end))
        cur = end + timedelta(seconds=1)

    print(f"[MultiTF] Fetching {len(chunks)} chunks of {interval_min}m NIFTY data...")
    all_frames = []
    for i, (cf, ct) in enumerate(chunks):
        chunk = _fetch_dhan_intraday_chunk(cf, ct, interval_min, dhan_client)
        if not chunk.empty:
            all_frames.append(chunk)
        if i < len(chunks) - 1:
            time.sleep(delay_seconds)

    if not all_frames:
        print("[MultiTF] Dhan intraday unavailable, falling back to yfinance...")
        return _yf_download_nifty(start_date, end_date, interval_min)

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"[MultiTF] {len(combined)} intraday bars fetched.")
    return combined


def fetch_nifty_daily(start_date: str, end_date: str, dhan_client=None) -> pd.DataFrame:
    """
    Fetch NIFTY daily OHLCV.
    Tries Dhan daily API first, falls back to yfinance.
    """
    payload = {
        "securityId":      NIFTY_SECURITY_ID,
        "exchangeSegment": NIFTY_EXCHANGE_SEG,
        "instrument":      NIFTY_INSTRUMENT,
        "expiryCode":      0,
        "oi":              False,
        "fromDate":        start_date,
        "toDate":          end_date,
    }

    if dhan_client is not None:
        try:
            resp = dhan_client.historical_daily_data(
                security_id=NIFTY_SECURITY_ID,
                exchange_segment=NIFTY_EXCHANGE_SEG,
                instrument_type=NIFTY_INSTRUMENT,
                from_date=start_date,
                to_date=end_date,
            )
            if isinstance(resp, dict) and resp.get("status") == "success":
                df = _parse_dhan_response(resp.get("data", resp))
                if not df.empty:
                    daily = df.resample("D").agg(
                        {"Open": "first", "High": "max", "Low": "min",
                         "Close": "last", "Volume": "sum"}
                    ).dropna(subset=["Close"])
                    daily = daily[daily["Close"] > 0]
                    if not daily.empty:
                        return daily
        except Exception as e:
            print(f"[MultiTF] Dhan daily failed: {e}")

    # yfinance fallback
    return _yf_download_nifty_daily(start_date, end_date)


def _yf_download_nifty(start: str, end: str, interval_min: int) -> pd.DataFrame:
    """yfinance fallback for intraday (max 60 days per call for 1h)."""
    # yfinance supports: 1m (7d), 5m (60d), 15m (60d), 1h (730d), 1d
    yf_interval_map = {1: "1m", 5: "5m", 15: "15m", 25: "5m", 60: "1h"}
    yf_interval = yf_interval_map.get(interval_min, "1h")
    try:
        df = yf.download(
            NIFTY_YFINANCE_SYMBOL, start=start, end=end,
            interval=yf_interval, progress=False, auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            # Resample 5m to 25m if needed
            if interval_min == 25 and yf_interval == "5m":
                df = _resample_ohlcv(df, "25min")
        return df
    except Exception as e:
        print(f"[MultiTF] yfinance intraday failed: {e}")
        return pd.DataFrame()


def _yf_download_nifty_daily(start: str, end: str) -> pd.DataFrame:
    """yfinance daily OHLCV for NIFTY."""
    try:
        df = yf.download(NIFTY_YFINANCE_SYMBOL, start=start, end=end,
                          progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"[MultiTF] yfinance daily failed: {e}")
        return pd.DataFrame()


# ── Resampling ────────────────────────────────────────────────────────────────

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample minute/daily OHLCV to target rule."""
    if df.empty or rule is None:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    resampled = df.resample(rule).agg(
        {k: v for k, v in agg.items() if k in df.columns}
    ).dropna(subset=["Close"])
    return resampled[resampled["Close"] > 0]


# ── Derived price fields ───────────────────────────────────────────────────────

def _get_price_series(df: pd.DataFrame, field: str) -> pd.Series:
    """Return the requested price series from OHLCV DataFrame."""
    o = df["Open"]  if "Open"   in df.columns else df["Close"]
    h = df["High"]  if "High"   in df.columns else df["Close"]
    l = df["Low"]   if "Low"    in df.columns else df["Close"]
    c = df["Close"] if "Close"  in df.columns else pd.Series(dtype=float)
    v = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
    if field == "Close":   return c
    if field == "Open":    return o
    if field == "High":    return h
    if field == "Low":     return l
    if field == "Volume":  return v
    if field == "OHLC4":   return (o + h + l + c) / 4
    if field == "HL2":     return (h + l) / 2
    if field == "OC2":     return (o + c) / 2
    return c  # default


# ── Indicator computation ────────────────────────────────────────────────────

def _compute_indicator(df: pd.DataFrame, operand: Operand) -> pd.Series:
    """
    Compute the indicator specified by `operand` on `df` (OHLCV DataFrame).
    Returns a pd.Series aligned to df.index.
    """
    ind = operand.indicator
    src = _get_price_series(df, operand.source)
    p1  = int(operand.period)
    p2  = operand.period2   # may be float (multiplier for SuperTrend)
    p3  = int(operand.period3)

    if ind == "EMA":
        return src.ewm(span=p1, adjust=False).mean()

    if ind == "SMA":
        return src.rolling(p1).mean()

    if ind == "RSI":
        delta = src.diff()
        gain  = delta.clip(lower=0).rolling(p1).mean()
        loss  = (-delta.clip(upper=0)).rolling(p1).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    if ind == "ATR":
        hi = df["High"] if "High" in df.columns else src
        lo = df["Low"]  if "Low"  in df.columns else src
        cl = df["Close"] if "Close" in df.columns else src
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift(1)).abs(),
            (lo - cl.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=p1, adjust=False).mean()

    if ind in ("MACD", "MACD_Signal", "MACD_Hist"):
        fast = int(p1); slow = int(p2); sig = int(p3)
        ema_fast = src.ewm(span=fast, adjust=False).mean()
        ema_slow = src.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        sig_line  = macd_line.ewm(span=sig, adjust=False).mean()
        hist      = macd_line - sig_line
        if ind == "MACD":        return macd_line
        if ind == "MACD_Signal": return sig_line
        return hist

    if ind == "SuperTrend":
        period = p1; mult = float(p2)
        hi = df["High"] if "High" in df.columns else src
        lo = df["Low"]  if "Low"  in df.columns else src
        cl = df["Close"] if "Close" in df.columns else src
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift(1)).abs(),
            (lo - cl.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr  = tr.ewm(span=period, adjust=False).mean()
        hl2  = (hi + lo) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        st = _compute_supertrend_series(cl.values, upper.values, lower.values)
        return pd.Series(st, index=df.index)

    if ind in ("BB_Upper", "BB_Lower", "BB_Mid"):
        period = p1; dev = float(p2)
        mid   = src.rolling(period).mean()
        sigma = src.rolling(period).std()
        if ind == "BB_Upper": return mid + dev * sigma
        if ind == "BB_Lower": return mid - dev * sigma
        return mid

    if ind in ("Stoch_K", "Stoch_D"):
        hi = df["High"] if "High" in df.columns else src
        lo = df["Low"]  if "Low"  in df.columns else src
        cl = df["Close"] if "Close" in df.columns else src
        lo_min = lo.rolling(p1).min()
        hi_max = hi.rolling(p1).max()
        k = 100 * (cl - lo_min) / (hi_max - lo_min + 1e-9)
        d = k.rolling(int(p2)).mean()
        return k if ind == "Stoch_K" else d

    if ind in ("ADX", "DI_Plus", "DI_Minus"):
        hi = df["High"] if "High" in df.columns else src
        lo = df["Low"]  if "Low"  in df.columns else src
        cl = df["Close"] if "Close" in df.columns else src
        try:
            import ta
            adx_ind = ta.trend.ADXIndicator(hi, lo, cl, window=p1)
            if ind == "ADX":      return adx_ind.adx()
            if ind == "DI_Plus":  return adx_ind.adx_pos()
            return adx_ind.adx_neg()
        except Exception:
            return pd.Series(0.0, index=df.index)

    if ind == "VWAP":
        hi = df["High"] if "High" in df.columns else src
        lo = df["Low"]  if "Low"  in df.columns else src
        cl = df["Close"] if "Close" in df.columns else src
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
        tp  = (hi + lo + cl) / 3
        return (tp * vol).cumsum() / vol.cumsum()

    if ind == "OBV":
        cl  = df["Close"] if "Close" in df.columns else src
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
        direction = cl.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (direction * vol).cumsum()

    if ind == "Volume_MA":
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
        return vol.rolling(p1).mean()

    # Fallback
    return src


def _compute_supertrend_series(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    n  = len(close)
    st = np.empty(n)
    st[0] = upper[0]
    for i in range(1, n):
        if close[i] > upper[i - 1]:
            st[i] = lower[i]
        elif close[i] < lower[i - 1]:
            st[i] = upper[i]
        else:
            st[i] = st[i - 1]
    return st


# ── Operand value resolution ──────────────────────────────────────────────────

def _resolve_operand(
    operand: Operand,
    tf_data: Dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Resolve an Operand to a daily pd.Series (indexed by date_index).
    Checks previous closed bar for each timeframe.
    """
    if operand.kind == "value":
        return pd.Series(operand.value, index=date_index)

    df = tf_data.get(operand.timeframe, pd.DataFrame())
    if df.empty:
        return pd.Series(np.nan, index=date_index)

    if operand.kind == "price":
        raw = _get_price_series(df, operand.price_field)
    else:  # indicator
        raw = _compute_indicator(df, operand)

    # For each day in date_index, get the value of the PREVIOUS closed bar
    # For daily TF: shift(1) handles it; for intraday: resample to daily last, then shift
    if operand.timeframe in ("1D", "1W", "1M"):
        # Already daily/weekly/monthly — take last value of previous bar
        daily_last = raw.resample("D").last().ffill()
        # previous bar = shift(1)
        prev = daily_last.shift(1)
    else:
        # Intraday: get the last bar value for each calendar day, then shift to previous day
        daily_last = raw.resample("D").last().ffill()
        prev = daily_last.shift(1)

    # Reindex to date_index (fill forward for non-trading days)
    result = prev.reindex(date_index, method="ffill")
    return result


# ── Condition evaluation ─────────────────────────────────────────────────────

def _evaluate_condition(
    cond: MultiTFCondition,
    tf_data: Dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Evaluate a single condition across all dates.
    Returns a boolean pd.Series.
    """
    left  = _resolve_operand(cond.left,  tf_data, date_index)
    right = _resolve_operand(cond.right, tf_data, date_index)

    op = cond.operator
    if op == ">":           return (left > right).fillna(False)
    if op == "<":           return (left < right).fillna(False)
    if op == ">=":          return (left >= right).fillna(False)
    if op == "<=":          return (left <= right).fillna(False)
    if op == "==":          return (left == right).fillna(False)
    if op == "crosses_above":
        return ((left > right) & (left.shift(1) <= right.shift(1))).fillna(False)
    if op == "crosses_below":
        return ((left < right) & (left.shift(1) >= right.shift(1))).fillna(False)
    return pd.Series(False, index=date_index)


def evaluate_rule(
    rule: MultiTFRule,
    tf_data: Dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Evaluate a complete rule (all conditions combined with AND/OR).
    Returns a boolean pd.Series indexed by date_index.
    """
    if rule.is_empty():
        return pd.Series(False, index=date_index)

    results = [_evaluate_condition(c, tf_data, date_index) for c in rule.conditions]

    if rule.logic == "OR":
        combined = results[0].copy()
        for r in results[1:]:
            combined = combined | r
    else:  # AND (default)
        combined = results[0].copy()
        for r in results[1:]:
            combined = combined & r

    return combined


# ── Main Engine ───────────────────────────────────────────────────────────────

class MultiTFEngine:
    """
    Orchestrates data fetching and signal evaluation for multi-timeframe conditions.

    Usage:
        engine = MultiTFEngine(start_date, end_date, entry_rule, exit_rule, dhan_client)
        result = engine.run()
        # result['entry_signals'] -> daily boolean Series
        # result['exit_signals']  -> daily boolean Series
        # result['tf_data']       -> dict of {tf: DataFrame}
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        entry_rule: MultiTFRule,
        exit_rule: MultiTFRule,
        dhan_client=None,
    ):
        self.start_date  = start_date
        self.end_date    = end_date
        self.entry_rule  = entry_rule
        self.exit_rule   = exit_rule
        self.dhan_client = dhan_client
        self._tf_data: Dict[str, pd.DataFrame] = {}

    def _needed_timeframes(self) -> set:
        """Collect all unique timeframes referenced in entry + exit rules."""
        tfs = set()
        for rule in (self.entry_rule, self.exit_rule):
            for cond in rule.conditions:
                if cond.left.kind != "value":
                    tfs.add(cond.left.timeframe)
                if cond.right.kind != "value":
                    tfs.add(cond.right.timeframe)
        return tfs

    def fetch_all_data(self, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Fetch and cache all required timeframe data."""
        tfs = self._needed_timeframes()
        result = {}
        total = len(tfs)
        for i, tf in enumerate(tfs):
            if progress_callback:
                progress_callback(i + 1, total, f"Fetching {tf} data...")

            if tf in DAILY_INTERVALS:
                df = fetch_nifty_daily(self.start_date, self.end_date, self.dhan_client)
                if tf == "1W":
                    df = _resample_ohlcv(df, RESAMPLE_RULES["1W"])
                elif tf == "1M":
                    df = _resample_ohlcv(df, RESAMPLE_RULES["1M"])
            else:
                interval_min = INTRADAY_INTERVALS.get(tf, 5)
                df = fetch_nifty_intraday(
                    self.start_date, self.end_date,
                    interval_min=interval_min,
                    dhan_client=self.dhan_client,
                )
                rule = RESAMPLE_RULES.get(tf)
                if rule and rule not in (None, "1m"):
                    df = _resample_ohlcv(df, rule)

            result[tf] = df
            print(f"[MultiTF] TF {tf}: {len(df)} bars fetched.")

        self._tf_data = result
        return result

    def run(
        self, progress_callback=None
    ) -> dict:
        """
        Fetch data and evaluate entry/exit signals.
        Returns dict with entry_signals, exit_signals, tf_data, date_index.
        """
        tf_data = self.fetch_all_data(progress_callback)

        # Build a daily date index from the 1D data (or longest available)
        daily_df = tf_data.get("1D") or fetch_nifty_daily(
            self.start_date, self.end_date, self.dhan_client
        )
        if daily_df.empty:
            raise RuntimeError("Failed to fetch NIFTY daily data.")

        date_index = daily_df.index

        entry_signals = evaluate_rule(self.entry_rule, tf_data, date_index)
        exit_signals  = evaluate_rule(self.exit_rule,  tf_data, date_index)

        return {
            "entry_signals": entry_signals,
            "exit_signals":  exit_signals,
            "tf_data":       tf_data,
            "date_index":    date_index,
            "daily_df":      daily_df,
        }
