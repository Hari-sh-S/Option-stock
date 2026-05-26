"""
core/signal_builder.py
Entry/Exit Signal Builder — rule-based condition engine.

Two usage modes:
  1. Programmatic: construct Condition / SignalRule objects directly
  2. UI-driven: serialize/deserialize rules as JSON for Streamlit widgets

Supported operators:
  >  <  >=  <=  ==  !=
  crosses_above  crosses_below   (detect one-bar crossovers)

Supported left/right operands:
  - Any column name present in the DataFrame (e.g. 'EMA_20', 'RSI_14', 'Close')
  - A numeric literal (e.g. 30, 0.5)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ─── Condition dataclass ──────────────────────────────────────────────────────

@dataclass
class Condition:
    """A single comparison condition applied row-by-row to a DataFrame."""
    left: str                   # column name  (e.g. 'RSI_14', 'Close')
    operator: str               # one of OPERATORS
    right: Union[str, float]    # column name  OR  numeric scalar

    OPERATORS = ['>', '<', '>=', '<=', '==', '!=', 'crosses_above', 'crosses_below']

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Condition':
        return cls(**d)


# ─── SignalRule dataclass ─────────────────────────────────────────────────────

@dataclass
class SignalRule:
    """
    A set of Conditions joined by AND/OR logic.
    Evaluates to a boolean Series when applied to a DataFrame.
    """
    conditions: List[Condition] = field(default_factory=list)
    logic: str = 'AND'          # 'AND' or 'OR'
    name: str = ''

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'logic': self.logic,
            'conditions': [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SignalRule':
        return cls(
            name=d.get('name', ''),
            logic=d.get('logic', 'AND'),
            conditions=[Condition.from_dict(c) for c in d.get('conditions', [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> 'SignalRule':
        return cls.from_dict(json.loads(s))

    def is_empty(self) -> bool:
        return len(self.conditions) == 0


# ─── SignalBuilder engine ─────────────────────────────────────────────────────

class SignalBuilder:
    """
    Evaluates entry/exit SignalRules against a prepared OHLCV+indicator DataFrame.

    Usage:
        sb = SignalBuilder(df)
        entry_series = sb.evaluate(entry_rule)  # pd.Series[bool]
        exit_series  = sb.evaluate(exit_rule)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(self, rule: SignalRule) -> pd.Series:
        """
        Apply a SignalRule to the DataFrame.
        Returns a boolean Series (True = condition met on that bar).
        """
        if rule.is_empty():
            # No conditions → always True (all bars trigger)
            return pd.Series(True, index=self.df.index)

        masks: List[pd.Series] = [self._eval_condition(c) for c in rule.conditions]

        if rule.logic == 'OR':
            result = masks[0].copy()
            for m in masks[1:]:
                result = result | m
        else:  # AND (default)
            result = masks[0].copy()
            for m in masks[1:]:
                result = result & m

        return result

    def _eval_condition(self, c: Condition) -> pd.Series:
        """Evaluate a single Condition against self.df."""
        left_series = self._resolve_operand(c.left)
        right_series = self._resolve_operand(c.right)

        if c.operator == '>':
            return left_series > right_series
        elif c.operator == '<':
            return left_series < right_series
        elif c.operator == '>=':
            return left_series >= right_series
        elif c.operator == '<=':
            return left_series <= right_series
        elif c.operator == '==':
            return left_series == right_series
        elif c.operator == '!=':
            return left_series != right_series
        elif c.operator == 'crosses_above':
            # Current bar: left > right; Previous bar: left <= right
            prev_left = left_series.shift(1)
            prev_right = right_series.shift(1) if isinstance(right_series, pd.Series) else right_series
            return (left_series > right_series) & (prev_left <= prev_right)
        elif c.operator == 'crosses_below':
            prev_left = left_series.shift(1)
            prev_right = right_series.shift(1) if isinstance(right_series, pd.Series) else right_series
            return (left_series < right_series) & (prev_left >= prev_right)
        else:
            raise ValueError(f"Unknown operator: {c.operator}")

    def _resolve_operand(self, operand: Union[str, float]) -> Union[pd.Series, float]:
        """Return column Series if operand is a column name, else return numeric."""
        if isinstance(operand, (int, float)):
            return operand
        if isinstance(operand, str):
            try:
                # Try parsing as float first
                return float(operand)
            except ValueError:
                pass
            if operand in self.df.columns:
                col = self.df[operand]
                return col.squeeze() if isinstance(col, pd.DataFrame) else col
            raise KeyError(
                f"Column '{operand}' not found in DataFrame. "
                f"Available: {list(self.df.columns[:20])}"
            )
        raise TypeError(f"Unsupported operand type: {type(operand)}")

    # ── Signal summary ────────────────────────────────────────────────────────

    def get_latest_signal(self, rule: SignalRule) -> bool:
        """Returns True if the rule fires on the most recent bar."""
        mask = self.evaluate(rule)
        return bool(mask.iloc[-1]) if len(mask) > 0 else False

    def get_signal_dates(self, rule: SignalRule) -> pd.DatetimeIndex:
        """Returns all dates where the rule fires."""
        mask = self.evaluate(rule)
        return mask[mask].index

    def get_entry_exit_series(
        self,
        entry_rule: SignalRule,
        exit_rule: SignalRule,
    ) -> pd.DataFrame:
        """
        Compute a position series from entry/exit rules.
        Returns a DataFrame with columns: ['Entry', 'Exit', 'Position']
          Position: 1 = in trade, 0 = flat
        """
        entry = self.evaluate(entry_rule).astype(int)
        exit_ = self.evaluate(exit_rule).astype(int)

        n = len(self.df)
        position = np.zeros(n, dtype=int)
        in_trade = False

        for i in range(n):
            if not in_trade and entry.iloc[i]:
                in_trade = True
            elif in_trade and exit_.iloc[i]:
                in_trade = False
            position[i] = 1 if in_trade else 0

        result = pd.DataFrame({
            'Entry_Signal': entry,
            'Exit_Signal': exit_,
            'Position': position,
        }, index=self.df.index)

        return result


# ─── Pre-built Strategy Templates ─────────────────────────────────────────────

STRATEGY_TEMPLATES: Dict[str, Dict[str, SignalRule]] = {

    "EMA Crossover (20/50)": {
        "entry": SignalRule(
            name="EMA 20/50 Crossover Entry",
            logic="AND",
            conditions=[
                Condition("EMA_20", "crosses_above", "EMA_50"),
            ],
        ),
        "exit": SignalRule(
            name="EMA 20/50 Crossover Exit",
            logic="AND",
            conditions=[
                Condition("EMA_20", "crosses_below", "EMA_50"),
            ],
        ),
    },

    "Golden / Death Cross (50/200)": {
        "entry": SignalRule(
            name="Golden Cross Entry",
            logic="AND",
            conditions=[
                Condition("EMA_50", "crosses_above", "EMA_200"),
            ],
        ),
        "exit": SignalRule(
            name="Death Cross Exit",
            logic="AND",
            conditions=[
                Condition("EMA_50", "crosses_below", "EMA_200"),
            ],
        ),
    },

    "RSI Reversal": {
        "entry": SignalRule(
            name="RSI Oversold Entry",
            logic="AND",
            conditions=[
                Condition("RSI_14", "<", 35),
                Condition("Close", ">", "EMA_200"),
            ],
        ),
        "exit": SignalRule(
            name="RSI Overbought Exit",
            logic="OR",
            conditions=[
                Condition("RSI_14", ">", 70),
            ],
        ),
    },

    "MACD Signal": {
        "entry": SignalRule(
            name="MACD Cross Up",
            logic="AND",
            conditions=[
                Condition("MACD", "crosses_above", "MACD_Signal"),
                Condition("MACD", "<", 0),   # bullish divergence zone
            ],
        ),
        "exit": SignalRule(
            name="MACD Cross Down",
            logic="AND",
            conditions=[
                Condition("MACD", "crosses_below", "MACD_Signal"),
            ],
        ),
    },

    "SuperTrend Flip": {
        "entry": SignalRule(
            name="SuperTrend Bullish",
            logic="AND",
            conditions=[
                Condition("Supertrend_Signal", "==", 1),
            ],
        ),
        "exit": SignalRule(
            name="SuperTrend Bearish",
            logic="AND",
            conditions=[
                Condition("Supertrend_Signal", "==", -1),
            ],
        ),
    },

    "Bollinger Band Mean Reversion": {
        "entry": SignalRule(
            name="BB Squeeze Entry",
            logic="AND",
            conditions=[
                Condition("Close", "<", "BB_Low"),
                Condition("RSI_14", "<", 40),
            ],
        ),
        "exit": SignalRule(
            name="BB Mean Reversion Exit",
            logic="OR",
            conditions=[
                Condition("Close", ">", "BB_Mid"),
            ],
        ),
    },

    "Momentum Breakout": {
        "entry": SignalRule(
            name="52W High Breakout",
            logic="AND",
            conditions=[
                Condition("Close", ">=", "52W_High"),
                Condition("Volume_Ratio", ">", 1.5),
            ],
        ),
        "exit": SignalRule(
            name="SuperTrend Bearish",
            logic="AND",
            conditions=[
                Condition("Supertrend_Signal", "==", -1),
            ],
        ),
    },

    "Nifty Call Buy (EMA + RSI)": {
        "entry": SignalRule(
            name="Nifty ATM Call Entry",
            logic="AND",
            conditions=[
                Condition("Close", ">", "EMA_20"),
                Condition("EMA_20", ">", "EMA_50"),
                Condition("RSI_14", ">", 50),
            ],
        ),
        "exit": SignalRule(
            name="Nifty Call Exit",
            logic="OR",
            conditions=[
                Condition("Close", "<", "EMA_20"),
                Condition("RSI_14", "<", 40),
            ],
        ),
    },

    "Nifty Put Buy (Bearish Momentum)": {
        "entry": SignalRule(
            name="Nifty ATM Put Entry",
            logic="AND",
            conditions=[
                Condition("Close", "<", "EMA_20"),
                Condition("EMA_20", "<", "EMA_50"),
                Condition("RSI_14", "<", 50),
            ],
        ),
        "exit": SignalRule(
            name="Nifty Put Exit",
            logic="OR",
            conditions=[
                Condition("Close", ">", "EMA_20"),
                Condition("RSI_14", ">", 60),
            ],
        ),
    },
}


# ─── Available indicator columns for UI dropdown ──────────────────────────────

AVAILABLE_INDICATORS = [
    # Price
    "Close", "Open", "High", "Low",
    # EMAs
    "EMA_9", "EMA_20", "EMA_50", "EMA_100", "EMA_200",
    # SMAs
    "SMA_20", "SMA_50", "SMA_100", "SMA_200",
    # Momentum
    "RSI_14", "MACD", "MACD_Signal", "MACD_Diff",
    # Volatility
    "BB_High", "BB_Low", "BB_Mid", "BB_Width", "BB_Pct", "ATR_14",
    # Trend
    "Supertrend", "Supertrend_Signal",
    # Levels
    "52W_High", "52W_Low",
    # Volume
    "Volume", "Volume_MA20", "Volume_Ratio",
    # ADX
    "ADX_14", "DI_Plus", "DI_Minus",
    # Stochastic
    "Stoch_K", "Stoch_D",
]

AVAILABLE_OPERATORS = [
    ">", "<", ">=", "<=", "==", "!=", "crosses_above", "crosses_below"
]
