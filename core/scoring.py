"""
core/scoring.py
Extended ScoreParser — adds Entry Signal Strength metric support
on top of the original scoring.py.

Fully backward-compatible with portfolio_engine.py.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Pull original ScoreParser so we extend it
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scoring import ScoreParser as _BaseScoreParser


class ScoreParser(_BaseScoreParser):
    """
    Extended ScoreParser — all original functionality preserved, plus:
      - 'Entry Signal Strength' metric support
      - Options-mode metrics (placeholder for IV Rank etc. via Dhan API)
      - `get_scanner_scoring_options()` for the Scanner UI
    """

    # Additional metric names supported in scanner context
    SCANNER_EXTRA_METRICS = [
        "Entry Signal Strength",       # % of last 20 bars where entry fired (0-100)
        "Days Since Entry",            # Recency — lower = fresher signal
    ]

    def __init__(self):
        super().__init__()

        # Extend metric_groups with scanner extras
        self.metric_groups["Signal Quality"] = [
            "Entry Signal Strength",
            "Days Since Entry",
        ]

        # Extend allowed_metrics
        self.allowed_metrics.extend(self.SCANNER_EXTRA_METRICS)

    def parse_and_calculate_scanner(self, formula: str, row: dict) -> float:
        """
        Like parse_and_calculate but also handles scanner-extra metrics
        (Entry Signal Strength, Days Since Entry).
        """
        # Inject extra metrics as numeric literals before evaluation
        processed = formula

        for metric in self.SCANNER_EXTRA_METRICS:
            val = row.get(metric, 0)
            if isinstance(val, pd.Series):
                val = val.iloc[0] if len(val) > 0 else 0
            if pd.isna(val) or val is None:
                val = 0
            # Escape metric name for regex substitution
            escaped = re.escape(metric)
            processed = re.sub(escaped, str(float(val)), processed)

        return self.parse_and_calculate(processed, row)

    def get_scanner_scoring_options(self) -> dict:
        """
        Returns scoring metric options for the Scanner UI.
        Combines existing momentum metrics with new signal quality metrics.
        """
        return {
            "Performance": [
                "1 Month Performance",
                "3 Month Performance",
                "6 Month Performance",
                "1 Year Performance",
            ],
            "Risk-Adjusted": [
                "3 Month Sharpe",
                "6 Month Sharpe",
                "1 Year Sharpe",
                "3 Month Sortino",
                "6 Month Calmar",
            ],
            "Volatility (lower=better)": [
                "1 Month Volatility",
                "3 Month Volatility",
                "6 Month Volatility",
            ],
            "Drawdown (lower=better)": [
                "3 Month Max Drawdown",
                "6 Month Max Drawdown",
                "1 Year Max Drawdown",
            ],
            "Signal Quality": [
                "Entry Signal Strength",
            ],
            "Price Levels": [
                "1 Year Distance From High",
                "6 Month Distance From Low",
            ],
        }

    def build_weighted_score(self, df: pd.DataFrame, weights: dict) -> pd.Series:
        """
        Build a 0–100 composite score from weighted percentile ranks.

        Args:
            df      : Scanner results DataFrame
            weights : {metric_name: (weight, 'asc'|'desc')}
                      'asc' = higher metric = better rank
                      'desc' = lower metric = better rank

        Returns:
            pd.Series of scores (0–100)
        """
        total_weight = sum(w for w, _ in weights.values())
        score = pd.Series(0.0, index=df.index)

        for metric, (weight, direction) in weights.items():
            if metric not in df.columns:
                continue
            raw = df[metric].fillna(0)
            if direction == "asc":
                ranked = raw.rank(pct=True) * 100
            else:
                ranked = (1 - raw.rank(pct=True)) * 100
            score += (weight / total_weight) * ranked

        return score.round(1)
