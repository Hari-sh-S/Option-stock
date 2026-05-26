"""
Option-Stock Scanner — Core Business Logic Package
Framework-agnostic: compatible with Streamlit and future Plotly Dash migration.
"""
from .indicators import IndicatorLibrary
from .scoring import ScoreParser
from .universe import UniverseManager
from .signal_builder import SignalBuilder, Condition, SignalRule, STRATEGY_TEMPLATES
from .scanner import Scanner

__all__ = [
    "IndicatorLibrary",
    "ScoreParser",
    "UniverseManager",
    "SignalBuilder",
    "Condition",
    "SignalRule",
    "STRATEGY_TEMPLATES",
    "Scanner",
]
