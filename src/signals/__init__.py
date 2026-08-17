"""Module 1 — Signal / Alpha."""

from src.signals.evaluate import signal_verdict
from src.signals.rules import CONTROL_RULES, RULE_SPECS, is_control, next_bar_simple_return

__all__ = [
    "CONTROL_RULES",
    "RULE_SPECS",
    "is_control",
    "next_bar_simple_return",
    "signal_verdict",
]
