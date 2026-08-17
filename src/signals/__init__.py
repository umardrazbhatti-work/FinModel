"""Module 1 — Signal / Alpha."""

from src.signals.evaluate import signal_verdict
from src.signals.labels import class_from_return, forward_simple_return
from src.signals.rules import CONTROL_RULES, RULE_SPECS, is_control, next_bar_simple_return
from src.signals.s2_eval import s2_verdict

__all__ = [
    "CONTROL_RULES",
    "RULE_SPECS",
    "class_from_return",
    "forward_simple_return",
    "is_control",
    "next_bar_simple_return",
    "s2_verdict",
    "signal_verdict",
]
