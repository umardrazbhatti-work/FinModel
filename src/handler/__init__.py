"""Module 2 — Trade Handler (locked). Vol forecast → size. Never direction."""

from src.handler.sizing import SizingConfig, SizingResult, size_from_vol
from src.handler.volatility import (
    HANDLER_VERSION,
    MODULE_ID,
    HandlerDecision,
    VolatilityTradeHandler,
    VolForecast,
)

__all__ = [
    "HANDLER_VERSION",
    "MODULE_ID",
    "HandlerDecision",
    "SizingConfig",
    "SizingResult",
    "VolForecast",
    "VolatilityTradeHandler",
    "size_from_vol",
]
