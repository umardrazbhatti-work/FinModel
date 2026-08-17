"""Inverse-vol sizing for the locked Trade Handler (Module 2).

Never chooses direction. Maps a vol forecast to a size multiplier and a
stand-aside flag so a Signal can be throttled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SizingConfig:
    min_multiplier: float = 0.0
    max_multiplier: float = 2.0
    max_log_width: float = 1.5
    max_rv: Optional[float] = None
    min_ref_rv: float = 1e-8
    eps: float = 1e-12


@dataclass(frozen=True)
class SizingResult:
    size_multiplier: float
    stand_aside: bool
    reason: str
    forecast_rv: float
    ref_rv: float
    log_width: float


def size_from_vol(
    log_q10: float,
    log_q50: float,
    log_q90: float,
    ref_rv: float,
    cfg: Optional[SizingConfig] = None,
) -> SizingResult:
    """Map log-RV quantiles + a reference RV to a size multiplier.

    ``size_multiplier = clip(ref_rv / exp(q50), min, max)``.
    Stand aside if the q90–q10 width is too large, forecast RV is capped,
    or the multiplier collapses to 0.
    """
    cfg = cfg or SizingConfig()
    log_q10 = float(log_q10)
    log_q50 = float(log_q50)
    log_q90 = float(log_q90)
    ref = max(float(ref_rv), cfg.min_ref_rv)
    log_width = log_q90 - log_q10
    forecast_rv = float(np.exp(log_q50))

    if not np.isfinite(forecast_rv) or forecast_rv <= cfg.eps:
        return SizingResult(0.0, True, "non_finite_or_zero_forecast_rv", forecast_rv, ref, log_width)
    if not np.isfinite(log_width) or log_width < 0:
        return SizingResult(0.0, True, "invalid_quantile_order", forecast_rv, ref, log_width)
    if log_width > cfg.max_log_width:
        return SizingResult(0.0, True, "uncertainty_width", forecast_rv, ref, log_width)
    if cfg.max_rv is not None and forecast_rv > cfg.max_rv:
        return SizingResult(0.0, True, "forecast_rv_cap", forecast_rv, ref, log_width)

    raw = ref / max(forecast_rv, cfg.eps)
    mult = float(np.clip(raw, cfg.min_multiplier, cfg.max_multiplier))
    if mult <= cfg.eps:
        return SizingResult(0.0, True, "zero_multiplier", forecast_rv, ref, log_width)
    return SizingResult(mult, False, "ok", forecast_rv, ref, log_width)


def realized_vol_from_closes(closes: np.ndarray, eps: float = 1e-12) -> float:
    """Realized vol of log-returns on a close series (handler reference)."""
    closes = np.asarray(closes, dtype=np.float64)
    if closes.size < 2:
        return float("nan")
    rets = np.diff(np.log(np.maximum(closes, eps)))
    return float(np.sqrt(np.mean(np.square(rets)) + eps))


def quantile_index(quantiles: Tuple[float, ...] | list, q: float) -> int:
    qs = [float(x) for x in quantiles]
    if q not in qs:
        raise ValueError(f"quantile {q} not in {qs}")
    return qs.index(q)
