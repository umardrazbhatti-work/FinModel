"""Declared ATR-regime filters on a fixed base-rule list. Not fitted on unseen."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import atr
from src.signals.catalog.spec import REGISTRY, RuleSpec, get_rule, positions_for, register

# Pre-declared bases (must exist before register_all).
VOL_BASES: List[str] = [
    "sma_12_h12",
    "sma_24_h12",
    "sma_48_h12",
    "sma_72_h12",
    "sma_120_h12",
    "ema_12_h12",
    "ema_24_h12",
    "ema_48_h12",
    "dual_sma_12_48_h12",
    "dual_sma_24_72_h12",
    "dual_ema_12_26_h12",
    "donchian_24_h12",
    "donchian_48_h12",
    "donchian_72_h12",
    "donchian_close_24_h12",
    "rsi14_30_70_h12",
    "rsi14_20_80_h12",
    "bb20_2p0_fade_h12",
    "z24_1p5_h12",
    "consec_3_rev_h12",
    "roc_12_k0_h12",
    "roc_24_k0_h12",
    "macd_12_26_9_h12",
    "sess_london_long_h12",
    "sess_ny_long_h12",
    "sess_london_L_asia_S_h12",
    "orb_london_2h_both_h12",
    "persist_h12_k2",
    "smafade_24_20p_h12",
    "dual_sma_8_24_h12",
    "donchian_12_h12",
    "ema_72_h12",
    "rsi7_30_70_h12",
    "bb20_1p5_fade_h12",
    "z48_2p0_h12",
    "hour_08_long_h12",
    "hour_14_long_h12",
    "dow_0_long_h12",
    "block_8_12_long_h12",
    "roc_8_k1_h12",
]


def _atr_regime_mask(df: pd.DataFrame, atr_window: int, regime: str) -> np.ndarray:
    a = atr(df, atr_window)
    med = pd.Series(a).rolling(500, min_periods=100).median().to_numpy()
    ok = np.isfinite(a) & np.isfinite(med)
    if regime == "high":
        return ok & (a >= med)
    return ok & (a < med)


def _filtered(
    df: pd.DataFrame,
    base_id: str,
    atr_window: int,
    regime: str,
    train_mask: np.ndarray | None = None,
    returns: np.ndarray | None = None,
    cost: float = 0.0001,
    seed: int = 42,
    **_: object,
) -> np.ndarray:
    spec = get_rule(base_id)
    if train_mask is None:
        train_mask = np.zeros(len(df), dtype=bool)
    if returns is None:
        returns = np.full(len(df), np.nan)
    pos = positions_for(
        spec, df, train_mask=train_mask, returns=returns, cost=cost, seed=seed
    ).copy()
    keep = _atr_regime_mask(df, atr_window, regime)
    pos[~keep] = 0.0
    return pos


def register_all(hold: int = 12) -> None:
    for base in VOL_BASES:
        if base not in REGISTRY:
            raise KeyError(f"vol base missing: {base}")
        for aw in (14, 24):
            for regime in ("high", "low"):
                register(
                    RuleSpec(
                        rule_id=f"vol_{regime}_atr{aw}__{base}",
                        family="vol",
                        hold=hold,
                        fn=_filtered,
                        needs_train=False,
                        kwargs={"base_id": base, "atr_window": aw, "regime": regime},
                    )
                )
