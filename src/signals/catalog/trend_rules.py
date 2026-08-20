"""Trend / drift: SMA, EMA, dual crossovers. Past-only rolling windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import close_arr, ema, sign_pos, sma
from src.signals.catalog.spec import RuleSpec, register

SMA_WINDOWS = [6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 60, 72, 84, 96, 108, 120, 144, 168, 192]
EMA_WINDOWS = [6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 168]
DUAL_SMA = [
    (4, 24),
    (6, 24),
    (8, 24),
    (8, 48),
    (12, 24),
    (12, 48),
    (12, 72),
    (16, 48),
    (16, 72),
    (24, 48),
    (24, 72),
    (24, 96),
    (24, 120),
    (24, 168),
    (36, 96),
    (36, 120),
    (48, 120),
    (48, 168),
    (12, 36),
    (8, 36),
]
DUAL_EMA = [
    (8, 24),
    (12, 26),
    (12, 48),
    (16, 48),
    (24, 72),
    (24, 120),
    (8, 48),
    (12, 72),
]


def _vs_ma(df: pd.DataFrame, window: int, kind: str = "sma", mode: str = "both", **_: object) -> np.ndarray:
    c = close_arr(df)
    ma = sma(c, window) if kind == "sma" else ema(c, window)
    valid = np.isfinite(ma)
    long = valid & (c > ma)
    short = valid & (c < ma)
    if mode == "long":
        short = np.zeros(len(c), dtype=bool)
    elif mode == "short":
        long = np.zeros(len(c), dtype=bool)
    return sign_pos(long, short)


def _dual_ma(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    kind: str = "sma",
    **_: object,
) -> np.ndarray:
    c = close_arr(df)
    f = sma(c, fast) if kind == "sma" else ema(c, fast)
    s = sma(c, slow) if kind == "sma" else ema(c, slow)
    valid = np.isfinite(f) & np.isfinite(s)
    return sign_pos(valid & (f > s), valid & (f < s))


def register_all(hold: int = 12) -> None:
    for w in SMA_WINDOWS:
        register(
            RuleSpec(
                rule_id=f"sma_{w}_h{hold}",
                family="trend",
                hold=hold,
                fn=_vs_ma,
                kwargs={"window": w, "kind": "sma", "mode": "both"},
            )
        )
    for w in EMA_WINDOWS:
        register(
            RuleSpec(
                rule_id=f"ema_{w}_h{hold}",
                family="trend",
                hold=hold,
                fn=_vs_ma,
                kwargs={"window": w, "kind": "ema", "mode": "both"},
            )
        )
    for w in (12, 24, 48, 72, 120):
        register(
            RuleSpec(
                rule_id=f"sma_{w}_longonly_h{hold}",
                family="trend",
                hold=hold,
                fn=_vs_ma,
                kwargs={"window": w, "kind": "sma", "mode": "long"},
            )
        )
        register(
            RuleSpec(
                rule_id=f"sma_{w}_shortonly_h{hold}",
                family="trend",
                hold=hold,
                fn=_vs_ma,
                kwargs={"window": w, "kind": "sma", "mode": "short"},
            )
        )
    for fast, slow in DUAL_SMA:
        register(
            RuleSpec(
                rule_id=f"dual_sma_{fast}_{slow}_h{hold}",
                family="trend",
                hold=hold,
                fn=_dual_ma,
                kwargs={"fast": fast, "slow": slow, "kind": "sma"},
            )
        )
    for fast, slow in DUAL_EMA:
        register(
            RuleSpec(
                rule_id=f"dual_ema_{fast}_{slow}_h{hold}",
                family="trend",
                hold=hold,
                fn=_dual_ma,
                kwargs={"fast": fast, "slow": slow, "kind": "ema"},
            )
        )
