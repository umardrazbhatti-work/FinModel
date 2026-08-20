"""Momentum / persist / MACD / ROC. Past returns only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import close_arr, ema, sign_pos
from src.signals.catalog.spec import RuleSpec, register
from src.signals.labels import persist_positions


ROC_WINDOWS = [4, 6, 8, 12, 16, 24, 36, 48, 72]
MACD = [(12, 26, 9), (8, 21, 5), (5, 35, 5), (16, 48, 12), (24, 52, 9)]


def _persist(df: pd.DataFrame, horizon: int, cost: float, k: float, **_: object) -> np.ndarray:
    return persist_positions(close_arr(df), int(horizon), float(cost), float(k))


def _roc(df: pd.DataFrame, window: int, k: float = 0.0, cost: float = 0.0001, **_: object) -> np.ndarray:
    c = close_arr(df)
    prev = np.roll(c, int(window))
    prev[: int(window)] = np.nan
    r = c / np.maximum(prev, 1e-12) - 1.0
    thresh = float(k) * float(cost)
    pos = np.zeros(len(c), dtype=np.float64)
    ok = np.isfinite(r)
    if thresh <= 0:
        pos[ok & (r > 0)] = 1.0
        pos[ok & (r < 0)] = -1.0
    else:
        pos[ok & (r >= thresh)] = 1.0
        pos[ok & (r <= -thresh)] = -1.0
    return pos


def _macd(df: pd.DataFrame, fast: int, slow: int, signal: int, **_: object) -> np.ndarray:
    c = close_arr(df)
    line = ema(c, fast) - ema(c, slow)
    sig = ema(line, signal)
    valid = np.isfinite(line) & np.isfinite(sig)
    return sign_pos(valid & (line > sig), valid & (line < sig))


def _roc_vs_roc(df: pd.DataFrame, fast: int, slow: int, **_: object) -> np.ndarray:
    c = close_arr(df)

    def roc(w: int) -> np.ndarray:
        prev = np.roll(c, int(w))
        out = c / np.maximum(prev, 1e-12) - 1.0
        out[: int(w)] = np.nan
        return out

    a = roc(fast)
    b = roc(slow)
    valid = np.isfinite(a) & np.isfinite(b)
    return sign_pos(valid & (a > b), valid & (a < b))


def register_all(hold: int = 12) -> None:
    for k in (0, 1, 2, 3):
        register(
            RuleSpec(
                rule_id=f"persist_h{hold}_k{k}",
                family="momentum",
                hold=hold,
                fn=_persist,
                kwargs={"horizon": hold, "k": float(k)},
            )
        )
    for w in ROC_WINDOWS:
        for k in (0, 1, 2):
            register(
                RuleSpec(
                    rule_id=f"roc_{w}_k{k}_h{hold}",
                    family="momentum",
                    hold=hold,
                    fn=_roc,
                    kwargs={"window": w, "k": float(k)},
                )
            )
    for fast, slow, sig in MACD:
        register(
            RuleSpec(
                rule_id=f"macd_{fast}_{slow}_{sig}_h{hold}",
                family="momentum",
                hold=hold,
                fn=_macd,
                kwargs={"fast": fast, "slow": slow, "signal": sig},
            )
        )
    for fast, slow in ((4, 12), (8, 24), (12, 48), (24, 72)):
        register(
            RuleSpec(
                rule_id=f"dualroc_{fast}_{slow}_h{hold}",
                family="momentum",
                hold=hold,
                fn=_roc_vs_roc,
                kwargs={"fast": fast, "slow": slow},
            )
        )
