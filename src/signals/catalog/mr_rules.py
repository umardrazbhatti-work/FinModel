"""Mean-reversion: RSI, Bollinger fade, z-score, consecutive-bar reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import close_arr, rolling_std, rsi, sign_pos, sma
from src.signals.catalog.spec import RuleSpec, register

RSI_PERIODS = [6, 7, 9, 14, 21, 28]
RSI_LEVELS = [(20, 80), (25, 75), (30, 70), (35, 65)]
BB_WINDOWS = [10, 20, 30, 50]
BB_K = [1.5, 2.0, 2.5]
Z_WINDOWS = [12, 24, 36, 48, 72]
Z_THRESH = [1.0, 1.5, 2.0]
CONSEC = [2, 3, 4, 5, 6]


def _rsi_fade(
    df: pd.DataFrame,
    period: int,
    lo: float,
    hi: float,
    mode: str = "both",
    **_: object,
) -> np.ndarray:
    r = rsi(close_arr(df), period)
    valid = np.isfinite(r)
    long = valid & (r < float(lo))
    short = valid & (r > float(hi))
    if mode == "long":
        short = np.zeros(len(r), dtype=bool)
    elif mode == "short":
        long = np.zeros(len(r), dtype=bool)
    return sign_pos(long, short)


def _bb_fade(df: pd.DataFrame, window: int, k: float, **_: object) -> np.ndarray:
    c = close_arr(df)
    mid = sma(c, window)
    sd = rolling_std(c, window)
    valid = np.isfinite(mid) & np.isfinite(sd) & (sd > 0)
    lower = mid - float(k) * sd
    upper = mid + float(k) * sd
    return sign_pos(valid & (c < lower), valid & (c > upper))


def _zscore_fade(df: pd.DataFrame, window: int, thresh: float, **_: object) -> np.ndarray:
    c = close_arr(df)
    mid = sma(c, window)
    sd = rolling_std(c, window)
    z = (c - mid) / sd
    valid = np.isfinite(z)
    t = float(thresh)
    return sign_pos(valid & (z < -t), valid & (z > t))


def _consec_reversal(df: pd.DataFrame, n_bars: int, **_: object) -> np.ndarray:
    c = close_arr(df)
    up = np.zeros(len(c), dtype=np.int32)
    down = np.zeros(len(c), dtype=np.int32)
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            up[i] = up[i - 1] + 1
            down[i] = 0
        elif c[i] < c[i - 1]:
            down[i] = down[i - 1] + 1
            up[i] = 0
    # fade after N completed bars (use prior streak so we do not peek)
    n = int(n_bars)
    prev_down = np.zeros(len(c), dtype=np.int32)
    prev_up = np.zeros(len(c), dtype=np.int32)
    prev_down[1:] = down[:-1]
    prev_up[1:] = up[:-1]
    return sign_pos(prev_down >= n, prev_up >= n)


def _sma_fade(df: pd.DataFrame, window: int, thresh: float, **_: object) -> np.ndarray:
    c = close_arr(df)
    mid = sma(c, window)
    gap = (c - mid) / np.maximum(c, 1e-12)
    valid = np.isfinite(gap)
    t = float(thresh)
    return sign_pos(valid & (gap < -t), valid & (gap > t))


def register_all(hold: int = 12) -> None:
    for p in RSI_PERIODS:
        for lo, hi in RSI_LEVELS:
            register(
                RuleSpec(
                    rule_id=f"rsi{p}_{int(lo)}_{int(hi)}_h{hold}",
                    family="mr",
                    hold=hold,
                    fn=_rsi_fade,
                    kwargs={"period": p, "lo": lo, "hi": hi, "mode": "both"},
                )
            )
    for p in (14, 21):
        register(
            RuleSpec(
                rule_id=f"rsi{p}_30_70_longonly_h{hold}",
                family="mr",
                hold=hold,
                fn=_rsi_fade,
                kwargs={"period": p, "lo": 30.0, "hi": 70.0, "mode": "long"},
            )
        )
        register(
            RuleSpec(
                rule_id=f"rsi{p}_30_70_shortonly_h{hold}",
                family="mr",
                hold=hold,
                fn=_rsi_fade,
                kwargs={"period": p, "lo": 30.0, "hi": 70.0, "mode": "short"},
            )
        )
    for w in BB_WINDOWS:
        for k in BB_K:
            tag = str(k).replace(".", "p")
            register(
                RuleSpec(
                    rule_id=f"bb{w}_{tag}_fade_h{hold}",
                    family="mr",
                    hold=hold,
                    fn=_bb_fade,
                    kwargs={"window": w, "k": k},
                )
            )
    for w in Z_WINDOWS:
        for t in Z_THRESH:
            tag = str(t).replace(".", "p")
            register(
                RuleSpec(
                    rule_id=f"z{w}_{tag}_h{hold}",
                    family="mr",
                    hold=hold,
                    fn=_zscore_fade,
                    kwargs={"window": w, "thresh": t},
                )
            )
    for n in CONSEC:
        register(
            RuleSpec(
                rule_id=f"consec_{n}_rev_h{hold}",
                family="mr",
                hold=hold,
                fn=_consec_reversal,
                kwargs={"n_bars": n},
            )
        )
    for w in (12, 24, 48, 72):
        for t in (0.001, 0.002, 0.003):
            tag = str(int(t * 10000))
            register(
                RuleSpec(
                    rule_id=f"smafade_{w}_{tag}p_h{hold}",
                    family="mr",
                    hold=hold,
                    fn=_sma_fade,
                    kwargs={"window": w, "thresh": t},
                )
            )
