"""Donchian / opening-range breakouts. Prior-window extrema only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import (
    atr,
    close_arr,
    hours_utc,
    prior_rolling_max,
    prior_rolling_min,
    sign_pos,
)
from src.signals.catalog.spec import RuleSpec, register

DONCHIAN_LB = [6, 8, 12, 16, 20, 24, 30, 36, 48, 60, 72, 96, 120]
ORB_HOURS = [1, 2, 3, 4]


def _donchian(
    df: pd.DataFrame,
    lookback: int,
    use: str = "hl",
    mode: str = "both",
    **_: object,
) -> np.ndarray:
    close = close_arr(df)
    if use == "hl":
        hi = prior_rolling_max(df["high"].to_numpy(dtype=np.float64), lookback)
        lo = prior_rolling_min(df["low"].to_numpy(dtype=np.float64), lookback)
    else:
        hi = prior_rolling_max(close, lookback)
        lo = prior_rolling_min(close, lookback)
    valid = np.isfinite(hi) & np.isfinite(lo)
    long = valid & (close > hi)
    short = valid & (close < lo)
    if mode == "long":
        short = np.zeros(len(close), dtype=bool)
    elif mode == "short":
        long = np.zeros(len(close), dtype=bool)
    return sign_pos(long, short)


def _donchian_atr(
    df: pd.DataFrame,
    lookback: int,
    atr_window: int = 14,
    q: float = 0.5,
    **_: object,
) -> np.ndarray:
    pos = _donchian(df, lookback=lookback, use="hl", mode="both")
    a = atr(df, atr_window)
    thresh = pd.Series(a).rolling(500, min_periods=100).median().to_numpy()
    quiet = np.isfinite(a) & np.isfinite(thresh) & (a < thresh)
    pos[quiet] = 0.0
    return pos


def _london_orb(df: pd.DataFrame, n_hours: int = 2, side: str = "both", **_: object) -> np.ndarray:
    """Break of the first n London hours' range (07:00 UTC), same calendar day."""
    ts = pd.to_datetime(df["time"], utc=True)
    day = ts.dt.floor("D")
    hour = hours_utc(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    n = len(df)
    pos = np.zeros(n, dtype=np.float64)
    in_orb = (hour >= 7) & (hour < 7 + int(n_hours))
    after = hour >= 7 + int(n_hours)
    # range from prior orb bars of the same day only
    orb_hi = np.full(n, np.nan)
    orb_lo = np.full(n, np.nan)
    for d, idx in pd.Series(np.arange(n)).groupby(day).groups.items():
        ix = np.asarray(idx)
        orb_ix = ix[in_orb[ix]]
        if orb_ix.size == 0:
            continue
        hi = float(np.max(high[orb_ix]))
        lo = float(np.min(low[orb_ix]))
        later = ix[after[ix]]
        orb_hi[later] = hi
        orb_lo[later] = lo
    valid = np.isfinite(orb_hi)
    long = valid & (close > orb_hi)
    short = valid & (close < orb_lo)
    if side == "long":
        short = np.zeros(n, dtype=bool)
    elif side == "short":
        long = np.zeros(n, dtype=bool)
    return sign_pos(long, short)


def register_all(hold: int = 12) -> None:
    for lb in DONCHIAN_LB:
        register(
            RuleSpec(
                rule_id=f"donchian_{lb}_h{hold}",
                family="breakout",
                hold=hold,
                fn=_donchian,
                kwargs={"lookback": lb, "use": "hl", "mode": "both"},
            )
        )
        register(
            RuleSpec(
                rule_id=f"donchian_close_{lb}_h{hold}",
                family="breakout",
                hold=hold,
                fn=_donchian,
                kwargs={"lookback": lb, "use": "close", "mode": "both"},
            )
        )
    for lb in (12, 24, 48, 72, 120):
        register(
            RuleSpec(
                rule_id=f"donchian_{lb}_longonly_h{hold}",
                family="breakout",
                hold=hold,
                fn=_donchian,
                kwargs={"lookback": lb, "use": "hl", "mode": "long"},
            )
        )
        register(
            RuleSpec(
                rule_id=f"donchian_{lb}_shortonly_h{hold}",
                family="breakout",
                hold=hold,
                fn=_donchian,
                kwargs={"lookback": lb, "use": "hl", "mode": "short"},
            )
        )
    for lb in (24, 48, 72):
        register(
            RuleSpec(
                rule_id=f"donchian_{lb}_atrhi_h{hold}",
                family="breakout",
                hold=hold,
                fn=_donchian_atr,
                kwargs={"lookback": lb, "atr_window": 14, "q": 0.5},
            )
        )
    for n in ORB_HOURS:
        for side in ("both", "long", "short"):
            register(
                RuleSpec(
                    rule_id=f"orb_london_{n}h_{side}_h{hold}",
                    family="breakout",
                    hold=hold,
                    fn=_london_orb,
                    kwargs={"n_hours": n, "side": side},
                )
            )
