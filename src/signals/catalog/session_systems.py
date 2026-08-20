"""Session and level systems used in retail forex: Asian box, daily pivots, round numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import close_arr, hours_utc, sign_pos
from src.signals.catalog.spec import RuleSpec, register


def _asian_box(df: pd.DataFrame, asia_lo: int = 0, asia_hi: int = 7, trade_hi: int = 16, **_: object) -> np.ndarray:
    """Break of 00:00-07:00 UTC range during London (common London-breakout recipe)."""
    ts = pd.to_datetime(df["time"], utc=True)
    day = ts.dt.floor("D")
    hour = hours_utc(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    n = len(df)
    box_hi = np.full(n, np.nan)
    box_lo = np.full(n, np.nan)
    in_asia = (hour >= asia_lo) & (hour < asia_hi)
    in_trade = (hour >= asia_hi) & (hour < trade_hi)
    for _, idx in pd.Series(np.arange(n)).groupby(day.to_numpy()).groups.items():
        ix = np.asarray(idx)
        a = ix[in_asia[ix]]
        if a.size == 0:
            continue
        hi = float(np.max(high[a]))
        lo = float(np.min(low[a]))
        t = ix[in_trade[ix]]
        box_hi[t] = hi
        box_lo[t] = lo
    ok = np.isfinite(box_hi)
    return sign_pos(ok & (close > box_hi), ok & (close < box_lo))


def _daily_pivot(df: pd.DataFrame, mode: str = "trend", **_: object) -> np.ndarray:
    """Classic floor-trader pivots from the *previous* calendar day only."""
    ts = pd.to_datetime(df["time"], utc=True)
    day = ts.dt.floor("D")
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    g = pd.DataFrame({"day": day, "h": high, "l": low, "c": close}).groupby("day", sort=True)
    daily = g.agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    prev = daily.shift(1)
    p = (prev["h"] + prev["l"] + prev["c"]) / 3.0
    r1 = 2.0 * p - prev["l"]
    s1 = 2.0 * p - prev["h"]
    day_key = day
    pv = day_key.map(p)
    r1v = day_key.map(r1)
    s1v = day_key.map(s1)
    pv = np.asarray(pv, dtype=np.float64)
    r1v = np.asarray(r1v, dtype=np.float64)
    s1v = np.asarray(s1v, dtype=np.float64)
    ok = np.isfinite(pv)
    if mode == "fade":
        return sign_pos(ok & (close <= s1v), ok & (close >= r1v))
    return sign_pos(ok & (close > pv), ok & (close < pv))


def _round_number(df: pd.DataFrame, step: float = 0.0050, mode: str = "fade", **_: object) -> np.ndarray:
    """React to 50/00 figures (BabyPips Cowabunga targets used as fade/break levels)."""
    c = close_arr(df)
    nearest = np.round(c / step) * step
    dist = c - nearest
    atr_proxy = pd.Series(c).diff().abs().rolling(24, min_periods=24).mean().to_numpy()
    near = np.isfinite(atr_proxy) & (np.abs(dist) <= np.maximum(atr_proxy, step * 0.15))
    if mode == "break":
        return sign_pos(near & (dist > 0), near & (dist < 0))
    return sign_pos(near & (dist < 0), near & (dist > 0))


def _fifty_pip_london(df: pd.DataFrame, trigger: float = 0.0010, **_: object) -> np.ndarray:
    """50-pips-a-day cousin: after 07:00 UTC close, follow a 10+ pip break of that print."""
    ts = pd.to_datetime(df["time"], utc=True)
    day = ts.dt.floor("D")
    hour = hours_utc(df)
    close = close_arr(df)
    n = len(df)
    ref = np.full(n, np.nan)
    for _, idx in pd.Series(np.arange(n)).groupby(day.to_numpy()).groups.items():
        ix = np.asarray(idx)
        seven = ix[hour[ix] == 7]
        if seven.size == 0:
            continue
        r = float(close[seven[0]])
        later = ix[hour[ix] > 7]
        ref[later] = r
    ok = np.isfinite(ref)
    t = float(trigger)
    return sign_pos(ok & (close >= ref + t), ok & (close <= ref - t))


def register_all(hold: int = 12) -> None:
    register(
        RuleSpec(
            rule_id=f"asian_box_0007_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_asian_box,
            kwargs={"asia_lo": 0, "asia_hi": 7, "trade_hi": 16},
            note="Asian range 00-07 UTC, break during London",
        )
    )
    register(
        RuleSpec(
            rule_id=f"asian_box_0008_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_asian_box,
            kwargs={"asia_lo": 0, "asia_hi": 8, "trade_hi": 16},
        )
    )
    register(
        RuleSpec(
            rule_id=f"asian_box_0006_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_asian_box,
            kwargs={"asia_lo": 0, "asia_hi": 6, "trade_hi": 12},
        )
    )
    register(
        RuleSpec(
            rule_id=f"pivot_trend_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_daily_pivot,
            kwargs={"mode": "trend"},
            note="Prior-day classic pivot trend",
        )
    )
    register(
        RuleSpec(
            rule_id=f"pivot_fade_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_daily_pivot,
            kwargs={"mode": "fade"},
        )
    )
    register(
        RuleSpec(
            rule_id=f"round_50_fade_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_round_number,
            kwargs={"step": 0.0050, "mode": "fade"},
        )
    )
    register(
        RuleSpec(
            rule_id=f"round_00_fade_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_round_number,
            kwargs={"step": 0.0100, "mode": "fade"},
        )
    )
    register(
        RuleSpec(
            rule_id=f"round_50_brk_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_round_number,
            kwargs={"step": 0.0050, "mode": "break"},
        )
    )
    register(
        RuleSpec(
            rule_id=f"london_10pip_from_7utc_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_fifty_pip_london,
            kwargs={"trigger": 0.0010},
            note="Follow 10-pip break of 07:00 UTC close",
        )
    )
    register(
        RuleSpec(
            rule_id=f"london_20pip_from_7utc_h{hold}",
            family="session_sys",
            hold=hold,
            fn=_fifty_pip_london,
            kwargs={"trigger": 0.0020},
        )
    )
