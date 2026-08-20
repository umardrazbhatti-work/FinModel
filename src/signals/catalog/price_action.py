"""Price-action recipes: engulfing, pin bar, inside-bar break, NR7 break."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import close_arr, prior_rolling_max, prior_rolling_min, sign_pos
from src.signals.catalog.spec import RuleSpec, register


def _engulfing(df: pd.DataFrame, **_: object) -> np.ndarray:
    o = df["open"].to_numpy(dtype=np.float64)
    c = close_arr(df)
    prev_o = np.roll(o, 1)
    prev_c = np.roll(c, 1)
    prev_o[0] = np.nan
    prev_c[0] = np.nan
    prev_bear = prev_c < prev_o
    prev_bull = prev_c > prev_o
    curr_bull = c > o
    curr_bear = c < o
    bull = curr_bull & prev_bear & (c >= np.maximum(prev_o, prev_c)) & (o <= np.minimum(prev_o, prev_c))
    bear = curr_bear & prev_bull & (c <= np.minimum(prev_o, prev_c)) & (o >= np.maximum(prev_o, prev_c))
    return sign_pos(bull, bear)


def _pinbar(df: pd.DataFrame, wick_mult: float = 2.0, **_: object) -> np.ndarray:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = close_arr(df)
    body = np.abs(c - o)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    rng = np.maximum(h - l, 1e-12)
    bull = (lower >= float(wick_mult) * np.maximum(body, 1e-12)) & (lower > upper) & ((c - l) / rng > 0.6)
    bear = (upper >= float(wick_mult) * np.maximum(body, 1e-12)) & (upper > lower) & ((h - c) / rng > 0.6)
    return sign_pos(bull, bear)


def _inside_break(df: pd.DataFrame, **_: object) -> np.ndarray:
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = close_arr(df)
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_h[0] = np.nan
    prev_l[0] = np.nan
    inside = (h <= prev_h) & (l >= prev_l)
    # break of the inside bar's *prior* mother bar on this close
    mother_h = np.where(inside, prev_h, np.nan)
    mother_l = np.where(inside, prev_l, np.nan)
    # hold mother levels forward one bar? trade the close that *is* the inside
    # Standard: after an inside bar, next bar break. Use previous-bar inside.
    was_inside = np.roll(inside, 1)
    was_inside[0] = False
    mh = np.roll(prev_h, 1)
    ml = np.roll(prev_l, 1)
    mh[0] = np.nan
    ml[0] = np.nan
    ok = was_inside & np.isfinite(mh)
    return sign_pos(ok & (c > mh), ok & (c < ml))


def _nr7_break(df: pd.DataFrame, n: int = 7, **_: object) -> np.ndarray:
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = close_arr(df)
    rng = h - l
    min_prev = prior_rolling_min(rng, n)
    nr = np.isfinite(min_prev) & (rng <= min_prev)
    was = np.roll(nr, 1)
    was[0] = False
    hh = prior_rolling_max(h, 1)
    ll = prior_rolling_min(l, 1)
    # break yesterday's NR bar
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    return sign_pos(was & (c > prev_h), was & (c < prev_l))


def register_all(hold: int = 12) -> None:
    register(RuleSpec(rule_id=f"pa_engulfing_h{hold}", family="price_action", hold=hold, fn=_engulfing))
    register(
        RuleSpec(
            rule_id=f"pa_pinbar_h{hold}",
            family="price_action",
            hold=hold,
            fn=_pinbar,
            kwargs={"wick_mult": 2.0},
        )
    )
    register(
        RuleSpec(
            rule_id=f"pa_pinbar_3x_h{hold}",
            family="price_action",
            hold=hold,
            fn=_pinbar,
            kwargs={"wick_mult": 3.0},
        )
    )
    register(RuleSpec(rule_id=f"pa_inside_brk_h{hold}", family="price_action", hold=hold, fn=_inside_break))
    register(
        RuleSpec(
            rule_id=f"pa_nr7_brk_h{hold}",
            family="price_action",
            hold=hold,
            fn=_nr7_break,
            kwargs={"n": 7},
            note="NR7 range contraction break",
        )
    )
    register(
        RuleSpec(
            rule_id=f"pa_nr4_brk_h{hold}",
            family="price_action",
            hold=hold,
            fn=_nr7_break,
            kwargs={"n": 4},
        )
    )
