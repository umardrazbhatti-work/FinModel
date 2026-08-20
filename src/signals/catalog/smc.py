"""Leakage-safe SMC / ICT / Raja-Banks-style structure.

Public recipes (mechanical stand-ins, no lookahead):
- FVG: 3-candle imbalance; long/short on retrace into the last unfilled gap
- Order block: last opposite candle before an ATR displacement
- Liquidity sweep: wick through prior N-bar high/low, close back inside
- BOS: close beyond prior N-bar swing
- CHoCH: BOS against the last BOS direction
- OTE: 62-79% retrace of last N-bar impulse, trade with impulse
- Killzones: London 07-10 UTC, NY 13-16 UTC, Silver Bullet 14-15 / 15-16 UTC
- Raja Banks public clips teach SMC structure (BOS/CHoCH + OB/FVG in session).
  Encoded as those confluences, not a licensed product.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import atr, close_arr, hours_utc, prior_rolling_max, prior_rolling_min, sign_pos
from src.signals.catalog.spec import RuleSpec, register

KZ = {
    "any": (0, 24),
    "london": (7, 10),
    "ny": (13, 16),
    "sb14": (14, 15),
    "sb15": (15, 16),
    "asia": (0, 7),
}


def _kz_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    lo, hi = KZ[name]
    h = hours_utc(df)
    if lo <= hi:
        return (h >= lo) & (h < hi)
    return (h >= lo) | (h < hi)


def _apply_kz_side(pos: np.ndarray, df: pd.DataFrame, kz: str, side: str) -> np.ndarray:
    out = pos.copy()
    if kz != "any":
        out[~_kz_mask(df, kz)] = 0.0
    if side == "long":
        out[out < 0] = 0.0
    elif side == "short":
        out[out > 0] = 0.0
    return out


def _fvg_retrace(df: pd.DataFrame, max_age: int = 24, **_: object) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    n = len(df)
    pos = np.zeros(n, dtype=np.float64)
    b_lo = b_hi = s_lo = s_hi = np.nan
    b_i = s_i = -10**9
    age = int(max_age)
    for i in range(2, n):
        if i - b_i <= age and np.isfinite(b_lo):
            if low[i] <= b_lo:
                b_i = -10**9
            elif low[i] <= b_hi and close[i] >= b_lo:
                pos[i] = 1.0
        if i - s_i <= age and np.isfinite(s_hi):
            if high[i] >= s_hi:
                s_i = -10**9
            elif high[i] >= s_lo and close[i] <= s_hi:
                pos[i] = -1.0
        if low[i] > high[i - 2]:
            b_lo, b_hi, b_i = high[i - 2], low[i], i
        if high[i] < low[i - 2]:
            s_lo, s_hi, s_i = high[i], low[i - 2], i
    return pos


def _sweep(df: pd.DataFrame, lookback: int = 13, **_: object) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    ph = prior_rolling_max(high, lookback)
    pl = prior_rolling_min(low, lookback)
    ok = np.isfinite(ph) & np.isfinite(pl)
    # sweep high then close back below = short; sweep low then close back above = long
    return sign_pos(ok & (low < pl) & (close > pl), ok & (high > ph) & (close < ph))


def _bos_follow(df: pd.DataFrame, lookback: int = 13, **_: object) -> np.ndarray:
    close = close_arr(df)
    ph = prior_rolling_max(df["high"].to_numpy(dtype=np.float64), lookback)
    pl = prior_rolling_min(df["low"].to_numpy(dtype=np.float64), lookback)
    ok = np.isfinite(ph) & np.isfinite(pl)
    return sign_pos(ok & (close > ph), ok & (close < pl))


def _choch(df: pd.DataFrame, lookback: int = 13, **_: object) -> np.ndarray:
    """Trade the flip: after a BOS, the opposite BOS is CHoCH -> follow new side."""
    bos = _bos_follow(df, lookback)
    last = 0.0
    pos = np.zeros(len(bos))
    for i, b in enumerate(bos):
        if b != 0 and b != last:
            if last != 0:
                pos[i] = b
            last = b
    return pos


def _ote(df: pd.DataFrame, lookback: int = 24, lo: float = 0.62, hi: float = 0.79, **_: object) -> np.ndarray:
    close = close_arr(df)
    ph = prior_rolling_max(df["high"].to_numpy(dtype=np.float64), lookback)
    pl = prior_rolling_min(df["low"].to_numpy(dtype=np.float64), lookback)
    rng = ph - pl
    ok = np.isfinite(rng) & (rng > 0)
    # impulse up if last close > mid of prior range; buy OTE pullback
    retr_from_high = (ph - close) / rng
    retr_from_low = (close - pl) / rng
    long = ok & (close > (pl + 0.5 * rng)) & (retr_from_high >= lo) & (retr_from_high <= hi)
    short = ok & (close < (pl + 0.5 * rng)) & (retr_from_low >= lo) & (retr_from_low <= hi)
    return sign_pos(long, short)


def _order_block(df: pd.DataFrame, disp: float = 1.5, atr_n: int = 14, **_: object) -> np.ndarray:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = close_arr(df)
    a = atr(df, atr_n)
    n = len(c)
    pos = np.zeros(n)
    ob_lo = ob_hi = np.nan
    ob_side = 0.0
    for i in range(1, n):
        if np.isfinite(ob_lo) and ob_side != 0:
            if ob_side > 0 and l[i] <= ob_hi and c[i] >= ob_lo:
                pos[i] = 1.0
            elif ob_side < 0 and h[i] >= ob_lo and c[i] <= ob_hi:
                pos[i] = -1.0
            if (ob_side > 0 and c[i] < ob_lo) or (ob_side < 0 and c[i] > ob_hi):
                ob_side = 0.0
        if not np.isfinite(a[i]):
            continue
        body = c[i] - o[i]
        if body > float(disp) * a[i]:
            # last bearish candle before this bullish displacement
            j = i - 1
            while j >= 0 and c[j] >= o[j]:
                j -= 1
            if j >= 0:
                ob_lo, ob_hi, ob_side = l[j], h[j], 1.0
        elif body < -float(disp) * a[i]:
            j = i - 1
            while j >= 0 and c[j] <= o[j]:
                j -= 1
            if j >= 0:
                ob_lo, ob_hi, ob_side = l[j], h[j], -1.0
    return pos


def _pd_array(df: pd.DataFrame, lookback: int = 48, **_: object) -> np.ndarray:
    """Discount = buy below 50% of last swing; premium = sell above."""
    close = close_arr(df)
    ph = prior_rolling_max(df["high"].to_numpy(dtype=np.float64), lookback)
    pl = prior_rolling_min(df["low"].to_numpy(dtype=np.float64), lookback)
    mid = (ph + pl) / 2.0
    ok = np.isfinite(mid)
    return sign_pos(ok & (close < mid), ok & (close > mid))


def _wrap(fn, kz: str, side: str, **params):
    def _inner(df, **kw):
        pos = fn(df, **params)
        return _apply_kz_side(pos, df, kz, side)

    return _inner


def register_all(hold: int = 12) -> None:
    ages = (8, 12, 24, 48)
    sweeps = (5, 8, 13, 21)
    kzs = ("any", "london", "ny", "sb14", "sb15")
    sides = ("both", "long", "short")

    for age in ages:
        for kz in kzs:
            for side in sides:
                register(
                    RuleSpec(
                        rule_id=f"smc_fvg_a{age}_{kz}_{side}_h{hold}",
                        family="smc",
                        hold=hold,
                        fn=_wrap(_fvg_retrace, kz, side, max_age=age),
                        note="SMC FVG retrace",
                    )
                )
    for lb in sweeps:
        for kz in kzs:
            for side in sides:
                register(
                    RuleSpec(
                        rule_id=f"smc_sweep_{lb}_{kz}_{side}_h{hold}",
                        family="smc",
                        hold=hold,
                        fn=_wrap(_sweep, kz, side, lookback=lb),
                        note="SMC liquidity sweep",
                    )
                )
                register(
                    RuleSpec(
                        rule_id=f"smc_bos_{lb}_{kz}_{side}_h{hold}",
                        family="smc",
                        hold=hold,
                        fn=_wrap(_bos_follow, kz, side, lookback=lb),
                        note="SMC BOS follow",
                    )
                )
                register(
                    RuleSpec(
                        rule_id=f"smc_choch_{lb}_{kz}_{side}_h{hold}",
                        family="smc",
                        hold=hold,
                        fn=_wrap(_choch, kz, side, lookback=lb),
                        note="SMC CHoCH",
                    )
                )
    for disp in (1.0, 1.5, 2.0):
        for kz in ("any", "london", "ny"):
            for side in sides:
                tag = str(disp).replace(".", "p")
                register(
                    RuleSpec(
                        rule_id=f"smc_ob_{tag}_{kz}_{side}_h{hold}",
                        family="smc",
                        hold=hold,
                        fn=_wrap(_order_block, kz, side, disp=disp),
                        note="SMC order block retest",
                    )
                )
    for lb in (13, 24, 48):
        for lo, hi in ((0.62, 0.79), (0.705, 0.79), (0.50, 0.705)):
            for kz in ("any", "london", "ny"):
                tag = f"{int(lo*1000)}_{int(hi*1000)}"
                register(
                    RuleSpec(
                        rule_id=f"smc_ote_{lb}_{tag}_{kz}_h{hold}",
                        family="smc",
                        hold=hold,
                        fn=_wrap(_ote, kz, "both", lookback=lb, lo=lo, hi=hi),
                        note="ICT OTE",
                    )
                )
    for lb in (24, 48, 72):
        for kz in kzs:
            register(
                RuleSpec(
                    rule_id=f"smc_pd_{lb}_{kz}_h{hold}",
                    family="smc",
                    hold=hold,
                    fn=_wrap(_pd_array, kz, "both", lookback=lb),
                    note="Premium/discount",
                )
            )

    # Raja Banks public SMC structure: CHoCH + killzone, sweep+FVG-style via sweep in KZ, OB in KZ
    for lb in (8, 13, 21):
        for kz in ("london", "ny", "sb14"):
            register(
                RuleSpec(
                    rule_id=f"raja_choch_{lb}_{kz}_h{hold}",
                    family="raja_banks",
                    hold=hold,
                    fn=_wrap(_choch, kz, "both", lookback=lb),
                    note="Raja Banks / SMC structure CHoCH in killzone",
                )
            )
            register(
                RuleSpec(
                    rule_id=f"raja_sweep_{lb}_{kz}_h{hold}",
                    family="raja_banks",
                    hold=hold,
                    fn=_wrap(_sweep, kz, "both", lookback=lb),
                    note="Raja Banks / SMC sweep in killzone",
                )
            )
    for disp in (1.5, 2.0):
        for kz in ("london", "ny"):
            tag = str(disp).replace(".", "p")
            register(
                RuleSpec(
                    rule_id=f"raja_ob_{tag}_{kz}_h{hold}",
                    family="raja_banks",
                    hold=hold,
                    fn=_wrap(_order_block, kz, "both", disp=disp),
                    note="Raja Banks / SMC OB in killzone",
                )
            )
