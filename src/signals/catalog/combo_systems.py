"""Published multi-indicator recipes: squeeze, Hull, Vortex, Camarilla, trend+RSI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import (
    atr,
    close_arr,
    ema,
    hours_utc,
    rolling_std,
    rsi,
    sign_pos,
    sma,
    stochastic_k,
    supertrend,
)
from src.signals.catalog.spec import RuleSpec, register


def _wma(x: np.ndarray, window: int) -> np.ndarray:
    w = np.arange(1, int(window) + 1, dtype=np.float64)
    return (
        pd.Series(x)
        .rolling(int(window), min_periods=int(window))
        .apply(lambda v: float(np.dot(v, w) / w.sum()), raw=True)
        .to_numpy()
    )


def _hull(df: pd.DataFrame, window: int = 16, **_: object) -> np.ndarray:
    c = close_arr(df)
    n = int(window)
    half = max(n // 2, 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    raw = 2.0 * _wma(c, half) - _wma(c, n)
    hma = _wma(raw, sqrt_n)
    prev = np.roll(hma, 1)
    prev[0] = np.nan
    ok = np.isfinite(hma) & np.isfinite(prev)
    return sign_pos(ok & (hma > prev), ok & (hma < prev))


def _ttm_squeeze(df: pd.DataFrame, window: int = 20, **_: object) -> np.ndarray:
    """Long/short after BB was inside Keltner (squeeze on) then closes outside BB."""
    c = close_arr(df)
    mid = sma(c, window)
    bb = 2.0 * rolling_std(c, window)
    kel = 1.5 * atr(df, window)
    squeezed = np.isfinite(bb) & np.isfinite(kel) & (bb < kel)
    was = np.roll(squeezed, 1)
    was[0] = False
    released = was & (~squeezed)
    return sign_pos(released & (c > mid), released & (c < mid))


def _vortex(df: pd.DataFrame, period: int = 14, **_: object) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    prev_low = np.roll(low, 1)
    prev_high = np.roll(high, 1)
    prev_close = np.roll(close, 1)
    prev_low[0] = np.nan
    prev_high[0] = np.nan
    prev_close[0] = np.nan
    plus = np.abs(high - prev_low)
    minus = np.abs(low - prev_high)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    vp = pd.Series(plus).rolling(int(period), min_periods=int(period)).sum()
    vm = pd.Series(minus).rolling(int(period), min_periods=int(period)).sum()
    trs = pd.Series(tr).rolling(int(period), min_periods=int(period)).sum()
    vip = (vp / trs).to_numpy()
    vim = (vm / trs).to_numpy()
    ok = np.isfinite(vip) & np.isfinite(vim)
    return sign_pos(ok & (vip > vim), ok & (vim > vip))


def _camarilla(df: pd.DataFrame, mode: str = "fade", **_: object) -> np.ndarray:
    ts = pd.to_datetime(df["time"], utc=True)
    day = ts.dt.floor("D")
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    g = pd.DataFrame({"day": day, "h": high, "l": low, "c": close}).groupby("day", sort=True)
    daily = g.agg(h=("h", "max"), l=("l", "min"), c=("c", "last")).shift(1)
    rng = daily["h"] - daily["l"]
    r3 = daily["c"] + rng * 1.1 / 4.0
    s3 = daily["c"] - rng * 1.1 / 4.0
    r3v = np.asarray(day.map(r3), dtype=np.float64)
    s3v = np.asarray(day.map(s3), dtype=np.float64)
    ok = np.isfinite(r3v) & np.isfinite(s3v)
    if mode == "break":
        return sign_pos(ok & (close > r3v), ok & (close < s3v))
    return sign_pos(ok & (close < s3v), ok & (close > r3v))


def _sma200_rsi(df: pd.DataFrame, rsi_lo: float, rsi_hi: float, **_: object) -> np.ndarray:
    c = close_arr(df)
    m = sma(c, 200)
    r = rsi(c, 14)
    ok = np.isfinite(m) & np.isfinite(r)
    return sign_pos(ok & (c > m) & (r < rsi_lo), ok & (c < m) & (r > rsi_hi))


def _st_rsi(df: pd.DataFrame, st_n: int, st_m: float, rsi_mid: float = 50.0, **_: object) -> np.ndarray:
    t = supertrend(df, st_n, st_m)
    r = rsi(close_arr(df), 14)
    ok = np.isfinite(r)
    return sign_pos(ok & (t > 0) & (r > rsi_mid), ok & (t < 0) & (r < rsi_mid))


def _macdaddy(df: pd.DataFrame, **_: object) -> np.ndarray:
    """BabyPips MACD(addy)-style: MACD hist and stochastic agree."""
    c = close_arr(df)
    hist = (ema(c, 12) - ema(c, 26)) - ema(ema(c, 12) - ema(c, 26), 9)
    k = stochastic_k(df, 14, 3)
    ok = np.isfinite(hist) & np.isfinite(k)
    return sign_pos(ok & (hist > 0) & (k > 50.0), ok & (hist < 0) & (k < 50.0))


def _ny_box(df: pd.DataFrame, **_: object) -> np.ndarray:
    """Break of 12:00-14:00 UTC range into the NY afternoon (common NY ORB)."""
    ts = pd.to_datetime(df["time"], utc=True)
    day = ts.dt.floor("D")
    hour = hours_utc(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    n = len(df)
    bh = np.full(n, np.nan)
    bl = np.full(n, np.nan)
    in_orb = (hour >= 12) & (hour < 14)
    after = (hour >= 14) & (hour < 21)
    for _, idx in pd.Series(np.arange(n)).groupby(day.to_numpy()).groups.items():
        ix = np.asarray(idx)
        o = ix[in_orb[ix]]
        if o.size == 0:
            continue
        t = ix[after[ix]]
        bh[t] = float(np.max(high[o]))
        bl[t] = float(np.min(low[o]))
    ok = np.isfinite(bh)
    return sign_pos(ok & (close > bh), ok & (close < bl))


def register_all(hold: int = 12) -> None:
    for n in (9, 16, 21, 55):
        register(
            RuleSpec(
                rule_id=f"hull_{n}_h{hold}",
                family="combo",
                hold=hold,
                fn=_hull,
                kwargs={"window": n},
                note="Hull MA slope",
            )
        )
    register(
        RuleSpec(
            rule_id=f"ttm_squeeze_20_h{hold}",
            family="combo",
            hold=hold,
            fn=_ttm_squeeze,
            kwargs={"window": 20},
            note="TTM squeeze release",
        )
    )
    register(
        RuleSpec(
            rule_id=f"vortex_14_h{hold}",
            family="combo",
            hold=hold,
            fn=_vortex,
            kwargs={"period": 14},
        )
    )
    register(
        RuleSpec(
            rule_id=f"vortex_21_h{hold}",
            family="combo",
            hold=hold,
            fn=_vortex,
            kwargs={"period": 21},
        )
    )
    register(
        RuleSpec(
            rule_id=f"camarilla_fade_h{hold}",
            family="combo",
            hold=hold,
            fn=_camarilla,
            kwargs={"mode": "fade"},
            note="Camarilla S3/R3 fade",
        )
    )
    register(
        RuleSpec(
            rule_id=f"camarilla_brk_h{hold}",
            family="combo",
            hold=hold,
            fn=_camarilla,
            kwargs={"mode": "break"},
        )
    )
    register(
        RuleSpec(
            rule_id=f"sma200_rsi30_h{hold}",
            family="combo",
            hold=hold,
            fn=_sma200_rsi,
            kwargs={"rsi_lo": 30.0, "rsi_hi": 70.0},
            note="200 SMA filter + RSI 30/70",
        )
    )
    register(
        RuleSpec(
            rule_id=f"sma200_rsi40_h{hold}",
            family="combo",
            hold=hold,
            fn=_sma200_rsi,
            kwargs={"rsi_lo": 40.0, "rsi_hi": 60.0},
        )
    )
    register(
        RuleSpec(
            rule_id=f"st10_rsi_h{hold}",
            family="combo",
            hold=hold,
            fn=_st_rsi,
            kwargs={"st_n": 10, "st_m": 3.0, "rsi_mid": 50.0},
        )
    )
    register(
        RuleSpec(
            rule_id=f"macdaddy_h{hold}",
            family="combo",
            hold=hold,
            fn=_macdaddy,
            note="BabyPips MACD + stochastic agree",
        )
    )
    register(
        RuleSpec(
            rule_id=f"ny_orb_12_14_h{hold}",
            family="combo",
            hold=hold,
            fn=_ny_box,
            note="NY 12-14 UTC opening range break",
        )
    )
