"""Structure entries for the R-engine: trend / breakout only. First-bar fire."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.signals.catalog.helpers import (
    adx_di,
    atr,
    close_arr,
    ema,
    hours_utc,
    prior_rolling_max,
    prior_rolling_min,
    sma,
    supertrend,
)


def first_fire(level: np.ndarray) -> np.ndarray:
    """Keep only the first bar of a new non-zero run. Stops overtrading a held regime."""
    out = np.zeros(len(level), dtype=np.float64)
    prev = 0.0
    for i, v in enumerate(level):
        if v != 0.0 and v != prev:
            out[i] = v
        prev = v if v != 0.0 else prev
        if v == 0.0:
            prev = 0.0
    return out


def session_mask(df: pd.DataFrame, which: str) -> np.ndarray:
    h = hours_utc(df)
    if which == "ov":
        return (h >= 12) & (h < 16)
    if which == "lon":
        return (h >= 7) & (h < 16)
    if which == "ny":
        return (h >= 12) & (h < 21)
    if which == "lny":
        return (h >= 7) & (h < 21)
    return np.ones(len(df), dtype=bool)


def atr_above_median(a: np.ndarray, window: int = 500) -> np.ndarray:
    med = pd.Series(a).rolling(int(window), min_periods=max(int(window) // 5, 50)).median().to_numpy()
    return np.isfinite(a) & np.isfinite(med) & (a > med)


def _don(df: pd.DataFrame, lookback: int) -> np.ndarray:
    close = close_arr(df)
    hi = prior_rolling_max(df["high"].to_numpy(dtype=np.float64), lookback)
    lo = prior_rolling_min(df["low"].to_numpy(dtype=np.float64), lookback)
    ok = np.isfinite(hi) & np.isfinite(lo)
    lvl = np.zeros(len(close), dtype=np.float64)
    lvl[ok & (close > hi)] = 1.0
    lvl[ok & (close < lo)] = -1.0
    return lvl


def _keltner(df: pd.DataFrame, window: int = 20, mult: float = 2.0) -> np.ndarray:
    c = close_arr(df)
    mid = ema(c, window)
    band = float(mult) * atr(df, window)
    up = mid + band
    dn = mid - band
    ok = np.isfinite(up) & np.isfinite(dn)
    lvl = np.zeros(len(c), dtype=np.float64)
    lvl[ok & (c > up)] = 1.0
    lvl[ok & (c < dn)] = -1.0
    return lvl


def _ma_cross(df: pd.DataFrame, fast: int, slow: int) -> np.ndarray:
    c = close_arr(df)
    f = ema(c, fast)
    s = ema(c, slow)
    ok = np.isfinite(f) & np.isfinite(s)
    lvl = np.zeros(len(c), dtype=np.float64)
    lvl[ok & (f > s)] = 1.0
    lvl[ok & (f < s)] = -1.0
    return lvl


def _ema_pull_resume(df: pd.DataFrame, fast: int = 21, slow: int = 50) -> np.ndarray:
    """Trend continuation: fast>slow, close was below fast, now closes back above fast."""
    c = close_arr(df)
    f = ema(c, fast)
    s = ema(c, slow)
    ok = np.isfinite(f) & np.isfinite(s)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    prev_f = np.roll(f, 1)
    prev_f[0] = np.nan
    long = ok & (f > s) & (prev_c < prev_f) & (c > f)
    short = ok & (f < s) & (prev_c > prev_f) & (c < f)
    lvl = np.zeros(len(c), dtype=np.float64)
    lvl[long] = 1.0
    lvl[short] = -1.0
    return lvl


def _london_orb(df: pd.DataFrame, n_hours: int = 2) -> np.ndarray:
    ts = pd.to_datetime(df["time"], utc=True)
    day = ts.dt.floor("D")
    hour = hours_utc(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    n = len(df)
    lvl = np.zeros(n, dtype=np.float64)
    in_orb = (hour >= 7) & (hour < 7 + int(n_hours))
    after = hour >= 7 + int(n_hours)
    for _, idx in pd.Series(np.arange(n)).groupby(day).groups.items():
        ix = np.asarray(idx)
        orb_ix = ix[in_orb[ix]]
        if orb_ix.size == 0:
            continue
        oh = float(np.nanmax(high[orb_ix]))
        ol = float(np.nanmin(low[orb_ix]))
        fired = False
        for j in ix[after[ix]]:
            if fired:
                break
            if close[j] > oh:
                lvl[j] = 1.0
                fired = True
            elif close[j] < ol:
                lvl[j] = -1.0
                fired = True
    return lvl


def _st_flip(df: pd.DataFrame, aw: int = 10, mult: float = 3.0) -> np.ndarray:
    return supertrend(df, aw, mult)


def _adx_don(df: pd.DataFrame, lookback: int = 20, adx_min: float = 25.0) -> np.ndarray:
    don = _don(df, lookback)
    adx, pdi, mdi = adx_di(df, 14)
    ok = np.isfinite(adx) & (adx >= float(adx_min))
    long = ok & (don > 0) & (pdi > mdi)
    short = ok & (don < 0) & (mdi > pdi)
    lvl = np.zeros(len(don), dtype=np.float64)
    lvl[long] = 1.0
    lvl[short] = -1.0
    return lvl


ENTRY_FNS = {
    "don20": lambda df: _don(df, 20),
    "don55": lambda df: _don(df, 55),
    "kelt20": lambda df: _keltner(df, 20, 2.0),
    "ema12_48": lambda df: _ma_cross(df, 12, 48),
    "ema21_55": lambda df: _ma_cross(df, 21, 55),
    "pull21_50": lambda df: _ema_pull_resume(df, 21, 50),
    "st10_3": lambda df: _st_flip(df, 10, 3.0),
    "orb2": lambda df: _london_orb(df, 2),
    "adxdon20": lambda df: _adx_don(df, 20, 25.0),
}


@dataclass(frozen=True)
class RSpec:
    rule_id: str
    entry: str
    sl_mult: float
    rr: float
    session: str
    trail: bool
    max_hold: int = 72
    adx_min: float = 20.0
    risk_frac: float = 0.01


def list_rspecs() -> List[RSpec]:
    specs: List[RSpec] = []
    for entry in ENTRY_FNS:
        adx_min = 0.0 if entry == "orb2" else 20.0
        for sl in (1.0, 1.5, 2.0):
            for rr in (2.0, 2.5, 3.0):
                for sess in ("ov", "lny"):
                    for trail in (False, True):
                        sl_t = str(sl).replace(".", "p")
                        rr_t = str(rr).replace(".", "p")
                        tr = "trail" if trail else "fixed"
                        rid = f"r_{entry}_sl{sl_t}_rr{rr_t}_{sess}_{tr}"
                        specs.append(
                            RSpec(
                                rule_id=rid,
                                entry=entry,
                                sl_mult=sl,
                                rr=rr,
                                session=sess,
                                trail=trail,
                                adx_min=adx_min,
                            )
                        )
    return specs


def build_fire_map(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    a = atr(df, 14)
    vol_ok = atr_above_median(a, 500)
    adx, _, _ = adx_di(df, 14)
    struct = np.isfinite(adx) & (adx >= 20.0)
    out: Dict[str, np.ndarray] = {}
    for name, fn in ENTRY_FNS.items():
        raw = fn(df)
        fire = first_fire(raw)
        mask = vol_ok.copy()
        if name != "orb2":
            mask &= struct
        fire = fire.copy()
        fire[~mask] = 0.0
        out[name] = fire
    out["_atr"] = a
    out["_sess_ov"] = session_mask(df, "ov").astype(np.float64)
    out["_sess_lny"] = session_mask(df, "lny").astype(np.float64)
    return out


def fire_for(spec: RSpec, built: Dict[str, np.ndarray]) -> np.ndarray:
    fire = built[spec.entry].copy()
    sess = built["_sess_ov"] if spec.session == "ov" else built["_sess_lny"]
    fire[sess <= 0] = 0.0
    return fire
