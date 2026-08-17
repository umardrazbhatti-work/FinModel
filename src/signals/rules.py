"""Module 1 / S-1 — explicit long/short/flat rules. No neural net.

Position at bar t uses only information available at t.
The return scored is the next bar (t → t+1).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# +1 long, -1 short, 0 flat
PosFn = Callable[..., np.ndarray]

CONTROL_RULES = ("always_flat", "always_long", "coin_flip")


def next_bar_simple_return(close: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    out = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) > 1:
        prev = np.maximum(close[:-1], eps)
        out[:-1] = close[1:] / prev - 1.0
    return out


def always_flat(df: pd.DataFrame, **_: object) -> np.ndarray:
    return np.zeros(len(df), dtype=np.float64)


def always_long(df: pd.DataFrame, **_: object) -> np.ndarray:
    return np.ones(len(df), dtype=np.float64)


def coin_flip(df: pd.DataFrame, seed: int = 42, **_: object) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.choice(np.array([-1.0, 1.0]), size=len(df))


def _hours(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["time"], utc=True)
    return ts.dt.hour.to_numpy()


def session_london(df: pd.DataFrame, **_: object) -> np.ndarray:
    """Long 07:00–15:59 UTC (London cash), else flat."""
    h = _hours(df)
    return np.where((h >= 7) & (h < 16), 1.0, 0.0)


def session_ny(df: pd.DataFrame, **_: object) -> np.ndarray:
    """Long 12:00–20:59 UTC (New York cash), else flat."""
    h = _hours(df)
    return np.where((h >= 12) & (h < 21), 1.0, 0.0)


def session_overlap(df: pd.DataFrame, **_: object) -> np.ndarray:
    """Long London/NY overlap 12:00–15:59 UTC, else flat."""
    h = _hours(df)
    return np.where((h >= 12) & (h < 16), 1.0, 0.0)


def session_london_short_asia(df: pd.DataFrame, **_: object) -> np.ndarray:
    """Long London, short Asia 00:00–06:59 UTC, else flat."""
    h = _hours(df)
    pos = np.zeros(len(df), dtype=np.float64)
    pos[(h >= 7) & (h < 16)] = 1.0
    pos[h < 7] = -1.0
    return pos


def tod_london_open(df: pd.DataFrame, **_: object) -> np.ndarray:
    """Long first four London hours 07:00–10:59 UTC."""
    h = _hours(df)
    return np.where((h >= 7) & (h < 11), 1.0, 0.0)


def breakout(df: pd.DataFrame, lookback: int = 24, **_: object) -> np.ndarray:
    """Donchian: long if close breaks prior N-bar high, short if prior N-bar low."""
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(df)
    pos = np.zeros(n, dtype=np.float64)
    lb = int(lookback)
    if n <= lb:
        return pos
    # rolling max/min of the *previous* lb bars (no current bar)
    for i in range(lb, n):
        prior_hi = np.max(high[i - lb : i])
        prior_lo = np.min(low[i - lb : i])
        if close[i] > prior_hi:
            pos[i] = 1.0
        elif close[i] < prior_lo:
            pos[i] = -1.0
    return pos


def sma_drift(df: pd.DataFrame, window: int = 24, **_: object) -> np.ndarray:
    """Long if close > SMA(window), short if close < SMA(window)."""
    close = df["close"].to_numpy(dtype=np.float64)
    s = pd.Series(close).rolling(int(window), min_periods=int(window)).mean().to_numpy()
    pos = np.zeros(len(df), dtype=np.float64)
    valid = np.isfinite(s)
    pos[valid & (close > s)] = 1.0
    pos[valid & (close < s)] = -1.0
    return pos


def tod_train_hours(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    returns: np.ndarray,
    cost: float = 0.0001,
    **_: object,
) -> np.ndarray:
    """Long hours whose *train* mean raw return exceeds one-way cost; else flat.

    Hours are chosen on train only (leakage-safe). Applied to the full series.
    """
    hours = _hours(df)
    pos = np.zeros(len(df), dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool)
    rets = np.asarray(returns, dtype=np.float64)
    ok = train & np.isfinite(rets)
    if not ok.any():
        return pos
    chosen = []
    for h in range(24):
        m = ok & (hours == h)
        if m.sum() < 20:
            continue
        if float(np.mean(rets[m])) > float(cost):
            chosen.append(h)
    if not chosen:
        return pos
    pos[np.isin(hours, np.array(chosen))] = 1.0
    return pos


RULE_SPECS: Dict[str, Dict] = {
    "always_flat": {"fn": always_flat, "control": True, "kwargs": {}},
    "always_long": {"fn": always_long, "control": True, "kwargs": {}},
    "coin_flip": {"fn": coin_flip, "control": True, "kwargs": {}},
    "session_london": {"fn": session_london, "control": False, "kwargs": {}},
    "session_ny": {"fn": session_ny, "control": False, "kwargs": {}},
    "session_overlap": {"fn": session_overlap, "control": False, "kwargs": {}},
    "session_london_short_asia": {"fn": session_london_short_asia, "control": False, "kwargs": {}},
    "tod_london_open": {"fn": tod_london_open, "control": False, "kwargs": {}},
    "breakout_24": {"fn": breakout, "control": False, "kwargs": {"lookback": 24}},
    "breakout_48": {"fn": breakout, "control": False, "kwargs": {"lookback": 48}},
    "sma_drift_24": {"fn": sma_drift, "control": False, "kwargs": {"window": 24}},
    "tod_train_hours": {"fn": tod_train_hours, "control": False, "kwargs": {}, "needs_train": True},
}


def is_control(name: str) -> bool:
    return bool(RULE_SPECS[name]["control"])
