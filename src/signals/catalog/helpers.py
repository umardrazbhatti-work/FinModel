"""Past-only indicators. No current-bar lookahead in rolling extrema."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hours_utc(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["time"], utc=True)
    return ts.dt.hour.to_numpy()


def dow_utc(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["time"], utc=True)
    return ts.dt.dayofweek.to_numpy()


def month_utc(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["time"], utc=True)
    return ts.dt.month.to_numpy()


def close_arr(df: pd.DataFrame) -> np.ndarray:
    return df["close"].to_numpy(dtype=np.float64)


def sma(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(int(window), min_periods=int(window)).mean().to_numpy()


def ema(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).ewm(span=int(window), adjust=False, min_periods=int(window)).mean().to_numpy()


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(int(window), min_periods=int(window)).std(ddof=0).to_numpy()


def prior_rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Max of the previous `window` bars (excludes current)."""
    return (
        pd.Series(x)
        .shift(1)
        .rolling(int(window), min_periods=int(window))
        .max()
        .to_numpy()
    )


def prior_rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(x)
        .shift(1)
        .rolling(int(window), min_periods=int(window))
        .min()
        .to_numpy()
    )


def true_range(df: pd.DataFrame) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    prev = np.empty_like(close)
    prev[0] = close[0]
    prev[1:] = close[:-1]
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def atr(df: pd.DataFrame, window: int = 14) -> np.ndarray:
    return sma(true_range(df), window)


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0.0).rolling(int(window), min_periods=int(window)).mean()
    loss = (-delta.clip(upper=0.0)).rolling(int(window), min_periods=int(window)).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.to_numpy(dtype=np.float64)


def sign_pos(cond_long: np.ndarray, cond_short: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(cond_long), dtype=np.float64)
    pos[np.asarray(cond_long, dtype=bool)] = 1.0
    pos[np.asarray(cond_short, dtype=bool)] = -1.0
    return pos


def smma(x: np.ndarray, window: int) -> np.ndarray:
    s = sma(x, window)
    n = int(window)
    out = s.copy()
    for i in range(n, len(x)):
        if np.isfinite(out[i - 1]) and np.isfinite(x[i]):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def stochastic_k(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    hh = pd.Series(high).rolling(int(period), min_periods=int(period)).max().to_numpy()
    ll = pd.Series(low).rolling(int(period), min_periods=int(period)).min().to_numpy()
    raw = 100.0 * (close - ll) / np.maximum(hh - ll, 1e-12)
    if int(smooth) <= 1:
        return raw
    return sma(raw, smooth)


def williams_r(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    hh = pd.Series(high).rolling(int(period), min_periods=int(period)).max().to_numpy()
    ll = pd.Series(low).rolling(int(period), min_periods=int(period)).min().to_numpy()
    return -100.0 * (hh - close) / np.maximum(hh - ll, 1e-12)


def cci(df: pd.DataFrame, period: int = 20) -> np.ndarray:
    tp = (
        df["high"].to_numpy(dtype=np.float64)
        + df["low"].to_numpy(dtype=np.float64)
        + df["close"].to_numpy(dtype=np.float64)
    ) / 3.0
    mid = sma(tp, period)
    md = pd.Series(np.abs(tp - mid)).rolling(int(period), min_periods=int(period)).mean().to_numpy()
    return (tp - mid) / np.maximum(0.015 * md, 1e-12)


def adx_di(df: pd.DataFrame, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = close_arr(df)
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_s = smma(tr, period)
    pdi = 100.0 * smma(plus_dm, period) / np.maximum(atr_s, 1e-12)
    mdi = 100.0 * smma(minus_dm, period) / np.maximum(atr_s, 1e-12)
    dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    return smma(dx, period), pdi, mdi


def supertrend(df: pd.DataFrame, atr_window: int = 10, mult: float = 3.0) -> np.ndarray:
    """+1 below price (uptrend), -1 above price (downtrend). Past-only trail."""
    close = close_arr(df)
    hl2 = (df["high"].to_numpy(dtype=np.float64) + df["low"].to_numpy(dtype=np.float64)) / 2.0
    a = atr(df, atr_window)
    upper = hl2 + float(mult) * a
    lower = hl2 - float(mult) * a
    n = len(close)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
            continue
        if i == 0 or not np.isfinite(fu[i - 1]):
            fu[i] = upper[i]
            fl[i] = lower[i]
            trend[i] = 1.0 if close[i] >= hl2[i] else -1.0
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if close[i] > fu[i]:
            trend[i] = 1.0
        elif close[i] < fl[i]:
            trend[i] = -1.0
        else:
            trend[i] = trend[i - 1]
    return trend


def heikin_ashi_close(df: pd.DataFrame) -> np.ndarray:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = close_arr(df)
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_c, ha_o
