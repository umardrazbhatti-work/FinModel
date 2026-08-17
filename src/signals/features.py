"""Leakage-safe tabular features for S-2 logistic Signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.labels import past_simple_return


def ohlc_features(df: pd.DataFrame, eps: float = 1e-12) -> np.ndarray:
    """Past-only OHLC features. Shape [N, F]."""
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    hour = pd.to_datetime(df["time"], utc=True).dt.hour.to_numpy(dtype=np.float64)
    r1 = past_simple_return(close, 1, eps)
    r4 = past_simple_return(close, 4, eps)
    r12 = past_simple_return(close, 12, eps)
    r24 = past_simple_return(close, 24, eps)
    rng = (high - low) / np.maximum(close, eps)
    sma = pd.Series(close).rolling(24, min_periods=24).mean().to_numpy()
    gap = (close - sma) / np.maximum(close, eps)
    vol = pd.Series(r1).rolling(24, min_periods=24).std().to_numpy()
    ang = 2.0 * np.pi * hour / 24.0
    cols = [r1, r4, r12, r24, rng, gap, vol, np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


def event_features(df: pd.DataFrame) -> np.ndarray:
    """Calendar / event columns already on the aligned parquet (past-as-of-bar)."""
    names = ["high_impact_count", "usd_events", "eur_events", "vix"]
    cols = []
    n = len(df)
    for name in names:
        if name not in df.columns:
            cols.append(np.zeros(n, dtype=np.float64))
            continue
        s = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
        # ffill then 0: early nans are "no reading yet", not future
        s = pd.Series(s).ffill().fillna(0.0).to_numpy(dtype=np.float64)
        cols.append(s)
    return np.column_stack(cols)


def combine_features(df: pd.DataFrame, use_events: bool) -> np.ndarray:
    x = ohlc_features(df)
    if use_events:
        x = np.column_stack([x, event_features(df)])
    return x
