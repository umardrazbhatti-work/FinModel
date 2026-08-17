"""S-2 labels: forward return, sign, and large-move classes.

All arrays are aligned to bar t. Label at t uses close[t+H] (future) —
models may not see that. Persistence uses only close[t] vs close[t-H].
"""

from __future__ import annotations

import numpy as np


def forward_simple_return(close: np.ndarray, horizon: int, eps: float = 1e-12) -> np.ndarray:
    """r_t = close[t+H] / close[t] - 1. Last H bars are NaN."""
    close = np.asarray(close, dtype=np.float64)
    h = int(horizon)
    out = np.full(len(close), np.nan, dtype=np.float64)
    if h < 1 or len(close) <= h:
        return out
    prev = np.maximum(close[:-h], eps)
    out[:-h] = close[h:] / prev - 1.0
    return out


def past_simple_return(close: np.ndarray, horizon: int, eps: float = 1e-12) -> np.ndarray:
    """r^{past}_t = close[t] / close[t-H] - 1. First H bars are NaN."""
    close = np.asarray(close, dtype=np.float64)
    h = int(horizon)
    out = np.full(len(close), np.nan, dtype=np.float64)
    if h < 1 or len(close) <= h:
        return out
    prev = np.maximum(close[:-h], eps)
    out[h:] = close[h:] / prev - 1.0
    return out


def class_from_return(r: np.ndarray, cost: float, k: float) -> np.ndarray:
    """Map a return to {-1, 0, +1}.

    k == 0: sign only (flat only if r is 0 / non-finite).
    k > 0: flat if |r| < k * cost.
    """
    r = np.asarray(r, dtype=np.float64)
    y = np.zeros(len(r), dtype=np.float64)
    finite = np.isfinite(r)
    if float(k) <= 0:
        y[finite & (r > 0)] = 1.0
        y[finite & (r < 0)] = -1.0
        return y
    thresh = float(k) * float(cost)
    y[finite & (r >= thresh)] = 1.0
    y[finite & (r <= -thresh)] = -1.0
    return y


def persist_positions(close: np.ndarray, horizon: int, cost: float, k: float) -> np.ndarray:
    """Sign of the last H-bar move; flat if it was not a large move (k>0)."""
    return class_from_return(past_simple_return(close, horizon), cost=cost, k=k)


def oracle_positions(fwd: np.ndarray, cost: float, k: float) -> np.ndarray:
    """Perfect foresight of the label. Diagnostic ceiling — not a Signal."""
    return class_from_return(fwd, cost=cost, k=k)


def nonoverlap_indices(start: int, end: int, horizon: int, n: int) -> np.ndarray:
    """Entry bars for non-overlapping H-bar holds. Last entry needs t+H in range."""
    h = int(horizon)
    last = min(int(end), int(n) - h)
    if last <= start or h < 1:
        return np.array([], dtype=int)
    return np.arange(int(start), last, h, dtype=int)
