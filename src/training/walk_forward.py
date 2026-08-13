"""Walk-forward fold generation on the primary timeframe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """Single walk-forward fold defined on primary TF bar indices / timestamps."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start_idx: int
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    test_start_idx: int
    test_end_idx: int


def generate_walk_forward_folds(
    primary_timestamps: np.ndarray,
    min_train_bars: int = 5000,
    val_bars: int = 1000,
    test_bars: int = 1000,
    step_bars: int = 500,
    purge_bars: int = 24,
    mode: Literal["expanding", "rolling"] = "expanding",
    max_folds: Optional[int] = None,
) -> List[WalkForwardFold]:
    """
    Generate deterministic walk-forward folds on the primary timestamp index.

    Layout per fold (indices on primary TF):
        [train) | purge | [val) | purge | [test)
    For expanding mode, train always starts at index 0.
    For rolling mode, train length is fixed to min_train_bars.
    """
    n = len(primary_timestamps)
    if n < min_train_bars + purge_bars + val_bars + purge_bars + test_bars:
        raise ValueError(
            f"Not enough bars ({n}) for walk-forward with "
            f"train={min_train_bars}, val={val_bars}, test={test_bars}, purge={purge_bars}"
        )

    def ts_at(i: int) -> pd.Timestamp:
        i = min(max(i, 0), n - 1)
        return pd.Timestamp(primary_timestamps[i], tz="UTC")

    folds: List[WalkForwardFold] = []
    # first test block starts after min train + purges + val
    # train: [0, train_end)
    # purge
    # val: [val_start, val_end)
    # purge
    # test: [test_start, test_end)

    train_end = min_train_bars
    fold_id = 0

    while True:
        val_start = train_end + purge_bars
        val_end = val_start + val_bars
        test_start = val_end + purge_bars
        test_end = test_start + test_bars

        if test_end > n:
            break

        if mode == "expanding":
            train_start = 0
        else:
            train_start = max(0, train_end - min_train_bars)

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=ts_at(train_start),
            train_end=ts_at(train_end),
            val_start=ts_at(val_start),
            val_end=ts_at(val_end),
            test_start=ts_at(test_start),
            test_end=ts_at(test_end) if test_end < n else ts_at(n - 1) + pd.Timedelta(hours=1),
            train_start_idx=train_start,
            train_end_idx=train_end,
            val_start_idx=val_start,
            val_end_idx=val_end,
            test_start_idx=test_start,
            test_end_idx=test_end,
        )
        folds.append(fold)
        fold_id += 1

        if max_folds is not None and len(folds) >= max_folds:
            break

        train_end += step_bars

    if not folds:
        raise ValueError("Walk-forward produced zero folds; check configuration.")
    return folds
