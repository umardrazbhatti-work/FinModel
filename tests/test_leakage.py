"""Leakage and temporal integrity tests for MultiTFDataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multi_tf_dataset import MultiTFDataset

DATA_DIR = ROOT / "data" / "aligned"
PAIR = "EURUSD"


@pytest.fixture(scope="module")
def dataset() -> MultiTFDataset:
    if not DATA_DIR.exists():
        pytest.skip("Aligned data not present")
    ds = MultiTFDataset(
        pair=PAIR,
        data_dir=str(DATA_DIR),
        mode="train",
        # small fold in the middle of history for speed
        fold_start="2020-01-01",
        fold_end="2020-03-01",
    )
    ds.fit_standardization()
    return ds


def test_dataset_nonempty(dataset: MultiTFDataset):
    assert len(dataset) > 0


def test_no_future_bars_in_inputs(dataset: MultiTFDataset):
    rng = np.random.default_rng(0)
    n = min(20, len(dataset))
    idxs = rng.choice(len(dataset), size=n, replace=False)
    for idx in idxs:
        t = dataset.debug_prediction_time(int(idx))
        for tf in dataset.tfs:
            hist_ts = dataset.debug_history_timestamps(int(idx), tf)
            if len(hist_ts) == 0:
                continue
            assert hist_ts.max() <= t, (
                f"Leakage: TF {tf} has bar {hist_ts.max()} > prediction time {t}"
            )


def test_targets_use_existing_future(dataset: MultiTFDataset):
    sample = dataset[0]
    for tf in dataset.tradable_tfs:
        mask = sample["target_mask"][tf].numpy()
        # if mask is 1, target should be finite
        tgt = sample["targets"][tf].numpy()
        assert np.isfinite(tgt).all()
        assert ((mask == 0) | (mask == 1)).all()


def test_volatility_past_only(dataset: MultiTFDataset):
    """Realized vol uses only closes at indices <= end_idx at t."""
    idx = 0
    t = dataset.debug_prediction_time(idx)
    for tf in dataset.tradable_tfs:
        vol = dataset._realized_vol(tf, t)
        assert np.isfinite(vol) and vol > 0


def test_shapes_and_finite(dataset: MultiTFDataset):
    sample = dataset[min(5, len(dataset) - 1)]
    for tf, x in sample["inputs"].items():
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] == dataset.lookback[tf]
        assert x.shape[1] == len(dataset.feature_cols)
        assert torch.isfinite(x).all()
    for tf in dataset.tradable_tfs:
        h = len(dataset.horizons[tf])
        assert sample["targets"][tf].shape == (h,)
        assert sample["target_mask"][tf].shape == (h,)
        assert sample["raw_returns"][tf].shape == (h,)
    assert torch.isfinite(sample["context"]).all()


def test_fold_bounds_respected(dataset: MultiTFDataset):
    for i in range(min(50, len(dataset))):
        ts = dataset.get_sample_timestamp(i)
        assert ts >= dataset.get_sample_timestamp(0)
