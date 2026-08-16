"""Series M-A specialist helpers: 4h dataset, folds, wall-clock, go/nogo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import SingleTFPatchModel
from src.data import MultiTFDataset, horizon_wall_clock, multi_tf_collate, tf_bar_hours
from src.evaluation.metrics import specialist_rv_verdict
from src.training import generate_walk_forward_folds
from src.utils.config import load_config

DATA_DIR = ROOT / "data" / "aligned"
CFG_4H = ROOT / "configs" / "eurusd_rv_ma_4h.yaml"
CFG_30M = ROOT / "configs" / "eurusd_rv_ma_30m.yaml"


def test_tf_bar_hours_known_clocks():
    assert tf_bar_hours("1h") == 1.0
    assert tf_bar_hours("4h") == 4.0
    assert tf_bar_hours("30m") == 0.5
    assert tf_bar_hours("1d") == 24.0
    assert tf_bar_hours("daily") == 24.0


def test_horizon_wall_clock_4h():
    wall = horizon_wall_clock("4h", [4, 12])
    assert wall[0]["bars"] == 4
    assert wall[0]["hours"] == 16.0
    assert wall[0]["days"] == pytest.approx(16.0 / 24.0)
    assert wall[1]["hours"] == 48.0
    assert wall[1]["days"] == pytest.approx(2.0)


def test_horizon_wall_clock_30m():
    wall = horizon_wall_clock("30m", [4, 12])
    assert wall[0]["hours"] == 2.0
    assert wall[1]["hours"] == 6.0


def test_specialist_verdict_requires_har_and_corr():
    ok = specialist_rv_verdict(0.40, 1.0, 0.10, 0.20, 0.12, require_har=True)
    assert ok["pass"] is True
    assert ok["beats_har"] is True

    lose_har = specialist_rv_verdict(0.40, 1.0, 0.13, 0.20, 0.12, require_har=True)
    assert lose_har["pass"] is False
    assert lose_har["beats_har"] is False

    lose_corr = specialist_rv_verdict(0.10, 0.2, 0.10, 0.20, 0.12, require_har=True)
    assert lose_corr["pass_corr"] is False
    assert lose_corr["pass"] is False

    skip_har = specialist_rv_verdict(0.40, 1.0, 0.10, 0.20, None, require_har=False)
    assert skip_har["pass"] is True
    assert skip_har["beats_har"] is None


def test_4h_config_is_single_tf_rv():
    cfg = load_config(CFG_4H)
    assert cfg["data"]["pair"] == "EURUSD"
    assert cfg["data"]["tfs"] == ["4h"]
    assert cfg["data"]["primary_tf"] == "4h"
    assert cfg["data"]["tradable_tfs"] == ["4h"]
    assert cfg["data"]["target_type"] == "realized_vol"
    assert cfg["data"]["horizons"]["4h"] == [4, 12]
    assert cfg["evaluation"]["require_har"] is True
    assert cfg["walk_forward"]["max_folds"] == 6


def test_30m_config_is_single_tf_rv():
    cfg = load_config(CFG_30M)
    assert cfg["data"]["pair"] == "EURUSD"
    assert cfg["data"]["tfs"] == ["30m"]
    assert cfg["data"]["primary_tf"] == "30m"
    assert cfg["data"]["tradable_tfs"] == ["30m"]
    assert cfg["data"]["target_type"] == "realized_vol"
    assert cfg["data"]["horizons"]["30m"] == [4, 12]
    assert cfg["evaluation"]["har_windows"] == [8, 24, 48, 240]
    assert cfg["evaluation"]["require_har"] is True
    assert cfg["walk_forward"]["max_folds"] == 6


def test_4h_dataset_rv_forward():
    if not DATA_DIR.exists():
        pytest.skip("data missing")
    cfg = load_config(CFG_4H)
    ds = MultiTFDataset(
        pair=cfg["data"]["pair"],
        data_dir=str(DATA_DIR),
        tfs=cfg["data"]["tfs"],
        primary_tf=cfg["data"]["primary_tf"],
        lookback=cfg["data"]["lookback"],
        horizons=cfg["data"]["horizons"],
        quantiles=cfg["data"]["quantiles"],
        feature_cols=cfg["data"].get("feature_cols"),
        context_cols=[],
        target_type="realized_vol",
        tradable_tfs=["4h"],
        rv_log_transform=True,
        fold_start="2018-01-01",
        fold_end="2018-04-01",
    )
    assert len(ds) > 0
    ds.fit_standardization()
    cfg["data"]["context_cols"] = list(ds.context_cols)
    cfg["data"]["feature_cols"] = list(ds.feature_cols)
    cfg["data"]["tradable_tfs"] = ["4h"]
    cfg["data"]["horizons"] = {"4h": list(ds.horizons["4h"])}

    loader = DataLoader(ds, batch_size=4, collate_fn=multi_tf_collate, shuffle=False)
    batch = next(iter(loader))
    assert set(batch["inputs"].keys()) == {"4h"}
    assert set(batch["targets"].keys()) == {"4h"}
    assert batch["inputs"]["4h"].shape[1] == 42
    y = batch["targets"]["4h"].numpy()
    assert np.isfinite(y).all()

    # No future 4h bar in the input window
    t0 = ds.debug_prediction_time(0)
    hist = ds.debug_history_timestamps(0, "4h")
    assert hist.max() <= t0

    model = SingleTFPatchModel(cfg, primary_tf="4h")
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert set(out["predictions"].keys()) == {"4h"}
    assert out["predictions"]["4h"].shape == (4, 2, 3)


def test_4h_walk_forward_six_folds():
    if not DATA_DIR.exists():
        pytest.skip("data missing")
    cfg = load_config(CFG_4H)
    ds = MultiTFDataset(
        pair=cfg["data"]["pair"],
        data_dir=str(DATA_DIR),
        tfs=["4h"],
        primary_tf="4h",
        lookback=cfg["data"]["lookback"],
        horizons=cfg["data"]["horizons"],
        tradable_tfs=["4h"],
        target_type="realized_vol",
        context_cols=[],
    )
    wf = cfg["walk_forward"]
    folds = generate_walk_forward_folds(
        primary_timestamps=ds.get_primary_timestamps(),
        min_train_bars=wf["min_train_bars"],
        val_bars=wf["val_bars"],
        test_bars=wf["test_bars"],
        step_bars=wf["step_bars"],
        purge_bars=wf["purge_bars"],
        mode=wf.get("mode", "expanding"),
        max_folds=wf.get("max_folds"),
    )
    assert len(folds) == 6
    assert folds[0].train_start < folds[0].test_start
    assert folds[-1].test_end_idx <= len(ds.get_primary_timestamps())


def test_30m_dataset_rv_forward():
    if not DATA_DIR.exists():
        pytest.skip("data missing")
    cfg = load_config(CFG_30M)
    ds = MultiTFDataset(
        pair=cfg["data"]["pair"],
        data_dir=str(DATA_DIR),
        tfs=cfg["data"]["tfs"],
        primary_tf=cfg["data"]["primary_tf"],
        lookback=cfg["data"]["lookback"],
        horizons=cfg["data"]["horizons"],
        quantiles=cfg["data"]["quantiles"],
        feature_cols=cfg["data"].get("feature_cols"),
        context_cols=[],
        target_type="realized_vol",
        tradable_tfs=["30m"],
        rv_log_transform=True,
        fold_start="2018-01-01",
        fold_end="2018-02-01",
    )
    assert len(ds) > 0
    ds.fit_standardization()
    cfg["data"]["context_cols"] = list(ds.context_cols)
    cfg["data"]["feature_cols"] = list(ds.feature_cols)
    cfg["data"]["tradable_tfs"] = ["30m"]
    cfg["data"]["horizons"] = {"30m": list(ds.horizons["30m"])}

    loader = DataLoader(ds, batch_size=4, collate_fn=multi_tf_collate, shuffle=False)
    batch = next(iter(loader))
    assert set(batch["inputs"].keys()) == {"30m"}
    assert set(batch["targets"].keys()) == {"30m"}
    assert batch["inputs"]["30m"].shape[1] == 48
    assert np.isfinite(batch["targets"]["30m"].numpy()).all()
    t0 = ds.debug_prediction_time(0)
    hist = ds.debug_history_timestamps(0, "30m")
    assert hist.max() <= t0

    model = SingleTFPatchModel(cfg, primary_tf="30m")
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert set(out["predictions"].keys()) == {"30m"}
    assert out["predictions"]["30m"].shape == (4, 2, 3)


def test_30m_walk_forward_six_folds():
    if not DATA_DIR.exists():
        pytest.skip("data missing")
    cfg = load_config(CFG_30M)
    ds = MultiTFDataset(
        pair=cfg["data"]["pair"],
        data_dir=str(DATA_DIR),
        tfs=["30m"],
        primary_tf="30m",
        lookback=cfg["data"]["lookback"],
        horizons=cfg["data"]["horizons"],
        tradable_tfs=["30m"],
        target_type="realized_vol",
        context_cols=[],
    )
    wf = cfg["walk_forward"]
    folds = generate_walk_forward_folds(
        primary_timestamps=ds.get_primary_timestamps(),
        min_train_bars=wf["min_train_bars"],
        val_bars=wf["val_bars"],
        test_bars=wf["test_bars"],
        step_bars=wf["step_bars"],
        purge_bars=wf["purge_bars"],
        mode=wf.get("mode", "expanding"),
        max_folds=wf.get("max_folds"),
    )
    assert len(folds) == 6
    assert folds[-1].test_end_idx <= len(ds.get_primary_timestamps())
