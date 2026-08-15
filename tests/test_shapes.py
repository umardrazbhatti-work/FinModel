"""Shape and forward-pass tests for MTP-Transformer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MultiTFDataset, multi_tf_collate
from src.models import MTPTransformer
from src.utils.config import load_config

DATA_DIR = ROOT / "data" / "aligned"


def _small_cfg():
    cfg = load_config(ROOT / "configs" / "eurusd_1h.yaml")
    cfg["data"]["data_dir"] = str(DATA_DIR)
    return cfg


def test_forward_shapes():
    if not DATA_DIR.exists():
        import pytest
        pytest.skip("data missing")

    cfg = _small_cfg()
    ds = MultiTFDataset(
        pair=cfg["data"]["pair"],
        data_dir=str(DATA_DIR),
        tfs=cfg["data"]["tfs"],
        primary_tf=cfg["data"]["primary_tf"],
        lookback=cfg["data"]["lookback"],
        horizons=cfg["data"]["horizons"],
        quantiles=cfg["data"]["quantiles"],
        cost_threshold=cfg["data"]["cost_threshold"],
        feature_cols=cfg["data"].get("feature_cols"),
        context_cols=cfg["data"].get("context_cols"),
        fold_start="2021-01-01",
        fold_end="2021-02-01",
    )
    ds.fit_standardization()
    cfg["data"]["context_cols"] = list(ds.context_cols)
    cfg["data"]["feature_cols"] = list(ds.feature_cols)

    loader = DataLoader(ds, batch_size=4, collate_fn=multi_tf_collate, shuffle=False)
    batch = next(iter(loader))
    model = MTPTransformer(cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch)

    assert "predictions" in out and "gate_weights" in out
    gates = out["gate_weights"]
    assert torch.allclose(gates.sum(), torch.tensor(1.0), atol=1e-5)

    for tf in ["30m", "1h", "4h"]:
        pred = out["predictions"][tf]
        n_h = len(cfg["data"]["horizons"][tf])
        n_q = len(cfg["data"]["quantiles"])
        assert pred.shape == (batch["targets"][tf].shape[0], n_h, n_q)
        assert pred.shape[1:] == batch["targets"][tf].shape[1:] + (n_q,)

    n_params = model.count_parameters()
    assert n_params < 2_000_000


def test_rv_multi_tf_forward_primary_only():
    """MTP and single-TF both predict only 1h RV when tradable_tfs=[1h]."""
    if not DATA_DIR.exists():
        import pytest
        pytest.skip("data missing")

    cfg = load_config(ROOT / "configs" / "eurusd_rv_multi_tf.yaml")
    cfg["data"]["data_dir"] = str(DATA_DIR)
    ds = MultiTFDataset(
        pair=cfg["data"]["pair"],
        data_dir=str(DATA_DIR),
        tfs=cfg["data"]["tfs"],
        primary_tf=cfg["data"]["primary_tf"],
        lookback=cfg["data"]["lookback"],
        horizons=cfg["data"]["horizons"],
        quantiles=cfg["data"]["quantiles"],
        cost_threshold=cfg["data"]["cost_threshold"],
        feature_cols=cfg["data"].get("feature_cols"),
        context_cols=cfg["data"].get("context_cols") or [],
        target_type="realized_vol",
        tradable_tfs=["1h"],
        rv_log_transform=True,
        fold_start="2017-01-01",
        fold_end="2017-02-01",
    )
    ds.fit_standardization()
    cfg["data"]["context_cols"] = list(ds.context_cols)
    cfg["data"]["feature_cols"] = list(ds.feature_cols)
    cfg["data"]["tradable_tfs"] = list(ds.tradable_tfs)
    cfg["data"]["horizons"] = {k: list(v) for k, v in ds.horizons.items() if k in ds.tradable_tfs}

    loader = DataLoader(ds, batch_size=4, collate_fn=multi_tf_collate, shuffle=False)
    batch = next(iter(loader))
    assert set(batch["targets"].keys()) == {"1h"}
    assert set(batch["inputs"].keys()) >= {"5m", "15m", "30m", "1h", "4h", "1d"}

    from src.baselines import SingleTFPatchModel

    mtp = MTPTransformer(cfg)
    stf = SingleTFPatchModel(cfg, primary_tf="1h")
    mtp.eval()
    stf.eval()
    with torch.no_grad():
        mtp_out = mtp(batch)
        stf_out = stf(batch)
    n_h = len(cfg["data"]["horizons"]["1h"])
    n_q = len(cfg["data"]["quantiles"])
    assert set(mtp_out["predictions"].keys()) == {"1h"}
    assert mtp_out["predictions"]["1h"].shape == (4, n_h, n_q)
    assert stf_out["predictions"]["1h"].shape == (4, n_h, n_q)
    assert torch.allclose(mtp_out["gate_weights"].sum(), torch.tensor(1.0), atol=1e-5)


def test_collate_batch_keys():
    if not DATA_DIR.exists():
        import pytest
        pytest.skip("data missing")
    ds = MultiTFDataset(
        pair="EURUSD",
        data_dir=str(DATA_DIR),
        fold_start="2021-06-01",
        fold_end="2021-07-01",
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=multi_tf_collate)
    batch = next(iter(loader))
    assert set(batch.keys()) >= {
        "inputs",
        "context",
        "targets",
        "target_mask",
        "raw_returns",
        "timestamp",
        "pair",
    }
