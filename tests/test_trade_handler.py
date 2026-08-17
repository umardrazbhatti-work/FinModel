"""Module 2 — Trade Handler lock tests: sizing math + no direction."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import SingleTFPatchModel
from src.handler import (
    HANDLER_VERSION,
    MODULE_ID,
    VolatilityTradeHandler,
    size_from_vol,
)
from src.handler.sizing import SizingConfig, realized_vol_from_closes
from src.utils.config import load_config

CFG = ROOT / "configs" / "handler_eurusd_1h.yaml"


def test_handler_config_is_locked_1h_rv():
    cfg = load_config(CFG)
    assert cfg["handler"]["module"] == 2
    assert cfg["data"]["pair"] == "EURUSD"
    assert cfg["data"]["tfs"] == ["1h"]
    assert cfg["data"]["primary_tf"] == "1h"
    assert cfg["data"]["target_type"] == "realized_vol"
    assert cfg["data"]["horizons"]["1h"] == [4, 12]


def test_size_high_vol_shrinks():
    # forecast RV = e^0 = 1, ref = 0.5 → raw 0.5
    r = size_from_vol(0.0, 0.0, 0.2, ref_rv=0.5)
    assert r.stand_aside is False
    assert r.reason == "ok"
    assert r.size_multiplier == pytest.approx(0.5)


def test_size_low_vol_grows_but_clips():
    # forecast RV = e^{-2} ≈ 0.135, ref = 1 → raw ≈ 7.4 → clip 2
    r = size_from_vol(-2.2, -2.0, -1.8, ref_rv=1.0, cfg=SizingConfig(max_multiplier=2.0))
    assert r.stand_aside is False
    assert r.size_multiplier == pytest.approx(2.0)


def test_size_wide_band_stands_aside():
    r = size_from_vol(-1.0, 0.0, 2.0, ref_rv=1.0, cfg=SizingConfig(max_log_width=1.5))
    assert r.stand_aside is True
    assert r.size_multiplier == 0.0
    assert r.reason == "uncertainty_width"


def test_size_invalid_order_stands_aside():
    r = size_from_vol(1.0, 0.0, -1.0, ref_rv=1.0)
    assert r.stand_aside is True
    assert r.reason == "invalid_quantile_order"


def test_realized_vol_from_closes():
    closes = np.exp(np.linspace(0, 0.1, 21))
    rv = realized_vol_from_closes(closes)
    assert rv > 0
    assert math.isfinite(rv)


def _dummy_batch(cfg, n=1):
    lookback = int(cfg["data"]["lookback"]["1h"])
    n_feat = len(cfg["data"]["feature_cols"])
    x = torch.randn(n, lookback, n_feat)
    return {
        "inputs": {"1h": x},
        "context": torch.zeros(n, 0),
        "timestamp": ["2018-01-01T00:00:00+00:00"] * n,
        "pair": ["EURUSD"] * n,
    }


def test_handler_never_sets_side():
    cfg = load_config(CFG)
    model = SingleTFPatchModel(cfg, primary_tf="1h")
    handler = VolatilityTradeHandler(model, cfg, device="cpu")
    batch = _dummy_batch(cfg)
    closes = np.linspace(1.10, 1.12, int(cfg["data"]["lookback"]["1h"]))
    dec = handler.decide(batch, closes=closes)
    assert dec.side is None
    assert dec.to_dict()["side"] is None
    assert dec.module == MODULE_ID
    assert dec.handler_version == HANDLER_VERSION
    assert dec.pair == "EURUSD"
    assert dec.forecast.tf == "1h"
    assert set(dec.forecast.log_quantiles[1]) == {"q10", "q50", "q90"}


def test_handler_from_config_untrained_still_decides():
    cfg = load_config(CFG)
    handler = VolatilityTradeHandler.from_config(CFG, checkpoint=None, device="cpu")
    batch = _dummy_batch(cfg)
    closes = np.linspace(1.10, 1.11, int(cfg["data"]["lookback"]["1h"]))
    dec = handler.decide(batch, closes=closes)
    assert dec.side is None
    assert dec.size_multiplier >= 0.0
    assert isinstance(dec.stand_aside, bool)
