"""Module 1 / S-2 — labels, persist leakage, logistic train-only, verdict."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.labels import (
    class_from_return,
    forward_simple_return,
    nonoverlap_indices,
    persist_positions,
)
from src.signals.logistic import logistic_positions
from src.signals.s2_eval import s2_verdict, score_nonoverlap_trades
from src.utils.config import load_config

CFG = ROOT / "configs" / "signal_s2_eurusd_1h.yaml"


def test_s2_config_sweep_no_handler():
    cfg = load_config(CFG)
    assert cfg["signal"]["module"] == 1
    assert cfg["signal"]["experiment"] == "S-2"
    assert 12 in cfg["signal"]["horizons"]
    assert 0 in cfg["signal"]["k_list"] or 0.0 in cfg["signal"]["k_list"]


def test_forward_return_uses_future_close_only_in_label():
    close = np.array([1.0, 1.0, 1.2, 1.2], dtype=np.float64)
    r = forward_simple_return(close, 2)
    assert r[0] == pytest.approx(0.2)
    assert np.isnan(r[-1])
    assert np.isnan(r[-2])


def test_persist_does_not_use_future():
    close = np.array([1.0, 1.0, 1.0, 1.3, 1.3], dtype=np.float64)
    pos = persist_positions(close, horizon=2, cost=0.0001, k=0)
    # at t=2, past is close[2]/close[0]-1 = 0 -> flat
    assert pos[2] == 0.0
    # at t=3, past is 1.3/1.0-1 > 0 -> long (uses t=3 and t=1, not t+H)
    assert pos[3] == 1.0


def test_large_move_flat_inside_band():
    r = np.array([0.00005, 0.0003, -0.0003], dtype=np.float64)
    y = class_from_return(r, cost=0.0001, k=2)
    assert y[0] == 0.0
    assert y[1] == 1.0
    assert y[2] == -1.0


def test_nonoverlap_stride():
    idx = nonoverlap_indices(10, 40, horizon=12, n=100)
    assert list(idx) == [10, 22, 34]


def test_score_round_trip_cost():
    pos = np.ones(20)
    fwd = np.full(20, 0.001)
    s = score_nonoverlap_trades(pos, fwd, 0, 20, horizon=4, cost=0.0001, periods_per_year=6048)
    # net per trade = 0.001 - 0.0002
    assert s["n_trades"] == 4
    assert s["expectancy"] == pytest.approx(0.0008)


def test_logistic_ignores_test_only_pattern():
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = np.zeros(n)
    # Test half: feature 0 perfectly predicts sign; train half is noise labels
    y[:100] = rng.choice([-1.0, 1.0], size=100)
    y[100:] = np.where(x[100:, 0] > 0, 1.0, -1.0)
    train = np.zeros(n, dtype=bool)
    train[:100] = True
    pos = logistic_positions(x, y, train, seed=0)
    # Should not recover the test-only rule perfectly; accuracy on test should not be ~1
    test_acc = float((pos[100:] == y[100:]).mean())
    assert test_acc < 0.9


def test_oracle_cannot_win_verdict():
    summary = pd.DataFrame(
        [
            {
                "key": "h12_always_long",
                "horizon": 12,
                "k": 0,
                "model": "always_long",
                "control": True,
                "oracle": False,
                "mean_expectancy": -0.001,
                "frac_folds_pos": 0.16,
                "beats_always_long": False,
                "beats_coin_flip": True,
            },
            {
                "key": "h12_coin_flip",
                "horizon": 12,
                "k": 0,
                "model": "coin_flip",
                "control": True,
                "oracle": False,
                "mean_expectancy": -0.002,
                "frac_folds_pos": 0.0,
                "beats_always_long": False,
                "beats_coin_flip": False,
            },
            {
                "key": "h12_k0_oracle",
                "horizon": 12,
                "k": 0,
                "model": "oracle",
                "control": False,
                "oracle": True,
                "mean_expectancy": 0.01,
                "frac_folds_pos": 1.0,
                "beats_always_long": True,
                "beats_coin_flip": True,
            },
            {
                "key": "h12_k0_persist",
                "horizon": 12,
                "k": 0,
                "model": "persist",
                "control": False,
                "oracle": False,
                "mean_expectancy": -0.0005,
                "frac_folds_pos": 0.16,
                "beats_always_long": True,
                "beats_coin_flip": True,
            },
        ]
    )
    v = s2_verdict(summary, 0.5)
    assert v["pass"] is False
    assert v["winning_keys"] == []
