"""Module 1 / S-1 — rule leakage, controls, and go/nogo logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.evaluate import signal_verdict
from src.signals.rules import (
    RULE_SPECS,
    always_flat,
    always_long,
    breakout,
    coin_flip,
    next_bar_simple_return,
    session_london,
    sma_drift,
    tod_train_hours,
)
from src.utils.config import load_config

CFG = ROOT / "configs" / "signal_s1_eurusd_1h.yaml"


def _toy_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2018-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="h")
    close = 1.1 + np.cumsum(rng.normal(0, 0.0002, size=n))
    high = close + 0.0003
    low = close - 0.0003
    return pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n),
        }
    )


def test_s1_config_is_module_1_no_net():
    cfg = load_config(CFG)
    assert cfg["signal"]["module"] == 1
    assert cfg["signal"]["experiment"] == "S-1"
    assert cfg["data"]["pair"] == "EURUSD"
    assert cfg["data"]["tf"] == "1h"
    assert cfg["walk_forward"]["max_folds"] == 6
    for name in cfg["signal"]["rules"]:
        assert name in RULE_SPECS


def test_next_bar_return_no_lookahead_in_formula():
    close = np.array([1.0, 1.1, 1.0], dtype=np.float64)
    r = next_bar_simple_return(close)
    assert r[0] == pytest.approx(0.1)
    assert r[1] == pytest.approx(1.0 / 1.1 - 1.0)
    assert np.isnan(r[-1])


def test_controls_shapes_and_values():
    df = _toy_df(24)
    assert (always_flat(df) == 0).all()
    assert (always_long(df) == 1).all()
    flip = coin_flip(df, seed=1)
    assert set(np.unique(flip)).issubset({-1.0, 1.0})
    assert (coin_flip(df, seed=1) == flip).all()


def test_session_london_uses_utc_hour_only():
    df = _toy_df(24)
    pos = session_london(df)
    hours = pd.to_datetime(df["time"], utc=True).dt.hour.to_numpy()
    assert pos[hours == 8][0] == 1.0
    assert pos[hours == 3][0] == 0.0
    assert pos[hours == 17][0] == 0.0


def test_breakout_uses_prior_bars_only():
    n = 30
    close = np.ones(n)
    high = np.ones(n)
    low = np.ones(n)
    # bar 24 closes above all prior highs
    high[:24] = 1.0
    close[24] = 1.5
    high[24] = 1.6
    df = pd.DataFrame(
        {
            "time": pd.date_range("2018-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
        }
    )
    pos = breakout(df, lookback=24)
    assert pos[23] == 0.0
    assert pos[24] == 1.0


def test_sma_drift_needs_full_window():
    df = _toy_df(30)
    pos = sma_drift(df, window=24)
    assert (pos[:23] == 0).all()
    assert set(np.unique(pos[23:])).issubset({-1.0, 0.0, 1.0})


def test_tod_train_hours_ignores_test_when_choosing():
    n = 240
    t0 = pd.Timestamp("2018-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="h")
    close = np.ones(n, dtype=np.float64)
    # Huge positive next-bar return only at hour 3 in the TEST half
    # Train hours 0-119: hour 3 is flat. Test 120-239: hour 3 jumps.
    rets = np.zeros(n)
    hours = times.hour.to_numpy()
    test = np.arange(n) >= 120
    rets[test & (hours == 3)] = 0.05
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )
    train_mask = np.arange(n) < 120
    pos = tod_train_hours(df, train_mask=train_mask, returns=rets, cost=0.0001)
    # Hour 3 should NOT be selected (train mean is 0)
    assert (pos[hours == 3] == 0).all()


def test_verdict_requires_beat_controls_and_majority():
    summary = pd.DataFrame(
        [
            {"rule": "always_flat", "control": True, "mean_expectancy": 0.0, "frac_folds_pos": 0.0},
            {"rule": "always_long", "control": True, "mean_expectancy": 0.00001, "frac_folds_pos": 0.5},
            {"rule": "coin_flip", "control": True, "mean_expectancy": -0.00002, "frac_folds_pos": 0.17},
            {"rule": "session_london", "control": False, "mean_expectancy": 0.00002, "frac_folds_pos": 0.67},
            {"rule": "breakout_24", "control": False, "mean_expectancy": -0.00001, "frac_folds_pos": 0.17},
        ]
    )
    v = signal_verdict(summary, 0.0, 0.5, ["always_flat", "always_long", "coin_flip"])
    assert v["pass"] is True
    assert v["winning_rules"] == ["session_london"]

    # loses to always_long
    summary.loc[summary["rule"] == "session_london", "mean_expectancy"] = 0.000005
    v2 = signal_verdict(summary, 0.0, 0.5, ["always_flat", "always_long", "coin_flip"])
    assert v2["pass"] is False
