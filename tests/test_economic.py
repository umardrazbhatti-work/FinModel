"""Unit tests for economic metrics, especially max-drawdown."""

from __future__ import annotations

import numpy as np

from src.evaluation.economic import (
    compute_economic_metrics,
    max_drawdown_from_wealth,
    median_threshold_signal,
    score_position_returns,
    wealth_curve_from_returns,
)


def test_wealth_curve_starts_at_one():
    w = wealth_curve_from_returns(np.array([0.01, -0.02, 0.03]))
    assert w[0] == 1.0
    assert abs(w[-1] - (1.0 + 0.02)) < 1e-12


def test_max_drawdown_simple_path():
    # +10%, then -50% of peak from 1.1 → 0.55 is -50% of peak 1.1 ≈ -0.5
    # wealth: 1 → 1.1 → 0.55
    wealth = np.array([1.0, 1.1, 0.55])
    mdd = max_drawdown_from_wealth(wealth)
    assert mdd == -0.5


def test_max_drawdown_no_explosion_on_tiny_peak():
    """Regression: old code divided by near-zero cumsum peak → 1e9+ DD."""
    # Many tiny alternating returns around zero
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 1e-4, size=500)
    scored = score_position_returns(
        positions=np.ones(500),
        raw_returns=rets,
        cost=0.0,
        periods_per_year=252 * 24,
    )
    mdd = scored["max_drawdown"]
    assert np.isfinite(mdd)
    # Fractional MDD must be in a sane band for small FX-like noise
    assert -1.0 <= mdd <= 0.0
    assert abs(mdd) < 0.5  # 50% wealth wipe on pure noise is already huge; 1e9 impossible


def test_max_drawdown_always_non_positive():
    rets = np.array([0.01, 0.02, -0.015, 0.005, -0.03, 0.01])
    scored = score_position_returns(np.ones(len(rets)), rets, cost=0.0, periods_per_year=252)
    assert scored["max_drawdown"] <= 0.0 + 1e-15


def test_flat_positions_zero_pnl():
    rets = np.array([0.01, -0.02, 0.03])
    scored = score_position_returns(np.zeros(3), rets, cost=0.0001, periods_per_year=252)
    assert scored["total_return"] == 0.0
    assert scored["max_drawdown"] == 0.0
    assert scored["pct_flat"] == 1.0


def test_median_threshold_signal():
    pred = np.array([[[ -0.5, -0.2, 0.1], [0.0, 0.0, 0.0], [-0.1, 0.5, 1.0]]])  # [1,3,3]
    pos = median_threshold_signal(pred, threshold=0.1, median_idx=1)
    assert pos.shape == (1, 3)
    assert pos[0, 0] == -1.0
    assert pos[0, 1] == 0.0
    assert pos[0, 2] == 1.0


def test_compute_economic_metrics_shapes():
    n, h, q = 100, 3, 3
    rng = np.random.default_rng(1)
    predictions = {"1h": rng.normal(0, 0.5, size=(n, h, q)).astype(np.float32)}
    raw = {"1h": rng.normal(0, 0.001, size=(n, h)).astype(np.float32)}
    masks = {"1h": np.ones((n, h), dtype=np.float32)}
    out = compute_economic_metrics(
        predictions, raw, masks, cost=1e-4, signal_threshold=0.1
    )
    primary = out["per_tf"]["1h"]["primary"]
    assert primary["n"] == n
    assert -1.5 <= primary["max_drawdown"] <= 0.0
    assert np.isfinite(primary["sharpe"])
    assert "pct_long" in primary
