"""Tests for lagged-RV / HAR-OLS baselines (no leakage, OLS recovers a linear HAR)."""

from __future__ import annotations

import numpy as np

from src.baselines.har_rv import (
    build_har_matrix,
    collect_future_rv_targets,
    fit_har_ols,
    future_realized_vol,
    past_realized_vol,
    persistence_point,
    predict_har_quantiles,
    run_classical_rv_baselines,
)


def _gbm_closes(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.001, size=n - 1)
    log_p = np.concatenate([[0.0], np.cumsum(rets)])
    return np.exp(log_p).astype(np.float64)


def test_past_rv_uses_only_history():
    closes = np.array([1.0, 1.01, 1.03, 1.02, 1.05, 1.04], dtype=np.float64)
    end = 3
    past = past_realized_vol(closes, end, window=2)
    # Recompute on a truncated series so any look-ahead would change the value
    past_trunc = past_realized_vol(closes[: end + 1], end, window=2)
    assert np.isfinite(past)
    assert abs(past - past_trunc) < 1e-12
    # Future window after end must not match past
    fut = future_realized_vol(closes, end, horizon=2)
    assert np.isfinite(fut)
    assert abs(past - fut) > 1e-8


def test_future_rv_matches_manual_window():
    closes = _gbm_closes(80)
    end, h = 20, 4
    got = future_realized_vol(closes, end, h)
    sl = closes[end : end + h + 1]
    manual = float(np.sqrt(np.mean(np.diff(np.log(sl)) ** 2)))
    assert abs(got - manual) < 1e-12


def test_har_ols_recovers_linear_map():
    rng = np.random.default_rng(1)
    n = 200
    # Fake design: intercept + 2 features
    x = np.column_stack(
        [
            np.ones(n),
            rng.normal(size=n),
            rng.normal(size=n),
        ]
    )
    true_beta = np.array([0.2, -0.5, 0.3])
    y = (x @ true_beta)[:, None]
    mask = np.ones((n, 1))
    fitted = fit_har_ols(x, y, mask, quantiles=[0.1, 0.5, 0.9])
    assert np.allclose(fitted["beta"][0], true_beta, atol=1e-6)
    pred = predict_har_quantiles(x, fitted)
    # Median offset is residual median ≈ 0
    assert np.allclose(pred[:, 0, 1], y[:, 0], atol=1e-6)


def test_persistence_is_past_same_horizon():
    closes = _gbm_closes(100)
    idxs = list(range(30, 50))
    pred, mask = persistence_point(closes, idxs, horizons=[4, 12], log_transform=True)
    assert pred.shape == (20, 2)
    assert mask.all()
    for j, i in enumerate(idxs):
        rv4 = past_realized_vol(closes, i, 4)
        assert abs(pred[j, 0] - np.log(rv4 + 1e-8)) < 1e-12


def test_classical_runner_shapes_and_finite():
    closes = _gbm_closes(300)
    train_idx = list(range(130, 200))
    test_idx = list(range(200, 240))
    out = run_classical_rv_baselines(
        closes=closes,
        train_end_indices=train_idx,
        test_end_indices=test_idx,
        horizons=[4, 12],
        quantiles=[0.1, 0.5, 0.9],
        windows=[4, 12, 24],
        primary_tf="1h",
    )
    har = out["har_ols"]["predictions"]["1h"]
    pers = out["persistence"]["predictions"]["1h"]
    assert har.shape == (40, 2, 3)
    assert pers.shape == (40, 2, 3)
    assert np.isfinite(har).all()
    assert np.isfinite(pers).all()


def test_collect_targets_mask_on_short_future():
    closes = _gbm_closes(30)
    # last index cannot form H=12
    y, m = collect_future_rv_targets(closes, [20, 25], horizons=[4, 12])
    assert m[0, 0] == 1.0
    assert m[1, 1] == 0.0  # 25+12 >= 30
    x = build_har_matrix(closes, [20], windows=[4, 12])
    assert x.shape == (1, 3)
    assert np.isfinite(x).all()
