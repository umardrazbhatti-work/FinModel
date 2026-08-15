"""Classical realized-vol baselines: lagged-RV persistence and HAR-style OLS.

Past-only features. Fit on the train fold; residual quantiles give a 3-point
predictive distribution so pinball is comparable to the neural models.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


DEFAULT_HAR_WINDOWS: Tuple[int, ...] = (4, 12, 24, 120)


def realized_vol_window(
    closes: np.ndarray,
    start_idx: int,
    end_idx: int,
    eps: float = 1e-8,
) -> float:
    """√mean(r²) of log-returns on closes[start_idx : end_idx+1] (inclusive)."""
    if end_idx <= start_idx or start_idx < 0 or end_idx >= len(closes):
        return float("nan")
    window = np.clip(closes[start_idx : end_idx + 1].astype(np.float64), eps, None)
    if len(window) < 2:
        return float("nan")
    rets = np.diff(np.log(window))
    if len(rets) == 0:
        return float("nan")
    rv = float(np.sqrt(np.mean(rets ** 2)))
    if not np.isfinite(rv) or rv < eps:
        return float(eps)
    return rv


def past_realized_vol(
    closes: np.ndarray,
    end_idx: int,
    window: int,
    eps: float = 1e-8,
) -> float:
    """Past-only RV of the last `window` log-returns ending at `end_idx`."""
    if window < 1:
        return float("nan")
    return realized_vol_window(closes, end_idx - window, end_idx, eps=eps)


def future_realized_vol(
    closes: np.ndarray,
    end_idx: int,
    horizon: int,
    eps: float = 1e-8,
) -> float:
    """Future RV of the next `horizon` log-returns after `end_idx`."""
    if horizon < 1:
        return float("nan")
    return realized_vol_window(closes, end_idx, end_idx + horizon, eps=eps)


def _maybe_log(rv: float, log_transform: bool, eps: float) -> float:
    if not np.isfinite(rv):
        return float("nan")
    if log_transform:
        return float(np.log(rv + eps))
    return float(rv)


def har_design_row(
    closes: np.ndarray,
    end_idx: int,
    windows: Sequence[int],
    log_transform: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Intercept + past log-RV (or RV) at each window. NaN if any window missing."""
    feats = [1.0]
    for w in windows:
        rv = past_realized_vol(closes, end_idx, int(w), eps=eps)
        feats.append(_maybe_log(rv, log_transform, eps))
    return np.asarray(feats, dtype=np.float64)


def build_har_matrix(
    closes: np.ndarray,
    end_indices: Sequence[int],
    windows: Sequence[int],
    log_transform: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    rows = [
        har_design_row(closes, int(i), windows, log_transform=log_transform, eps=eps)
        for i in end_indices
    ]
    return np.vstack(rows) if rows else np.zeros((0, 1 + len(windows)), dtype=np.float64)


def collect_future_rv_targets(
    closes: np.ndarray,
    end_indices: Sequence[int],
    horizons: Sequence[int],
    log_transform: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Targets [N, H] and mask [N, H] for future RV (optionally log)."""
    n = len(end_indices)
    h_n = len(horizons)
    y = np.zeros((n, h_n), dtype=np.float64)
    mask = np.zeros((n, h_n), dtype=np.float64)
    for j, idx in enumerate(end_indices):
        for hi, h in enumerate(horizons):
            rv = future_realized_vol(closes, int(idx), int(h), eps=eps)
            if not np.isfinite(rv):
                continue
            y[j, hi] = _maybe_log(rv, log_transform, eps)
            mask[j, hi] = 1.0
    return y, mask


def _ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta.astype(np.float64)


def _residual_quantiles(
    resid: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    if resid.size == 0:
        return np.zeros(len(quantiles), dtype=np.float64)
    qs = np.quantile(resid, np.asarray(quantiles, dtype=np.float64))
    return qs.astype(np.float64)


def fit_har_ols(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    quantiles: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Per-horizon OLS + residual quantile offsets.

    Returns dict with `beta` [H, K] and `q_offset` [H, Q].
    """
    n_h = y_train.shape[1]
    n_q = len(quantiles)
    n_k = x_train.shape[1]
    beta = np.zeros((n_h, n_k), dtype=np.float64)
    q_off = np.zeros((n_h, n_q), dtype=np.float64)
    for h in range(n_h):
        valid = (
            (mask_train[:, h] > 0.5)
            & np.isfinite(y_train[:, h])
            & np.all(np.isfinite(x_train), axis=1)
        )
        if int(valid.sum()) < n_k + 2:
            continue
        b = _ols_fit(x_train[valid], y_train[valid, h])
        beta[h] = b
        fitted = x_train[valid] @ b
        q_off[h] = _residual_quantiles(y_train[valid, h] - fitted, quantiles)
    return {"beta": beta, "q_offset": q_off}


def predict_har_quantiles(
    x: np.ndarray,
    fitted: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return [N, H, Q] quantile predictions. Invalid rows → NaN."""
    beta = fitted["beta"]
    q_off = fitted["q_offset"]
    n = x.shape[0]
    n_h, n_q = q_off.shape
    out = np.full((n, n_h, n_q), np.nan, dtype=np.float64)
    ok = np.all(np.isfinite(x), axis=1)
    if not np.any(ok):
        return out
    point = x[ok] @ beta.T  # [N_ok, H]
    out[ok] = point[:, :, None] + q_off[None, :, :]
    return out


def persistence_point(
    closes: np.ndarray,
    end_indices: Sequence[int],
    horizons: Sequence[int],
    log_transform: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """ŷ_H = past RV over the same window H. Returns pred [N,H] and mask."""
    n = len(end_indices)
    h_n = len(horizons)
    pred = np.zeros((n, h_n), dtype=np.float64)
    mask = np.zeros((n, h_n), dtype=np.float64)
    for j, idx in enumerate(end_indices):
        for hi, h in enumerate(horizons):
            rv = past_realized_vol(closes, int(idx), int(h), eps=eps)
            if not np.isfinite(rv):
                continue
            pred[j, hi] = _maybe_log(rv, log_transform, eps)
            mask[j, hi] = 1.0
    return pred, mask


def persistence_quantiles(
    train_point: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    test_point: np.ndarray,
    test_point_mask: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    """Add train residual quantiles to persistence point forecasts."""
    n, n_h = test_point.shape
    n_q = len(quantiles)
    out = np.full((n, n_h, n_q), np.nan, dtype=np.float64)
    for h in range(n_h):
        tr = (
            (train_mask[:, h] > 0.5)
            & np.isfinite(train_y[:, h])
            & np.isfinite(train_point[:, h])
        )
        te = (test_point_mask[:, h] > 0.5) & np.isfinite(test_point[:, h])
        if int(tr.sum()) < 5:
            continue
        q_off = _residual_quantiles(train_y[tr, h] - train_point[tr, h], quantiles)
        out[te, h, :] = test_point[te, h][:, None] + q_off[None, :]
    return out


def run_classical_rv_baselines(
    closes: np.ndarray,
    train_end_indices: Sequence[int],
    test_end_indices: Sequence[int],
    horizons: Sequence[int],
    quantiles: Sequence[float],
    windows: Optional[Sequence[int]] = None,
    log_transform: bool = True,
    eps: float = 1e-8,
    primary_tf: str = "1h",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit HAR-OLS + persistence on train indices; predict test.

    Returns dict keyed by baseline name, each with predictions {tf: [N,H,Q]}
    plus targets/masks for the test fold (same alignment).
    """
    windows = tuple(windows or DEFAULT_HAR_WINDOWS)
    y_tr, m_tr = collect_future_rv_targets(
        closes, train_end_indices, horizons, log_transform=log_transform, eps=eps
    )
    y_te, m_te = collect_future_rv_targets(
        closes, test_end_indices, horizons, log_transform=log_transform, eps=eps
    )
    x_tr = build_har_matrix(
        closes, train_end_indices, windows, log_transform=log_transform, eps=eps
    )
    x_te = build_har_matrix(
        closes, test_end_indices, windows, log_transform=log_transform, eps=eps
    )
    har_fit = fit_har_ols(x_tr, y_tr, m_tr, quantiles)
    har_pred = predict_har_quantiles(x_te, har_fit)

    pers_tr, pers_tr_m = persistence_point(
        closes, train_end_indices, horizons, log_transform=log_transform, eps=eps
    )
    pers_te, pers_te_m = persistence_point(
        closes, test_end_indices, horizons, log_transform=log_transform, eps=eps
    )
    pers_pred = persistence_quantiles(
        pers_tr, y_tr, m_tr, pers_te, pers_te_m, quantiles
    )

    # HAR/persistence can be NaN on a few short-history rows; fill with train mean
    # so pinball/corr helpers see a full array (mask still drops invalid y).
    for pred in (har_pred, pers_pred):
        for h in range(pred.shape[1]):
            valid_tr = m_tr[:, h] > 0.5
            fill = float(np.nanmean(y_tr[valid_tr, h])) if valid_tr.any() else 0.0
            nan_rows = ~np.isfinite(pred[:, h, :]).all(axis=1)
            pred[nan_rows, h, :] = fill

    return {
        "har_ols": {
            "predictions": {primary_tf: har_pred.astype(np.float32)},
            "targets": {primary_tf: y_te.astype(np.float32)},
            "masks": {primary_tf: m_te.astype(np.float32)},
            "beta": har_fit["beta"],
            "q_offset": har_fit["q_offset"],
            "windows": np.asarray(windows, dtype=np.int32),
        },
        "persistence": {
            "predictions": {primary_tf: pers_pred.astype(np.float32)},
            "targets": {primary_tf: y_te.astype(np.float32)},
            "masks": {primary_tf: m_te.astype(np.float32)},
        },
    }
