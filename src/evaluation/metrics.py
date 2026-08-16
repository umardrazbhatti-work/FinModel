"""Statistical evaluation metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _pinball_np(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    quantiles: List[float],
) -> float:
    """pred [N,H,Q], target [N,H], mask [N,H]."""
    q = np.asarray(quantiles, dtype=np.float64).reshape(1, 1, -1)
    t = target[..., None]
    errors = t - pred
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    m = mask[..., None]
    valid = m.sum()
    if valid == 0:
        return float("nan")
    return float((loss * m).sum() / (valid * pred.shape[-1]))


def compute_statistical_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    quantiles: List[float],
) -> dict:
    """
    Compute pinball, coverage, directional accuracy per TF and overall.
    """
    q_levels = list(quantiles)
    try:
        median_idx = q_levels.index(0.5)
    except ValueError:
        median_idx = len(q_levels) // 2

    per_tf = {}
    pinballs = []
    coverages = {str(q): [] for q in q_levels}
    dir_accs = []

    for tf, pred in predictions.items():
        tgt = targets[tf]
        m = masks[tf]
        pb = _pinball_np(pred, tgt, m, q_levels)
        pinballs.append(pb)

        tf_cov = {}
        for qi, q in enumerate(q_levels):
            # coverage: fraction of y <= q_hat
            valid = m > 0.5
            if valid.sum() == 0:
                cov = float("nan")
            else:
                cov = float((tgt[valid] <= pred[..., qi][valid]).mean())
            tf_cov[str(q)] = cov
            coverages[str(q)].append(cov)

        # directional accuracy on median
        med = pred[..., median_idx]
        valid = m > 0.5
        if valid.sum() == 0:
            da = float("nan")
        else:
            da = float((np.sign(med[valid]) == np.sign(tgt[valid])).mean())
        dir_accs.append(da)

        # simple CRPS approx: mean absolute pinball across quantiles (scaled)
        crps = pb  # pinball average is a CRPS proxy for discrete quantiles

        per_tf[tf] = {
            "pinball": pb,
            "coverage": tf_cov,
            "directional_accuracy": da,
            "crps_proxy": crps,
        }

    overall = {
        "pinball": float(np.nanmean(pinballs)) if pinballs else float("nan"),
        "directional_accuracy": float(np.nanmean(dir_accs)) if dir_accs else float("nan"),
        "coverage": {
            q: float(np.nanmean(vals)) if vals else float("nan")
            for q, vals in coverages.items()
        },
    }
    return {"overall": overall, "per_tf": per_tf}


def baseline_predict_zero(
    targets: Dict[str, np.ndarray],
    quantiles: List[float],
) -> Dict[str, np.ndarray]:
    """Always predict 0 for all quantiles."""
    out = {}
    n_q = len(quantiles)
    for tf, tgt in targets.items():
        n, h = tgt.shape
        out[tf] = np.zeros((n, h, n_q), dtype=np.float32)
    return out


def corr_and_r2(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """Pearson corr and R^2 of pred vs y on mask>0.5."""
    m = mask > 0.5
    if m.sum() < 10:
        return {
            "corr": float("nan"),
            "r2": float("nan"),
            "n": int(m.sum()),
            "std_pred": float("nan"),
            "std_y": float("nan"),
        }
    p, t = pred[m], y[m]
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {
        "corr": corr,
        "r2": float(r2),
        "n": int(m.sum()),
        "std_pred": float(np.std(p)),
        "std_y": float(np.std(t)),
    }


def specialist_rv_verdict(
    mean_corr: float,
    frac_folds_pass_corr: float,
    mean_pinball: float,
    mean_hist_mean_pinball: float,
    mean_har_pinball: Optional[float] = None,
    success_corr: float = 0.15,
    require_har: bool = True,
) -> dict:
    """Series M-A go/nogo: corr bar + beat hist-mean + beat HAR (when present)."""
    pass_corr = bool(
        np.isfinite(mean_corr)
        and mean_corr > float(success_corr)
        and frac_folds_pass_corr >= 0.5
    )
    beats_hist = bool(
        np.isfinite(mean_pinball)
        and np.isfinite(mean_hist_mean_pinball)
        and mean_pinball < mean_hist_mean_pinball
    )
    if mean_har_pinball is None or not np.isfinite(mean_har_pinball):
        beats_har: Optional[bool] = None
        har_ok = not require_har
    else:
        beats_har = bool(mean_pinball < float(mean_har_pinball))
        har_ok = beats_har
    return {
        "pass_corr": pass_corr,
        "beats_hist_mean": beats_hist,
        "beats_har": beats_har,
        "pass": bool(pass_corr and beats_hist and har_ok),
        "require_har": bool(require_har),
        "success_corr_threshold": float(success_corr),
    }


def evaluate_rv_skill(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    quantiles: List[float],
    primary_tf: str,
) -> dict:
    """OOS corr / R^2 of q50 vs target on each horizon of the primary TF."""
    try:
        med_i = list(quantiles).index(0.5)
    except ValueError:
        med_i = len(quantiles) // 2
    pred = predictions[primary_tf]
    y = targets[primary_tf]
    m = masks[primary_tf]
    per_h = []
    for h in range(pred.shape[1]):
        sk = corr_and_r2(pred[:, h, med_i], y[:, h], m[:, h])
        per_h.append({"horizon_idx": h, **sk})
    corrs = [x["corr"] for x in per_h if np.isfinite(x["corr"])]
    return {
        "per_horizon": per_h,
        "mean_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "best_horizon_corr": float(np.max(corrs)) if corrs else float("nan"),
    }


def baseline_historical_mean(
    train_targets: Dict[str, np.ndarray],
    train_masks: Dict[str, np.ndarray],
    test_targets: Dict[str, np.ndarray],
    quantiles: List[float],
) -> Dict[str, np.ndarray]:
    """Predict historical mean (for all quantiles) from train fold targets."""
    out = {}
    n_q = len(quantiles)
    for tf, tgt_test in test_targets.items():
        tr = train_targets[tf]
        m = train_masks[tf]
        means = []
        for h in range(tr.shape[1]):
            valid = m[:, h] > 0.5
            if valid.sum() == 0:
                means.append(0.0)
            else:
                means.append(float(tr[valid, h].mean()))
        means_arr = np.asarray(means, dtype=np.float32)  # [H]
        n = tgt_test.shape[0]
        pred = np.tile(means_arr.reshape(1, -1, 1), (n, 1, n_q))
        out[tf] = pred
    return out
