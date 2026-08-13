"""Economic evaluation metrics from quantile predictions."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def median_threshold_signal(
    predictions: np.ndarray,
    threshold: float = 0.1,
    median_idx: int = 1,
) -> np.ndarray:
    """
    predictions: [N, H, Q] → positions [N, H] in {-1, 0, +1}
    """
    med = predictions[..., median_idx]
    pos = np.zeros_like(med, dtype=np.float64)
    pos[med > threshold] = 1.0
    pos[med < -threshold] = -1.0
    return pos


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(np.abs(peak), 1e-12)
    return float(dd.min())


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = returns.mean()
    sigma = returns.std()
    if sigma < 1e-12:
        return float("nan")
    return float(np.sqrt(periods_per_year) * mu / sigma)


def compute_economic_metrics(
    predictions: Dict[str, np.ndarray],
    raw_returns: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    cost: float,
    signal_fn: Optional[Callable] = None,
    signal_threshold: float = 0.1,
    quantiles: Optional[list] = None,
    periods_per_year: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Convert quantile forecasts to simple threshold signals and score after costs.

    Net return_t = position_t * raw_return_t - cost * |position_t - position_{t-1}|
    """
    quantiles = quantiles or [0.1, 0.5, 0.9]
    try:
        median_idx = list(quantiles).index(0.5)
    except ValueError:
        median_idx = len(quantiles) // 2

    if signal_fn is None:
        def signal_fn(pred):  # type: ignore[misc]
            return median_threshold_signal(pred, threshold=signal_threshold, median_idx=median_idx)

    default_ppy = {
        "30m": 24 * 2 * 252,
        "1h": 24 * 252,
        "4h": 6 * 252,
    }
    if periods_per_year is None:
        periods_per_year = default_ppy

    per_tf = {}
    for tf, pred in predictions.items():
        if tf not in raw_returns:
            continue
        raw = raw_returns[tf]
        m = masks[tf]
        positions = signal_fn(pred)  # [N, H]
        # evaluate each horizon separately; also report primary h=0 aggregate
        horizon_metrics = []
        for h in range(positions.shape[1]):
            pos = positions[:, h].astype(np.float64)
            ret = raw[:, h].astype(np.float64)
            valid = m[:, h] > 0.5
            pos = pos[valid]
            ret = ret[valid]
            if len(pos) == 0:
                horizon_metrics.append({"horizon_idx": h, "n": 0})
                continue
            pos_change = np.abs(np.diff(pos, prepend=0.0))
            net = pos * ret - cost * pos_change
            equity = np.cumsum(net)
            total_ret = float(equity[-1]) if len(equity) else 0.0
            ppy = periods_per_year.get(tf, 24 * 252)
            sharpe = _sharpe(net, ppy)
            mdd = _max_drawdown(np.concatenate([[0.0], equity]))
            turnover = float(pos_change.mean())
            nonzero = pos != 0
            hit = float(((pos[nonzero] * ret[nonzero]) > 0).mean()) if nonzero.any() else float("nan")
            gains = net[net > 0].sum()
            losses = -net[net < 0].sum()
            profit_factor = float(gains / losses) if losses > 1e-12 else float("inf")
            calmar = float(total_ret / abs(mdd)) if abs(mdd) > 1e-12 else float("nan")
            horizon_metrics.append(
                {
                    "horizon_idx": h,
                    "n": int(len(net)),
                    "total_return": total_ret,
                    "sharpe": sharpe,
                    "max_drawdown": mdd,
                    "calmar": calmar,
                    "turnover": turnover,
                    "hit_rate": hit,
                    "profit_factor": profit_factor,
                    "mean_net_return": float(net.mean()),
                }
            )
        # primary summary = first horizon
        primary = horizon_metrics[0] if horizon_metrics else {}
        per_tf[tf] = {"primary": primary, "horizons": horizon_metrics}

    # overall: average primary sharpe across TFs
    sharpes = [
        v["primary"].get("sharpe", float("nan"))
        for v in per_tf.values()
        if v.get("primary")
    ]
    overall = {
        "mean_sharpe": float(np.nanmean(sharpes)) if sharpes else float("nan"),
        "cost": cost,
        "signal_threshold": signal_threshold,
    }
    return {"overall": overall, "per_tf": per_tf}
