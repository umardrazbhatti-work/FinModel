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


def wealth_curve_from_returns(returns: np.ndarray) -> np.ndarray:
    """
    Additive PnL wealth curve: W_0 = 1, W_t = 1 + sum_{i=1..t} r_i.

    Used for relative drawdown. Suitable when r_i are small bar returns
    (not already compounded).
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    wealth = np.empty(len(r) + 1, dtype=np.float64)
    wealth[0] = 1.0
    if len(r):
        wealth[1:] = 1.0 + np.cumsum(r)
    return wealth


def max_drawdown_from_wealth(wealth: np.ndarray) -> float:
    """
    Maximum drawdown as a negative fraction of peak wealth.

    MDD = min_t (W_t - peak_t) / peak_t  ∈ (-∞, 0], typically in [-1, 0]
    when wealth stays positive.

    Peak is floored at a tiny epsilon so a near-zero peak never explodes
    the ratio (the previous bug that produced 1e9+ drawdowns).
    """
    w = np.asarray(wealth, dtype=np.float64).reshape(-1)
    if len(w) == 0:
        return float("nan")
    peak = np.maximum.accumulate(w)
    # Relative to running peak; never divide by ~0
    denom = np.maximum(peak, 1e-12)
    dd = (w - peak) / denom
    return float(np.min(dd))


def max_drawdown_abs_from_wealth(wealth: np.ndarray) -> float:
    """Absolute peak-to-trough drop in wealth units (negative or zero)."""
    w = np.asarray(wealth, dtype=np.float64).reshape(-1)
    if len(w) == 0:
        return float("nan")
    peak = np.maximum.accumulate(w)
    return float(np.min(w - peak))


def _max_drawdown(equity: np.ndarray) -> float:
    """
    Backward-compatible name.

    Prefer passing a wealth curve starting at 1.0. If a pure cumsum equity
    curve starting at 0 is passed, we convert via wealth = 1 + equity.
    """
    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    if len(eq) == 0:
        return float("nan")
    # Heuristic: if first point is ~0 and values look like cumsum PnL, shift to wealth
    if abs(eq[0]) < 1e-15 and np.nanmax(np.abs(eq)) < 50:
        wealth = 1.0 + eq
    else:
        wealth = eq
    return max_drawdown_from_wealth(wealth)


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=0))
    if sigma < 1e-12:
        return float("nan")
    return float(np.sqrt(periods_per_year) * mu / sigma)


def score_position_returns(
    positions: np.ndarray,
    raw_returns: np.ndarray,
    cost: float,
    periods_per_year: float,
) -> dict:
    """
    Score a 1-D position series against aligned raw returns after costs.

    Net_t = pos_t * raw_t - cost * |pos_t - pos_{t-1}|  (pos_{-1}=0)
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    ret = np.asarray(raw_returns, dtype=np.float64).reshape(-1)
    if len(pos) != len(ret):
        raise ValueError(f"positions ({len(pos)}) and returns ({len(ret)}) length mismatch")
    if len(pos) == 0:
        return {
            "n": 0,
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "max_drawdown_abs": float("nan"),
            "calmar": float("nan"),
            "turnover": float("nan"),
            "hit_rate": float("nan"),
            "profit_factor": float("nan"),
            "mean_net_return": float("nan"),
            "pct_long": float("nan"),
            "pct_short": float("nan"),
            "pct_flat": float("nan"),
            "final_wealth": float("nan"),
        }

    pos_change = np.abs(np.diff(pos, prepend=0.0))
    net = pos * ret - float(cost) * pos_change
    wealth = wealth_curve_from_returns(net)
    total_ret = float(net.sum())
    mdd = max_drawdown_from_wealth(wealth)
    mdd_abs = max_drawdown_abs_from_wealth(wealth)
    sharpe = _sharpe(net, periods_per_year)
    turnover = float(pos_change.mean())

    nonzero = pos != 0
    if nonzero.any():
        hit = float(((pos[nonzero] * ret[nonzero]) > 0).mean())
    else:
        hit = float("nan")

    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    if losses > 1e-12:
        profit_factor = gains / losses
    elif gains > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    # Calmar: total PnL / |fractional MDD| (window-level, not annualized)
    if np.isfinite(mdd) and abs(mdd) > 1e-12:
        calmar = float(total_ret / abs(mdd))
    else:
        calmar = float("nan")

    n = len(net)
    return {
        "n": int(n),
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "max_drawdown_abs": mdd_abs,
        "calmar": calmar,
        "turnover": turnover,
        "hit_rate": hit,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else profit_factor,
        "mean_net_return": float(net.mean()),
        "pct_long": float((pos > 0).mean()),
        "pct_short": float((pos < 0).mean()),
        "pct_flat": float((pos == 0).mean()),
        "final_wealth": float(wealth[-1]),
    }


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

    Max drawdown is a fractional peak-to-trough on wealth W=1+cumsum(net)
    (not a ratio against a near-zero cumsum peak).
    """
    quantiles = quantiles or [0.1, 0.5, 0.9]
    try:
        median_idx = list(quantiles).index(0.5)
    except ValueError:
        median_idx = len(quantiles) // 2

    if signal_fn is None:

        def signal_fn(pred):  # type: ignore[misc]
            return median_threshold_signal(
                pred, threshold=signal_threshold, median_idx=median_idx
            )

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
        horizon_metrics = []
        ppy = float(periods_per_year.get(tf, 24 * 252))
        for h in range(positions.shape[1]):
            pos = positions[:, h]
            ret = raw[:, h]
            valid = m[:, h] > 0.5
            pos_v = pos[valid]
            ret_v = ret[valid]
            if len(pos_v) == 0:
                horizon_metrics.append({"horizon_idx": h, "n": 0})
                continue
            scored = score_position_returns(pos_v, ret_v, cost=cost, periods_per_year=ppy)
            horizon_metrics.append({"horizon_idx": h, **scored})

        primary = horizon_metrics[0] if horizon_metrics else {}
        per_tf[tf] = {"primary": primary, "horizons": horizon_metrics}

    sharpes = [
        v["primary"].get("sharpe", float("nan"))
        for v in per_tf.values()
        if v.get("primary")
    ]
    mdds = [
        v["primary"].get("max_drawdown", float("nan"))
        for v in per_tf.values()
        if v.get("primary")
    ]
    overall = {
        "mean_sharpe": float(np.nanmean(sharpes)) if sharpes else float("nan"),
        "mean_max_drawdown": float(np.nanmean(mdds)) if mdds else float("nan"),
        "cost": cost,
        "signal_threshold": signal_threshold,
    }
    return {"overall": overall, "per_tf": per_tf}
