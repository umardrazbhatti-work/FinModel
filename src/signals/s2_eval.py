"""S-2 scoring: non-overlapping H-bar holds after round-trip cost."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.signals.labels import nonoverlap_indices


def score_nonoverlap_trades(
    positions: np.ndarray,
    fwd_return: np.ndarray,
    start: int,
    end: int,
    horizon: int,
    cost: float,
    periods_per_year: float,
) -> Dict[str, Any]:
    """One round-trip cost (2 * cost) when |pos|=1. Stride = horizon."""
    pos = np.asarray(positions, dtype=np.float64)
    fwd = np.asarray(fwd_return, dtype=np.float64)
    idx = nonoverlap_indices(start, end, horizon, n=len(pos))
    if idx.size == 0:
        return {
            "n_trades": 0,
            "expectancy": float("nan"),
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "pct_long": float("nan"),
            "pct_short": float("nan"),
            "pct_flat": float("nan"),
            "pass_fold": False,
        }
    p = pos[idx]
    r = fwd[idx]
    ok = np.isfinite(r)
    p = p[ok]
    r = r[ok]
    if p.size == 0:
        return {
            "n_trades": 0,
            "expectancy": float("nan"),
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "pct_long": float("nan"),
            "pct_short": float("nan"),
            "pct_flat": float("nan"),
            "pass_fold": False,
        }
    traded = np.abs(p) > 0
    net = p * r - (2.0 * float(cost)) * np.abs(p)
    n = int(net.size)
    mu = float(net.mean())
    sig = float(net.std(ddof=0))
    trades_per_year = float(periods_per_year) / float(horizon)
    if n >= 2 and sig > 1e-12:
        sharpe = float(np.sqrt(trades_per_year) * mu / sig)
    else:
        sharpe = float("nan")
    if traded.any():
        hit = float(((p[traded] * r[traded]) > 0).mean())
    else:
        hit = float("nan")
    return {
        "n_trades": n,
        "expectancy": mu,
        "total_return": float(net.sum()),
        "sharpe": sharpe,
        "hit_rate": hit,
        "pct_long": float((p > 0).mean()),
        "pct_short": float((p < 0).mean()),
        "pct_flat": float((p == 0).mean()),
        "pass_fold": bool(np.isfinite(mu) and mu > 0),
    }


def s2_verdict(summary: pd.DataFrame, min_frac_folds: float = 0.5) -> Dict[str, Any]:
    """PASS if any non-control, non-oracle row clears expectancy + majority + beats same-horizon controls."""
    winners: List[Dict[str, Any]] = []
    details = []
    for _, row in summary.iterrows():
        rec = {
            "key": str(row["key"]),
            "horizon": int(row["horizon"]),
            "k": float(row["k"]),
            "model": str(row["model"]),
            "control": bool(row["control"]),
            "oracle": bool(row["oracle"]),
            "mean_expectancy": float(row["mean_expectancy"]),
            "frac_folds_pos": float(row["frac_folds_pos"]),
            "beats_always_long": bool(row["beats_always_long"]),
            "beats_coin_flip": bool(row["beats_coin_flip"]),
            "pass": False,
        }
        if (not rec["control"]) and (not rec["oracle"]):
            ok = (
                rec["mean_expectancy"] > 0.0
                and rec["frac_folds_pos"] >= float(min_frac_folds)
                and rec["beats_always_long"]
                and rec["beats_coin_flip"]
            )
            rec["pass"] = bool(ok)
            if ok:
                winners.append(rec)
        details.append(rec)

    cand = summary.loc[(~summary["control"]) & (~summary["oracle"])]
    best = None
    if len(cand):
        br = cand.sort_values("mean_expectancy", ascending=False).iloc[0]
        best = {
            "key": str(br["key"]),
            "mean_expectancy": float(br["mean_expectancy"]),
            "frac_folds_pos": float(br["frac_folds_pos"]),
        }
    return {
        "module": 1,
        "experiment": "S-2",
        "pass": bool(winners),
        "winning_keys": [w["key"] for w in winners],
        "winners": winners,
        "best_non_control": best,
        "n_rows": int(len(summary)),
        "interpretation": (
            "PASS if a non-oracle, non-control (horizon, k, model) has mean OOS "
            "non-overlapping expectancy > 0 after 2-way cost, majority folds, "
            "and beats always-long and coin-flip on that horizon. "
            "Oracle is a ceiling only. If PASS: that label is the first Signal; "
            "then (and only then) a Kaggle net on that one label is allowed. "
            "If FAIL: do not attach the Handler; do not train a net on these labels."
        ),
    }
