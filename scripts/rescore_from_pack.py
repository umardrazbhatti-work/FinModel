#!/usr/bin/env python
"""
Re-score economic metrics from an analysis pack's predictions_sample.csv files.

Uses the fixed wealth-curve max-drawdown. Note: predictions_sample is an evenly
spaced subset (default 500 rows/TF/horizon), so Sharpe/DD are approximate but
use the same signal rule as training-time eval — and are no longer numerically absurd.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.economic import score_position_returns

PPY = {"30m": 24 * 2 * 252, "1h": 24 * 252, "4h": 6 * 252}


def score_sample_frame(
    df: pd.DataFrame,
    cost: float,
    signal_threshold: float,
) -> pd.DataFrame:
    rows = []
    for (fold_id, tf, h_idx), g in df.groupby(["fold_id", "tf", "horizon_idx"], sort=True):
        g = g.sort_values("row")
        # Economic PnL uses all rows with raw_return; mask only affects training loss.
        # For trading score we can either:
        #  (a) trade only when mask=1 (model was trained on those), or
        #  (b) always mark to market.
        # Use (b) for full-path economics, but also report mask-valid subset.
        q50 = g["q50"].to_numpy(dtype=np.float64)
        raw = g["raw_return"].to_numpy(dtype=np.float64)
        mask = g["mask"].to_numpy(dtype=np.float64) if "mask" in g.columns else np.ones(len(g))

        pos = np.zeros_like(q50)
        pos[q50 > signal_threshold] = 1.0
        pos[q50 < -signal_threshold] = -1.0

        # Full sample path (chronological subsample)
        full = score_position_returns(
            pos, raw, cost=cost, periods_per_year=float(PPY.get(str(tf), 24 * 252))
        )
        # Masked path (only bars model considered "tradable" in loss)
        valid = mask > 0.5
        if valid.sum() >= 2:
            masked = score_position_returns(
                pos[valid],
                raw[valid],
                cost=cost,
                periods_per_year=float(PPY.get(str(tf), 24 * 252)),
            )
        else:
            masked = {k: float("nan") for k in full}

        rows.append(
            {
                "fold_id": int(fold_id),
                "tf": tf,
                "horizon_idx": int(h_idx),
                "n_sample": int(len(g)),
                "n_mask_valid": int(valid.sum()),
                "signal_threshold": signal_threshold,
                "cost": cost,
                "mean_q50": float(np.mean(q50)),
                "pct_long": full["pct_long"],
                "pct_short": full["pct_short"],
                "pct_flat": full["pct_flat"],
                "sharpe": full["sharpe"],
                "total_return": full["total_return"],
                "max_drawdown": full["max_drawdown"],
                "max_drawdown_abs": full["max_drawdown_abs"],
                "final_wealth": full["final_wealth"],
                "hit_rate": full["hit_rate"],
                "profit_factor": full["profit_factor"],
                "turnover": full["turnover"],
                "calmar": full["calmar"],
                "sharpe_mask_valid": masked.get("sharpe"),
                "total_return_mask_valid": masked.get("total_return"),
                "max_drawdown_mask_valid": masked.get("max_drawdown"),
            }
        )
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> dict:
    # Primary: 1h horizon 0 if present, else mean across primary horizons
    primary = scored[(scored["tf"] == "1h") & (scored["horizon_idx"] == 0)]
    if primary.empty:
        primary = scored[scored["horizon_idx"] == 0]

    def _mean(col: str) -> float:
        return float(primary[col].mean()) if len(primary) and col in primary else float("nan")

    # Multi-TF pinball comparison is outside this file; economic only here
    by_fold = (
        primary.groupby("fold_id")
        .agg(
            sharpe=("sharpe", "mean"),
            total_return=("total_return", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            pct_short=("pct_short", "mean"),
            pct_long=("pct_long", "mean"),
            pct_flat=("pct_flat", "mean"),
        )
        .reset_index()
    )

    return {
        "n_folds_primary_1h_h0": int(len(primary)),
        "mean_sharpe_1h_h0": _mean("sharpe"),
        "mean_total_return_1h_h0": _mean("total_return"),
        "mean_max_drawdown_1h_h0": _mean("max_drawdown"),
        "mean_pct_short_1h_h0": _mean("pct_short"),
        "mean_pct_long_1h_h0": _mean("pct_long"),
        "mean_pct_flat_1h_h0": _mean("pct_flat"),
        "worst_max_drawdown_1h_h0": float(primary["max_drawdown"].min()) if len(primary) else float("nan"),
        "best_sharpe_1h_h0": float(primary["sharpe"].max()) if len(primary) else float("nan"),
        "per_fold_primary": by_fold.to_dict(orient="records"),
        "note": (
            "Scored from predictions_sample (evenly spaced subset per fold). "
            "Max drawdown is fractional on wealth=1+cumsum(net); values must be in ~[-1,0]."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-dir",
        type=str,
        required=True,
        help="Path to extracted analysis pack (contains fold_*/predictions_sample.csv)",
    )
    parser.add_argument("--cost", type=float, default=0.0001)
    parser.add_argument("--signal-threshold", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: pack-dir/rescore_fixed_econ)",
    )
    args = parser.parse_args()

    pack = Path(args.pack_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pack / "rescore_fixed_econ"
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for p in sorted(pack.glob("fold_*/predictions_sample.csv")):
        parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(f"No fold_*/predictions_sample.csv under {pack}")

    df = pd.concat(parts, ignore_index=True)
    scored = score_sample_frame(df, cost=args.cost, signal_threshold=args.signal_threshold)
    scored.to_csv(out_dir / "economic_rescored_by_horizon.csv", index=False)

    summary = summarize(scored)
    summary["cost"] = args.cost
    summary["signal_threshold"] = args.signal_threshold
    summary["pack_dir"] = str(pack)
    (out_dir / "economic_rescore_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Compact primary table
    primary = scored[(scored["tf"] == "1h") & (scored["horizon_idx"] == 0)].copy()
    primary.to_csv(out_dir / "economic_primary_1h_h0.csv", index=False)

    # All horizons mean by fold for 1h
    h1 = scored[scored["tf"] == "1h"].copy()
    h1.to_csv(out_dir / "economic_1h_all_horizons.csv", index=False)

    print("Wrote:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
