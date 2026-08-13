"""Result aggregation and reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from src.utils.io import ensure_dir, save_json


def summarize_fold_results(fold_id: int, stats: dict, econ: dict) -> pd.DataFrame:
    rows = []
    overall_s = stats.get("overall", {})
    overall_e = econ.get("overall", {})
    rows.append(
        {
            "fold_id": fold_id,
            "scope": "overall",
            "tf": "all",
            "pinball": overall_s.get("pinball"),
            "directional_accuracy": overall_s.get("directional_accuracy"),
            "mean_sharpe": overall_e.get("mean_sharpe"),
        }
    )
    for tf, s in stats.get("per_tf", {}).items():
        e_primary = econ.get("per_tf", {}).get(tf, {}).get("primary", {})
        rows.append(
            {
                "fold_id": fold_id,
                "scope": "tf",
                "tf": tf,
                "pinball": s.get("pinball"),
                "directional_accuracy": s.get("directional_accuracy"),
                "sharpe": e_primary.get("sharpe"),
                "total_return": e_primary.get("total_return"),
                "max_drawdown": e_primary.get("max_drawdown"),
                "hit_rate": e_primary.get("hit_rate"),
                "turnover": e_primary.get("turnover"),
                "profit_factor": e_primary.get("profit_factor"),
            }
        )
    return pd.DataFrame(rows)


def write_experiment_summary(
    fold_rows: List[pd.DataFrame],
    output_dir: Union[str, Path],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    if fold_rows:
        all_df = pd.concat(fold_rows, ignore_index=True)
    else:
        all_df = pd.DataFrame()

    stats_path = output_dir / "summary_stats.csv"
    econ_path = output_dir / "summary_economic.csv"

    if not all_df.empty:
        stats_cols = [c for c in ["fold_id", "scope", "tf", "pinball", "directional_accuracy"] if c in all_df.columns]
        econ_cols = [
            c
            for c in [
                "fold_id",
                "scope",
                "tf",
                "sharpe",
                "mean_sharpe",
                "total_return",
                "max_drawdown",
                "hit_rate",
                "turnover",
                "profit_factor",
            ]
            if c in all_df.columns
        ]
        all_df[stats_cols].to_csv(stats_path, index=False)
        all_df[econ_cols].to_csv(econ_path, index=False)
    else:
        all_df.to_csv(stats_path, index=False)
        all_df.to_csv(econ_path, index=False)

    meta = {"n_folds": len(fold_rows)}
    if extra:
        meta.update(extra)
    meta_path = output_dir / "summary_meta.json"
    save_json(meta, meta_path)
    return {"stats": stats_path, "economic": econ_path, "meta": meta_path}
