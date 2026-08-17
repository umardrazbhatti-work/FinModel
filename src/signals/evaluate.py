"""Module 1 go/nogo for costed Signals."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def signal_verdict(
    summary: pd.DataFrame,
    min_mean_expectancy: float,
    min_frac_folds: float,
    control_names: List[str],
) -> Dict[str, Any]:
    """A candidate PASSes if mean OOS expectancy > 0, majority folds, beats controls."""
    winners: List[str] = []
    details = []
    control_means = {
        n: float(summary.loc[summary["rule"] == n, "mean_expectancy"].iloc[0])
        for n in control_names
        if n in set(summary["rule"])
    }
    al = control_means.get("always_long", 0.0)
    cf = control_means.get("coin_flip", 0.0)

    for _, row in summary.iterrows():
        name = str(row["rule"])
        rec = {
            "rule": name,
            "control": bool(row["control"]),
            "mean_expectancy": float(row["mean_expectancy"]),
            "frac_folds_pos": float(row["frac_folds_pos"]),
            "beats_always_long": float(row["mean_expectancy"]) > al,
            "beats_coin_flip": float(row["mean_expectancy"]) > cf,
            "pass": False,
        }
        if not row["control"]:
            ok = (
                rec["mean_expectancy"] > float(min_mean_expectancy)
                and rec["frac_folds_pos"] >= float(min_frac_folds)
                and rec["beats_always_long"]
                and rec["beats_coin_flip"]
            )
            rec["pass"] = bool(ok)
            if ok:
                winners.append(name)
        details.append(rec)

    best = None
    cand = summary.loc[~summary["control"]]
    if len(cand):
        best_row = cand.sort_values("mean_expectancy", ascending=False).iloc[0]
        best = {
            "rule": str(best_row["rule"]),
            "mean_expectancy": float(best_row["mean_expectancy"]),
            "frac_folds_pos": float(best_row["frac_folds_pos"]),
        }

    return {
        "module": 1,
        "experiment": "S-1",
        "pass": bool(winners),
        "winning_rules": winners,
        "best_non_control": best,
        "controls": control_means,
        "rules": details,
        "interpretation": (
            "PASS if a non-control rule has mean OOS expectancy > 0, "
            "majority folds positive, and beats always-long and coin-flip. "
            "If PASS: that rule is the first Signal; then combine with the locked Handler. "
            "If FAIL: these rules are not Signals; next is S-2 (do not attach the Handler)."
        ),
    }
