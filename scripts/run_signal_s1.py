#!/usr/bin/env python
"""Module 1 / S-1 — costed rule-baseline Signals on EURUSD 1h.

No training. Replay explicit long/short/flat rules on the same expanding
walk-forward folds as the locked 1h handler. Scores expectancy after costs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.economic import score_position_returns
from src.signals.evaluate import signal_verdict
from src.signals.rules import RULE_SPECS, is_control, next_bar_simple_return
from src.training import generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.signal_s1")


def _data_path(cfg: Dict[str, Any]) -> Path:
    p = Path(cfg["data"]["data_dir"])
    if not p.is_absolute():
        p = ROOT / p
    pair = cfg["data"]["pair"]
    tf = cfg["data"]["tf"]
    stem = "daily" if tf in {"1d", "daily"} else tf
    return p / f"{pair}_{stem}_aligned.parquet"


def load_ohlc(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = _data_path(cfg)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "time" not in df.columns:
        raise KeyError("expected column 'time'")
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _positions_for_rule(
    name: str,
    df: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    returns: np.ndarray,
    cost: float,
    seed: int,
) -> np.ndarray:
    spec = RULE_SPECS[name]
    kwargs = dict(spec.get("kwargs") or {})
    fn = spec["fn"]
    if spec.get("needs_train"):
        return fn(df, train_mask=train_mask, returns=returns, cost=cost, seed=seed, **kwargs)
    return fn(df, seed=seed, **kwargs)


def write_report(
    path: Path,
    verdict: Dict[str, Any],
    summary: pd.DataFrame,
    cost: float,
    n_folds: int,
    runtime_h: float,
) -> None:
    lines = [
        "# Module 1 / S-1 — costed rule-baseline Signals (EURUSD 1h)",
        "",
        f"- Module: **1 Signal / Alpha** | experiment: **S-1**",
        f"- Pair / TF: EURUSD 1h | hold: next 1 bar | cost (one-way): {cost}",
        f"- Folds: {n_folds} expanding (same bar protocol as the locked 1h handler)",
        f"- Official PASS: **{verdict['pass']}**",
        f"- Winning rules: {verdict['winning_rules'] or 'none'}",
        f"- Runtime hours: {runtime_h:.4f}",
        "",
        "## Summary by rule",
        "",
        summary.to_string(index=False),
        "",
        "## Interpretation",
        "",
        verdict["interpretation"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/signal_s1_eurusd_1h.yaml")
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["project"]["seed"]))
    cost = float(cfg["data"]["cost"])
    ppy = float(cfg["signal"]["periods_per_year"])
    rules = list(cfg["signal"]["rules"])
    unknown = [r for r in rules if r not in RULE_SPECS]
    if unknown:
        raise ValueError(f"unknown rules: {unknown}")

    df = load_ohlc(cfg)
    rets = next_bar_simple_return(df["close"].to_numpy())
    # walk-forward helper expects tz-naive UTC (dataset timestamps are naive)
    ts = df["time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    wf = dict(cfg["walk_forward"])
    if args.max_folds is not None:
        wf["max_folds"] = int(args.max_folds)
    folds = generate_walk_forward_folds(primary_timestamps=ts, **wf)
    logger.info("Loaded %d bars, %d folds, %d rules", len(df), len(folds), len(rules))

    exp_dir = ensure_dir(ROOT / cfg["output"]["dir"] / cfg["output"]["experiment_name"])
    save_config(cfg, exp_dir / "config.yaml")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    seed0 = int(cfg["project"]["seed"])

    for fold in folds:
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[fold.train_start_idx : fold.train_end_idx] = True
        test_sl = slice(fold.test_start_idx, fold.test_end_idx)
        logger.info("===== Fold %d test %s -> %s =====", fold.fold_id, fold.test_start, fold.test_end)

        for name in rules:
            pos = _positions_for_rule(
                name,
                df,
                train_mask=train_mask,
                returns=rets,
                cost=cost,
                seed=seed0 + fold.fold_id,
            )
            pos_t = pos[test_sl]
            ret_t = rets[test_sl]
            valid = np.isfinite(ret_t)
            scored = score_position_returns(
                pos_t[valid], ret_t[valid], cost=cost, periods_per_year=ppy
            )
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "rule": name,
                    "control": is_control(name),
                    "test_start": str(fold.test_start),
                    "test_end": str(fold.test_end),
                    "expectancy": scored["mean_net_return"],
                    "total_return": scored["total_return"],
                    "sharpe": scored["sharpe"],
                    "max_drawdown": scored["max_drawdown"],
                    "hit_rate": scored["hit_rate"],
                    "turnover": scored["turnover"],
                    "pct_long": scored["pct_long"],
                    "pct_short": scored["pct_short"],
                    "pct_flat": scored["pct_flat"],
                    "final_wealth": scored["final_wealth"],
                    "n": scored["n"],
                    "pass_fold": bool(
                        np.isfinite(scored["mean_net_return"])
                        and scored["mean_net_return"] > 0
                    ),
                }
            )
            logger.info(
                "Fold %d %-28s exp=%+.6e sharpe=%+.3f long=%.2f flat=%.2f",
                fold.fold_id,
                name,
                scored["mean_net_return"],
                scored["sharpe"] if np.isfinite(scored["sharpe"]) else float("nan"),
                scored["pct_long"],
                scored["pct_flat"],
            )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(exp_dir / "01_fold_overview.csv", index=False)

    agg_rows = []
    for name in rules:
        sub = fold_df[fold_df["rule"] == name]
        agg_rows.append(
            {
                "rule": name,
                "control": is_control(name),
                "mean_expectancy": float(sub["expectancy"].mean()),
                "median_expectancy": float(sub["expectancy"].median()),
                "frac_folds_pos": float(sub["pass_fold"].mean()),
                "folds_pos": int(sub["pass_fold"].sum()),
                "n_folds": int(len(sub)),
                "mean_sharpe": float(sub["sharpe"].mean()),
                "mean_total_return": float(sub["total_return"].mean()),
                "mean_max_drawdown": float(sub["max_drawdown"].mean()),
                "mean_pct_long": float(sub["pct_long"].mean()),
                "mean_pct_flat": float(sub["pct_flat"].mean()),
            }
        )
    summary = pd.DataFrame(agg_rows)
    summary.to_csv(exp_dir / "02_summary_by_rule.csv", index=False)

    ev = cfg["evaluation"]
    verdict = signal_verdict(
        summary,
        min_mean_expectancy=float(ev["min_mean_expectancy"]),
        min_frac_folds=float(ev["min_frac_folds"]),
        control_names=["always_flat", "always_long", "coin_flip"],
    )
    verdict["n_folds"] = len(folds)
    verdict["cost"] = cost
    verdict["pair"] = cfg["data"]["pair"]
    verdict["tf"] = cfg["data"]["tf"]
    verdict["runtime_hours"] = (time.time() - t0) / 3600.0
    save_json(verdict, exp_dir / "10_go_nogo_signal_s1.json")
    write_report(
        exp_dir / "00_signal_report.md",
        verdict,
        summary,
        cost=cost,
        n_folds=len(folds),
        runtime_h=verdict["runtime_hours"],
    )
    logger.info(
        "S-1 PASS=%s winners=%s | wrote %s",
        verdict["pass"],
        verdict["winning_rules"],
        exp_dir,
    )


if __name__ == "__main__":
    main()
