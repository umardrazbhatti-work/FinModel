#!/usr/bin/env python
"""Module 1 / S-2 — costed direction / large-move sweep on EURUSD 1h.

One local replay. No Transformer. Oracle is a ceiling, not a Signal.
Kaggle is not used unless a non-oracle cell PASSes and we then train one net.
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

from src.signals.features import combine_features
from src.signals.labels import (
    class_from_return,
    forward_simple_return,
    oracle_positions,
    persist_positions,
)
from src.signals.logistic import logistic_positions
from src.signals.rules import always_flat, always_long, coin_flip
from src.signals.s2_eval import s2_verdict, score_nonoverlap_trades
from src.training import generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.signal_s2")


def load_ohlc(cfg: Dict[str, Any]) -> pd.DataFrame:
    p = Path(cfg["data"]["data_dir"])
    if not p.is_absolute():
        p = ROOT / p
    pair = cfg["data"]["pair"]
    tf = cfg["data"]["tf"]
    stem = "daily" if tf in {"1d", "daily"} else tf
    path = p / f"{pair}_{stem}_aligned.parquet"
    df = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def write_report(path: Path, verdict: Dict[str, Any], summary: pd.DataFrame, runtime_h: float) -> None:
    show = summary.sort_values(["horizon", "oracle", "control", "mean_expectancy"], ascending=[True, True, False, False])
    lines = [
        "# Module 1 / S-2 — direction / large-move sweep (EURUSD 1h)",
        "",
        f"- Module: **1 Signal / Alpha** | experiment: **S-2**",
        f"- Official PASS: **{verdict['pass']}**",
        f"- Winning keys: {verdict['winning_keys'] or 'none'}",
        f"- Best non-control: {verdict.get('best_non_control')}",
        f"- Runtime hours: {runtime_h:.4f}",
        "",
        "## Summary (all cells)",
        "",
        show.to_string(index=False),
        "",
        "## Interpretation",
        "",
        verdict["interpretation"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/signal_s2_eurusd_1h.yaml")
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["project"]["seed"]))
    cost = float(cfg["data"]["cost"])
    ppy = float(cfg["signal"]["periods_per_year"])
    horizons = [int(h) for h in cfg["signal"]["horizons"]]
    k_list = [float(k) for k in cfg["signal"]["k_list"]]
    model_names = list(cfg["signal"]["models"])

    df = load_ohlc(cfg)
    close = df["close"].to_numpy(dtype=np.float64)
    ts = df["time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    feat_ohlc = combine_features(df, use_events=False)
    feat_evt = combine_features(df, use_events=True)

    wf = dict(cfg["walk_forward"])
    if args.max_folds is not None:
        wf["max_folds"] = int(args.max_folds)
    folds = generate_walk_forward_folds(primary_timestamps=ts, **wf)
    logger.info(
        "Loaded %d bars, %d folds, horizons=%s k=%s models=%s",
        len(df),
        len(folds),
        horizons,
        k_list,
        model_names,
    )

    exp_dir = ensure_dir(ROOT / cfg["output"]["dir"] / cfg["output"]["experiment_name"])
    save_config(cfg, exp_dir / "config.yaml")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    seed0 = int(cfg["project"]["seed"])

    for fold in folds:
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[fold.train_start_idx : fold.train_end_idx] = True
        logger.info(
            "===== Fold %d test %s -> %s =====",
            fold.fold_id,
            fold.test_start,
            fold.test_end,
        )
        rng_pos = coin_flip(df, seed=seed0 + fold.fold_id)

        for h in horizons:
            fwd = forward_simple_return(close, h)
            control_pos = {
                "always_flat": always_flat(df),
                "always_long": always_long(df),
                "coin_flip": rng_pos,
            }
            for cname, cpos in control_pos.items():
                scored = score_nonoverlap_trades(
                    cpos, fwd, fold.test_start_idx, fold.test_end_idx, h, cost, ppy
                )
                rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "horizon": h,
                        "k": 0.0,
                        "model": cname,
                        "key": f"h{h}_{cname}",
                        "control": True,
                        "oracle": False,
                        **scored,
                    }
                )

            for k in k_list:
                y = class_from_return(fwd, cost=cost, k=k)
                ktag = int(k) if float(k).is_integer() else k
                specs = []
                if "persist" in model_names:
                    specs.append(("persist", persist_positions(close, h, cost, k), False, False))
                if "logistic_ohlc" in model_names:
                    specs.append(
                        (
                            "logistic_ohlc",
                            logistic_positions(feat_ohlc, y, train_mask, seed=seed0 + fold.fold_id),
                            False,
                            False,
                        )
                    )
                if "logistic_evt" in model_names:
                    specs.append(
                        (
                            "logistic_evt",
                            logistic_positions(feat_evt, y, train_mask, seed=seed0 + fold.fold_id),
                            False,
                            False,
                        )
                    )
                specs.append(("oracle", oracle_positions(fwd, cost, k), False, True))
                for mname, mpos, is_ctrl, is_oracle in specs:
                    scored = score_nonoverlap_trades(
                        mpos, fwd, fold.test_start_idx, fold.test_end_idx, h, cost, ppy
                    )
                    rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "horizon": h,
                            "k": float(k),
                            "model": mname,
                            "key": f"h{h}_k{ktag}_{mname}",
                            "control": is_ctrl,
                            "oracle": is_oracle,
                            **scored,
                        }
                    )
                    logger.info(
                        "Fold %d %-22s exp=%+.4e n=%s fold_ok=%s",
                        fold.fold_id,
                        f"h{h}_k{ktag}_{mname}",
                        scored["expectancy"],
                        scored["n_trades"],
                        scored["pass_fold"],
                    )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(exp_dir / "01_fold_overview.csv", index=False)

    agg_rows = []
    for key, sub in fold_df.groupby("key", sort=False):
        al_key = f"h{int(sub['horizon'].iloc[0])}_always_long"
        cf_key = f"h{int(sub['horizon'].iloc[0])}_coin_flip"
        al = float(fold_df.loc[fold_df["key"] == al_key, "expectancy"].mean())
        cf = float(fold_df.loc[fold_df["key"] == cf_key, "expectancy"].mean())
        mean_exp = float(sub["expectancy"].mean())
        agg_rows.append(
            {
                "key": key,
                "horizon": int(sub["horizon"].iloc[0]),
                "k": float(sub["k"].iloc[0]),
                "model": str(sub["model"].iloc[0]),
                "control": bool(sub["control"].iloc[0]),
                "oracle": bool(sub["oracle"].iloc[0]),
                "mean_expectancy": mean_exp,
                "median_expectancy": float(sub["expectancy"].median()),
                "frac_folds_pos": float(sub["pass_fold"].mean()),
                "folds_pos": int(sub["pass_fold"].sum()),
                "n_folds": int(len(sub)),
                "mean_sharpe": float(sub["sharpe"].mean()),
                "mean_n_trades": float(sub["n_trades"].mean()),
                "beats_always_long": mean_exp > al,
                "beats_coin_flip": mean_exp > cf,
            }
        )
    summary = pd.DataFrame(agg_rows)
    summary.to_csv(exp_dir / "02_summary_by_cell.csv", index=False)

    verdict = s2_verdict(summary, min_frac_folds=float(cfg["evaluation"]["min_frac_folds"]))
    verdict["n_folds"] = len(folds)
    verdict["cost"] = cost
    verdict["pair"] = cfg["data"]["pair"]
    verdict["runtime_hours"] = (time.time() - t0) / 3600.0
    save_json(verdict, exp_dir / "10_go_nogo_signal_s2.json")
    write_report(exp_dir / "00_signal_report.md", verdict, summary, verdict["runtime_hours"])
    logger.info("S-2 PASS=%s winners=%s | wrote %s", verdict["pass"], verdict["winning_keys"], exp_dir)


if __name__ == "__main__":
    main()
