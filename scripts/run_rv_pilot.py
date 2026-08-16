#!/usr/bin/env python
"""
Single-TF realized-volatility specialist (Series M-A / Stage 1 pilots).

Trains SingleTFPatchModel on one TF predicting that TF's own future log-RV.
Reports pinball + OOS corr(q50, y) vs hist-mean, HAR-OLS, and lagged-RV persistence.

Series M-A success (default): mean primary-horizon corr > 0.15, majority folds
pass, and pinball < hist-mean and < HAR on that TF.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import SingleTFPatchModel
from src.baselines.har_rv import run_classical_rv_baselines
from src.data import MultiTFDataset, horizon_wall_clock, multi_tf_collate, tf_bar_hours
from src.evaluation.metrics import (
    baseline_historical_mean,
    baseline_predict_zero,
    compute_statistical_metrics,
    evaluate_rv_skill,
    specialist_rv_verdict,
)
from src.losses import MultiQuantilePinballLoss
from src.training import MTPTrainer, build_optimizer_and_scheduler, generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.rv_pilot")


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        device = requested
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        return device
    if not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        logger.info("GPU: %s (compute %d.%d)", name, major, minor)
        x = torch.ones(1, device="cuda")
        _ = (x * 2).item()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CUDA is visible but no kernel can run on this GPU. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return "cuda"


def build_dataset(cfg, fold_start, fold_end, mode: str) -> MultiTFDataset:
    data = cfg["data"]
    data_path = Path(data["data_dir"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    return MultiTFDataset(
        pair=data["pair"],
        data_dir=str(data_path),
        tfs=data["tfs"],
        primary_tf=data["primary_tf"],
        lookback=data["lookback"],
        horizons=data["horizons"],
        quantiles=data["quantiles"],
        cost_threshold=data.get("cost_threshold", 1e-4),
        feature_cols=data.get("feature_cols"),
        context_cols=data.get("context_cols") or [],
        mode=mode,
        fold_start=fold_start,
        fold_end=fold_end,
        vol_window=data.get("vol_window", 24),
        target_clip=data.get("target_clip"),
        target_type=data.get("target_type", "realized_vol"),
        tradable_tfs=data.get("tradable_tfs"),
        rv_log_transform=bool(data.get("rv_log_transform", True)),
    )


def score_model(
    predictions,
    targets,
    masks,
    quantiles,
    primary_tf: str,
    primary_h: int,
) -> Dict[str, Any]:
    stats = compute_statistical_metrics(predictions, targets, masks, quantiles)
    skill = evaluate_rv_skill(predictions, targets, masks, quantiles, primary_tf)
    primary = (
        skill["per_horizon"][primary_h]
        if primary_h < len(skill["per_horizon"])
        else skill["per_horizon"][-1]
    )
    std_y = primary.get("std_y") or 0.0
    std_ratio = (
        float(primary["std_pred"]) / (float(std_y) + 1e-12)
        if primary.get("std_pred") is not None and np.isfinite(std_y)
        else float("nan")
    )
    return {
        "pinball": stats["overall"]["pinball"],
        "primary_h_corr": primary["corr"],
        "primary_h_r2": primary["r2"],
        "primary_h_std_ratio": std_ratio,
        "mean_corr_all_h": skill["mean_corr"],
        "skill": skill,
        "stats": stats,
    }


def _history_df(history: List[dict], fold_id: int) -> pd.DataFrame:
    rows = []
    for row in history or []:
        rec = {"fold_id": fold_id}
        rec.update({k: v for k, v in row.items() if k != "gate_weights"})
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pilot_eurusd_rv_single_tf.yaml",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--skip-har", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    set_seed(int(cfg["project"]["seed"]))
    device = resolve_device(args.device)
    logger.info(
        "Device: %s | target_type=%s | primary_tf=%s",
        device,
        cfg["data"].get("target_type"),
        cfg["data"].get("primary_tf"),
    )

    if str(cfg["data"].get("target_type", "")).lower() not in ("realized_vol", "rv"):
        logger.warning("Config target_type is not realized_vol — specialist expects RV targets")

    base = build_dataset(cfg, None, None, mode="train")
    primary_ts = base.get_primary_timestamps()
    wf = dict(cfg["walk_forward"])
    if args.max_folds is not None:
        wf["max_folds"] = args.max_folds
    folds = generate_walk_forward_folds(
        primary_timestamps=primary_ts,
        min_train_bars=wf["min_train_bars"],
        val_bars=wf["val_bars"],
        test_bars=wf["test_bars"],
        step_bars=wf["step_bars"],
        purge_bars=wf["purge_bars"],
        mode=wf.get("mode", "expanding"),
        max_folds=wf.get("max_folds"),
    )
    logger.info("Generated %d folds", len(folds))

    out_cfg = cfg["output"]
    exp_dir = ensure_dir(ROOT / out_cfg["dir"] / out_cfg["experiment_name"])
    save_config(cfg, exp_dir / "config.yaml")

    quantiles = list(cfg["data"]["quantiles"])
    primary_tf = cfg["data"]["primary_tf"]
    success_corr = float(cfg.get("evaluation", {}).get("success_corr", 0.15))
    primary_h = int(cfg.get("evaluation", {}).get("primary_eval_horizon_idx", 1))
    ev = cfg.get("evaluation", {})
    har_windows = list(ev.get("har_windows") or [4, 12, 24, 120])
    require_har = bool(ev.get("require_har", True)) and not args.skip_har
    sample_rows = int(ev.get("prediction_sample_rows", 500))
    max_epochs = args.max_epochs or int(cfg["training"]["max_epochs"])
    horizon_bars = list(cfg["data"]["horizons"].get(primary_tf) or [4, 12])
    wall = horizon_wall_clock(primary_tf, horizon_bars)
    bar_hours = tf_bar_hours(primary_tf)
    logger.info(
        "Wall-clock horizons on %s (%.4g h/bar): %s",
        primary_tf,
        bar_hours,
        wall,
    )

    fold_rows = []
    t0 = time.time()

    for fold in folds:
        logger.info("===== Fold %d =====", fold.fold_id)
        fold_dir = ensure_dir(exp_dir / f"fold_{fold.fold_id}")
        train_ds = build_dataset(cfg, fold.train_start, fold.train_end, mode="train")
        val_ds = build_dataset(cfg, fold.val_start, fold.val_end, mode="val")
        test_ds = build_dataset(cfg, fold.test_start, fold.test_end, mode="test")
        stats_pack = train_ds.fit_standardization()
        val_ds.set_standardization(*stats_pack)
        test_ds.set_standardization(*stats_pack)

        local_cfg = dict(cfg)
        local_cfg["data"] = dict(cfg["data"])
        local_cfg["data"]["context_cols"] = list(train_ds.context_cols)
        local_cfg["data"]["feature_cols"] = list(train_ds.feature_cols)
        local_cfg["data"]["tradable_tfs"] = list(train_ds.tradable_tfs)
        local_cfg["data"]["horizons"] = {
            k: list(v) for k, v in train_ds.horizons.items() if k in train_ds.tradable_tfs
        }
        save_config(local_cfg, fold_dir / "config.yaml")

        tr = cfg["training"]
        kw = dict(
            batch_size=tr["batch_size"],
            num_workers=tr.get("num_workers", 0),
            collate_fn=multi_tf_collate,
        )
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
        val_loader = DataLoader(val_ds, shuffle=False, **kw)
        test_loader = DataLoader(test_ds, shuffle=False, **kw)
        logger.info(
            "Samples train=%d val=%d test=%d",
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )

        model = SingleTFPatchModel(local_cfg, primary_tf=primary_tf)
        loss_fn = MultiQuantilePinballLoss(
            quantiles=quantiles,
            tradable_tfs=list(train_ds.tradable_tfs),
            entropy_weight=0.0,
        )
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, tr, max_epochs=max_epochs
        )
        trainer = MTPTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            grad_clip=float(tr["grad_clip"]),
            log_every=int(tr.get("log_every", 50)),
            scheduler=scheduler,
        )
        fit = trainer.fit_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            early_stopping_patience=int(tr["early_stopping_patience"]),
            checkpoint_dir=fold_dir,
            fold_id=fold.fold_id,
        )
        test_out = trainer.evaluate(test_loader)
        model_score = score_model(
            test_out["predictions"],
            test_out["targets"],
            test_out["masks"],
            quantiles,
            primary_tf,
            primary_h,
        )
        _history_df(fit.get("history") or [], fold.fold_id).to_csv(
            fold_dir / "history.csv", index=False
        )

        train_out = trainer.evaluate(train_loader)
        zero_preds = baseline_predict_zero(test_out["targets"], quantiles)
        mean_preds = baseline_historical_mean(
            train_out["targets"],
            train_out["masks"],
            test_out["targets"],
            quantiles,
        )
        zero_stats = compute_statistical_metrics(
            zero_preds, test_out["targets"], test_out["masks"], quantiles
        )
        mean_stats = compute_statistical_metrics(
            mean_preds, test_out["targets"], test_out["masks"], quantiles
        )

        har_score = None
        pers_score = None
        if not args.skip_har:
            classical = run_classical_rv_baselines(
                closes=train_ds.closes[primary_tf],
                train_end_indices=train_ds.sample_indices,
                test_end_indices=test_ds.sample_indices,
                horizons=list(train_ds.horizons[primary_tf]),
                quantiles=quantiles,
                windows=har_windows,
                log_transform=bool(local_cfg["data"].get("rv_log_transform", True)),
                eps=float(getattr(train_ds, "eps", 1e-8)),
                primary_tf=primary_tf,
            )
            har_score = score_model(
                classical["har_ols"]["predictions"],
                test_out["targets"],
                test_out["masks"],
                quantiles,
                primary_tf,
                primary_h,
            )
            pers_score = score_model(
                classical["persistence"]["predictions"],
                test_out["targets"],
                test_out["masks"],
                quantiles,
                primary_tf,
                primary_h,
            )
            save_json(
                {
                    "har_ols": {
                        k: har_score[k]
                        for k in (
                            "pinball",
                            "primary_h_corr",
                            "primary_h_r2",
                            "mean_corr_all_h",
                        )
                    },
                    "persistence": {
                        k: pers_score[k]
                        for k in (
                            "pinball",
                            "primary_h_corr",
                            "primary_h_r2",
                            "mean_corr_all_h",
                        )
                    },
                    "har_windows": har_windows,
                    "har_beta": classical["har_ols"]["beta"].tolist(),
                    "skill_har": har_score["skill"],
                    "skill_persistence": pers_score["skill"],
                },
                fold_dir / "classical_metrics.json",
            )
            try:
                med_i = quantiles.index(0.5)
            except ValueError:
                med_i = len(quantiles) // 2
            n = min(sample_rows, test_out["targets"][primary_tf].shape[0])
            dump = {
                "y_h0": test_out["targets"][primary_tf][:n, 0],
                "y_h1": test_out["targets"][primary_tf][:n, 1]
                if test_out["targets"][primary_tf].shape[1] > 1
                else np.full(n, np.nan),
                "stf_q50_h0": test_out["predictions"][primary_tf][:n, 0, med_i],
                "stf_q50_h1": test_out["predictions"][primary_tf][:n, 1, med_i]
                if test_out["predictions"][primary_tf].shape[1] > 1
                else np.full(n, np.nan),
                "har_q50_h0": classical["har_ols"]["predictions"][primary_tf][:n, 0, med_i],
                "persist_q50_h0": classical["persistence"]["predictions"][primary_tf][
                    :n, 0, med_i
                ],
            }
            if classical["har_ols"]["predictions"][primary_tf].shape[1] > 1:
                dump["har_q50_h1"] = classical["har_ols"]["predictions"][primary_tf][
                    :n, 1, med_i
                ]
                dump["persist_q50_h1"] = classical["persistence"]["predictions"][
                    primary_tf
                ][:n, 1, med_i]
            pd.DataFrame(dump).to_csv(fold_dir / "predictions_sample.csv", index=False)

        row = {
            "fold_id": fold.fold_id,
            "best_epoch": fit.get("best_epoch"),
            "best_val_pinball": fit.get("best_val_pinball"),
            "test_pinball": model_score["pinball"],
            "baseline_zero_pinball": zero_stats["overall"]["pinball"],
            "baseline_mean_pinball": mean_stats["overall"]["pinball"],
            "har_pinball": har_score["pinball"] if har_score else None,
            "har_primary_h_corr": har_score["primary_h_corr"] if har_score else None,
            "persist_pinball": pers_score["pinball"] if pers_score else None,
            "persist_primary_h_corr": pers_score["primary_h_corr"]
            if pers_score
            else None,
            "mean_corr_all_h": model_score["mean_corr_all_h"],
            "primary_h_corr": model_score["primary_h_corr"],
            "primary_h_r2": model_score["primary_h_r2"],
            "primary_h_std_ratio": model_score["primary_h_std_ratio"],
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "pass_corr": bool(
                np.isfinite(model_score["primary_h_corr"])
                and model_score["primary_h_corr"] > success_corr
            ),
        }
        fold_rows.append(row)
        save_json(
            {
                **row,
                "skill": model_score["skill"],
                "stats": model_score["stats"],
                "har_skill": har_score["skill"] if har_score else None,
                "persist_skill": pers_score["skill"] if pers_score else None,
                "fold_window": {
                    "train": [str(fold.train_start), str(fold.train_end)],
                    "val": [str(fold.val_start), str(fold.val_end)],
                    "test": [str(fold.test_start), str(fold.test_end)],
                },
            },
            fold_dir / "metrics.json",
        )
        logger.info(
            "Fold %d | pinball=%.4f zero=%.4f mean=%.4f har=%s | primary_corr=%.4f r2=%.4f best_epoch=%s",
            fold.fold_id,
            row["test_pinball"],
            row["baseline_zero_pinball"],
            row["baseline_mean_pinball"],
            f"{row['har_pinball']:.4f}" if row["har_pinball"] is not None else "na",
            row["primary_h_corr"] if np.isfinite(row["primary_h_corr"]) else float("nan"),
            row["primary_h_r2"] if np.isfinite(row["primary_h_r2"]) else float("nan"),
            row["best_epoch"],
        )

    overview = pd.DataFrame(fold_rows)
    overview.to_csv(exp_dir / "01_fold_overview.csv", index=False)

    def _mean(col: str) -> float:
        if col not in overview.columns or overview[col].dropna().empty:
            return float("nan")
        return float(overview[col].mean())

    mean_corr = _mean("primary_h_corr")
    frac_pass = float(overview["pass_corr"].mean()) if len(overview) else float("nan")
    mean_pb = _mean("test_pinball")
    mean_hist = _mean("baseline_mean_pinball")
    mean_har = _mean("har_pinball")
    verdict = specialist_rv_verdict(
        mean_corr=mean_corr,
        frac_folds_pass_corr=frac_pass,
        mean_pinball=mean_pb,
        mean_hist_mean_pinball=mean_hist,
        mean_har_pinball=None if args.skip_har else mean_har,
        success_corr=success_corr,
        require_har=require_har,
    )
    go = {
        "series": "M-A",
        "target_type": "realized_vol",
        "model": "single_tf_patch",
        "pair": cfg["data"]["pair"],
        "primary_tf": primary_tf,
        "horizons_bars": horizon_bars,
        "horizons_wall_clock": wall,
        "bar_hours": bar_hours,
        "primary_horizon_idx": primary_h,
        "primary_horizon_hours": (
            wall[primary_h]["hours"] if primary_h < len(wall) else wall[-1]["hours"]
        ),
        "har_windows": har_windows,
        "mean_primary_h_corr": mean_corr,
        "median_primary_h_corr": float(overview["primary_h_corr"].median()),
        "frac_folds_pass_corr": frac_pass,
        "mean_test_pinball": mean_pb,
        "mean_zero_pinball": _mean("baseline_zero_pinball"),
        "mean_hist_mean_pinball": mean_hist,
        "mean_har_pinball": mean_har,
        "mean_persist_pinball": _mean("persist_pinball"),
        "mean_har_primary_h_corr": _mean("har_primary_h_corr"),
        "mean_persist_primary_h_corr": _mean("persist_primary_h_corr"),
        "runtime_hours": (time.time() - t0) / 3600.0,
        "n_folds": len(overview),
        "interpretation": (
            "PASS if mean OOS corr(q50, log-RV) > threshold, majority folds pass, "
            "and pinball < hist-mean and < HAR on this TF. "
            "If PASS: this clock is a Series M specialist. "
            "If FAIL: park this TF; do not feed it into M-B/C/D."
        ),
        **verdict,
    }
    save_json(go, exp_dir / "10_go_nogo_rv_pilot.json")

    pd.DataFrame(
        [
            {"model": "single_tf", "mean_pinball": mean_pb, "mean_primary_corr": mean_corr},
            {
                "model": "har_ols",
                "mean_pinball": go["mean_har_pinball"],
                "mean_primary_corr": go["mean_har_primary_h_corr"],
            },
            {
                "model": "persistence",
                "mean_pinball": go["mean_persist_pinball"],
                "mean_primary_corr": go["mean_persist_primary_h_corr"],
            },
            {
                "model": "hist_mean",
                "mean_pinball": mean_hist,
                "mean_primary_corr": 0.0,
            },
        ]
    ).to_csv(exp_dir / "02_model_vs_baselines.csv", index=False)

    report = [
        f"# Single-TF realized-vol specialist report ({primary_tf})",
        "",
        f"- Series: M-A | pair: {cfg['data']['pair']} | primary TF: **{primary_tf}**",
        f"- RV horizons (bars): {horizon_bars}",
        f"- Wall-clock horizons: {wall}",
        f"- Primary eval horizon idx: {primary_h} "
        f"(~{go['primary_horizon_hours']:.1f} hours)",
        f"- HAR windows (bars): {har_windows}",
        f"- Mean primary-horizon corr(q50, y): **{mean_corr:.4f}** (threshold {success_corr})",
        f"- Frac folds pass corr: **{frac_pass:.2f}**",
        f"- Mean test pinball: **{mean_pb:.4f}** "
        f"(hist-mean {mean_hist:.4f}; HAR {mean_har:.4f})",
        f"- Beats hist-mean: {go['beats_hist_mean']} | beats HAR: {go['beats_har']}",
        f"- Specialist PASS: **{go['pass']}**",
        f"- Runtime hours: {go['runtime_hours']:.3f}",
        "",
        "## Per fold",
        "",
        overview.to_string(index=False),
        "",
    ]
    (exp_dir / "00_pilot_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info(
        "Specialist PASS=%s tf=%s mean_corr=%.4f pinball=%.4f har=%.4f | wrote %s",
        go["pass"],
        primary_tf,
        mean_corr,
        mean_pb,
        mean_har,
        exp_dir,
    )


if __name__ == "__main__":
    main()
