#!/usr/bin/env python
"""
Single-TF realized-volatility pilot (Stage 1 target pivot).

Trains SingleTFPatchModel only. Reports pinball + OOS corr(q50, y) / R^2.
Success (default): mean corr on primary horizon > 0.15 across folds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import SingleTFPatchModel
from src.data import MultiTFDataset, multi_tf_collate
from src.evaluation.metrics import (
    baseline_historical_mean,
    baseline_predict_zero,
    compute_statistical_metrics,
)
from src.losses import MultiQuantilePinballLoss
from src.training import MTPTrainer, build_optimizer_and_scheduler, generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.rv_pilot")


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


def corr_and_r2(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    m = mask > 0.5
    if m.sum() < 10:
        return {"corr": float("nan"), "r2": float("nan"), "n": int(m.sum())}
    p, t = pred[m], y[m]
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"corr": corr, "r2": float(r2), "n": int(m.sum()), "std_pred": float(np.std(p)), "std_y": float(np.std(t))}


def evaluate_rv_skill(predictions, targets, masks, quantiles, primary_tf: str) -> dict:
    try:
        med_i = list(quantiles).index(0.5)
    except ValueError:
        med_i = len(quantiles) // 2
    per_h = []
    pred = predictions[primary_tf]
    y = targets[primary_tf]
    m = masks[primary_tf]
    for h in range(pred.shape[1]):
        sk = corr_and_r2(pred[:, h, med_i], y[:, h], m[:, h])
        per_h.append({"horizon_idx": h, **sk})
    corrs = [x["corr"] for x in per_h if np.isfinite(x["corr"])]
    return {
        "per_horizon": per_h,
        "mean_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "best_horizon_corr": float(np.max(corrs)) if corrs else float("nan"),
    }


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
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    set_seed(int(cfg["project"]["seed"]))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | target_type=%s", device, cfg["data"].get("target_type"))

    if str(cfg["data"].get("target_type", "")).lower() not in ("realized_vol", "rv"):
        logger.warning("Config target_type is not realized_vol — pilot expects RV targets")

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

    quantiles = cfg["data"]["quantiles"]
    primary_tf = cfg["data"]["primary_tf"]
    success_corr = float(cfg.get("evaluation", {}).get("success_corr", 0.15))
    primary_h = int(cfg.get("evaluation", {}).get("primary_eval_horizon_idx", 1))
    max_epochs = args.max_epochs or int(cfg["training"]["max_epochs"])

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
        stats = compute_statistical_metrics(
            test_out["predictions"],
            test_out["targets"],
            test_out["masks"],
            quantiles,
        )
        skill = evaluate_rv_skill(
            test_out["predictions"],
            test_out["targets"],
            test_out["masks"],
            quantiles,
            primary_tf,
        )

        # Baselines on RV scale
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
        # hist-mean corr is 0 by construction for constant; still report pinball

        primary_skill = skill["per_horizon"][primary_h] if primary_h < len(skill["per_horizon"]) else skill["per_horizon"][-1]
        row = {
            "fold_id": fold.fold_id,
            "best_epoch": fit.get("best_epoch"),
            "best_val_pinball": fit.get("best_val_pinball"),
            "test_pinball": stats["overall"]["pinball"],
            "baseline_zero_pinball": zero_stats["overall"]["pinball"],
            "baseline_mean_pinball": mean_stats["overall"]["pinball"],
            "mean_corr_all_h": skill["mean_corr"],
            "primary_h_corr": primary_skill["corr"],
            "primary_h_r2": primary_skill["r2"],
            "primary_h_std_ratio": (
                primary_skill["std_pred"] / (primary_skill["std_y"] + 1e-12)
                if primary_skill.get("std_y")
                else float("nan")
            ),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "pass_corr": bool(
                np.isfinite(primary_skill["corr"]) and primary_skill["corr"] > success_corr
            ),
        }
        fold_rows.append(row)
        save_json(
            {
                **row,
                "skill": skill,
                "stats": stats,
                "fold_window": {
                    "train": [str(fold.train_start), str(fold.train_end)],
                    "val": [str(fold.val_start), str(fold.val_end)],
                    "test": [str(fold.test_start), str(fold.test_end)],
                },
            },
            fold_dir / "metrics.json",
        )
        logger.info(
            "Fold %d | pinball=%.4f zero=%.4f mean=%.4f | primary_corr=%.4f r2=%.4f best_epoch=%s",
            fold.fold_id,
            row["test_pinball"],
            row["baseline_zero_pinball"],
            row["baseline_mean_pinball"],
            row["primary_h_corr"] if np.isfinite(row["primary_h_corr"]) else float("nan"),
            row["primary_h_r2"] if np.isfinite(row["primary_h_r2"]) else float("nan"),
            row["best_epoch"],
        )

    overview = pd.DataFrame(fold_rows)
    overview.to_csv(exp_dir / "01_fold_overview.csv", index=False)

    mean_corr = float(overview["primary_h_corr"].mean())
    frac_pass = float(overview["pass_corr"].mean())
    go = {
        "target_type": "realized_vol",
        "model": "single_tf_patch",
        "success_corr_threshold": success_corr,
        "primary_horizon_idx": primary_h,
        "mean_primary_h_corr": mean_corr,
        "median_primary_h_corr": float(overview["primary_h_corr"].median()),
        "frac_folds_pass_corr": frac_pass,
        "mean_test_pinball": float(overview["test_pinball"].mean()),
        "mean_zero_pinball": float(overview["baseline_zero_pinball"].mean()),
        "mean_hist_mean_pinball": float(overview["baseline_mean_pinball"].mean()),
        "pass": bool(mean_corr > success_corr and frac_pass >= 0.5),
        "runtime_hours": (time.time() - t0) / 3600.0,
        "n_folds": len(overview),
        "interpretation": (
            "PASS if mean OOS corr(q50, RV) > threshold and majority of folds pass. "
            "If PASS: plumbing confirmed; revisit multi-TF from a learnable target. "
            "If FAIL: feature/horizon set still weak before any multi-TF work."
        ),
    }
    save_json(go, exp_dir / "10_go_nogo_rv_pilot.json")

    report = [
        "# Single-TF realized-vol pilot report",
        "",
        f"- Pair: {cfg['data']['pair']}",
        f"- Primary TF: {primary_tf}",
        f"- RV horizons (bars): {cfg['data']['horizons'].get(primary_tf)}",
        f"- Mean primary-horizon corr(q50, y): **{mean_corr:.4f}** (threshold {success_corr})",
        f"- Frac folds pass: **{frac_pass:.2f}**",
        f"- Pilot PASS: **{go['pass']}**",
        f"- Mean test pinball: {go['mean_test_pinball']:.4f} (zero {go['mean_zero_pinball']:.4f})",
        f"- Runtime hours: {go['runtime_hours']:.3f}",
        "",
        "## Per fold",
        "",
        overview.to_string(index=False),
        "",
    ]
    (exp_dir / "00_pilot_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("Pilot PASS=%s mean_corr=%.4f | wrote %s", go["pass"], mean_corr, exp_dir)


if __name__ == "__main__":
    main()
