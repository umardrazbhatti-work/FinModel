#!/usr/bin/env python
"""Run full walk-forward experiment and export a rich analysis pack."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import SingleTFPatchModel
from src.data import MultiTFDataset, multi_tf_collate
from src.evaluation import (
    compute_economic_metrics,
    compute_statistical_metrics,
    summarize_fold_results,
)
from src.evaluation.artifacts import (
    history_to_dataframe,
    package_analysis_zip,
    save_fold_artifacts,
    write_analysis_pack,
)
from src.evaluation.metrics import baseline_historical_mean, baseline_predict_zero
from src.losses import MultiQuantilePinballLoss
from src.models import MTPTransformer
from src.training import MTPTrainer, build_optimizer_and_scheduler, generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.walk_forward")


def resolve_device(requested: str | None) -> str:
    """Pick a working device; fail fast if CUDA is advertised but unusable (e.g. P100 + new torch)."""
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
    except Exception as exc:  # noqa: BLE001 — surface as clear operator error
        raise RuntimeError(
            "CUDA is visible but no kernel can run on this GPU "
            f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'}). "
            "Common on Kaggle Tesla P100 with recent PyTorch (needs sm_70+). "
            "Fix: switch Kaggle accelerator to GPU T4, or install torch==2.1.2+cu118 "
            "before training. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return "cuda"


def build_dataset(cfg, fold_start, fold_end, mode: str) -> MultiTFDataset:
    data = cfg["data"]
    data_dir = data["data_dir"]
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = ROOT / data_dir
    return MultiTFDataset(
        pair=data["pair"],
        data_dir=str(data_path),
        tfs=data["tfs"],
        primary_tf=data["primary_tf"],
        lookback=data["lookback"],
        horizons=data["horizons"],
        quantiles=data["quantiles"],
        cost_threshold=data["cost_threshold"],
        feature_cols=data.get("feature_cols"),
        context_cols=data.get("context_cols"),
        mode=mode,
        fold_start=fold_start,
        fold_end=fold_end,
        vol_window=data.get("vol_window", 24),
        target_clip=data.get("target_clip", 5.0),
        target_type=data.get("target_type", "return"),
        tradable_tfs=data.get("tradable_tfs"),
        rv_log_transform=bool(data.get("rv_log_transform", True)),
    )


def make_loaders(train_ds, val_ds, test_ds, cfg):
    tr = cfg["training"]
    kw = dict(
        batch_size=tr["batch_size"],
        num_workers=tr.get("num_workers", 0),
        collate_fn=multi_tf_collate,
    )
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


def train_model_on_fold(
    model,
    train_loader,
    val_loader,
    cfg,
    device,
    fold_id,
    ckpt_dir,
    max_epochs: int | None = None,
    patience: int | None = None,
):
    tr = cfg["training"]
    loss_fn = MultiQuantilePinballLoss(
        quantiles=cfg["data"]["quantiles"],
        tradable_tfs=list(cfg["data"].get("tradable_tfs") or ["30m", "1h", "4h"]),
        entropy_weight=float(cfg["model"].get("gate_entropy_weight", 0.01)),
    )
    epochs = int(max_epochs if max_epochs is not None else tr["max_epochs"])
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, tr, max_epochs=epochs
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
        max_epochs=epochs,
        early_stopping_patience=int(
            patience if patience is not None else tr["early_stopping_patience"]
        ),
        checkpoint_dir=ckpt_dir,
        fold_id=fold_id,
    )
    return trainer, fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward MTP experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_eurusd_12h.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--skip-single-tf-baseline", action="store_true")
    parser.add_argument(
        "--pack-zip",
        type=str,
        default=None,
        help="Optional path for analysis zip (default: outputs/<exp>/<exp>_analysis_pack.zip)",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include .pt checkpoints in analysis zip (larger download)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    set_seed(int(cfg["project"]["seed"]))
    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    t0 = time.time()

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

    # Persist fold schedule for analysis
    out_cfg = cfg["output"]
    exp_dir = ensure_dir(ROOT / out_cfg["dir"] / out_cfg["experiment_name"])
    save_config(cfg, exp_dir / "config.yaml")
    fold_sched = []
    for f in folds:
        fold_sched.append(
            {
                "fold_id": f.fold_id,
                "train_start": str(f.train_start),
                "train_end": str(f.train_end),
                "val_start": str(f.val_start),
                "val_end": str(f.val_end),
                "test_start": str(f.test_start),
                "test_end": str(f.test_end),
                "train_start_idx": f.train_start_idx,
                "train_end_idx": f.train_end_idx,
                "val_start_idx": f.val_start_idx,
                "val_end_idx": f.val_end_idx,
                "test_start_idx": f.test_start_idx,
                "test_end_idx": f.test_end_idx,
            }
        )
    save_json({"folds": fold_sched}, exp_dir / "17_fold_schedule.json")
    pd.DataFrame(fold_sched).to_csv(exp_dir / "17_fold_schedule.csv", index=False)

    quantiles = cfg["data"]["quantiles"]
    horizons = cfg["data"]["horizons"]
    sample_rows = int(cfg.get("evaluation", {}).get("prediction_sample_rows", 500))
    fold_tables = []
    notes = []
    fold_metrics_list = []
    all_histories = []

    mtp_max_epochs = args.max_epochs or int(cfg["training"]["max_epochs"])
    baseline_max_epochs = int(
        cfg["training"].get("baseline_max_epochs", min(40, mtp_max_epochs))
    )
    baseline_patience = int(
        cfg["training"].get(
            "baseline_early_stopping_patience",
            cfg["training"]["early_stopping_patience"],
        )
    )

    for fold in folds:
        logger.info("===== Fold %d =====", fold.fold_id)
        fold_dir = ensure_dir(exp_dir / f"fold_{fold.fold_id}")
        fold_t0 = time.time()

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
        save_config(local_cfg, fold_dir / "config.yaml")

        train_loader, val_loader, test_loader = make_loaders(
            train_ds, val_ds, test_ds, local_cfg
        )
        logger.info(
            "Samples train=%d val=%d test=%d",
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )

        model = MTPTransformer(local_cfg)
        trainer, fit = train_model_on_fold(
            model,
            train_loader,
            val_loader,
            local_cfg,
            device,
            fold.fold_id,
            fold_dir,
            max_epochs=mtp_max_epochs,
        )
        test_out = trainer.evaluate(test_loader)
        stats = compute_statistical_metrics(
            test_out["predictions"], test_out["targets"], test_out["masks"], quantiles
        )
        econ = compute_economic_metrics(
            predictions=test_out["predictions"],
            raw_returns=test_out["raw_returns"],
            masks=test_out["masks"],
            cost=float(cfg["evaluation"]["cost_per_trade"]),
            signal_threshold=float(cfg["evaluation"]["signal_threshold"]),
            quantiles=quantiles,
        )

        # Baselines (zero / hist-mean are free; single-TF is trained)
        train_out = trainer.evaluate(train_loader)
        zero_preds = baseline_predict_zero(test_out["targets"], quantiles)
        mean_preds = baseline_historical_mean(
            train_out["targets"], train_out["masks"], test_out["targets"], quantiles
        )
        zero_stats = compute_statistical_metrics(
            zero_preds, test_out["targets"], test_out["masks"], quantiles
        )
        mean_stats = compute_statistical_metrics(
            mean_preds, test_out["targets"], test_out["masks"], quantiles
        )

        single_tf_pinball = None
        if not args.skip_single_tf_baseline:
            st_model = SingleTFPatchModel(
                local_cfg, primary_tf=cfg["data"]["primary_tf"]
            )
            st_trainer, st_fit = train_model_on_fold(
                st_model,
                train_loader,
                val_loader,
                local_cfg,
                device,
                fold.fold_id,
                fold_dir / "single_tf",
                max_epochs=baseline_max_epochs,
                patience=baseline_patience,
            )
            st_out = st_trainer.evaluate(test_loader)
            st_stats = compute_statistical_metrics(
                st_out["predictions"], st_out["targets"], st_out["masks"], quantiles
            )
            single_tf_pinball = st_stats["overall"]["pinball"]
            # save baseline history
            st_hist = history_to_dataframe(st_fit.get("history") or [], fold.fold_id)
            st_hist.to_csv(fold_dir / "single_tf_history.csv", index=False)
            save_json(
                {
                    "baseline": "single_tf",
                    "test_pinball": single_tf_pinball,
                    "best_val_pinball": st_fit.get("best_val_pinball"),
                    "best_epoch": st_fit.get("best_epoch"),
                },
                fold_dir / "single_tf_metrics.json",
            )

        primary_tf = cfg["evaluation"].get("primary_eval_tf", "1h")
        mean_sharpe_primary = None
        if primary_tf in econ.get("per_tf", {}):
            mean_sharpe_primary = (
                econ["per_tf"][primary_tf].get("primary", {}).get("sharpe")
            )

        metrics = {
            "fold_id": fold.fold_id,
            "best_val_pinball": fit["best_val_pinball"],
            "best_epoch": fit["best_epoch"],
            "test_pinball": stats["overall"]["pinball"],
            "test_dir_acc": stats["overall"]["directional_accuracy"],
            "test_stats": stats,
            "test_economic": econ,
            "baseline_zero_pinball": zero_stats["overall"]["pinball"],
            "baseline_mean_pinball": mean_stats["overall"]["pinball"],
            "baseline_single_tf_pinball": single_tf_pinball,
            "gate_weights": fit.get("gate_weights"),
            "n_params": model.count_parameters(),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "train_seconds": time.time() - fold_t0,
            "mean_sharpe_primary": mean_sharpe_primary,
            "fold_window": {
                "train": [str(fold.train_start), str(fold.train_end)],
                "val": [str(fold.val_start), str(fold.val_end)],
                "test": [str(fold.test_start), str(fold.test_end)],
            },
        }
        save_json(metrics, fold_dir / "metrics.json")

        save_fold_artifacts(
            fold_dir=fold_dir,
            fold_id=fold.fold_id,
            fit_result=fit,
            metrics=metrics,
            stats=stats,
            econ=econ,
            gate_tf_names=local_cfg["data"]["tfs"],
            predictions=test_out["predictions"],
            targets=test_out["targets"],
            masks=test_out["masks"],
            raw_returns=test_out["raw_returns"],
            sample_rows=sample_rows,
            quantiles=quantiles,
            horizons=horizons,
        )

        hist_df = history_to_dataframe(fit.get("history") or [], fold.fold_id)
        all_histories.append(hist_df)
        fold_tables.append(summarize_fold_results(fold.fold_id, stats, econ))
        fold_metrics_list.append(metrics)

        note = (
            f"Fold {fold.fold_id}: pinball={stats['overall']['pinball']:.5f} "
            f"zero={zero_stats['overall']['pinball']:.5f} "
            f"mean={mean_stats['overall']['pinball']:.5f} "
            f"single_tf={single_tf_pinball} "
            f"best_epoch={fit.get('best_epoch')} "
            f"sec={metrics['train_seconds']:.1f}"
        )
        notes.append(note)
        logger.info(note)

    runtime = time.time() - t0
    write_analysis_pack(
        exp_dir=exp_dir,
        cfg=cfg,
        fold_metrics=fold_metrics_list,
        all_histories=all_histories,
        fold_tables=fold_tables,
        notes=notes,
        runtime_seconds=runtime,
        device=device,
    )

    include_ckpt = bool(
        args.include_checkpoints
        or cfg.get("output", {}).get("include_checkpoints_in_pack", False)
    )
    zip_path = (
        Path(args.pack_zip)
        if args.pack_zip
        else exp_dir / f"{out_cfg['experiment_name']}_analysis_pack.zip"
    )
    if not zip_path.is_absolute():
        zip_path = ROOT / zip_path
    pack_info = package_analysis_zip(
        exp_dir=exp_dir,
        zip_path=zip_path,
        include_checkpoints=include_ckpt,
    )
    pack_info_path = exp_dir / "18_pack_info.json"
    save_json(pack_info, pack_info_path)
    # Append pack_info into the zip so the download is self-describing
    import zipfile

    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        arc = f"{exp_dir.name}/18_pack_info.json"
        # replace if present
        if arc not in zf.namelist():
            zf.write(pack_info_path, arcname=arc)
    # refresh file count for logs
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    pack_info["n_files"] = len(names)
    pack_info["files"] = names
    save_json(pack_info, pack_info_path)

    logger.info(
        "Walk-forward complete in %.2f h | analysis zip: %s (%d files)",
        runtime / 3600.0,
        pack_info["zip_path"],
        pack_info["n_files"],
    )
    for fp in names[:50]:
        logger.info("pack: %s", fp)


if __name__ == "__main__":
    main()
