#!/usr/bin/env python
"""Train a single walk-forward fold (or fold 0 by default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MultiTFDataset, multi_tf_collate
from src.evaluation import compute_economic_metrics, compute_statistical_metrics
from src.losses import MultiQuantilePinballLoss
from src.models import MTPTransformer
from src.training import MTPTrainer, generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.train_fold")


def build_dataset(cfg, fold_start, fold_end, mode: str) -> MultiTFDataset:
    data = cfg["data"]
    return MultiTFDataset(
        pair=data["pair"],
        data_dir=str(ROOT / data["data_dir"]),
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one MTP-Transformer fold")
    parser.add_argument("--config", type=str, default="configs/eurusd_1h.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build a base dataset to read primary timestamps
    base = build_dataset(cfg, None, None, mode="train")
    primary_ts = base.get_primary_timestamps()
    wf = cfg["walk_forward"]
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
    if args.fold < 0 or args.fold >= len(folds):
        raise SystemExit(f"Fold {args.fold} out of range 0..{len(folds)-1}")
    fold = folds[args.fold]
    logger.info(
        "Fold %s: train %s -> %s | val %s -> %s | test %s -> %s",
        fold.fold_id,
        fold.train_start,
        fold.train_end,
        fold.val_start,
        fold.val_end,
        fold.test_start,
        fold.test_end,
    )

    train_ds = build_dataset(cfg, fold.train_start, fold.train_end, mode="train")
    val_ds = build_dataset(cfg, fold.val_start, fold.val_end, mode="val")
    test_ds = build_dataset(cfg, fold.test_start, fold.test_end, mode="test")

    # Standardize using train fold only
    stats = train_ds.fit_standardization()
    val_ds.set_standardization(*stats)
    test_ds.set_standardization(*stats)

    logger.info("Samples train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    tr = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=tr["batch_size"],
        shuffle=True,
        num_workers=tr.get("num_workers", 0),
        collate_fn=multi_tf_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr["batch_size"],
        shuffle=False,
        num_workers=tr.get("num_workers", 0),
        collate_fn=multi_tf_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tr["batch_size"],
        shuffle=False,
        num_workers=tr.get("num_workers", 0),
        collate_fn=multi_tf_collate,
    )

    # Align model context size with dataset
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["context_cols"] = list(train_ds.context_cols)
    cfg["data"]["feature_cols"] = list(train_ds.feature_cols)

    model = MTPTransformer(cfg)
    logger.info("Parameters: %s", model.count_parameters())

    loss_fn = MultiQuantilePinballLoss(
        quantiles=cfg["data"]["quantiles"],
        tradable_tfs=["30m", "1h", "4h"],
        entropy_weight=float(cfg["model"].get("gate_entropy_weight", 0.01)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr["lr"]),
        weight_decay=float(tr["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    out_cfg = cfg["output"]
    exp_dir = ensure_dir(ROOT / out_cfg["dir"] / out_cfg["experiment_name"] / f"fold_{fold.fold_id}")
    save_config(cfg, exp_dir / "config.yaml")

    trainer = MTPTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        grad_clip=float(tr["grad_clip"]),
        log_every=int(tr.get("log_every", 50)),
        scheduler=scheduler,
    )
    max_epochs = args.max_epochs or int(tr["max_epochs"])
    fit_result = trainer.fit_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        early_stopping_patience=int(tr["early_stopping_patience"]),
        checkpoint_dir=exp_dir,
        fold_id=fold.fold_id,
    )

    # Evaluate on test
    test_out = trainer.evaluate(test_loader)
    quantiles = cfg["data"]["quantiles"]
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

    # Baselines on test
    from src.evaluation.metrics import baseline_historical_mean, baseline_predict_zero

    # Collect train targets for historical mean baseline
    train_eval = trainer.evaluate(train_loader)
    zero_preds = baseline_predict_zero(test_out["targets"], quantiles)
    mean_preds = baseline_historical_mean(
        train_eval["targets"], train_eval["masks"], test_out["targets"], quantiles
    )
    zero_stats = compute_statistical_metrics(
        zero_preds, test_out["targets"], test_out["masks"], quantiles
    )
    mean_stats = compute_statistical_metrics(
        mean_preds, test_out["targets"], test_out["masks"], quantiles
    )

    metrics = {
        "fold_id": fold.fold_id,
        "best_val_pinball": fit_result["best_val_pinball"],
        "best_epoch": fit_result["best_epoch"],
        "test_stats": stats,
        "test_economic": econ,
        "baseline_zero_pinball": zero_stats["overall"]["pinball"],
        "baseline_mean_pinball": mean_stats["overall"]["pinball"],
        "gate_weights": fit_result.get("gate_weights"),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_params": model.count_parameters(),
    }
    save_json(metrics, exp_dir / "metrics.json")

    # Gate weights CSV
    if fit_result.get("gate_weights") is not None:
        gw = fit_result["gate_weights"]
        tfs = cfg["data"]["tfs"]
        pd.DataFrame([{"tf": t, "weight": w} for t, w in zip(tfs, gw)]).to_csv(
            exp_dir / "gates.csv", index=False
        )

    logger.info(
        "Done fold %s | test pinball=%.5f | zero=%.5f | mean=%.5f",
        fold.fold_id,
        stats["overall"]["pinball"],
        zero_stats["overall"]["pinball"],
        mean_stats["overall"]["pinball"],
    )


if __name__ == "__main__":
    main()
