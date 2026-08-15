#!/usr/bin/env python
"""
Fair multi-TF vs single-TF realized-vol comparison (Stage 1.7).

Same target (log-RV), horizons (4, 12), folds, and optim.
Trains MTP (all TFs) and SingleTFPatchModel (1h) with identical epoch/patience.
Also scores lagged-RV persistence and HAR-OLS (CPU, no extra architecture).

Fair bar: MTP mean OOS pinball < single-TF, and primary-horizon corr not worse.
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
from src.data import MultiTFDataset, multi_tf_collate
from src.evaluation.metrics import (
    baseline_historical_mean,
    baseline_predict_zero,
    compute_statistical_metrics,
    evaluate_rv_skill,
)
from src.losses import MultiQuantilePinballLoss
from src.models import MTPTransformer
from src.training import MTPTrainer, build_optimizer_and_scheduler, generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.rv_comparison")


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


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model_on_fold(
    model,
    train_loader,
    val_loader,
    cfg,
    device: str,
    fold_id: int,
    ckpt_dir: Path,
    max_epochs: int,
    patience: int,
    entropy_weight: float,
):
    tradable = list(cfg["data"].get("tradable_tfs") or ["1h"])
    loss_fn = MultiQuantilePinballLoss(
        quantiles=cfg["data"]["quantiles"],
        tradable_tfs=tradable,
        entropy_weight=float(entropy_weight),
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg["training"], max_epochs=max_epochs
    )
    trainer = MTPTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        grad_clip=float(cfg["training"]["grad_clip"]),
        log_every=int(cfg["training"].get("log_every", 50)),
        scheduler=scheduler,
    )
    fit = trainer.fit_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        checkpoint_dir=ckpt_dir,
        fold_id=fold_id,
    )
    return trainer, fit


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


def _history_df(history: List[dict], fold_id: int, model: str) -> pd.DataFrame:
    rows = []
    for row in history or []:
        rec = {"fold_id": fold_id, "model": model}
        rec.update({k: v for k, v in row.items() if k != "gate_weights"})
        gw = row.get("gate_weights")
        if gw is not None:
            rec["gate_weights"] = str(gw)
        rows.append(rec)
    return pd.DataFrame(rows)


def _verdict(
    mean_mtp_pb: float,
    mean_stf_pb: float,
    mean_mtp_corr: float,
    mean_stf_corr: float,
    pinball_match_rel: float,
    corr_match_abs: float,
) -> Dict[str, Any]:
    delta_pb = float(mean_mtp_pb - mean_stf_pb)
    delta_corr = float(mean_mtp_corr - mean_stf_corr)
    rel_pb = delta_pb / (abs(mean_stf_pb) + 1e-12)
    beats_pb = delta_pb < 0
    beats_corr = delta_corr > 0
    matches_pb = abs(rel_pb) <= pinball_match_rel
    matches_corr = abs(delta_corr) <= corr_match_abs
    if beats_pb and beats_corr:
        label = "BEATS"
    elif (beats_pb or matches_pb) and (beats_corr or matches_corr):
        label = "MATCHES"
    else:
        label = "LOSES"
    return {
        "delta_pinball_mtp_minus_stf": delta_pb,
        "delta_corr_mtp_minus_stf": delta_corr,
        "rel_pinball_delta": rel_pb,
        "beats_pinball": beats_pb,
        "beats_corr": beats_corr,
        "matches_pinball": matches_pb,
        "matches_corr": matches_corr,
        "verdict": label,
        "fair_bar_pass": label in ("BEATS", "MATCHES"),
        "architecture_expansion_allowed": label == "BEATS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eurusd_rv_multi_tf.yaml",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--skip-single-tf", action="store_true")
    parser.add_argument("--skip-har", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    set_seed(int(cfg["project"]["seed"]))
    device = resolve_device(args.device)
    logger.info(
        "Device: %s | target_type=%s | tfs=%s",
        device,
        cfg["data"].get("target_type"),
        cfg["data"].get("tfs"),
    )

    if str(cfg["data"].get("target_type", "")).lower() not in ("realized_vol", "rv"):
        logger.warning("Config target_type is not realized_vol")

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
    primary_h = int(cfg.get("evaluation", {}).get("primary_eval_horizon_idx", 1))
    ev = cfg.get("evaluation", {})
    pinball_match_rel = float(ev.get("pinball_match_rel", 0.01))
    corr_match_abs = float(ev.get("corr_match_abs", 0.02))
    har_windows = list(ev.get("har_windows") or [4, 12, 24, 120])
    sample_rows = int(ev.get("prediction_sample_rows", 500))

    tr = cfg["training"]
    mtp_epochs = int(args.max_epochs or tr["max_epochs"])
    mtp_patience = int(tr["early_stopping_patience"])
    # Identical budget unless caller overrides max-epochs (applies to both)
    stf_epochs = int(args.max_epochs or tr.get("baseline_max_epochs", mtp_epochs))
    stf_patience = int(tr.get("baseline_early_stopping_patience", mtp_patience))
    if args.max_epochs is not None:
        stf_epochs = mtp_epochs
        stf_patience = min(stf_patience, mtp_patience)

    fold_rows: List[dict] = []
    t0 = time.time()

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
        local_cfg["data"]["tradable_tfs"] = list(train_ds.tradable_tfs)
        local_cfg["data"]["horizons"] = {
            k: list(v) for k, v in train_ds.horizons.items() if k in train_ds.tradable_tfs
        }
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

        # --- MTP ---
        mtp = MTPTransformer(local_cfg)
        logger.info("MTP params=%d", count_params(mtp))
        mtp_trainer, mtp_fit = train_model_on_fold(
            mtp,
            train_loader,
            val_loader,
            local_cfg,
            device,
            fold.fold_id,
            fold_dir,
            max_epochs=mtp_epochs,
            patience=mtp_patience,
            entropy_weight=float(local_cfg["model"].get("gate_entropy_weight", 0.01)),
        )
        mtp_test = mtp_trainer.evaluate(test_loader)
        mtp_score = score_model(
            mtp_test["predictions"],
            mtp_test["targets"],
            mtp_test["masks"],
            quantiles,
            primary_tf,
            primary_h,
        )
        _history_df(mtp_fit.get("history") or [], fold.fold_id, "mtp").to_csv(
            fold_dir / "history.csv", index=False
        )

        # --- single-TF (same loaders, same epochs) ---
        stf_score: Optional[Dict[str, Any]] = None
        stf_fit: Dict[str, Any] = {}
        if not args.skip_single_tf:
            stf = SingleTFPatchModel(local_cfg, primary_tf=primary_tf)
            logger.info("Single-TF params=%d", count_params(stf))
            stf_trainer, stf_fit = train_model_on_fold(
                stf,
                train_loader,
                val_loader,
                local_cfg,
                device,
                fold.fold_id,
                fold_dir / "single_tf",
                max_epochs=stf_epochs,
                patience=stf_patience,
                entropy_weight=0.0,
            )
            stf_test = stf_trainer.evaluate(test_loader)
            stf_score = score_model(
                stf_test["predictions"],
                stf_test["targets"],
                stf_test["masks"],
                quantiles,
                primary_tf,
                primary_h,
            )
            _history_df(stf_fit.get("history") or [], fold.fold_id, "single_tf").to_csv(
                fold_dir / "single_tf_history.csv", index=False
            )
            save_json(
                {
                    "baseline": "single_tf",
                    **{
                        k: stf_score[k]
                        for k in (
                            "pinball",
                            "primary_h_corr",
                            "primary_h_r2",
                            "primary_h_std_ratio",
                            "mean_corr_all_h",
                        )
                    },
                    "best_val_pinball": stf_fit.get("best_val_pinball"),
                    "best_epoch": stf_fit.get("best_epoch"),
                    "n_params": count_params(stf),
                    "skill": stf_score["skill"],
                    "stats": stf_score["stats"],
                },
                fold_dir / "single_tf_metrics.json",
            )

        # --- free neural baselines (zero / hist-mean) ---
        train_out = mtp_trainer.evaluate(train_loader)
        zero_preds = baseline_predict_zero(mtp_test["targets"], quantiles)
        mean_preds = baseline_historical_mean(
            train_out["targets"],
            train_out["masks"],
            mtp_test["targets"],
            quantiles,
        )
        zero_stats = compute_statistical_metrics(
            zero_preds, mtp_test["targets"], mtp_test["masks"], quantiles
        )
        mean_stats = compute_statistical_metrics(
            mean_preds, mtp_test["targets"], mtp_test["masks"], quantiles
        )

        # --- HAR / persistence (aligned to dataset sample order = test loader) ---
        har_score = None
        pers_score = None
        if not args.skip_har:
            closes = train_ds.closes[primary_tf]
            horizons = list(train_ds.horizons[primary_tf])
            classical = run_classical_rv_baselines(
                closes=closes,
                train_end_indices=train_ds.sample_indices,
                test_end_indices=test_ds.sample_indices,
                horizons=horizons,
                quantiles=quantiles,
                windows=har_windows,
                log_transform=bool(local_cfg["data"].get("rv_log_transform", True)),
                eps=float(getattr(train_ds, "eps", 1e-8)),
                primary_tf=primary_tf,
            )
            # Prefer neural-eval targets (identical y) for skill; HAR y is the same
            # construction. Score HAR preds against MTP test targets/masks.
            har_score = score_model(
                classical["har_ols"]["predictions"],
                mtp_test["targets"],
                mtp_test["masks"],
                quantiles,
                primary_tf,
                primary_h,
            )
            pers_score = score_model(
                classical["persistence"]["predictions"],
                mtp_test["targets"],
                mtp_test["masks"],
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

            # aligned sample dump
            try:
                med_i = quantiles.index(0.5)
            except ValueError:
                med_i = len(quantiles) // 2
            n = min(sample_rows, mtp_test["targets"][primary_tf].shape[0])
            dump = {
                "y_h0": mtp_test["targets"][primary_tf][:n, 0],
                "y_h1": mtp_test["targets"][primary_tf][:n, 1]
                if mtp_test["targets"][primary_tf].shape[1] > 1
                else np.full(n, np.nan),
                "mtp_q50_h0": mtp_test["predictions"][primary_tf][:n, 0, med_i],
                "mtp_q50_h1": mtp_test["predictions"][primary_tf][:n, 1, med_i]
                if mtp_test["predictions"][primary_tf].shape[1] > 1
                else np.full(n, np.nan),
            }
            if stf_score is not None:
                dump["stf_q50_h0"] = stf_test["predictions"][primary_tf][:n, 0, med_i]
                if stf_test["predictions"][primary_tf].shape[1] > 1:
                    dump["stf_q50_h1"] = stf_test["predictions"][primary_tf][:n, 1, med_i]
            dump["har_q50_h0"] = classical["har_ols"]["predictions"][primary_tf][:n, 0, med_i]
            if classical["har_ols"]["predictions"][primary_tf].shape[1] > 1:
                dump["har_q50_h1"] = classical["har_ols"]["predictions"][primary_tf][
                    :n, 1, med_i
                ]
            dump["persist_q50_h0"] = classical["persistence"]["predictions"][primary_tf][
                :n, 0, med_i
            ]
            if classical["persistence"]["predictions"][primary_tf].shape[1] > 1:
                dump["persist_q50_h1"] = classical["persistence"]["predictions"][
                    primary_tf
                ][:n, 1, med_i]
            pd.DataFrame(dump).to_csv(fold_dir / "predictions_sample.csv", index=False)

        gates = mtp_fit.get("gate_weights")
        if gates is not None:
            g = np.asarray(gates, dtype=np.float64).reshape(-1)
            pd.DataFrame(
                {"tf": list(local_cfg["data"]["tfs"])[: len(g)], "gate_weight": g}
            ).to_csv(fold_dir / "gates.csv", index=False)

        row = {
            "fold_id": fold.fold_id,
            "mtp_best_epoch": mtp_fit.get("best_epoch"),
            "mtp_best_val_pinball": mtp_fit.get("best_val_pinball"),
            "mtp_pinball": mtp_score["pinball"],
            "mtp_primary_h_corr": mtp_score["primary_h_corr"],
            "mtp_primary_h_r2": mtp_score["primary_h_r2"],
            "mtp_primary_h_std_ratio": mtp_score["primary_h_std_ratio"],
            "mtp_mean_corr_all_h": mtp_score["mean_corr_all_h"],
            "stf_best_epoch": stf_fit.get("best_epoch") if stf_score else None,
            "stf_pinball": stf_score["pinball"] if stf_score else None,
            "stf_primary_h_corr": stf_score["primary_h_corr"] if stf_score else None,
            "stf_primary_h_r2": stf_score["primary_h_r2"] if stf_score else None,
            "stf_mean_corr_all_h": stf_score["mean_corr_all_h"] if stf_score else None,
            "delta_pinball_mtp_minus_stf": (
                float(mtp_score["pinball"] - stf_score["pinball"])
                if stf_score
                else None
            ),
            "delta_corr_mtp_minus_stf": (
                float(mtp_score["primary_h_corr"] - stf_score["primary_h_corr"])
                if stf_score
                else None
            ),
            "mtp_wins_pinball": (
                bool(mtp_score["pinball"] < stf_score["pinball"])
                if stf_score
                else None
            ),
            "mtp_wins_corr": (
                bool(mtp_score["primary_h_corr"] > stf_score["primary_h_corr"])
                if stf_score
                else None
            ),
            "har_pinball": har_score["pinball"] if har_score else None,
            "har_primary_h_corr": har_score["primary_h_corr"] if har_score else None,
            "persist_pinball": pers_score["pinball"] if pers_score else None,
            "persist_primary_h_corr": pers_score["primary_h_corr"]
            if pers_score
            else None,
            "baseline_zero_pinball": zero_stats["overall"]["pinball"],
            "baseline_mean_pinball": mean_stats["overall"]["pinball"],
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "n_mtp_params": count_params(mtp),
            "train_seconds": time.time() - fold_t0,
        }
        fold_rows.append(row)

        save_json(
            {
                **row,
                "mtp_skill": mtp_score["skill"],
                "mtp_stats": mtp_score["stats"],
                "stf_skill": stf_score["skill"] if stf_score else None,
                "har_skill": har_score["skill"] if har_score else None,
                "persist_skill": pers_score["skill"] if pers_score else None,
                "gate_weights": gates,
                "fold_window": {
                    "train": [str(fold.train_start), str(fold.train_end)],
                    "val": [str(fold.val_start), str(fold.val_end)],
                    "test": [str(fold.test_start), str(fold.test_end)],
                },
            },
            fold_dir / "metrics.json",
        )
        logger.info(
            "Fold %d | MTP pb=%.4f corr=%.4f | STF pb=%s corr=%s | HAR pb=%s corr=%s | persist corr=%s | best_ep MTP/STF=%s/%s",
            fold.fold_id,
            row["mtp_pinball"],
            row["mtp_primary_h_corr"],
            f"{row['stf_pinball']:.4f}" if row["stf_pinball"] is not None else "na",
            f"{row['stf_primary_h_corr']:.4f}"
            if row["stf_primary_h_corr"] is not None
            else "na",
            f"{row['har_pinball']:.4f}" if row["har_pinball"] is not None else "na",
            f"{row['har_primary_h_corr']:.4f}"
            if row["har_primary_h_corr"] is not None
            else "na",
            f"{row['persist_primary_h_corr']:.4f}"
            if row["persist_primary_h_corr"] is not None
            else "na",
            row["mtp_best_epoch"],
            row["stf_best_epoch"],
        )

    overview = pd.DataFrame(fold_rows)
    overview.to_csv(exp_dir / "01_fold_overview.csv", index=False)

    def _mean(col: str) -> float:
        if col not in overview.columns or overview[col].dropna().empty:
            return float("nan")
        return float(overview[col].mean())

    mean_mtp_pb = _mean("mtp_pinball")
    mean_stf_pb = _mean("stf_pinball")
    mean_mtp_corr = _mean("mtp_primary_h_corr")
    mean_stf_corr = _mean("stf_primary_h_corr")
    verdict = _verdict(
        mean_mtp_pb,
        mean_stf_pb,
        mean_mtp_corr,
        mean_stf_corr,
        pinball_match_rel,
        corr_match_abs,
    )
    n_folds = len(overview)
    n_mtp_pb_wins = (
        int(overview["mtp_wins_pinball"].sum())
        if "mtp_wins_pinball" in overview and overview["mtp_wins_pinball"].notna().any()
        else 0
    )
    n_mtp_corr_wins = (
        int(overview["mtp_wins_corr"].sum())
        if "mtp_wins_corr" in overview and overview["mtp_wins_corr"].notna().any()
        else 0
    )

    go = {
        "target_type": "realized_vol",
        "comparison": "mtp_vs_single_tf",
        "primary_horizon_idx": primary_h,
        "n_folds": n_folds,
        "mean_mtp_pinball": mean_mtp_pb,
        "mean_stf_pinball": mean_stf_pb,
        "mean_har_pinball": _mean("har_pinball"),
        "mean_persist_pinball": _mean("persist_pinball"),
        "mean_hist_mean_pinball": _mean("baseline_mean_pinball"),
        "mean_mtp_primary_h_corr": mean_mtp_corr,
        "mean_stf_primary_h_corr": mean_stf_corr,
        "mean_har_primary_h_corr": _mean("har_primary_h_corr"),
        "mean_persist_primary_h_corr": _mean("persist_primary_h_corr"),
        "folds_mtp_wins_pinball": n_mtp_pb_wins,
        "folds_mtp_wins_corr": n_mtp_corr_wins,
        "pinball_match_rel": pinball_match_rel,
        "corr_match_abs": corr_match_abs,
        **verdict,
        "runtime_hours": (time.time() - t0) / 3600.0,
        "interpretation": (
            "BEATS = mean pinball lower AND mean primary corr higher than single-TF. "
            "MATCHES = within pinball_match_rel and/or corr_match_abs on the non-winning metric. "
            "LOSES = worse on the fair bar. Architecture expansion only if BEATS. "
            "HAR/persistence are context, not the Stage-1 fair bar."
        ),
    }
    save_json(go, exp_dir / "10_go_nogo_rv_multi_tf.json")

    summary_rows = [
        {
            "model": "mtp",
            "mean_pinball": mean_mtp_pb,
            "mean_primary_corr": mean_mtp_corr,
        },
        {
            "model": "single_tf",
            "mean_pinball": mean_stf_pb,
            "mean_primary_corr": mean_stf_corr,
        },
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
            "mean_pinball": go["mean_hist_mean_pinball"],
            "mean_primary_corr": 0.0,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(exp_dir / "02_model_vs_baselines.csv", index=False)

    report = [
        "# Multi-TF vs single-TF realized-vol comparison",
        "",
        f"- Pair: {cfg['data']['pair']}",
        f"- Input TFs: {cfg['data']['tfs']}",
        f"- Target: log-RV on {primary_tf}, horizons {cfg['data']['horizons'].get(primary_tf)}",
        f"- Primary eval horizon idx: {primary_h}",
        f"- Folds: {n_folds}",
        f"- Fair-bar verdict: **{go['verdict']}**",
        f"- MTP pinball: **{mean_mtp_pb:.4f}** vs single-TF **{mean_stf_pb:.4f}** "
        f"(Δ {go['delta_pinball_mtp_minus_stf']:+.4f})",
        f"- MTP primary corr: **{mean_mtp_corr:.4f}** vs single-TF **{mean_stf_corr:.4f}** "
        f"(Δ {go['delta_corr_mtp_minus_stf']:+.4f})",
        f"- Folds MTP wins pinball: {n_mtp_pb_wins}/{n_folds}; corr: {n_mtp_corr_wins}/{n_folds}",
        f"- HAR-OLS pinball/corr: {go['mean_har_pinball']:.4f} / {go['mean_har_primary_h_corr']:.4f}",
        f"- Persistence pinball/corr: {go['mean_persist_pinball']:.4f} / {go['mean_persist_primary_h_corr']:.4f}",
        f"- Hist-mean pinball: {go['mean_hist_mean_pinball']:.4f}",
        f"- Runtime hours: {go['runtime_hours']:.3f}",
        f"- Architecture expansion allowed: {go['architecture_expansion_allowed']}",
        "",
        "## Per fold",
        "",
        overview.to_string(index=False),
        "",
    ]
    (exp_dir / "00_comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info(
        "Verdict=%s MTP pb=%.4f STF pb=%.4f MTP corr=%.4f STF corr=%.4f | wrote %s",
        go["verdict"],
        mean_mtp_pb,
        mean_stf_pb,
        mean_mtp_corr,
        mean_stf_corr,
        exp_dir,
    )


if __name__ == "__main__":
    main()
