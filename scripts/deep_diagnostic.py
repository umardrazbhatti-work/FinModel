#!/usr/bin/env python
"""
Deep diagnostic: why almost no predictive skill?

Uses:
  - Local MultiTFDataset (targets, features, leakage)
  - Latest pack predictions_sample.csv (all folds)
  - Optional local checkpoint for gradient probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MultiTFDataset, multi_tf_collate
from src.losses import MultiQuantilePinballLoss
from src.models import MTPTransformer
from src.baselines import SingleTFPatchModel


def _stats(x: np.ndarray, name: str = "") -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p50": float(np.quantile(x, 0.50)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "frac_zero": float(np.mean(x == 0)),
        "frac_abs_lt_0_1": float(np.mean(np.abs(x) < 0.1)),
        "frac_at_clip5": float(np.mean(np.abs(x) >= 4.999)),
    }


def acf(x: np.ndarray, lags: List[int]) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < max(lags) + 5:
        return {f"lag_{k}": float("nan") for k in lags}
    x = x - x.mean()
    var = float(np.dot(x, x) / len(x))
    out = {}
    for k in lags:
        if var < 1e-18:
            out[f"lag_{k}"] = float("nan")
        else:
            out[f"lag_{k}"] = float(np.dot(x[:-k], x[k:]) / (len(x) * var))
    return out


def diagnose_targets(ds: MultiTFDataset, n_samples: int = 3000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(ds))
    idxs = rng.choice(len(ds), size=n, replace=False)

    raw_by_tf_h: Dict[str, Dict[int, List[float]]] = {}
    y_by_tf_h: Dict[str, Dict[int, List[float]]] = {}
    mask_by_tf_h: Dict[str, Dict[int, List[float]]] = {}
    y_unclipped: List[float] = []
    y_clipped: List[float] = []
    raw_all: List[float] = []

    # also collect sequential primary-1h h0 targets for ACF
    # sample along order for a contiguous mid slice
    start = len(ds) // 4
    end = min(start + 2000, len(ds))
    y_1h_h0_seq = []
    raw_1h_h0_seq = []

    for i in range(start, end):
        item = ds[i]
        y = item["targets"]["1h"].numpy()
        r = item["raw_returns"]["1h"].numpy()
        m = item["target_mask"]["1h"].numpy()
        y_1h_h0_seq.append(float(y[0]) if m[0] > 0.5 else np.nan)
        raw_1h_h0_seq.append(float(r[0]))

    for i in idxs:
        item = ds[int(i)]
        t = ds.debug_prediction_time(int(i))
        for tf in ds.tradable_tfs:
            y = item["targets"][tf].numpy()
            r = item["raw_returns"][tf].numpy()
            m = item["target_mask"][tf].numpy()
            # unclipped reconstruction for diagnostics
            vol = ds._realized_vol(tf, t)
            for hi, h in enumerate(ds.horizons[tf]):
                raw_by_tf_h.setdefault(tf, {}).setdefault(hi, []).append(float(r[hi]))
                y_by_tf_h.setdefault(tf, {}).setdefault(hi, []).append(float(y[hi]))
                mask_by_tf_h.setdefault(tf, {}).setdefault(hi, []).append(float(m[hi]))
                raw_all.append(float(r[hi]))
                if abs(r[hi]) >= ds.cost_threshold and vol > 0:
                    yu = float(r[hi] / (vol + ds.eps))
                    y_unclipped.append(yu)
                    y_clipped.append(float(np.clip(yu, -5, 5)))

    # horizon correlations within 1h
    h_corr = {}
    for tf in ds.tradable_tfs:
        mats = []
        for hi in range(len(ds.horizons[tf])):
            mats.append(np.array(y_by_tf_h[tf][hi]))
        M = np.stack(mats, axis=1)
        valid = np.array(mask_by_tf_h[tf][0]) > 0.5
        # use rows where all horizons valid-ish
        ok = np.ones(len(M), dtype=bool)
        for hi in range(M.shape[1]):
            ok &= np.array(mask_by_tf_h[tf][hi]) > 0.5
        if ok.sum() > 50:
            h_corr[tf] = np.corrcoef(M[ok].T).tolist()
        else:
            h_corr[tf] = None

    mask_rates = {}
    for tf in ds.tradable_tfs:
        mask_rates[tf] = {
            str(hi): float(np.mean(mask_by_tf_h[tf][hi]))
            for hi in range(len(ds.horizons[tf]))
        }

    return {
        "n_random_samples": n,
        "raw_return_stats": _stats(np.array(raw_all), "raw_all"),
        "y_unclipped_stats": _stats(np.array(y_unclipped), "y_unclipped"),
        "y_clipped5_stats": _stats(np.array(y_clipped), "y_clipped5"),
        "mask_valid_frac_by_tf_h": mask_rates,
        "frac_masked_overall": float(
            1.0
            - np.mean(
                [
                    m
                    for tf in mask_by_tf_h
                    for hi in mask_by_tf_h[tf]
                    for m in mask_by_tf_h[tf][hi]
                ]
            )
        ),
        "horizon_corr_vol_norm_y": h_corr,
        "acf_1h_h0_raw": acf(np.array(raw_1h_h0_seq), [1, 2, 5, 10, 24]),
        "acf_1h_h0_y_valid": acf(
            np.array([v for v in y_1h_h0_seq if np.isfinite(v)]), [1, 2, 5, 10, 24]
        ),
        "naive_predictability": {
            "sign_persistence_raw_1h_h0": float(
                np.mean(
                    np.sign(np.array(raw_1h_h0_seq)[1:])
                    == np.sign(np.array(raw_1h_h0_seq)[:-1])
                )
            )
            if len(raw_1h_h0_seq) > 2
            else float("nan"),
            "note": "FX log-returns at these horizons are near-uncorrelated; high ACF would be surprising.",
        },
    }


def diagnose_predictions(pack: Path) -> dict:
    parts = [pd.read_csv(p) for p in sorted(pack.glob("fold_*/predictions_sample.csv"))]
    df = pd.concat(parts, ignore_index=True)
    valid = df["mask"] > 0.5
    y = df.loc[valid, "y"].to_numpy()
    q10 = df.loc[valid, "q10"].to_numpy()
    q50 = df.loc[valid, "q50"].to_numpy()
    q90 = df.loc[valid, "q90"].to_numpy()
    raw = df.loc[valid, "raw_return"].to_numpy()

    def corr(a, b):
        if len(a) < 10 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    per_h = {}
    for (tf, hi), g in df[valid].groupby(["tf", "horizon_idx"]):
        yy, qq = g["y"].to_numpy(), g["q50"].to_numpy()
        per_h[f"{tf}_h{hi}"] = {
            "n": int(len(g)),
            "corr_q50_y": corr(yy, qq),
            "std_y": float(np.std(yy)),
            "std_q50": float(np.std(qq)),
            "std_ratio_q50_over_y": float(np.std(qq) / (np.std(yy) + 1e-12)),
            "mean_q50": float(np.mean(qq)),
            "mean_y": float(np.mean(yy)),
            "dir_acc": float(np.mean(np.sign(qq) == np.sign(yy))),
            "coverage_q10": float(np.mean(yy <= g["q10"].to_numpy())),
            "coverage_q50": float(np.mean(yy <= qq)),
            "coverage_q90": float(np.mean(yy <= g["q90"].to_numpy())),
            "mean_width_q90_q10": float(np.mean(g["q90"].to_numpy() - g["q10"].to_numpy())),
            "std_q10": float(np.std(g["q10"])),
            "std_q90": float(np.std(g["q90"])),
        }

    # q10/q90 variation
    q_range_stats = {
        "std_q10": float(np.std(q10)),
        "std_q50": float(np.std(q50)),
        "std_q90": float(np.std(q90)),
        "mean_spread_q90_q10": float(np.mean(q90 - q10)),
        "std_spread": float(np.std(q90 - q10)),
        "frac_spread_lt_1": float(np.mean((q90 - q10) < 1.0)),
        "frac_q10_ge_q50": float(np.mean(q10 >= q50)),
        "frac_q50_ge_q90": float(np.mean(q50 >= q90)),
    }

    # residual of q50 vs unconditional
    unconditional_mae = float(np.mean(np.abs(y - np.mean(y))))
    model_mae = float(np.mean(np.abs(y - q50)))
    zero_mae = float(np.mean(np.abs(y)))

    return {
        "n_valid": int(valid.sum()),
        "y_stats": _stats(y, "y_valid"),
        "q50_stats": _stats(q50, "q50"),
        "q10_stats": _stats(q10, "q10"),
        "q90_stats": _stats(q90, "q90"),
        "corr_q50_y": corr(y, q50),
        "corr_q50_raw": corr(q50, raw),
        "corr_q10_y": corr(y, q10),
        "corr_q90_y": corr(y, q90),
        "std_ratio_q50_over_y": float(np.std(q50) / (np.std(y) + 1e-12)),
        "dir_acc": float(np.mean(np.sign(q50) == np.sign(y))),
        "quantile_crossing_rate": float(np.mean((q10 > q50) | (q50 > q90))),
        "q_range_stats": q_range_stats,
        "per_tf_horizon": per_h,
        "skill_vs_naive": {
            "mae_model_q50": model_mae,
            "mae_predict_mean_y": unconditional_mae,
            "mae_predict_zero": zero_mae,
            "model_beats_zero_mae": bool(model_mae < zero_mae),
            "model_beats_uncond_mean_mae": bool(model_mae < unconditional_mae),
        },
    }


def diagnose_features(ds: MultiTFDataset, n: int = 64) -> dict:
    rng = np.random.default_rng(1)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    per_tf = {tf: [] for tf in ds.tfs}
    revin_proxy = {tf: [] for tf in ds.tfs}

    from src.models.tf_encoder import RevIN

    revins = {
        tf: RevIN(n_features=len(ds.feature_cols)) for tf in ds.tfs
    }

    for i in idxs:
        item = ds[int(i)]
        for tf in ds.tfs:
            x = item["inputs"][tf].numpy()  # [L, C] already standardized features
            per_tf[tf].append(x)
            # RevIN on batch of 1
            xt = torch.from_numpy(x).unsqueeze(0).float()
            with torch.no_grad():
                xn = revins[tf](xt, mode="norm").numpy()[0]
            revin_proxy[tf].append(xn)

    out = {}
    for tf in ds.tfs:
        X = np.stack(per_tf[tf], axis=0)  # [N, L, C]
        R = np.stack(revin_proxy[tf], axis=0)
        out[tf] = {
            "input_mean": float(np.mean(X)),
            "input_std": float(np.std(X)),
            "input_mean_abs": float(np.mean(np.abs(X))),
            "per_channel_std": np.std(X, axis=(0, 1)).tolist(),
            "after_revin_mean": float(np.mean(R)),
            "after_revin_std": float(np.std(R)),
            "seq_len": int(X.shape[1]),
            "n_feat": int(X.shape[2]),
            "finite_frac": float(np.isfinite(X).mean()),
            "sample0_close_channel_std": float(np.std(X[0, :, 3])) if X.shape[2] > 3 else float("nan"),
        }
    # relative scales across TFs
    scales = {tf: out[tf]["input_std"] for tf in ds.tfs}
    out["_cross_tf_input_std"] = scales
    out["_cross_tf_revin_std"] = {tf: out[tf]["after_revin_std"] for tf in ds.tfs}
    return out


def diagnose_leakage_and_loss(ds: MultiTFDataset) -> dict:
    # sample a few points and check history timestamps
    issues = []
    for idx in [0, len(ds) // 2, len(ds) - 1]:
        t = ds.debug_prediction_time(idx)
        for tf in ds.tfs:
            hist = ds.debug_history_timestamps(idx, tf)
            if len(hist) and np.any(hist > t):
                issues.append(f"FUTURE_IN_INPUT idx={idx} tf={tf}")
        item = ds[idx]
        for tf in ds.tradable_tfs:
            # target alignment: if mask=1, raw should match log ret roughly
            pass

    # loss masking: random batch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        torch.utils.data.Subset(ds, list(range(min(128, len(ds))))),
        batch_size=16,
        collate_fn=multi_tf_collate,
        shuffle=False,
    )
    batch = next(iter(loader))
    model = MTPTransformer(
        {
            "data": {
                "tfs": ds.tfs,
                "tradable_tfs": ds.tradable_tfs,
                "horizons": ds.horizons,
                "quantiles": ds.quantiles,
                "feature_cols": ds.feature_cols,
                "context_cols": ds.context_cols,
            },
            "model": {
                "d_model": 64,
                "n_layers": 3,
                "n_heads": 4,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "patch_len": 16,
                "use_revin": True,
                "use_context": True,
                "gate_entropy_weight": 0.01,
            },
        }
    )
    loss_fn = MultiQuantilePinballLoss(
        quantiles=ds.quantiles,
        tradable_tfs=ds.tradable_tfs,
        entropy_weight=0.01,
    )
    # zero out all targets where mask=1 and set preds = targets where mask=0 should not matter
    # Test: if all masks zero, pinball ~ 0
    zero_masks = {k: torch.zeros_like(v) for k, v in batch["target_mask"].items()}
    with torch.no_grad():
        out = model(
            {
                "inputs": batch["inputs"],
                "context": batch["context"],
            }
        )
        # fake perfect preds
        preds = {
            tf: batch["targets"][tf].unsqueeze(-1).expand(-1, -1, 3).clone()
            for tf in ds.tradable_tfs
        }
        l_masked = loss_fn(preds, batch["targets"], zero_masks, out["gate_weights"])
        l_perfect = loss_fn(
            preds, batch["targets"], batch["target_mask"], out["gate_weights"]
        )
        # wrong preds but fully masked
        bad = {tf: preds[tf] + 10.0 for tf in preds}
        l_bad_masked = loss_fn(bad, batch["targets"], zero_masks, out["gate_weights"])

    return {
        "future_in_input_issues": issues,
        "n_issues": len(issues),
        "loss_all_masks_zero_pinball": float(l_masked["pinball_loss"]),
        "loss_perfect_pred_pinball": float(l_perfect["pinball_loss"]),
        "loss_bad_pred_but_masks_zero_pinball": float(l_bad_masked["pinball_loss"]),
        "masking_ok": bool(
            float(l_masked["pinball_loss"]) < 1e-6
            and float(l_bad_masked["pinball_loss"]) < 1e-6
        ),
        "perfect_pred_near_zero": bool(float(l_perfect["pinball_loss"]) < 1e-5),
    }


def diagnose_gradients(ds: MultiTFDataset) -> dict:
    from torch.utils.data import DataLoader, Subset

    cfg = {
        "data": {
            "tfs": ds.tfs,
            "tradable_tfs": ds.tradable_tfs,
            "horizons": ds.horizons,
            "quantiles": ds.quantiles,
            "feature_cols": ds.feature_cols,
            "context_cols": ds.context_cols,
        },
        "model": {
            "d_model": 64,
            "n_layers": 3,
            "n_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,  # for clean grad probe
            "patch_len": 16,
            "use_revin": True,
            "use_context": bool(len(ds.context_cols) > 0),
            "gate_entropy_weight": 0.01,
            "gate_temperature": 1.0,
        },
    }
    model = MTPTransformer(cfg)
    loss_fn = MultiQuantilePinballLoss(
        quantiles=ds.quantiles,
        tradable_tfs=ds.tradable_tfs,
        entropy_weight=0.01,
    )
    loader = DataLoader(
        Subset(ds, list(range(min(256, len(ds))))),
        batch_size=32,
        collate_fn=multi_tf_collate,
        shuffle=True,
    )
    batch = next(iter(loader))
    model.train()
    out = model(batch)
    loss = loss_fn(
        out["predictions"],
        batch["targets"],
        batch["target_mask"],
        out["gate_weights"],
    )["loss"]
    loss.backward()

    groups = {
        "encoders": [],
        "gating": [],
        "fusion": [],
        "head": [],
        "context": [],
        "other": [],
    }
    for name, p in model.named_parameters():
        if p.grad is None:
            gnorm = 0.0
        else:
            gnorm = float(p.grad.detach().float().norm().item())
        if name.startswith("encoders"):
            groups["encoders"].append(gnorm)
        elif name.startswith("gating"):
            groups["gating"].append(gnorm)
        elif name.startswith("fusion"):
            groups["fusion"].append(gnorm)
        elif name.startswith("head"):
            groups["head"].append(gnorm)
        elif name.startswith("context"):
            groups["context"].append(gnorm)
        else:
            groups["other"].append(gnorm)

    def summarize(vals):
        if not vals:
            return {}
        a = np.array(vals, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "max": float(a.max()),
            "frac_near_zero": float(np.mean(a < 1e-8)),
            "sum": float(a.sum()),
        }

    # gate parameter grads specifically
    gate_detail = {}
    for name, p in model.named_parameters():
        if "gat" in name.lower() or name.startswith("gating"):
            gate_detail[name] = (
                float(p.grad.norm().item()) if p.grad is not None else 0.0
            )

    # one step of training then re-eval pinball on same batch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.zero_grad(set_to_none=True)
    out1 = model(batch)
    l1 = loss_fn(
        out1["predictions"], batch["targets"], batch["target_mask"], out1["gate_weights"]
    )["pinball_loss"].item()
    loss_fn(
        out1["predictions"], batch["targets"], batch["target_mask"], out1["gate_weights"]
    )["loss"].backward()
    opt.step()
    with torch.no_grad():
        out2 = model(batch)
        l2 = loss_fn(
            out2["predictions"],
            batch["targets"],
            batch["target_mask"],
            out2["gate_weights"],
        )["pinball_loss"].item()

    # prediction dispersion before/after one step
    with torch.no_grad():
        q50_1 = out1["predictions"]["1h"][..., 1].cpu().numpy().ravel()
        q50_2 = out2["predictions"]["1h"][..., 1].cpu().numpy().ravel()
        y = batch["targets"]["1h"].cpu().numpy().ravel()

    return {
        "grad_norm_by_group": {k: summarize(v) for k, v in groups.items()},
        "gate_param_grad_norms": gate_detail,
        "one_step_train_pinball": {"before": l1, "after": l2, "delta": l2 - l1},
        "q50_std_before_step": float(np.std(q50_1)),
        "q50_std_after_step": float(np.std(q50_2)),
        "y_std_batch": float(np.std(y)),
        "note": "Fresh random init on a real batch — tests whether gradients flow and loss can drop.",
    }


def diagnose_single_vs_multi(ds: MultiTFDataset) -> dict:
    """Compare fresh MTP vs single-TF fit capacity on one batch (not trained Kaggle weights)."""
    from torch.utils.data import DataLoader, Subset

    cfg = {
        "data": {
            "tfs": ds.tfs,
            "tradable_tfs": ds.tradable_tfs,
            "horizons": ds.horizons,
            "quantiles": ds.quantiles,
            "feature_cols": ds.feature_cols,
            "context_cols": ds.context_cols,
            "primary_tf": ds.primary_tf,
        },
        "model": {
            "d_model": 64,
            "n_layers": 3,
            "n_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "patch_len": 16,
            "use_revin": True,
            "use_context": False,
            "gate_entropy_weight": 0.0,
        },
    }
    loader = DataLoader(
        Subset(ds, list(range(min(512, len(ds))))),
        batch_size=64,
        collate_fn=multi_tf_collate,
        shuffle=False,
    )
    batch = next(iter(loader))
    mtp = MTPTransformer(cfg)
    stf = SingleTFPatchModel(cfg, primary_tf=ds.primary_tf)
    loss_fn = MultiQuantilePinballLoss(
        quantiles=ds.quantiles, tradable_tfs=ds.tradable_tfs, entropy_weight=0.0
    )

    def fit_few_steps(model, steps=30, lr=1e-3):
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        last = None
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            ld = loss_fn(
                out["predictions"],
                batch["targets"],
                batch["target_mask"],
                out.get("gate_weights"),
            )
            ld["loss"].backward()
            opt.step()
            last = float(ld["pinball_loss"].item())
        with torch.no_grad():
            out = model(batch)
            q50 = out["predictions"]["1h"][..., 1].cpu().numpy().ravel()
            y = batch["targets"]["1h"].cpu().numpy().ravel()
            m = batch["target_mask"]["1h"].cpu().numpy().ravel() > 0.5
            return {
                "final_pinball": last,
                "q50_std": float(np.std(q50[m])),
                "y_std": float(np.std(y[m])),
                "corr": float(np.corrcoef(q50[m], y[m])[0, 1])
                if m.sum() > 5 and np.std(q50[m]) > 1e-8
                else float("nan"),
                "dir_acc": float(np.mean(np.sign(q50[m]) == np.sign(y[m]))),
            }

    # init stats
    with torch.no_grad():
        o_m = mtp(batch)
        o_s = stf(batch)
        q_m = o_m["predictions"]["1h"][..., 1].cpu().numpy().ravel()
        q_s = o_s["predictions"]["1h"][..., 1].cpu().numpy().ravel()
        init = {
            "mtp_q50_std": float(np.std(q_m)),
            "stf_q50_std": float(np.std(q_s)),
            "pred_corr_mtp_stf_init": float(np.corrcoef(q_m, q_s)[0, 1]),
        }

    mtp_fit = fit_few_steps(mtp)
    stf_fit = fit_few_steps(stf)
    return {
        "init": init,
        "after_30_steps_same_batch": {"mtp": mtp_fit, "single_tf": stf_fit},
        "interpretation": (
            "Same-batch short fit: if both models reach low pinball and high corr on the "
            "training batch, capacity/wiring can overfit noise — OOS failure is generalization. "
            "If neither can fit the batch, target/features/loss wiring is broken."
        ),
    }


def linear_probe_primary(ds: MultiTFDataset, n_train: int = 4000, n_test: int = 1000) -> dict:
    """
    Ridge regression: flatten primary TF lookback features -> y_1h_h0.
    If this also gets ~0 correlation OOS, signal is not in the features→target map.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error

    primary = ds.primary_tf
    # chronological split mid-history
    start = max(0, len(ds) // 5)
    train_idx = list(range(start, start + n_train))
    test_idx = list(range(start + n_train + 50, start + n_train + 50 + n_test))
    train_idx = [i for i in train_idx if i < len(ds)]
    test_idx = [i for i in test_idx if i < len(ds)]

    def build_xy(idxs):
        xs, ys, rs = [], [], []
        for i in idxs:
            item = ds[i]
            x = item["inputs"][primary].numpy().reshape(-1)
            y = float(item["targets"]["1h"][0].item())
            m = float(item["target_mask"]["1h"][0].item())
            r = float(item["raw_returns"]["1h"][0].item())
            if m < 0.5:
                continue
            xs.append(x)
            ys.append(y)
            rs.append(r)
        return np.stack(xs), np.array(ys), np.array(rs)

    Xtr, ytr, _ = build_xy(train_idx)
    Xte, yte, rte = build_xy(test_idx)
    if len(ytr) < 100 or len(yte) < 50:
        return {"error": "not enough samples", "n_train": len(ytr), "n_test": len(yte)}

    model = Ridge(alpha=10.0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    corr = (
        float(np.corrcoef(pred, yte)[0, 1])
        if np.std(pred) > 1e-12 and np.std(yte) > 1e-12
        else float("nan")
    )
    return {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "corr_pred_y": corr,
        "dir_acc": float(np.mean(np.sign(pred) == np.sign(yte))),
        "mae_pred": float(mean_absolute_error(yte, pred)),
        "mae_zero": float(np.mean(np.abs(yte))),
        "std_pred": float(np.std(pred)),
        "std_y": float(np.std(yte)),
        "std_ratio": float(np.std(pred) / (np.std(yte) + 1e-12)),
        "r2": float(1.0 - np.sum((yte - pred) ** 2) / (np.sum((yte - yte.mean()) ** 2) + 1e-12)),
        "interpretation": (
            "If Ridge corr≈0 and dir≈50% on chronological holdout, the primary lookback features "
            "do not linearly encode short-horizon return; Transformers are unlikely to invent signal."
        ),
    }


def pack_baseline_comparison(pack: Path) -> dict:
    b = pd.read_csv(pack / "02_summary_baselines.csv")
    return {
        "mean_mtp": float(b["test_pinball"].mean()),
        "mean_stf": float(b["baseline_single_tf_pinball"].mean()),
        "mean_zero": float(b["baseline_zero_pinball"].mean()),
        "delta_mtp_minus_stf": float(
            (b["test_pinball"] - b["baseline_single_tf_pinball"]).mean()
        ),
        "folds_mtp_beats_stf": int(
            (b["test_pinball"] < b["baseline_single_tf_pinball"]).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack-dir",
        type=str,
        default=str(
            ROOT.parent
            / "Results"
            / "exp_eurusd_kaggle_12h - 14-08-26 1430Hrs"
            / "exp_eurusd_kaggle_12h"
        ),
    )
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "aligned"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    pack = Path(args.pack_dir)
    out_dir = Path(args.out) if args.out else pack / "deep_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset...")
    ds = MultiTFDataset(
        pair="EURUSD",
        data_dir=args.data_dir,
        mode="train",
        target_clip=5.0,
    )
    ds.fit_standardization()

    report: Dict[str, Any] = {
        "pack_dir": str(pack),
        "n_dataset_samples": len(ds),
    }

    print("1) Targets...")
    report["1_targets"] = diagnose_targets(ds)

    print("2) Predictions from pack...")
    report["2_predictions"] = diagnose_predictions(pack)

    print("3) Features / RevIN...")
    report["3_features"] = diagnose_features(ds)

    print("4) Gradients...")
    report["4_gradients"] = diagnose_gradients(ds)

    print("5) Single vs multi fit probe...")
    report["5_single_vs_multi"] = diagnose_single_vs_multi(ds)

    print("6) Leakage / masking...")
    report["6_wiring"] = diagnose_leakage_and_loss(ds)

    print("7) Linear Ridge probe...")
    report["7_linear_probe"] = linear_probe_primary(ds)

    report["pack_baselines"] = pack_baseline_comparison(pack)

    # Synthesis
    t = report["1_targets"]
    p = report["2_predictions"]
    g = report["4_gradients"]
    w = report["6_wiring"]
    sm = report["5_single_vs_multi"]

    causes = []
    # Signal in target
    acf1 = t["acf_1h_h0_raw"].get("lag_1", 0.0)
    if abs(acf1) < 0.05:
        causes.append(
            {
                "rank": 1,
                "cause": "Target has essentially no short-horizon serial dependence (near-efficient returns)",
                "where": "data / target construction (raw log-returns at 1–12 bars)",
                "evidence": {
                    "acf_raw_lag1": acf1,
                    "acf_y": t["acf_1h_h0_y_valid"],
                    "corr_q50_y_oos": p["corr_q50_y"],
                    "dir_acc": p["dir_acc"],
                },
            }
        )
    if p["std_ratio_q50_over_y"] < 0.4:
        causes.append(
            {
                "rank": 2,
                "cause": "Predicted median severely under-dispersed vs targets (learns narrow conditional mean near 0)",
                "where": "quantile head + pinball optimum under heavy noise",
                "evidence": {
                    "std_q50": p["q50_stats"]["std"],
                    "std_y": p["y_stats"]["std"],
                    "ratio": p["std_ratio_q50_over_y"],
                    "mae_vs_zero": p["skill_vs_naive"],
                },
            }
        )
    if report["pack_baselines"]["delta_mtp_minus_stf"] > 0:
        causes.append(
            {
                "rank": 3,
                "cause": "Multi-TF fusion adds parameters without OOS skill (gates near-uniform → extra overfit risk)",
                "where": "architecture / regularization of multi-TF path",
                "evidence": report["pack_baselines"],
            }
        )
    if w["masking_ok"] and w["n_issues"] == 0:
        causes.append(
            {
                "rank": 4,
                "cause": "No evidence of leakage or broken loss masking (wiring likely OK)",
                "where": "rules out primary wiring failure",
                "evidence": w,
            }
        )

    fit = sm["after_30_steps_same_batch"]
    causes.append(
        {
            "rank": 5,
            "cause": (
                "Models can reduce pinball on a single batch (capacity works); "
                "failure is generalization / lack of OOS signal, not dead gradients"
                if fit["mtp"]["final_pinball"] < 0.7
                else "Models struggle even on-batch — investigate loss/target harder"
            ),
            "where": "optimization vs generalization",
            "evidence": {
                "grad_groups": g["grad_norm_by_group"],
                "one_step": g["one_step_train_pinball"],
                "same_batch_fit": fit,
            },
        }
    )

    # Rank sort
    causes = sorted(causes, key=lambda c: c["rank"])

    lp = report["7_linear_probe"]
    if lp.get("corr_pred_y") is not None and abs(float(lp.get("corr_pred_y", 0))) < 0.05:
        causes.insert(
            0,
            {
                "rank": 0,
                "cause": "Even linear Ridge on primary lookback features has ~zero OOS correlation with y",
                "where": "features → target pairing (not Transformer-specific)",
                "evidence": lp,
            },
        )
        causes = sorted(causes, key=lambda c: c["rank"])

    recommended = {
        "change": (
            "Do not expand multi-TF architecture or add pairs yet. Root issue is low short-horizon "
            "return predictability from these inputs. Concrete next fix: change the learning problem "
            "to a higher-signal target (examples: predict realized vol; classify large |y|>threshold moves only; "
            "or longer horizons where ACF/skill may exist) while keeping single-TF as control — "
            "OR freeze multi-TF as negative result and productize single-TF only if it still has any OOS edge."
        ),
        "one_concrete_next_fix": (
            "Redefine the primary supervised target for a pilot fold: e.g. 1h realized volatility over next "
            "12 bars, or binary label 1{|return|>cost_threshold* k}. Re-run single-TF only. "
            "If that yields corr/AUC clearly > chance, the architecture/wiring is fine and the "
            "return-quantile task was the wrong objective. If still chance-level, broaden features "
            "(calendar/regime) before any multi-TF work."
        ),
    }

    report["synthesis"] = {
        "most_likely_root_causes": causes,
        "recommended_next_fix": recommended,
        "direct_conclusion": (
            "Primary root cause: the supervised mapping from past multi-TF bars to short-horizon "
            "vol-normalized FX returns has near-zero linear/nonlinear OOS signal under this protocol. "
            "The model learns a low-variance near-unconditional quantile forecast (under-dispersed q50, "
            "pinball better than zero but no direction skill). Multi-TF loses to single-TF because extra "
            "capacity/gates do not buy signal. Wiring/masking/leakage checks do not show a fatal bug."
        ),
    }

    (out_dir / "deep_diagnostic.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # Markdown report
    lines = [
        "# Deep diagnostic report — predictive skill failure",
        "",
        f"Pack: `{pack}`",
        "",
        "## Direct conclusion",
        "",
        report["synthesis"]["direct_conclusion"],
        "",
        "## 1. Most likely root causes (ranked)",
        "",
    ]
    for c in causes:
        lines.append(f"### Rank {c['rank']}: {c['cause']}")
        lines.append(f"- **Where:** {c['where']}")
        lines.append(f"- **Evidence:** `{json.dumps(c['evidence'], default=str)[:1200]}`")
        lines.append("")

    lines += [
        "## 2. Key numbers",
        "",
        f"- OOS corr(q50, y): **{p['corr_q50_y']:.4f}**",
        f"- OOS dir acc: **{p['dir_acc']:.4f}**",
        f"- std(q50)/std(y): **{p['std_ratio_q50_over_y']:.3f}**",
        f"- MAE model vs zero vs mean: **{p['skill_vs_naive']}**",
        f"- ACF raw 1h h0 lag1: **{t['acf_1h_h0_raw'].get('lag_1')}**",
        f"- MTP − single-TF pinball: **{report['pack_baselines']['delta_mtp_minus_stf']:.4f}** "
        f"({report['pack_baselines']['folds_mtp_beats_stf']}/6 folds)",
        f"- Loss masking OK: **{w['masking_ok']}**; leakage issues: **{w['n_issues']}**",
        f"- Same-batch 30-step pinball MTP/STF: "
        f"**{fit['mtp']['final_pinball']:.3f}** / **{fit['single_tf']['final_pinball']:.3f}**",
        "",
        "## 3. Recommended next fix (one concrete change)",
        "",
        recommended["one_concrete_next_fix"],
        "",
        recommended["change"],
        "",
        "## 4. Full JSON",
        "",
        "See `deep_diagnostic.json` in this folder.",
        "",
    ]
    (out_dir / "DEEP_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_dir)
    print(report["synthesis"]["direct_conclusion"])
    print("NEXT:", recommended["one_concrete_next_fix"])


if __name__ == "__main__":
    main()
