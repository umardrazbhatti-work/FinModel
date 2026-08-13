#!/usr/bin/env python
"""
Diagnose early-stopping / epoch-1–2 peak from an analysis pack + optional live data.

Checks:
  1) Train vs val pinball curves (overfit timing)
  2) Gate entropy / collapse
  3) Prediction bias (mean q50) and signal occupancy at threshold
  4) Vol-normalized target distribution from predictions_sample (and local data if present)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def analyze_curves(pack: Path) -> dict:
    curves_path = pack / "04_training_curves_all_folds.csv"
    if not curves_path.exists():
        return {"error": f"missing {curves_path.name}"}
    df = pd.read_csv(curves_path)
    fold_stats = []
    for fold_id, g in df.groupby("fold_id"):
        g = g.sort_values("epoch")
        best_i = int(g["val_pinball"].idxmin())
        best = g.loc[best_i]
        last = g.iloc[-1]
        # train-val gap at best and at end
        fold_stats.append(
            {
                "fold_id": int(fold_id),
                "n_epochs_ran": int(g["epoch"].max()),
                "best_epoch": int(best["epoch"]),
                "best_val_pinball": float(best["val_pinball"]),
                "train_pinball_at_best": float(best["train_pinball"]),
                "gap_at_best": float(best["val_pinball"] - best["train_pinball"]),
                "train_pinball_final": float(last["train_pinball"]),
                "val_pinball_final": float(last["val_pinball"]),
                "gap_final": float(last["val_pinball"] - last["train_pinball"]),
                "val_rise_from_best": float(last["val_pinball"] - best["val_pinball"]),
                "train_drop_after_best": float(best["train_pinball"] - last["train_pinball"]),
            }
        )
    fs = pd.DataFrame(fold_stats)
    return {
        "per_fold": fold_stats,
        "mean_best_epoch": float(fs["best_epoch"].mean()),
        "mean_n_epochs_ran": float(fs["n_epochs_ran"].mean()),
        "mean_gap_at_best": float(fs["gap_at_best"].mean()),
        "mean_gap_final": float(fs["gap_final"].mean()),
        "mean_val_rise_from_best": float(fs["val_rise_from_best"].mean()),
        "interpretation": (
            "If best_epoch is 1–2 and val_rise_from_best > 0 while train keeps falling, "
            "the model overfits quickly (or val is dominated by a simple scale fit that "
            "does not generalize as capacity is used)."
        ),
    }


def analyze_predictions(pack: Path, signal_threshold: float = 0.1) -> dict:
    parts = []
    for p in sorted(pack.glob("fold_*/predictions_sample.csv")):
        parts.append(pd.read_csv(p))
    if not parts:
        return {"error": "no predictions_sample.csv"}
    df = pd.concat(parts, ignore_index=True)

    y = df["y"].to_numpy(dtype=np.float64)
    mask = df["mask"].to_numpy(dtype=np.float64)
    q50 = df["q50"].to_numpy(dtype=np.float64)
    raw = df["raw_return"].to_numpy(dtype=np.float64)
    valid = mask > 0.5

    def _dist(x: np.ndarray) -> dict:
        if len(x) == 0:
            return {}
        return {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "p01": float(np.quantile(x, 0.01)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
            "frac_abs_gt_5": float(np.mean(np.abs(x) > 5)),
            "frac_abs_gt_10": float(np.mean(np.abs(x) > 10)),
            "frac_zero": float(np.mean(x == 0)),
        }

    pos = np.zeros_like(q50)
    pos[q50 > signal_threshold] = 1.0
    pos[q50 < -signal_threshold] = -1.0

    # Pinball residual diagnostics: does model mostly predict near-zero?
    return {
        "target_y_all": _dist(y),
        "target_y_mask_valid": _dist(y[valid]),
        "raw_return_all": _dist(raw),
        "q50_all": _dist(q50),
        "mask_valid_frac": float(valid.mean()),
        "signal_threshold": signal_threshold,
        "signal_occupancy": {
            "pct_long": float((pos > 0).mean()),
            "pct_short": float((pos < 0).mean()),
            "pct_flat": float((pos == 0).mean()),
        },
        "q50_bias": float(np.mean(q50)),
        "y_vs_q50_corr_valid": (
            float(np.corrcoef(y[valid], q50[valid])[0, 1])
            if valid.sum() > 5 and np.std(y[valid]) > 0 and np.std(q50[valid]) > 0
            else float("nan")
        ),
        "notes": [
            "y is vol-normalized log-return (raw / past_vol); large |y| means return >> recent vol.",
            "If q50 is systematically negative, threshold signals will be short-biased.",
            "If |y| regularly exceeds 5–10, heavy tails can dominate pinball and encourage under-dispersed quantiles.",
        ],
    }


def analyze_gates(pack: Path) -> dict:
    p = pack / "03_gate_weights_by_fold.csv"
    if not p.exists():
        return {"error": "missing gate weights"}
    g = pd.read_csv(p)
    # wide-ish: fold_id, tf, weight
    piv = g.pivot_table(index="fold_id", columns="tf", values="weight", aggfunc="mean")
    entropies = []
    maxes = []
    for _, row in piv.iterrows():
        w = row.to_numpy(dtype=np.float64)
        w = w / w.sum()
        ent = float(-np.sum(w * np.log(np.clip(w, 1e-12, None))))
        entropies.append(ent)
        maxes.append(float(w.max()))
    max_ent = float(np.log(piv.shape[1])) if piv.shape[1] else float("nan")
    return {
        "mean_weights": {c: float(piv[c].mean()) for c in piv.columns},
        "mean_entropy": float(np.mean(entropies)),
        "max_entropy_uniform": max_ent,
        "mean_max_gate": float(np.mean(maxes)),
        "near_uniform": bool(float(np.mean(maxes)) < 0.25),
    }


def analyze_baselines(pack: Path) -> dict:
    p2 = pack / "02_summary_baselines.csv"
    if not p2.exists():
        return {"error": "missing 02_summary_baselines.csv"}
    b = pd.read_csv(p2)
    b["delta_vs_single_tf"] = b["test_pinball"] - b["baseline_single_tf_pinball"]
    b["delta_vs_zero"] = b["test_pinball"] - b["baseline_zero_pinball"]
    b["delta_vs_mean"] = b["test_pinball"] - b["baseline_mean_pinball"]

    return {
        "mean_test_pinball": float(b["test_pinball"].mean()),
        "mean_single_tf_pinball": float(b["baseline_single_tf_pinball"].mean()),
        "mean_zero_pinball": float(b["baseline_zero_pinball"].mean()),
        "mean_hist_mean_pinball": float(b["baseline_mean_pinball"].mean()),
        "mean_delta_vs_single_tf": float(b["delta_vs_single_tf"].mean()),
        "folds_beat_single_tf": int((b["delta_vs_single_tf"] < 0).sum()),
        "folds_beat_zero": int((b["delta_vs_zero"] < 0).sum()),
        "n_folds": int(len(b)),
        "multi_tf_advances": bool((b["delta_vs_single_tf"] < 0).mean() > 0.5),
        "fair_comparison_rule": (
            "Multi-TF only advances if mean test pinball < single-TF under same "
            "walk-forward folds, epochs budget, and evaluation protocol."
        ),
    }


def optional_live_target_stats(data_dir: Path, pair: str = "EURUSD") -> dict:
    """Sample target distribution from local aligned data if available."""
    try:
        from src.data import MultiTFDataset
    except Exception as e:  # noqa: BLE001
        return {"skipped": True, "reason": str(e)}

    if not data_dir.exists():
        return {"skipped": True, "reason": f"no data_dir {data_dir}"}

    try:
        ds = MultiTFDataset(
            pair=pair,
            data_dir=str(data_dir),
            mode="train",
            fold_start=None,
            fold_end=None,
        )
    except Exception as e:  # noqa: BLE001
        return {"skipped": True, "reason": f"dataset init failed: {e}"}

    n = min(2000, len(ds))
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(ds), size=n, replace=False)
    ys = {tf: [] for tf in ds.tradable_tfs}
    masks = {tf: [] for tf in ds.tradable_tfs}
    raws = {tf: [] for tf in ds.tradable_tfs}
    for i in idxs:
        item = ds[int(i)]
        for tf in ds.tradable_tfs:
            ys[tf].append(item["targets"][tf].numpy())
            masks[tf].append(item["target_mask"][tf].numpy())
            raws[tf].append(item["raw_returns"][tf].numpy())

    out = {}
    for tf in ds.tradable_tfs:
        y = np.stack(ys[tf], axis=0).reshape(-1)
        m = np.stack(masks[tf], axis=0).reshape(-1)
        r = np.stack(raws[tf], axis=0).reshape(-1)
        valid = m > 0.5
        yv = y[valid]
        out[tf] = {
            "mask_valid_frac": float(valid.mean()),
            "y_mean": float(yv.mean()) if len(yv) else float("nan"),
            "y_std": float(yv.std()) if len(yv) else float("nan"),
            "y_p01": float(np.quantile(yv, 0.01)) if len(yv) else float("nan"),
            "y_p99": float(np.quantile(yv, 0.99)) if len(yv) else float("nan"),
            "y_max_abs": float(np.max(np.abs(yv))) if len(yv) else float("nan"),
            "raw_std": float(r[valid].std()) if valid.any() else float("nan"),
            "frac_abs_y_gt_5": float(np.mean(np.abs(yv) > 5)) if len(yv) else float("nan"),
            "frac_abs_y_gt_10": float(np.mean(np.abs(yv) > 10)) if len(yv) else float("nan"),
        }
    return {"skipped": False, "n_samples": n, "per_tf": out}


def write_markdown_report(out: dict, path: Path) -> None:
    lines = [
        "# Early-stop & metrics diagnosis",
        "",
        "## 1. Training curves (why epoch 1–2?)",
        "",
        f"- Mean best epoch: **{out['curves'].get('mean_best_epoch')}**",
        f"- Mean epochs ran: **{out['curves'].get('mean_n_epochs_ran')}**",
        f"- Mean train–val gap at best: **{out['curves'].get('mean_gap_at_best'):.4f}**"
        if isinstance(out["curves"].get("mean_gap_at_best"), float)
        else "",
        f"- Mean train–val gap at end: **{out['curves'].get('mean_gap_final'):.4f}**"
        if isinstance(out["curves"].get("mean_gap_final"), float)
        else "",
        f"- Mean val rise after best: **{out['curves'].get('mean_val_rise_from_best'):.4f}**"
        if isinstance(out["curves"].get("mean_val_rise_from_best"), float)
        else "",
        "",
        str(out["curves"].get("interpretation", "")),
        "",
        "### Per fold",
        "",
        "```",
        json.dumps(out["curves"].get("per_fold", []), indent=2),
        "```",
        "",
        "## 2. Target / prediction behavior",
        "",
        "```",
        json.dumps(out.get("predictions", {}), indent=2),
        "```",
        "",
        "## 3. Gates",
        "",
        "```",
        json.dumps(out.get("gates", {}), indent=2),
        "```",
        "",
        "## 4. Fair multi-TF vs single-TF",
        "",
        "```",
        json.dumps(out.get("baselines", {}), indent=2),
        "```",
        "",
        "## 5. Live data target check (if available)",
        "",
        "```",
        json.dumps(out.get("live_targets", {}), indent=2),
        "```",
        "",
        "## 6. Recommended controlled next steps (only after econ fix)",
        "",
        "1. Do **not** increase capacity / cross-attention yet.",
        "2. Multi-TF advances only if it beats single-TF pinball under same protocol.",
        "3. Candidate small upgrades (one at a time):",
        "   - Mild target winsorization (e.g. clip vol-normalized y to ±5 or ±8)",
        "   - Slightly stronger weight_decay / dropout",
        "   - Lower LR or cosine schedule with warmup",
        "   - Gate temperature / entropy weight retune (gates are near-uniform)",
        "   - Signal threshold sweep on fixed q50 (econ only; does not change model)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "aligned"))
    parser.add_argument("--signal-threshold", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    pack = Path(args.pack_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pack / "diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "curves": analyze_curves(pack),
        "predictions": analyze_predictions(pack, signal_threshold=args.signal_threshold),
        "gates": analyze_gates(pack),
        "baselines": analyze_baselines(pack),
        "live_targets": optional_live_target_stats(Path(args.data_dir)),
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, out_dir / "diagnosis_report.md")
    print("Wrote", out_dir)
    print(json.dumps({k: report[k] for k in ("curves", "baselines", "gates")}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
