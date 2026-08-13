"""Export a rich analysis pack (20-30 files) for offline review after Kaggle runs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, save_json


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return v
        return v
    except Exception:
        return float("nan")


def history_to_dataframe(history: List[Dict[str, Any]], fold_id: int) -> pd.DataFrame:
    rows = []
    for h in history:
        gates = h.get("gate_weights")
        row = {
            "fold_id": fold_id,
            "epoch": h.get("epoch"),
            "train_loss": h.get("train_loss"),
            "train_pinball": h.get("train_pinball"),
            "val_loss": h.get("val_loss"),
            "val_pinball": h.get("val_pinball"),
        }
        if isinstance(gates, (list, tuple)):
            for i, g in enumerate(gates):
                row[f"gate_{i}"] = g
        rows.append(row)
    return pd.DataFrame(rows)


def save_fold_artifacts(
    fold_dir: Path,
    fold_id: int,
    fit_result: Dict[str, Any],
    metrics: Dict[str, Any],
    stats: Dict[str, Any],
    econ: Dict[str, Any],
    gate_tf_names: Sequence[str],
    predictions: Optional[Dict[str, np.ndarray]] = None,
    targets: Optional[Dict[str, np.ndarray]] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    raw_returns: Optional[Dict[str, np.ndarray]] = None,
    sample_rows: int = 500,
    quantiles: Optional[List[float]] = None,
    horizons: Optional[Dict[str, List[int]]] = None,
) -> List[Path]:
    """Write per-fold analysis files. Returns list of written paths."""
    fold_dir = ensure_dir(fold_dir)
    written: List[Path] = []

    # 1) metrics.json (full)
    p = fold_dir / "metrics.json"
    save_json(metrics, p)
    written.append(p)

    # 2) history.csv
    hist = fit_result.get("history") or []
    hist_df = history_to_dataframe(hist, fold_id)
    p = fold_dir / "history.csv"
    hist_df.to_csv(p, index=False)
    written.append(p)

    # 3) gates.csv
    gw = fit_result.get("gate_weights")
    if gw is not None:
        names = list(gate_tf_names)
        if len(names) != len(gw):
            names = [f"tf_{i}" for i in range(len(gw))]
        gdf = pd.DataFrame({"tf": names, "weight": list(gw)})
        p = fold_dir / "gates.csv"
        gdf.to_csv(p, index=False)
        written.append(p)

    # 4) stats_detail.json
    p = fold_dir / "stats_detail.json"
    save_json({"stats": stats, "economic": econ}, p)
    written.append(p)

    # 5) pinball_by_tf.csv
    rows = []
    for tf, s in stats.get("per_tf", {}).items():
        rows.append(
            {
                "fold_id": fold_id,
                "tf": tf,
                "pinball": s.get("pinball"),
                "directional_accuracy": s.get("directional_accuracy"),
                "coverage_0.1": (s.get("coverage") or {}).get("0.1"),
                "coverage_0.5": (s.get("coverage") or {}).get("0.5"),
                "coverage_0.9": (s.get("coverage") or {}).get("0.9"),
            }
        )
    p = fold_dir / "pinball_by_tf.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    written.append(p)

    # 6) economic_by_horizon.csv
    erows = []
    for tf, block in econ.get("per_tf", {}).items():
        for h in block.get("horizons", []):
            erows.append({"fold_id": fold_id, "tf": tf, **h})
    p = fold_dir / "economic_by_horizon.csv"
    pd.DataFrame(erows).to_csv(p, index=False)
    written.append(p)

    # 7) prediction sample (compact, analysis-friendly)
    if predictions is not None and targets is not None and masks is not None:
        quantiles = quantiles or [0.1, 0.5, 0.9]
        try:
            med_i = list(quantiles).index(0.5)
        except ValueError:
            med_i = len(quantiles) // 2
        sample_parts = []
        for tf, pred in predictions.items():
            n = pred.shape[0]
            take = min(sample_rows, n)
            # evenly spaced indices across the test fold
            idxs = np.linspace(0, n - 1, take, dtype=int) if n > 0 else np.array([], dtype=int)
            for hi in range(pred.shape[1]):
                h_label = None
                if horizons and tf in horizons and hi < len(horizons[tf]):
                    h_label = horizons[tf][hi]
                for j in idxs:
                    sample_parts.append(
                        {
                            "fold_id": fold_id,
                            "tf": tf,
                            "row": int(j),
                            "horizon_idx": hi,
                            "horizon_bars": h_label,
                            "y": float(targets[tf][j, hi]),
                            "mask": float(masks[tf][j, hi]),
                            "q10": float(pred[j, hi, 0]) if pred.shape[-1] > 0 else float("nan"),
                            "q50": float(pred[j, hi, med_i]),
                            "q90": float(pred[j, hi, -1]) if pred.shape[-1] > 2 else float("nan"),
                            "raw_return": (
                                float(raw_returns[tf][j, hi])
                                if raw_returns is not None and tf in raw_returns
                                else float("nan")
                            ),
                        }
                    )
        p = fold_dir / "predictions_sample.csv"
        pd.DataFrame(sample_parts).to_csv(p, index=False)
        written.append(p)

    return written


def build_go_nogo(
    fold_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """STEP7 acceptance-style checklist from fold metrics."""
    if not fold_metrics:
        return {"pass": False, "checks": {}, "reason": "no folds"}

    pinballs = [_safe_float(m.get("test_pinball", m.get("test_stats", {}).get("overall", {}).get("pinball"))) for m in fold_metrics]
    zeros = [_safe_float(m.get("baseline_zero_pinball")) for m in fold_metrics]
    means = [_safe_float(m.get("baseline_mean_pinball")) for m in fold_metrics]
    single = [_safe_float(m.get("baseline_single_tf_pinball")) for m in fold_metrics if m.get("baseline_single_tf_pinball") is not None]

    beat_zero = [p < z for p, z in zip(pinballs, zeros) if np.isfinite(p) and np.isfinite(z)]
    beat_mean = [p < m for p, m in zip(pinballs, means) if np.isfinite(p) and np.isfinite(m)]

    # gate collapse: any fold with max gate > 0.85
    gate_ok = True
    max_gates = []
    for m in fold_metrics:
        gw = m.get("gate_weights")
        if gw is None:
            continue
        mx = float(np.max(gw))
        max_gates.append(mx)
        if mx > 0.85:
            gate_ok = False

    finite_losses = all(np.isfinite(p) for p in pinballs)

    checks = {
        "losses_finite": finite_losses,
        "mean_test_pinball": float(np.nanmean(pinballs)),
        "mean_zero_pinball": float(np.nanmean(zeros)),
        "mean_hist_mean_pinball": float(np.nanmean(means)),
        "mean_single_tf_pinball": float(np.nanmean(single)) if single else None,
        "frac_folds_beat_zero": float(np.mean(beat_zero)) if beat_zero else 0.0,
        "frac_folds_beat_hist_mean": float(np.mean(beat_mean)) if beat_mean else 0.0,
        "gates_not_collapsed": gate_ok,
        "max_gate_per_fold": max_gates,
        "n_folds": len(fold_metrics),
    }
    checks["pass_beat_zero_majority"] = checks["frac_folds_beat_zero"] >= 0.5
    checks["pass_beat_mean_majority"] = checks["frac_folds_beat_hist_mean"] >= 0.5
    checks["pass"] = bool(
        checks["losses_finite"]
        and checks["pass_beat_zero_majority"]
        and checks["gates_not_collapsed"]
    )
    return checks


def write_analysis_pack(
    exp_dir: Path,
    cfg: Dict[str, Any],
    fold_metrics: List[Dict[str, Any]],
    all_histories: List[pd.DataFrame],
    fold_tables: List[pd.DataFrame],
    notes: List[str],
    runtime_seconds: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, Path]:
    """
    Write experiment-level analysis files (target ~20-30 files total with fold files).
    """
    exp_dir = ensure_dir(exp_dir)
    paths: Dict[str, Path] = {}
    tfs = list(cfg.get("data", {}).get("tfs", []))
    quantiles = list(cfg.get("data", {}).get("quantiles", [0.1, 0.5, 0.9]))

    # --- fold overview ---
    overview_rows = []
    for m in fold_metrics:
        overview_rows.append(
            {
                "fold_id": m.get("fold_id"),
                "best_epoch": m.get("best_epoch"),
                "best_val_pinball": m.get("best_val_pinball"),
                "test_pinball": m.get("test_pinball"),
                "test_dir_acc": m.get("test_dir_acc"),
                "baseline_zero_pinball": m.get("baseline_zero_pinball"),
                "baseline_mean_pinball": m.get("baseline_mean_pinball"),
                "baseline_single_tf_pinball": m.get("baseline_single_tf_pinball"),
                "beat_zero": (
                    _safe_float(m.get("test_pinball")) < _safe_float(m.get("baseline_zero_pinball"))
                ),
                "beat_mean": (
                    _safe_float(m.get("test_pinball")) < _safe_float(m.get("baseline_mean_pinball"))
                ),
                "mean_sharpe_1h": m.get("mean_sharpe_primary"),
                "n_train": m.get("n_train"),
                "n_val": m.get("n_val"),
                "n_test": m.get("n_test"),
                "n_params": m.get("n_params"),
                "train_seconds": m.get("train_seconds"),
            }
        )
    overview = pd.DataFrame(overview_rows)
    p = exp_dir / "01_fold_overview.csv"
    overview.to_csv(p, index=False)
    paths["fold_overview"] = p

    # --- baselines summary ---
    p = exp_dir / "02_summary_baselines.csv"
    overview[
        [
            c
            for c in overview.columns
            if c
            in {
                "fold_id",
                "test_pinball",
                "baseline_zero_pinball",
                "baseline_mean_pinball",
                "baseline_single_tf_pinball",
                "beat_zero",
                "beat_mean",
            }
        ]
    ].to_csv(p, index=False)
    paths["baselines"] = p

    # --- gates by fold ---
    grow = []
    for m in fold_metrics:
        gw = m.get("gate_weights")
        if gw is None:
            continue
        names = tfs if len(tfs) == len(gw) else [f"tf_{i}" for i in range(len(gw))]
        for name, w in zip(names, gw):
            grow.append({"fold_id": m.get("fold_id"), "tf": name, "weight": w})
    p = exp_dir / "03_gate_weights_by_fold.csv"
    pd.DataFrame(grow).to_csv(p, index=False)
    paths["gates"] = p

    # --- training curves all folds ---
    if all_histories:
        curves = pd.concat(all_histories, ignore_index=True)
    else:
        curves = pd.DataFrame()
    p = exp_dir / "04_training_curves_all_folds.csv"
    curves.to_csv(p, index=False)
    paths["curves"] = p

    # --- stats / economic from fold tables ---
    if fold_tables:
        all_df = pd.concat(fold_tables, ignore_index=True)
    else:
        all_df = pd.DataFrame()
    p = exp_dir / "05_summary_stats.csv"
    all_df.to_csv(p, index=False)
    paths["stats"] = p

    # coverage long format
    cov_rows = []
    for m in fold_metrics:
        cov = (m.get("test_stats") or {}).get("overall", {}).get("coverage", {})
        for q, v in (cov or {}).items():
            cov_rows.append({"fold_id": m.get("fold_id"), "quantile": q, "coverage": v})
        for tf, s in ((m.get("test_stats") or {}).get("per_tf") or {}).items():
            for q, v in (s.get("coverage") or {}).items():
                cov_rows.append(
                    {"fold_id": m.get("fold_id"), "tf": tf, "quantile": q, "coverage": v}
                )
    p = exp_dir / "06_coverage_calibration.csv"
    pd.DataFrame(cov_rows).to_csv(p, index=False)
    paths["coverage"] = p

    # pinball by horizon if present in fold economic dumps - aggregate from fold files if available
    h_rows = []
    for m in fold_metrics:
        for tf, block in ((m.get("test_economic") or {}).get("per_tf") or {}).items():
            for h in block.get("horizons", []):
                h_rows.append({"fold_id": m.get("fold_id"), "tf": tf, **h})
    p = exp_dir / "07_economic_by_horizon_all_folds.csv"
    pd.DataFrame(h_rows).to_csv(p, index=False)
    paths["econ_horizon"] = p

    # pinball per tf all folds
    p_rows = []
    for m in fold_metrics:
        for tf, s in ((m.get("test_stats") or {}).get("per_tf") or {}).items():
            p_rows.append(
                {
                    "fold_id": m.get("fold_id"),
                    "tf": tf,
                    "pinball": s.get("pinball"),
                    "directional_accuracy": s.get("directional_accuracy"),
                }
            )
    p = exp_dir / "08_pinball_by_tf_all_folds.csv"
    pd.DataFrame(p_rows).to_csv(p, index=False)
    paths["pinball_tf"] = p

    # aggregate means
    agg = {
        "n_folds": len(fold_metrics),
        "mean_test_pinball": float(overview["test_pinball"].mean()) if len(overview) else None,
        "median_test_pinball": float(overview["test_pinball"].median()) if len(overview) else None,
        "worst_test_pinball": float(overview["test_pinball"].max()) if len(overview) else None,
        "best_test_pinball": float(overview["test_pinball"].min()) if len(overview) else None,
        "mean_zero_pinball": float(overview["baseline_zero_pinball"].mean()) if len(overview) else None,
        "mean_hist_mean_pinball": float(overview["baseline_mean_pinball"].mean()) if len(overview) else None,
        "frac_beat_zero": float(overview["beat_zero"].mean()) if len(overview) else None,
        "frac_beat_mean": float(overview["beat_mean"].mean()) if len(overview) else None,
        "device": device,
        "runtime_seconds": runtime_seconds,
        "runtime_hours": (runtime_seconds / 3600.0) if runtime_seconds else None,
        "quantiles": quantiles,
        "pair": cfg.get("data", {}).get("pair"),
        "primary_tf": cfg.get("data", {}).get("primary_tf"),
        "max_epochs": cfg.get("training", {}).get("max_epochs"),
        "max_folds": cfg.get("walk_forward", {}).get("max_folds"),
    }
    p = exp_dir / "09_aggregate_metrics.json"
    save_json(agg, p)
    paths["aggregate"] = p

    go = build_go_nogo(fold_metrics)
    p = exp_dir / "10_go_nogo.json"
    save_json(go, p)
    paths["go_nogo"] = p

    # human-readable acceptance checklist
    lines = [
        "# Go / No-Go Checklist (Experiment v1)",
        "",
        f"- Folds completed: {go.get('n_folds', go.get('checks', {}).get('n_folds'))}",
        f"- Losses finite: {go.get('losses_finite', go.get('checks', {}).get('losses_finite'))}",
        f"- Mean test pinball: {go.get('mean_test_pinball', go.get('checks', {}).get('mean_test_pinball'))}",
        f"- Mean zero baseline pinball: {go.get('mean_zero_pinball', go.get('checks', {}).get('mean_zero_pinball'))}",
        f"- Mean hist-mean baseline pinball: {go.get('mean_hist_mean_pinball', go.get('checks', {}).get('mean_hist_mean_pinball'))}",
        f"- Frac folds beat zero: {go.get('frac_folds_beat_zero', go.get('checks', {}).get('frac_folds_beat_zero'))}",
        f"- Frac folds beat hist-mean: {go.get('frac_folds_beat_hist_mean', go.get('checks', {}).get('frac_folds_beat_hist_mean'))}",
        f"- Gates not collapsed: {go.get('gates_not_collapsed', go.get('checks', {}).get('gates_not_collapsed'))}",
        f"- Overall PASS: {go.get('pass', go.get('checks', {}).get('pass'))}",
        "",
        "Interpretation notes:",
        "- Statistical edge vs trivial baselines is the first bar; economic Sharpe may still be weak in v1.",
        "- Gate collapse (one TF ~1.0) is a red flag even if pinball looks ok.",
        "- Use fold-level CSVs to see regime dependence (which folds fail).",
        "",
    ]
    # flatten if nested under checks
    if "checks" in go:
        c = go["checks"]
        lines = [
            "# Go / No-Go Checklist (Experiment v1)",
            "",
            f"- Folds completed: {c.get('n_folds')}",
            f"- Losses finite: {c.get('losses_finite')}",
            f"- Mean test pinball: {c.get('mean_test_pinball')}",
            f"- Mean zero baseline: {c.get('mean_zero_pinball')}",
            f"- Mean hist-mean baseline: {c.get('mean_hist_mean_pinball')}",
            f"- Mean single-TF baseline: {c.get('mean_single_tf_pinball')}",
            f"- Frac folds beat zero: {c.get('frac_folds_beat_zero')}",
            f"- Frac folds beat hist-mean: {c.get('frac_folds_beat_hist_mean')}",
            f"- Gates not collapsed: {c.get('gates_not_collapsed')}",
            f"- Max gate per fold: {c.get('max_gate_per_fold')}",
            f"- Overall PASS: {c.get('pass')}",
            "",
            "Notes:",
            "- First bar: beat trivial baselines out-of-sample.",
            "- Gate collapse is a design failure mode even with decent pinball.",
            "- Economic metrics can be weak in v1; still must be finite and computable.",
            "",
        ]
    p = exp_dir / "11_acceptance_checklist.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    paths["checklist"] = p

    # experiment report
    report = [
        "# MTP-Transformer Kaggle Experiment Report",
        "",
        f"- Pair: {cfg.get('data', {}).get('pair')}",
        f"- Primary TF: {cfg.get('data', {}).get('primary_tf')}",
        f"- Device: {device}",
        f"- Runtime hours: {agg.get('runtime_hours')}",
        f"- Max epochs (ceiling): {cfg.get('training', {}).get('max_epochs')}",
        f"- Early stopping patience: {cfg.get('training', {}).get('early_stopping_patience')}",
        f"- Folds: {len(fold_metrics)}",
        f"- Mean test pinball: {agg.get('mean_test_pinball')}",
        f"- Frac beat zero baseline: {agg.get('frac_beat_zero')}",
        f"- Frac beat hist-mean baseline: {agg.get('frac_beat_mean')}",
        f"- Go/No-Go pass: {go.get('checks', go).get('pass') if isinstance(go.get('checks', go), dict) else go.get('pass')}",
        "",
        "## Per-fold notes",
    ]
    for n in notes:
        report.append(f"- {n}")
    report.append("")
    report.append("## Files in this pack")
    report.append("See `12_manifest.json` for the full file list included in the download zip.")
    report.append("")
    p = exp_dir / "00_experiment_report.md"
    p.write_text("\n".join(report), encoding="utf-8")
    paths["report"] = p

    # notes.md
    p = exp_dir / "12_notes.md"
    p.write_text("# Notes\n\n" + "\n".join(f"- {n}" for n in notes) + "\n", encoding="utf-8")
    paths["notes"] = p

    # runtime log
    p = exp_dir / "13_runtime.json"
    save_json(
        {
            "device": device,
            "runtime_seconds": runtime_seconds,
            "runtime_hours": agg.get("runtime_hours"),
            "training": cfg.get("training"),
            "walk_forward": cfg.get("walk_forward"),
        },
        p,
    )
    paths["runtime"] = p

    # config snapshot already saved separately; also dump flat hyperparams
    p = exp_dir / "14_hyperparameters.json"
    save_json(
        {
            "model": cfg.get("model"),
            "training": cfg.get("training"),
            "walk_forward": cfg.get("walk_forward"),
            "data_lookback": cfg.get("data", {}).get("lookback"),
            "data_horizons": cfg.get("data", {}).get("horizons"),
            "quantiles": quantiles,
            "evaluation": cfg.get("evaluation"),
        },
        p,
    )
    paths["hyper"] = p

    # mean gate vector
    if grow:
        gdf = pd.DataFrame(grow)
        mean_g = gdf.groupby("tf", as_index=False)["weight"].mean()
        p = exp_dir / "15_mean_gate_weights.csv"
        mean_g.to_csv(p, index=False)
        paths["mean_gates"] = p

    # delta vs baselines
    if len(overview):
        ddf = overview.copy()
        ddf["delta_vs_zero"] = ddf["test_pinball"] - ddf["baseline_zero_pinball"]
        ddf["delta_vs_mean"] = ddf["test_pinball"] - ddf["baseline_mean_pinball"]
        if "baseline_single_tf_pinball" in ddf.columns:
            ddf["delta_vs_single_tf"] = ddf["test_pinball"] - ddf["baseline_single_tf_pinball"]
        p = exp_dir / "16_delta_vs_baselines.csv"
        ddf[
            [
                c
                for c in [
                    "fold_id",
                    "test_pinball",
                    "delta_vs_zero",
                    "delta_vs_mean",
                    "delta_vs_single_tf",
                ]
                if c in ddf.columns
            ]
        ].to_csv(p, index=False)
        paths["deltas"] = p

    return paths


def package_analysis_zip(
    exp_dir: Path,
    zip_path: Path,
    include_checkpoints: bool = False,
    include_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Create a downloadable zip of analysis artifacts (target 20-30 files).

    Excludes large .pt checkpoints unless include_checkpoints=True.
    """
    exp_dir = Path(exp_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect files
    files: List[Path] = []
    for p in sorted(exp_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(exp_dir)
        name = p.name
        if name.endswith(".pt") and not include_checkpoints:
            continue
        # skip python caches etc
        if "__pycache__" in rel.parts:
            continue
        files.append(p)

    # Prefer a clean analysis subset if there are way too many files
    # (keep all non-checkpoint files under exp_dir)
    manifest = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            rel = p.relative_to(exp_dir).as_posix()
            zf.write(p, arcname=f"{exp_dir.name}/{rel}")
            manifest.append(
                {
                    "path": f"{exp_dir.name}/{rel}",
                    "bytes": p.stat().st_size,
                    "suffix": p.suffix,
                }
            )

    # Write manifest both inside exp_dir and update zip if needed
    man_path = exp_dir / "12_manifest.json"
    save_json(
        {
            "n_files": len(manifest),
            "zip_path": str(zip_path),
            "include_checkpoints": include_checkpoints,
            "files": manifest,
        },
        man_path,
    )
    # Re-open zip to append manifest (or rewrite simpler: always include after)
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(man_path, arcname=f"{exp_dir.name}/12_manifest.json")

    return {
        "zip_path": str(zip_path),
        "n_files": len(manifest),
        "total_bytes": sum(m["bytes"] for m in manifest),
        "files": [m["path"] for m in manifest],
    }
