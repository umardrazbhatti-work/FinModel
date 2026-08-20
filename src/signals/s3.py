"""S-3: survivor Signal (side) x locked Handler (size). Handler never sets side."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data import MultiTFDataset, multi_tf_collate
from src.handler import VolatilityTradeHandler
from src.handler.sizing import realized_vol_from_closes, size_from_vol
from src.signals.catalog import build_catalog, get_rule, positions_for
from src.signals.factory import (
    PROJECT,
    attach_account,
    load_ohlc,
    results_dir,
    score_hold_window,
    unseen_month_periods,
    _month_index_bounds,
    pack_stamp,
)
from src.signals.labels import forward_simple_return
from src.signals.rules import next_bar_simple_return
from src.training import generate_walk_forward_folds
from src.utils.config import load_config, save_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

logger = get_logger("mtp.signal_s3")

DEFAULT_CKPT_ROOT = (
    PROJECT.parent
    / "Results"
    / "exp_eurusd_rv_single_tf_pilot_pilot_pack - 15-08-26 1600Hrs"
)


def _checkpoint(fold_id: int, root: Path) -> Path:
    p = root / f"fold_{int(fold_id)}" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


@torch.no_grad()
def infer_multipliers(
    handler: VolatilityTradeHandler,
    ds: MultiTFDataset,
    close: np.ndarray,
    bar_lo: int,
    bar_hi: int,
    lookback: int = 72,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Size multiplier per primary bar. Missing forecast -> 0 / stand aside."""
    n = len(close)
    mult = np.zeros(n, dtype=np.float64)
    aside = np.ones(n, dtype=bool)
    bar_to_j = {int(ds.sample_indices[j]): j for j in range(len(ds))}
    js: List[int] = []
    bars: List[int] = []
    for i in range(int(bar_lo), int(bar_hi)):
        j = bar_to_j.get(i)
        if j is not None:
            js.append(j)
            bars.append(i)
    if not js:
        return mult, aside

    h_idx = handler.primary_horizon_idx
    i10 = handler.quantiles.index(0.1)
    i50 = handler.quantiles.index(0.5)
    i90 = handler.quantiles.index(0.9)
    loader = DataLoader(
        Subset(ds, js),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=multi_tf_collate,
    )
    k = 0
    model = handler.model
    model.eval()
    for batch in loader:
        local = {
            "inputs": {tf: t.to(handler.device) for tf, t in batch["inputs"].items()},
        }
        if "context" in batch and torch.is_tensor(batch["context"]):
            local["context"] = batch["context"].to(handler.device)
        pred = model(local)["predictions"][handler.primary_tf]
        bsz = int(pred.shape[0])
        for b in range(bsz):
            bi = bars[k]
            sl = max(0, bi - int(lookback) + 1)
            ref = realized_vol_from_closes(close[sl : bi + 1])
            sz = size_from_vol(
                float(pred[b, h_idx, i10].cpu()),
                float(pred[b, h_idx, i50].cpu()),
                float(pred[b, h_idx, i90].cpu()),
                ref_rv=float(ref) if np.isfinite(ref) else float("nan"),
                cfg=handler.sizing_cfg,
            )
            if sz.stand_aside or not np.isfinite(sz.size_multiplier):
                mult[bi] = 0.0
                aside[bi] = True
            else:
                mult[bi] = float(sz.size_multiplier)
                aside[bi] = False
            k += 1
    return mult, aside


def _trade_stats(pos: np.ndarray, sized: np.ndarray, start: int, end: int, hold: int) -> Dict[str, Any]:
    from src.signals.labels import nonoverlap_indices

    if int(hold) <= 1:
        idx = np.arange(int(start), int(end))
    else:
        idx = nonoverlap_indices(int(start), int(end), int(hold), n=len(pos))
    if idx.size == 0:
        return {
            "n_slots": 0,
            "n_signal": 0,
            "n_traded": 0,
            "n_stood_aside": 0,
            "n_flat": 0,
            "mean_mult_on_signal": float("nan"),
        }
    p = np.asarray(pos[idx], dtype=np.float64)
    s = np.asarray(sized[idx], dtype=np.float64)
    sig = np.abs(p) > 0
    traded = np.abs(s) > 1e-12
    return {
        "n_slots": int(idx.size),
        "n_signal": int(sig.sum()),
        "n_traded": int(traded.sum()),
        "n_stood_aside": int((sig & ~traded).sum()),
        "n_flat": int((~sig).sum()),
        "mean_mult_on_signal": float(np.mean(np.abs(s[sig]) / np.maximum(np.abs(p[sig]), 1e-12)))
        if sig.any()
        else float("nan"),
    }


def run_s3(cfg: Dict[str, Any]) -> Dict[str, Any]:
    build_catalog()
    pair = cfg["data"]["pair"]
    tf = cfg["data"]["tf"]
    cost = float(cfg["data"]["cost"])
    ppy = float(cfg["signal"]["periods_per_year"])
    start_usd = float(cfg["signal"].get("start_usd", 100.0))
    n_months = int(cfg["signal"].get("unseen_months", 6))
    survivors = list(cfg["signal"]["survivors"])
    ckpt_root = Path(cfg["handler"]["checkpoint_root"])
    if not ckpt_root.is_absolute():
        ckpt_root = PROJECT / ckpt_root
    if not ckpt_root.exists():
        ckpt_root = DEFAULT_CKPT_ROOT
    handler_yaml = PROJECT / cfg["handler"]["config"]
    device = str(cfg["handler"].get("device", "cpu"))
    lookback = int(cfg["handler"].get("lookback", 72))
    batch_size = int(cfg["handler"].get("batch_size", 64))

    df = load_ohlc(Path(cfg["data"]["data_dir"]), pair, tf)
    close = df["close"].to_numpy(dtype=np.float64)
    next_r = next_bar_simple_return(close)
    ts = df["time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    wf = dict(cfg["walk_forward"])
    folds = generate_walk_forward_folds(primary_timestamps=ts, **wf)

    months = unseen_month_periods(df["time"], n_months)
    month_bounds = []
    unseen_lo = unseen_hi = None
    for per in months:
        b = _month_index_bounds(df["time"], per)
        if b is None:
            continue
        month_bounds.append((per, b[0], b[1]))
        unseen_lo = b[0] if unseen_lo is None else min(unseen_lo, b[0])
        unseen_hi = b[1] if unseen_hi is None else max(unseen_hi, b[1])

    data_cfg = load_config(handler_yaml)["data"]
    data_dir = Path(cfg["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT / data_dir
    ds = MultiTFDataset(
        pair=pair,
        data_dir=str(data_dir),
        tfs=data_cfg["tfs"],
        primary_tf=data_cfg["primary_tf"],
        lookback=data_cfg["lookback"],
        horizons=data_cfg["horizons"],
        quantiles=data_cfg["quantiles"],
        feature_cols=data_cfg.get("feature_cols"),
        context_cols=[],
        target_type="realized_vol",
        tradable_tfs=data_cfg.get("tradable_tfs"),
        rv_log_transform=bool(data_cfg.get("rv_log_transform", True)),
        vol_window=data_cfg.get("vol_window", 24),
    )
    n = len(close)
    fold_mults: Dict[int, np.ndarray] = {}
    logger.info("S-3 | %d folds | %d survivors | ckpt=%s", len(folds), len(survivors), ckpt_root)

    for fold in folds:
        logger.info("Handler fold %d %s -> %s", fold.fold_id, fold.test_start, fold.test_end)
        ds.fit_standardization(list(range(fold.train_start_idx, fold.train_end_idx)))
        handler = VolatilityTradeHandler.from_config(
            handler_yaml, checkpoint=_checkpoint(fold.fold_id, ckpt_root), device=device
        )
        m, _ = infer_multipliers(
            handler, ds, close, fold.test_start_idx, fold.test_end_idx, lookback, batch_size
        )
        fold_mults[fold.fold_id] = m

    unseen_mult = np.zeros(n, dtype=np.float64)
    if unseen_lo is not None and unseen_hi is not None:
        ds.fit_standardization(list(range(0, max(unseen_lo, 100))))
        last_fold = folds[-1].fold_id
        handler = VolatilityTradeHandler.from_config(
            handler_yaml, checkpoint=_checkpoint(last_fold, ckpt_root), device=device
        )
        unseen_mult, _ = infer_multipliers(
            handler, ds, close, unseen_lo, unseen_hi, lookback, batch_size
        )
        logger.info("Handler unseen %s -> %s using fold_%d ckpt", months[0], months[-1], last_fold)

    seed0 = int(cfg["project"]["seed"])
    fold_rows: List[Dict[str, Any]] = []
    month_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    fwd_cache: Dict[int, np.ndarray] = {}

    for spec_cfg in survivors:
        rid = str(spec_cfg["id"])
        hold = int(spec_cfg["hold"])
        spec = get_rule(rid)
        if hold not in fwd_cache:
            fwd_cache[hold] = next_r if hold <= 1 else forward_simple_return(close, hold)
        series = fwd_cache[hold]
        pos = positions_for(
            spec,
            df,
            train_mask=np.zeros(n, dtype=bool),
            returns=next_r,
            cost=cost,
            seed=seed0,
        )
        # train-using rules (none of the 3) already handled; logistic would need per-fold
        if spec.needs_train:
            raise ValueError(f"S-3 survivor {rid} needs_train; not wired")

        disc_raw = []
        disc_sz = []
        for fold in folds:
            m = fold_mults[fold.fold_id]
            sized = pos * m
            raw = score_hold_window(
                pos, series, fold.test_start_idx, fold.test_end_idx, hold, cost, ppy, start_usd
            )
            sz = score_hold_window(
                sized, series, fold.test_start_idx, fold.test_end_idx, hold, cost, ppy, start_usd
            )
            st = _trade_stats(pos, sized, fold.test_start_idx, fold.test_end_idx, hold)
            fold_rows.append(
                {
                    "rule_id": rid,
                    "hold": hold,
                    "fold_id": fold.fold_id,
                    "mode": "raw",
                    "test_start": str(fold.test_start),
                    "test_end": str(fold.test_end),
                    **raw,
                    **{f"t_{k}": v for k, v in st.items()},
                }
            )
            fold_rows.append(
                {
                    "rule_id": rid,
                    "hold": hold,
                    "fold_id": fold.fold_id,
                    "mode": "sized",
                    "test_start": str(fold.test_start),
                    "test_end": str(fold.test_end),
                    **sz,
                    **{f"t_{k}": v for k, v in st.items()},
                }
            )
            disc_raw.append(raw)
            disc_sz.append(sz)

        sized_u = pos * unseen_mult
        month_stats_raw = []
        month_stats_sz = []
        for per, a, b in month_bounds:
            st = _trade_stats(pos, sized_u, a, b, hold)
            raw_m = score_hold_window(pos, series, a, b, hold, cost, ppy, start_usd)
            sz_m = score_hold_window(sized_u, series, a, b, hold, cost, ppy, start_usd)
            for mode, sc in (("raw", raw_m), ("sized", sz_m)):
                month_rows.append(
                    {
                        "rule_id": rid,
                        "hold": hold,
                        "month": str(per),
                        "mode": mode,
                        **sc,
                        **{f"t_{k}": v for k, v in st.items()},
                    }
                )
            month_stats_raw.append(raw_m)
            month_stats_sz.append(sz_m)

        comb_raw = score_hold_window(pos, series, unseen_lo, unseen_hi, hold, cost, ppy, start_usd)
        comb_sz = score_hold_window(sized_u, series, unseen_lo, unseen_hi, hold, cost, ppy, start_usd)
        comb_tr = _trade_stats(pos, sized_u, unseen_lo, unseen_hi, hold)

        def _mean(xs, key):
            a = np.array([x[key] for x in xs], dtype=np.float64)
            return float(np.nanmean(a)) if a.size else float("nan")

        summaries.append(
            {
                "rule_id": rid,
                "hold": hold,
                "discovery_raw_exp": _mean(disc_raw, "expectancy"),
                "discovery_sized_exp": _mean(disc_sz, "expectancy"),
                "discovery_raw_profit_usd": _mean(disc_raw, "profit_usd"),
                "discovery_sized_profit_usd": _mean(disc_sz, "profit_usd"),
                "discovery_raw_folds_pos": int(sum(1 for x in disc_raw if x.get("pass_window"))),
                "discovery_sized_folds_pos": int(sum(1 for x in disc_sz if x.get("pass_window"))),
                "n_folds": len(folds),
                "unseen_raw_end_usd": comb_raw["end_usd"],
                "unseen_sized_end_usd": comb_sz["end_usd"],
                "unseen_raw_profit_usd": comb_raw["profit_usd"],
                "unseen_sized_profit_usd": comb_sz["profit_usd"],
                "unseen_raw_exp": comb_raw["expectancy"],
                "unseen_sized_exp": comb_sz["expectancy"],
                "unseen_months_green_raw": int(sum(1 for x in month_stats_raw if x.get("pass_window"))),
                "unseen_months_green_sized": int(sum(1 for x in month_stats_sz if x.get("pass_window"))),
                "unseen_n_months": len(month_bounds),
                **{f"unseen_{k}": v for k, v in comb_tr.items()},
                "sized_beats_raw_unseen": bool(
                    np.isfinite(comb_sz["profit_usd"])
                    and np.isfinite(comb_raw["profit_usd"])
                    and comb_sz["profit_usd"] > comb_raw["profit_usd"]
                ),
            }
        )

    return {
        "summary": pd.DataFrame(summaries),
        "fold_df": pd.DataFrame(fold_rows),
        "month_df": pd.DataFrame(month_rows),
        "cfg": cfg,
        "verdict": {
            "module": 1,
            "experiment": "S-3",
            "n_survivors": len(survivors),
            "start_usd": start_usd,
            "unseen_months": [str(m) for m, _, _ in month_bounds],
            "handler": "v1-eurusd-1h-rv",
            "checkpoint_root": str(ckpt_root),
        },
    }


def write_s3_pack(result: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    name = cfg["output"]["experiment_name"]
    out_local = ensure_dir(PROJECT / cfg["output"]["dir"] / name)
    save_config(cfg, out_local / "config.yaml")
    result["summary"].to_csv(out_local / "02_summary.csv", index=False)
    result["fold_df"].to_csv(out_local / "01_fold_overview.csv", index=False)
    result["month_df"].to_csv(out_local / "03_unseen_months.csv", index=False)
    save_json(result["verdict"], out_local / "10_go_nogo.json")

    s = result["summary"]
    m = result["month_df"]
    lines = [
        "# S-3 — Signal x locked Handler",
        "",
        "Handler sizes only. Signal still owns long/short/flat.",
        f"Start USD: **{result['verdict']['start_usd']}**. Unseen months: {result['verdict']['unseen_months']}.",
        "Unseen handler checkpoint = last discovery fold (fold_5), trained before 2019. No 2026 leak.",
        "",
        "## Combined 6-month unseen ($100)",
        "",
        s.to_string(index=False),
        "",
        "## Trades by month (sized vs raw)",
        "",
    ]
    for rid in s["rule_id"].tolist():
        lines.append(f"### {rid}")
        lines.append("")
        lines.append(
            "| month | mode | end USD | profit | slots | signal trades | actually traded | stood aside | mean size |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        sub = m[m["rule_id"] == rid].sort_values(["month", "mode"])
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['month']} | {r['mode']} | {r['end_usd']:.2f} | {r['profit_usd']:.2f} | "
                f"{int(r['t_n_slots'])} | {int(r['t_n_signal'])} | {int(r['t_n_traded'])} | "
                f"{int(r['t_n_stood_aside'])} | {r['t_mean_mult_on_signal']:.2f} |"
            )
        lines.append("")
    (out_local / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_local / "00_s3_report.md").write_text("\n".join(lines), encoding="utf-8")

    from shutil import copytree

    dest = ensure_dir(results_dir()) / f"{name} - {pack_stamp()}"
    if dest.exists():
        dest = results_dir() / f"{name} - {pack_stamp()}-s3"
    copytree(out_local, dest)
    return dest
