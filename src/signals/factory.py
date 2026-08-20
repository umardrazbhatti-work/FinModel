"""Module 1 rule factory: discovery folds + last-6-month unseen $100 account."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.economic import max_drawdown_from_wealth, score_position_returns, wealth_curve_from_returns
from src.signals.catalog import RuleSpec, build_catalog, list_rules, positions_for
from src.signals.labels import forward_simple_return, nonoverlap_indices
from src.signals.ledger import append_rows, write_scoreboard
from src.signals.rules import next_bar_simple_return
from src.training import generate_walk_forward_folds
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

logger = get_logger("mtp.rule_factory")

PROJECT = Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    return PROJECT.parent / "Results"


def pack_stamp() -> str:
    return datetime.now().strftime("%d-%m-%y %H%MHrs")


def load_ohlc(data_dir: Path, pair: str, tf: str) -> pd.DataFrame:
    stem = "daily" if tf in {"1d", "daily"} else tf
    path = Path(data_dir) / f"{pair}_{stem}_aligned.parquet"
    if not path.is_absolute():
        path = PROJECT / path
    df = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _naive_utc(times: pd.Series) -> pd.Series:
    ts = pd.to_datetime(times, utc=True)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def unseen_month_periods(times: pd.Series, n_months: int = 6) -> List[pd.Period]:
    last = pd.Timestamp(_naive_utc(times).max())
    end = last.to_period("M")
    return list(pd.period_range(end=end, periods=int(n_months), freq="M"))


def _month_index_bounds(times: pd.Series, period: pd.Period) -> Optional[Tuple[int, int]]:
    ts = _naive_utc(times)
    mask = ts.dt.to_period("M") == period
    idx = np.flatnonzero(mask.to_numpy())
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1] + 1)


def attach_account(scored: Dict[str, Any], start_usd: float) -> Dict[str, Any]:
    fw = scored.get("final_wealth")
    tr = scored.get("total_return")
    if fw is None or not np.isfinite(fw):
        fw = (1.0 + float(tr)) if tr is not None and np.isfinite(tr) else float("nan")
        scored["final_wealth"] = fw
    end = float(start_usd) * float(fw) if np.isfinite(fw) else float("nan")
    scored["start_usd"] = float(start_usd)
    scored["end_usd"] = end
    scored["profit_usd"] = (end - float(start_usd)) if np.isfinite(end) else float("nan")
    scored["profit_pct"] = ((end / float(start_usd)) - 1.0) * 100.0 if np.isfinite(end) else float("nan")
    mdd = scored.get("max_drawdown")
    scored["max_dd_usd"] = (
        float(start_usd) * abs(float(mdd)) if mdd is not None and np.isfinite(mdd) else float("nan")
    )
    return scored


def score_hold_window(
    pos: np.ndarray,
    ret_or_fwd: np.ndarray,
    start: int,
    end: int,
    hold: int,
    cost: float,
    periods_per_year: float,
    start_usd: float,
) -> Dict[str, Any]:
    """H=1: next-bar path. H>1: non-overlapping holds, 2-way cost when |pos|=1."""
    h = int(hold)
    if h <= 1:
        sl = slice(int(start), int(end))
        p = np.asarray(pos[sl], dtype=np.float64)
        r = np.asarray(ret_or_fwd[sl], dtype=np.float64)
        valid = np.isfinite(r)
        p = p[valid]
        r = r[valid]
        if p.size == 0:
            empty = {
                "n": 0,
                "n_active": 0,
                "expectancy": float("nan"),
                "total_return": float("nan"),
                "sharpe": float("nan"),
                "hit_rate": float("nan"),
                "pct_long": float("nan"),
                "pct_short": float("nan"),
                "pct_flat": float("nan"),
                "max_drawdown": float("nan"),
                "final_wealth": float("nan"),
                "pass_window": False,
            }
            return attach_account(empty, start_usd)
        scored = score_position_returns(p, r, cost=cost, periods_per_year=periods_per_year)
        out = {
            "n": int(scored["n"]),
            "n_active": int(np.sum(np.abs(p) > 0)),
            "expectancy": float(scored["mean_net_return"]),
            "total_return": float(scored["total_return"]),
            "sharpe": float(scored["sharpe"]) if np.isfinite(scored["sharpe"]) else float("nan"),
            "hit_rate": scored["hit_rate"],
            "pct_long": float(scored["pct_long"]),
            "pct_short": float(scored["pct_short"]),
            "pct_flat": float(scored["pct_flat"]),
            "max_drawdown": float(scored["max_drawdown"]),
            "final_wealth": float(scored["final_wealth"]),
            "pass_window": bool(np.isfinite(scored["mean_net_return"]) and scored["mean_net_return"] > 0),
        }
        return attach_account(out, start_usd)

    idx = nonoverlap_indices(int(start), int(end), h, n=len(pos))
    if idx.size == 0:
        empty = {
            "n": 0,
            "n_active": 0,
            "expectancy": float("nan"),
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "pct_long": float("nan"),
            "pct_short": float("nan"),
            "pct_flat": float("nan"),
            "max_drawdown": float("nan"),
            "final_wealth": float("nan"),
            "pass_window": False,
        }
        return attach_account(empty, start_usd)
    p = np.asarray(pos[idx], dtype=np.float64)
    r = np.asarray(ret_or_fwd[idx], dtype=np.float64)
    ok = np.isfinite(r)
    p = p[ok]
    r = r[ok]
    if p.size == 0:
        empty = {
            "n": 0,
            "n_active": 0,
            "expectancy": float("nan"),
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "hit_rate": float("nan"),
            "pct_long": float("nan"),
            "pct_short": float("nan"),
            "pct_flat": float("nan"),
            "max_drawdown": float("nan"),
            "final_wealth": float("nan"),
            "pass_window": False,
        }
        return attach_account(empty, start_usd)
    net = p * r - (2.0 * float(cost)) * np.abs(p)
    mu = float(net.mean())
    sig = float(net.std(ddof=0))
    trades_per_year = float(periods_per_year) / float(h)
    if net.size >= 2 and sig > 1e-12:
        sharpe = float(np.sqrt(trades_per_year) * mu / sig)
    else:
        sharpe = float("nan")
    traded = np.abs(p) > 0
    if traded.any():
        hit = float(((p[traded] * r[traded]) > 0).mean())
    else:
        hit = float("nan")
    wealth = wealth_curve_from_returns(net)
    out = {
        "n": int(net.size),
        "n_active": int(traded.sum()),
        "expectancy": mu,
        "total_return": float(net.sum()),
        "sharpe": sharpe,
        "hit_rate": hit,
        "pct_long": float((p > 0).mean()),
        "pct_short": float((p < 0).mean()),
        "pct_flat": float((p == 0).mean()),
        "max_drawdown": max_drawdown_from_wealth(wealth),
        "final_wealth": float(wealth[-1]),
        "pass_window": bool(np.isfinite(mu) and mu > 0),
    }
    return attach_account(out, start_usd)


def _control_means(summary: pd.DataFrame, hold: int) -> Tuple[float, float]:
    sub = summary[summary["hold"].astype(int) == int(hold)]
    al = sub.loc[sub["rule_id"].astype(str).str.startswith("always_long"), "discovery_mean_exp"]
    cf = sub.loc[sub["rule_id"].astype(str).str.startswith("coin_flip"), "discovery_mean_exp"]
    return (
        float(al.iloc[0]) if len(al) else 0.0,
        float(cf.iloc[0]) if len(cf) else 0.0,
    )


def apply_discovery_gate(
    row: Dict[str, Any],
    al: float,
    cf: float,
    min_exp: float,
    min_frac: float,
    max_flat: float,
    min_active: float,
) -> Dict[str, Any]:
    reasons: List[str] = []
    exp = row.get("discovery_mean_exp")
    frac = row.get("discovery_frac_pos")
    flat = row.get("discovery_pct_flat")
    active = row.get("discovery_mean_active")
    if row.get("control"):
        row["discovery_pass"] = False
        row["discovery_reason"] = "control"
        return row
    if exp is None or not np.isfinite(exp) or float(exp) <= float(min_exp):
        reasons.append("exp<=0")
    if frac is None or float(frac) < float(min_frac):
        reasons.append("folds")
    if exp is not None and np.isfinite(exp) and float(exp) <= al:
        reasons.append("vs_always_long")
    if exp is not None and np.isfinite(exp) and float(exp) <= cf:
        reasons.append("vs_coin_flip")
    if flat is not None and np.isfinite(flat) and float(flat) >= float(max_flat):
        reasons.append("flat_trick")
    if active is None or not np.isfinite(active) or float(active) < float(min_active):
        reasons.append("few_trades")
    row["discovery_pass"] = len(reasons) == 0
    row["discovery_reason"] = "pass" if not reasons else ",".join(reasons)
    return row


def apply_unseen_gate(
    row: Dict[str, Any],
    al_profit: float,
    cf_profit: float,
    min_months_green: int,
    max_flat: float,
    min_active: float,
) -> Dict[str, Any]:
    reasons: List[str] = []
    if row.get("control"):
        row["unseen_pass"] = False
        row["survivor"] = False
        row["unseen_reason"] = "control"
        return row
    exp = row.get("unseen_exp")
    profit = row.get("unseen_profit_usd")
    green = row.get("unseen_months_green")
    flat = row.get("unseen_pct_flat")
    active = row.get("unseen_n_active")
    n_m = row.get("unseen_n_months")
    if exp is None or not np.isfinite(exp) or float(exp) <= 0:
        reasons.append("exp<=0")
    if profit is None or not np.isfinite(profit) or float(profit) <= 0:
        reasons.append("usd<=0")
    need = int(min_months_green)
    if n_m is not None:
        need = min(need, int(n_m))
    if green is None or int(green) < need:
        reasons.append("months")
    if profit is not None and np.isfinite(profit) and float(profit) <= al_profit:
        reasons.append("vs_always_long")
    if profit is not None and np.isfinite(profit) and float(profit) <= cf_profit:
        reasons.append("vs_coin_flip")
    if flat is not None and np.isfinite(flat) and float(flat) >= float(max_flat):
        reasons.append("flat_trick")
    if active is None or not np.isfinite(active) or float(active) < float(min_active):
        reasons.append("few_trades")
    row["unseen_pass"] = len(reasons) == 0
    row["unseen_reason"] = "pass" if not reasons else ",".join(reasons)
    row["survivor"] = bool(row.get("discovery_pass") and row["unseen_pass"])
    return row


def _inject_controls(specs: List[RuleSpec], hold: int) -> List[RuleSpec]:
    have = {s.rule_id for s in specs}
    extra = []
    for cid in (f"always_flat_h{hold}", f"always_long_h{hold}", f"coin_flip_h{hold}"):
        if cid not in have:
            found = [s for s in list_rules(ids=[cid]) if s.hold == hold]
            extra.extend(found)
    return extra + specs


def run_wave(
    cfg: Dict[str, Any],
    *,
    max_folds: Optional[int] = None,
) -> Dict[str, Any]:
    build_catalog()
    pair = cfg["data"]["pair"]
    tf = cfg["data"]["tf"]
    cost = float(cfg["data"]["cost"])
    ppy = float(cfg["signal"]["periods_per_year"])
    start_usd = float(cfg["signal"].get("start_usd", 100.0))
    n_months = int(cfg["signal"].get("unseen_months", 6))
    wave = int(cfg["signal"]["wave"])
    families = list(cfg["signal"].get("families") or [])
    ids = list(cfg["signal"].get("rule_ids") or [])
    hold_filter = cfg["signal"].get("hold")
    hold_override = cfg["signal"].get("hold_override")
    if cfg["signal"].get("only_discovery_pass"):
        from src.signals.ledger import load_ledger

        led = load_ledger(results_dir())
        if led.empty or "discovery_pass" not in led.columns:
            raise ValueError("no discovery-pass rows in ledger")
        ids = [str(x) for x in led.loc[led["discovery_pass"] == True, "rule_id"].unique()]
        families = []
        hold_filter = None
    ev = cfg["evaluation"]

    data_dir = cfg["data"]["data_dir"]
    df = load_ohlc(Path(data_dir), pair, tf)
    close = df["close"].to_numpy(dtype=np.float64)
    next_r = next_bar_simple_return(close)
    ts = df["time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()

    specs = list_rules(families=families or None, hold=hold_filter, ids=ids or None)
    if not specs:
        raise ValueError(f"no rules for wave={wave} families={families} ids={ids}")

    holds = sorted({int(s.hold) for s in specs})
    score_holds = set(holds)
    if hold_override is not None:
        score_holds.add(int(hold_override))
    merged: List[RuleSpec] = []
    seen = set()
    for h in holds:
        chunk = [s for s in specs if int(s.hold) == h]
        chunk = _inject_controls(chunk, h)
        for s in chunk:
            if s.rule_id in seen:
                continue
            seen.add(s.rule_id)
            merged.append(s)
    specs = merged

    wf = dict(cfg["walk_forward"])
    if max_folds is not None:
        wf["max_folds"] = int(max_folds)
    folds = generate_walk_forward_folds(primary_timestamps=ts, **wf)

    fwd_cache = {h: (next_r if h <= 1 else forward_simple_return(close, h)) for h in score_holds}
    months = unseen_month_periods(df["time"], n_months)
    unseen_lo = None
    unseen_hi = None
    month_bounds: List[Tuple[pd.Period, int, int]] = []
    for per in months:
        b = _month_index_bounds(df["time"], per)
        if b is None:
            continue
        month_bounds.append((per, b[0], b[1]))
        unseen_lo = b[0] if unseen_lo is None else min(unseen_lo, b[0])
        unseen_hi = b[1] if unseen_hi is None else max(unseen_hi, b[1])

    seed0 = int(cfg["project"]["seed"])
    fold_rows: List[Dict[str, Any]] = []
    month_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    logger.info(
        "Wave %s | %d rules | %d folds | unseen months=%s | start_usd=%.2f",
        wave,
        len(specs),
        len(folds),
        [str(m) for m, _, _ in month_bounds],
        start_usd,
    )

    for spec in specs:
        h = int(hold_override) if hold_override is not None else int(spec.hold)
        series = fwd_cache[h]
        disc_exps = []
        disc_pass = []
        disc_flat = []
        disc_active = []
        disc_sharpe = []
        disc_usd = []

        cached_pos = None
        if not spec.needs_train:
            cached_pos = positions_for(
                spec,
                df,
                train_mask=np.zeros(len(df), dtype=bool),
                returns=next_r,
                cost=cost,
                seed=seed0,
            )

        for fold in folds:
            train_mask = np.zeros(len(df), dtype=bool)
            train_mask[fold.train_start_idx : fold.train_end_idx] = True
            if cached_pos is not None:
                pos = cached_pos
            else:
                pos = positions_for(
                    spec,
                    df,
                    train_mask=train_mask,
                    returns=next_r,
                    cost=cost,
                    seed=seed0 + fold.fold_id,
                )
            scored = score_hold_window(
                pos,
                series,
                fold.test_start_idx,
                fold.test_end_idx,
                h,
                cost,
                ppy,
                start_usd,
            )
            rec = {
                "wave": wave,
                "pair": pair,
                "fold_id": fold.fold_id,
                "rule_id": spec.rule_id,
                "family": spec.family,
                "hold": h,
                "control": bool(spec.control),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                **scored,
            }
            fold_rows.append(rec)
            disc_exps.append(scored["expectancy"])
            disc_pass.append(bool(scored["pass_window"]))
            disc_flat.append(scored["pct_flat"])
            disc_active.append(scored["n_active"])
            disc_sharpe.append(scored["sharpe"])
            disc_usd.append(scored["profit_usd"])

        # Unseen: train on everything before the 6-month window
        if cached_pos is not None:
            pos_u = cached_pos
        else:
            train_mask = np.zeros(len(df), dtype=bool)
            if unseen_lo is not None and unseen_lo > 0:
                train_mask[:unseen_lo] = True
            pos_u = positions_for(
                spec,
                df,
                train_mask=train_mask,
                returns=next_r,
                cost=cost,
                seed=seed0 + 100,
            )
        month_profits = []
        month_green = 0
        for per, a, b in month_bounds:
            ms = score_hold_window(pos_u, series, a, b, h, cost, ppy, start_usd)
            month_rows.append(
                {
                    "wave": wave,
                    "pair": pair,
                    "rule_id": spec.rule_id,
                    "hold": h,
                    "month": str(per),
                    **ms,
                }
            )
            if np.isfinite(ms["profit_usd"]):
                month_profits.append(float(ms["profit_usd"]))
            if ms["pass_window"]:
                month_green += 1

        if unseen_lo is not None and unseen_hi is not None:
            comb = score_hold_window(
                pos_u, series, unseen_lo, unseen_hi, h, cost, ppy, start_usd
            )
        else:
            comb = attach_account(
                {
                    "n": 0,
                    "n_active": 0,
                    "expectancy": float("nan"),
                    "total_return": float("nan"),
                    "sharpe": float("nan"),
                    "hit_rate": float("nan"),
                    "pct_long": float("nan"),
                    "pct_short": float("nan"),
                    "pct_flat": float("nan"),
                    "max_drawdown": float("nan"),
                    "final_wealth": float("nan"),
                    "pass_window": False,
                },
                start_usd,
            )

        def _nanmean(xs: Sequence) -> float:
            a = np.asarray(xs, dtype=np.float64)
            return float(np.nanmean(a)) if a.size else float("nan")

        summaries.append(
            {
                "wave": wave,
                "pair": pair,
                "rule_id": spec.rule_id,
                "family": spec.family,
                "hold": h,
                "control": bool(spec.control),
                "note": spec.note,
                "discovery_mean_exp": _nanmean(disc_exps),
                "discovery_frac_pos": float(np.mean(disc_pass)) if disc_pass else 0.0,
                "discovery_folds_pos": int(np.sum(disc_pass)),
                "n_folds": int(len(folds)),
                "discovery_pct_flat": _nanmean(disc_flat),
                "discovery_mean_active": _nanmean(disc_active),
                "discovery_mean_sharpe": _nanmean(disc_sharpe),
                "discovery_mean_profit_usd": _nanmean(disc_usd),
                "unseen_start": str(months[0]) if months else "",
                "unseen_end": str(months[-1]) if months else "",
                "unseen_n_months": int(len(month_bounds)),
                "unseen_months_green": int(month_green),
                "unseen_exp": comb["expectancy"],
                "unseen_n": comb["n"],
                "unseen_n_active": comb["n_active"],
                "unseen_pct_flat": comb["pct_flat"],
                "unseen_pct_long": comb["pct_long"],
                "unseen_pct_short": comb["pct_short"],
                "unseen_sharpe": comb["sharpe"],
                "unseen_hit": comb["hit_rate"],
                "unseen_start_usd": start_usd,
                "unseen_end_usd": comb["end_usd"],
                "unseen_profit_usd": comb["profit_usd"],
                "unseen_profit_pct": comb["profit_pct"],
                "unseen_max_dd": comb["max_drawdown"],
                "unseen_max_dd_usd": comb["max_dd_usd"],
            }
        )

    summary = pd.DataFrame(summaries)
    for h in sorted(score_holds):
        al, cf = _control_means(summary, h)
        for i, row in summary.iterrows():
            if int(row["hold"]) != h:
                continue
            rec = apply_discovery_gate(
                row.to_dict(),
                al,
                cf,
                min_exp=float(ev.get("min_mean_expectancy", 0.0)),
                min_frac=float(ev.get("min_frac_folds", 0.5)),
                max_flat=float(ev.get("max_pct_flat", 0.95)),
                min_active=float(ev.get("min_discovery_active", 30)),
            )
            for k, v in rec.items():
                summary.at[i, k] = v

        subh = summary[summary["hold"].astype(int) == int(h)]
        al_row = subh.loc[subh["rule_id"].astype(str).str.startswith("always_long"), "unseen_profit_usd"]
        cf_row = subh.loc[subh["rule_id"].astype(str).str.startswith("coin_flip"), "unseen_profit_usd"]
        al_p = float(al_row.iloc[0]) if len(al_row) and np.isfinite(al_row.iloc[0]) else 0.0
        cf_p = float(cf_row.iloc[0]) if len(cf_row) and np.isfinite(cf_row.iloc[0]) else 0.0
        for i, row in summary.iterrows():
            if int(row["hold"]) != h:
                continue
            rec = apply_unseen_gate(
                row.to_dict(),
                al_p,
                cf_p,
                min_months_green=int(ev.get("min_months_green", 4)),
                max_flat=float(ev.get("max_pct_flat", 0.95)),
                min_active=float(ev.get("min_unseen_active", 50)),
            )
            for k, v in rec.items():
                summary.at[i, k] = v

    n_pass = int(summary["discovery_pass"].sum())
    n_surv = int(summary["survivor"].sum())
    verdict = {
        "module": 1,
        "experiment": f"rule_factory_w{wave:02d}",
        "wave": wave,
        "families": families,
        "n_rules": int(len(specs)),
        "n_folds": int(len(folds)),
        "start_usd": start_usd,
        "unseen_months": [str(m) for m, _, _ in month_bounds],
        "discovery_pass_count": n_pass,
        "survivor_count": n_surv,
        "discovery_pass_ids": summary.loc[summary["discovery_pass"], "rule_id"].tolist(),
        "survivor_ids": summary.loc[summary["survivor"], "rule_id"].tolist(),
        "pair": pair,
        "tf": tf,
        "cost": cost,
    }
    return {
        "verdict": verdict,
        "summary": summary,
        "fold_df": pd.DataFrame(fold_rows),
        "month_df": pd.DataFrame(month_rows),
        "cfg": cfg,
        "wave": wave,
    }


def write_wave_pack(result: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    wave = int(result["wave"])
    name = cfg["output"]["experiment_name"]
    out_local = ensure_dir(PROJECT / cfg["output"]["dir"] / name)
    from src.utils.config import save_config

    save_config(cfg, out_local / "config.yaml")
    result["summary"].to_csv(out_local / "02_summary_by_rule.csv", index=False)
    result["fold_df"].to_csv(out_local / "01_fold_overview.csv", index=False)
    result["month_df"].to_csv(out_local / "03_unseen_months.csv", index=False)
    save_json(result["verdict"], out_local / "10_go_nogo.json")

    summary = result["summary"]
    v = result["verdict"]
    lines = [
        f"# Verdict — {name}",
        "",
        f"**Module:** 1 — Signal / Alpha | **Wave:** {wave}",
        f"**Rules scored:** {v['n_rules']}",
        f"**Discovery PASS:** {v['discovery_pass_count']}",
        f"**Survivors (discovery + 6-month unseen from $100):** {v['survivor_count']}",
        f"**Unseen months:** {v['unseen_months']}",
        f"**Start USD:** {v['start_usd']}",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Discovery PASS ids",
        "",
        ", ".join(v["discovery_pass_ids"]) or "none",
        "",
        "## Survivor ids",
        "",
        ", ".join(v["survivor_ids"]) or "none",
        "",
    ]
    (out_local / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_local / "00_wave_report.md").write_text("\n".join(lines), encoding="utf-8")

    stamp = pack_stamp()
    dest = ensure_dir(results_dir()) / f"{name} - {stamp}"
    if dest.exists():
        dest = results_dir() / f"{name} - {stamp}-{wave}"
    copytree(out_local, dest)
    return dest


def lock_wave(result: Dict[str, Any], pack_path: Path, scoreboard: bool = True) -> int:
    rows = result["summary"].to_dict(orient="records")
    for r in rows:
        r["pack"] = str(pack_path)
    n = append_rows(results_dir(), rows)
    if scoreboard:
        write_scoreboard(
            results_dir(),
            extra_lines=[
                f"Last pack: `{pack_path.name}`",
                f"Wave {result['wave']} locked {n} new rows.",
            ],
        )
    return n
