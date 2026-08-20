"""Walk-forward + unseen scoring for R-multiple systems."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.signals.factory import load_ohlc, results_dir, unseen_month_periods, _month_index_bounds
from src.signals.r_engine import simulate_r
from src.signals.r_entries import RSpec, build_fire_map, fire_for, list_rspecs
from src.training import generate_walk_forward_folds


BARS = {
    "min_payoff": 1.8,
    "min_pf": 1.5,
    "min_sharpe": 1.2,
    "max_dd": 0.25,
    "min_disc_trades": 20,
    "min_fold_trades": 8,
    "min_unseen_trades": 15,
    "min_frac_folds": 0.5,
}


def _nanmean(xs: Sequence) -> float:
    a = np.asarray(list(xs), dtype=np.float64)
    return float(np.nanmean(a)) if a.size else float("nan")


def apply_r_gates(row: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    if row.get("invalid_high_wr_low_payoff"):
        reasons.append("high_wr_low_payoff")
    e = row.get("discovery_expectancy_r")
    if e is None or not np.isfinite(e) or float(e) <= 0:
        reasons.append("e<=0")
    frac = row.get("discovery_frac_pos")
    if frac is None or float(frac) < BARS["min_frac_folds"]:
        reasons.append("folds")
    pay = row.get("discovery_payoff")
    if pay is None or not np.isfinite(pay) or float(pay) < BARS["min_payoff"]:
        reasons.append("payoff<1.8")
    pf = row.get("discovery_pf")
    if pf is None or not np.isfinite(pf) or float(pf) < BARS["min_pf"]:
        reasons.append("pf<1.5")
    sh = row.get("discovery_sharpe")
    if sh is None or not np.isfinite(sh) or float(sh) < BARS["min_sharpe"]:
        reasons.append("sharpe<1.2")
    dd = row.get("discovery_max_dd")
    if dd is None or not np.isfinite(dd) or float(dd) < -BARS["max_dd"]:
        reasons.append("dd>25pct")
    nt = row.get("discovery_n_trades")
    if nt is None or int(nt) < BARS["min_disc_trades"]:
        reasons.append("few_trades")
    aw = row.get("discovery_avg_win_r")
    al = row.get("discovery_avg_loss_r")
    if aw is None or al is None or not np.isfinite(aw) or not np.isfinite(al) or float(aw) <= float(al):
        reasons.append("avgwin<=avgloss")
    row["discovery_pass"] = len(reasons) == 0
    row["discovery_reason"] = "pass" if not reasons else ",".join(reasons)

    ureasons: List[str] = []
    ue = row.get("unseen_expectancy_r")
    if ue is None or not np.isfinite(ue) or float(ue) <= 0:
        ureasons.append("e<=0")
    up = row.get("unseen_payoff")
    if up is None or not np.isfinite(up) or float(up) < BARS["min_payoff"]:
        ureasons.append("payoff<1.8")
    upf = row.get("unseen_pf")
    if upf is None or not np.isfinite(upf) or float(upf) < BARS["min_pf"]:
        ureasons.append("pf<1.5")
    ush = row.get("unseen_sharpe")
    if ush is None or not np.isfinite(ush) or float(ush) < BARS["min_sharpe"]:
        ureasons.append("sharpe<1.2")
    udd = row.get("unseen_max_dd")
    if udd is None or not np.isfinite(udd) or float(udd) < -BARS["max_dd"]:
        ureasons.append("dd>25pct")
    unt = row.get("unseen_n_trades")
    if unt is None or int(unt) < BARS["min_unseen_trades"]:
        ureasons.append("few_trades")
    if row.get("unseen_profit_usd") is None or float(row.get("unseen_profit_usd") or 0) <= 0:
        ureasons.append("usd<=0")
    uaw = row.get("unseen_avg_win_r")
    ual = row.get("unseen_avg_loss_r")
    if uaw is None or ual is None or not np.isfinite(uaw) or not np.isfinite(ual) or float(uaw) <= float(ual):
        ureasons.append("avgwin<=avgloss")
    if row.get("unseen_invalid_high_wr"):
        ureasons.append("high_wr_low_payoff")
    row["unseen_pass"] = len(ureasons) == 0
    row["unseen_reason"] = "pass" if not ureasons else ",".join(ureasons)
    row["survivor"] = bool(row["discovery_pass"] and row["unseen_pass"])
    return row


def _pack_sim(sim: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}n_trades": int(sim["n_trades"]),
        f"{prefix}win_rate": sim["win_rate"],
        f"{prefix}avg_win_r": sim["avg_win_r"],
        f"{prefix}avg_loss_r": sim["avg_loss_r"],
        f"{prefix}payoff": sim["payoff"],
        f"{prefix}expectancy_r": sim["expectancy_r"],
        f"{prefix}pf": sim["profit_factor"],
        f"{prefix}sharpe": sim["sharpe"],
        f"{prefix}max_dd": sim["max_drawdown"],
        f"{prefix}end_usd": sim["end_usd"],
        f"{prefix}profit_usd": sim["profit_usd"],
    }


def score_spec(
    spec: RSpec,
    df: pd.DataFrame,
    built: Dict[str, np.ndarray],
    *,
    folds,
    unseen_lo: Optional[int],
    unseen_hi: Optional[int],
    month_bounds,
    spread: float,
    slip: float,
    start_usd: float,
    ppy: float,
) -> Dict[str, Any]:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    a = built["_atr"]
    fire = fire_for(spec, built)
    common = dict(
        sl_mult=spec.sl_mult,
        rr=spec.rr,
        trail=spec.trail,
        max_hold=spec.max_hold,
        spread=spread,
        slip=slip,
        risk_frac=spec.risk_frac,
        start_usd=start_usd,
        periods_per_year=ppy,
    )
    fold_e = []
    fold_ok = []
    fold_n = []
    fold_pay = []
    fold_pf = []
    fold_sh = []
    fold_dd = []
    fold_aw = []
    fold_al = []
    fold_wr = []
    invalid = False
    for fold in folds:
        sim = simulate_r(o, h, l, c, a, fire, lo=fold.test_start_idx, hi=fold.test_end_idx, **common)
        fold_e.append(sim["expectancy_r"])
        fold_ok.append(bool(sim["pass_e"]))
        fold_n.append(sim["n_trades"])
        fold_pay.append(sim["payoff"])
        fold_pf.append(sim["profit_factor"])
        fold_sh.append(sim["sharpe"])
        fold_dd.append(sim["max_drawdown"])
        fold_aw.append(sim["avg_win_r"])
        fold_al.append(sim["avg_loss_r"])
        fold_wr.append(sim["win_rate"])
        invalid = invalid or bool(sim["invalid_high_wr_low_payoff"])

    if unseen_lo is not None and unseen_hi is not None:
        usim = simulate_r(o, h, l, c, a, fire, lo=unseen_lo, hi=unseen_hi, **common)
    else:
        usim = simulate_r(o, h, l, c, a, fire, lo=0, hi=0, **common)

    month_green = 0
    month_rows = []
    for per, a0, b0 in month_bounds:
        ms = simulate_r(o, h, l, c, a, fire, lo=a0, hi=b0, **common)
        month_rows.append({"month": str(per), **_pack_sim(ms, "")})
        if ms["n_trades"] and np.isfinite(ms["expectancy_r"]) and ms["expectancy_r"] > 0:
            month_green += 1

    row: Dict[str, Any] = {
        "rule_id": spec.rule_id,
        "entry": spec.entry,
        "sl_mult": spec.sl_mult,
        "rr": spec.rr,
        "session": spec.session,
        "trail": spec.trail,
        "max_hold": spec.max_hold,
        "risk_frac": spec.risk_frac,
        "discovery_n_trades": int(np.nansum(fold_n)),
        "discovery_expectancy_r": _nanmean(fold_e),
        "discovery_frac_pos": float(np.mean(fold_ok)) if fold_ok else 0.0,
        "discovery_folds_pos": int(np.sum(fold_ok)),
        "n_folds": int(len(folds)),
        "discovery_payoff": _nanmean(fold_pay),
        "discovery_pf": _nanmean(fold_pf),
        "discovery_sharpe": _nanmean(fold_sh),
        "discovery_max_dd": _nanmean(fold_dd),
        "discovery_avg_win_r": _nanmean(fold_aw),
        "discovery_avg_loss_r": _nanmean(fold_al),
        "discovery_win_rate": _nanmean(fold_wr),
        "invalid_high_wr_low_payoff": invalid,
        "unseen_n_trades": int(usim["n_trades"]),
        "unseen_win_rate": usim["win_rate"],
        "unseen_avg_win_r": usim["avg_win_r"],
        "unseen_avg_loss_r": usim["avg_loss_r"],
        "unseen_payoff": usim["payoff"],
        "unseen_expectancy_r": usim["expectancy_r"],
        "unseen_pf": usim["profit_factor"],
        "unseen_sharpe": usim["sharpe"],
        "unseen_max_dd": usim["max_drawdown"],
        "unseen_end_usd": usim["end_usd"],
        "unseen_profit_usd": usim["profit_usd"],
        "unseen_months_green": month_green,
        "unseen_n_months": len(month_bounds),
        "unseen_invalid_high_wr": bool(usim["invalid_high_wr_low_payoff"]),
        "unseen_reasons": str(usim.get("reasons") or {}),
    }
    apply_r_gates(row)
    row["_unseen_sim"] = usim
    row["_month_rows"] = month_rows
    return row


def run_pair(
    pair: str,
    cost: float,
    slip: float,
    cfg: Dict[str, Any],
    specs: Optional[List[RSpec]] = None,
) -> pd.DataFrame:
    data_dir = cfg["data"]["data_dir"]
    tf = cfg["data"]["tf"]
    start_usd = float(cfg["signal"].get("start_usd", 100.0))
    ppy = float(cfg["signal"].get("periods_per_year", 6048))
    n_months = int(cfg["signal"].get("unseen_months", 6))
    df = load_ohlc(Path(data_dir), pair, tf)
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
    built = build_fire_map(df)
    specs = specs if specs is not None else list_rspecs()
    rows = []
    for spec in specs:
        row = score_spec(
            spec,
            df,
            built,
            folds=folds,
            unseen_lo=unseen_lo,
            unseen_hi=unseen_hi,
            month_bounds=month_bounds,
            spread=cost,
            slip=slip,
            start_usd=start_usd,
            ppy=ppy,
        )
        row["pair"] = pair
        row["cost"] = cost
        row["slip"] = slip
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["BARS", "apply_r_gates", "run_pair", "score_spec", "results_dir"]
