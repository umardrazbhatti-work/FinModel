#!/usr/bin/env python
"""Write R_BOOK.md + equity curves from R_LEDGER / R_ARTIFACTS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

RES = Path(__file__).resolve().parents[2] / "Results"


def _f(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def main() -> None:
    df = pd.read_csv(RES / "R_LEDGER.csv")
    art = {}
    ap = RES / "R_ARTIFACTS.json"
    if ap.exists():
        art = json.loads(ap.read_text(encoding="utf-8"))
    s = df[df["survivor"] == True].copy()
    s = s.sort_values("unseen_expectancy_r", ascending=False)
    lines = [
        "# R-multiple book — SL/TP, fixed 1% risk",
        "",
        "Protocol is **not** the 100k oscillator factory. Every trade has a predefined stop and target.",
        "Expectancy E = (WR × AvgWin) − (LR × AvgLoss) in **R**. Risk = **1%** of capital per trade.",
        "Costs: spread + slippage. Filters: ATR > median, ADX structure (except London ORB), London/NY session.",
        "Walk-forward 6 folds (2017-18) + unseen 2026-03..08 from **$100**.",
        "",
        "Mandatory bars: payoff ≥ 1.8, profit factor ≥ 1.5, Sharpe ≥ 1.2, max DD ≤ 25%, AvgWin > AvgLoss.",
        "High win-rate / low-payoff systems are **invalid**.",
        "",
        f"- Systems scored: **{len(df):,}**",
        f"- Discovery PASS: **{int(df['discovery_pass'].sum())}**",
        f"- Official survivors: **{len(s)}**",
        "",
        "## By pair",
        "",
        df.groupby("pair")
        .agg(n=("rule_id", "count"), disc=("discovery_pass", "sum"), surv=("survivor", "sum"))
        .to_string(),
        "",
        "## Survivors",
        "",
    ]
    disc = df[df["discovery_pass"] == True].copy()
    disc = disc.sort_values("unseen_expectancy_r", ascending=False)
    if s.empty:
        lines += [
            "None. No system cleared discovery **and** unseen under the R-multiple bars.",
            "",
            "## Discovery PASS that died on unseen (the only honest shortlist)",
            "",
            "These beat 2017-18 walk-forward on payoff / PF / Sharpe / DD / E.",
            "They are **not** tradeable until unseen also clears. n<15 in 6 months is too thin.",
            "",
            "| pair | rule | disc n | disc WR | disc payoff | disc E | unseen n | unseen WR | unseen payoff | unseen E | PF | Sharpe | $100 | why unseen failed |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for _, r in disc.iterrows():
            lines.append(
                f"| {r['pair']} | `{r['rule_id']}` | {int(r['discovery_n_trades'])} | "
                f"{_f(r['discovery_win_rate'], 3)} | {_f(r['discovery_payoff'], 2)} | "
                f"{_f(r['discovery_expectancy_r'], 3)} | {int(r['unseen_n_trades'])} | "
                f"{_f(r['unseen_win_rate'], 3)} | {_f(r['unseen_payoff'], 2)} | "
                f"{_f(r['unseen_expectancy_r'], 3)} | {_f(r['unseen_pf'], 2)} | "
                f"{_f(r['unseen_sharpe'], 2)} | {_f(r['unseen_end_usd'], 2)} | "
                f"{r.get('unseen_reason','')} |"
            )
        lines += [
            "",
            "## Highest unseen E with tiny n (do not promote — sample noise)",
            "",
            "| pair | rule | trades | WR | AvgW | AvgL | payoff | E (R) | PF | Sharpe | DD | $100 became | why not |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        top = df.sort_values("unseen_expectancy_r", ascending=False).head(15)
        for _, r in top.iterrows():
            lines.append(
                f"| {r['pair']} | `{r['rule_id']}` | {int(r['unseen_n_trades'])} | "
                f"{_f(r['unseen_win_rate'], 3)} | {_f(r['unseen_avg_win_r'], 2)} | "
                f"{_f(r['unseen_avg_loss_r'], 2)} | {_f(r['unseen_payoff'], 2)} | "
                f"{_f(r['unseen_expectancy_r'], 3)} | {_f(r['unseen_pf'], 2)} | "
                f"{_f(r['unseen_sharpe'], 2)} | {_f(r['unseen_max_dd'], 3)} | "
                f"{_f(r['unseen_end_usd'], 2)} | {r.get('discovery_reason','')}/{r.get('unseen_reason','')} |"
            )
    else:
        lines += [
            "| pair | rule | trades | WR | AvgW | AvgL | payoff | E (R) | PF | Sharpe | DD | $100 became |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in s.iterrows():
            lines.append(
                f"| {r['pair']} | `{r['rule_id']}` | {int(r['unseen_n_trades'])} | "
                f"{_f(r['unseen_win_rate'], 3)} | {_f(r['unseen_avg_win_r'], 2)} | "
                f"{_f(r['unseen_avg_loss_r'], 2)} | {_f(r['unseen_payoff'], 2)} | "
                f"{_f(r['unseen_expectancy_r'], 3)} | {_f(r['unseen_pf'], 2)} | "
                f"{_f(r['unseen_sharpe'], 2)} | {_f(r['unseen_max_dd'], 3)} | "
                f"{_f(r['unseen_end_usd'], 2)} |"
            )
        lines += ["", "## Trade distribution (unseen R)", ""]
        for _, r in s.iterrows():
            blob = (art.get(str(r["pair"])) or {}).get(str(r["rule_id"])) or {}
            rs = blob.get("r_list") or []
            if not rs:
                continue
            a = np.asarray(rs, dtype=float)
            lines += [
                f"### {r['pair']} `{r['rule_id']}`",
                "",
                f"- n={len(a)} mean={a.mean():.3f} median={np.median(a):.3f} "
                f"p10={np.quantile(a,0.1):.2f} p90={np.quantile(a,0.9):.2f}",
                f"- exits: {blob.get('reasons')}",
                "",
            ]

    lines += [
        "",
        "## Honest read",
        "",
        "- The 100k RSI/WillR/Stoch book is **invalid** under this protocol (no SL/TP, payoff ~1).",
        "- A survivor here must win in R, not by many tiny holds.",
        "- 3–9 trade unseen prints with WR 60–80% are **luck**, not an edge.",
        "- Closest real track: GBPJPY London ORB / ADX-Donchian, ~35% WR, ~3R target, E>0 on 2017-18; 2026 either too few trades or PF/Sharpe broke.",
        "- August 2026 is a partial month.",
        "",
    ]
    (RES / "R_BOOK.md").write_text("\n".join(lines), encoding="utf-8")

    if plt is None:
        print("wrote book (no matplotlib)")
        return
    fig_dir = RES / "r_equity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_rows = s if len(s) else df.sort_values("unseen_expectancy_r", ascending=False).head(6)
    for _, r in plot_rows.iterrows():
        blob = (art.get(str(r["pair"])) or {}).get(str(r["rule_id"])) or {}
        eq = blob.get("equity") or []
        rs = blob.get("r_list") or []
        if len(eq) < 2:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(eq, color="#1b6ca8")
        axes[0].set_title(f"{r['pair']} {r['rule_id']}  $100 -> {_f(r['unseen_end_usd'],2)}")
        axes[0].set_ylabel("equity $")
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(rs, bins=min(20, max(5, len(rs) // 2)), color="#444", alpha=0.85)
        axes[1].axvline(0, color="red", lw=1)
        axes[1].set_xlabel("trade R")
        axes[1].set_ylabel("count")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{r['pair']}_{r['rule_id']}.png", dpi=120)
        plt.close(fig)
    print("wrote", RES / "R_BOOK.md", "survivors", len(s))


if __name__ == "__main__":
    main()
