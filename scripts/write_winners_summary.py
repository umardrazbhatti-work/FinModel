#!/usr/bin/env python
"""Overall winners summary: pair, trades, P&L, hold, family."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "Results"


def main() -> None:
    df = pd.read_csv(RES / "RULE_LEDGER.csv", low_memory=False)
    if "pair" not in df.columns:
        df["pair"] = "EURUSD"
    df["pair"] = df["pair"].fillna("EURUSD")
    s = df[df["survivor"] == True].copy()
    s = s.sort_values(["unseen_profit_usd"], ascending=False)

    lines = [
        "# All official winners — full factory",
        "",
        "Gate: discovery 6 folds (mean exp>0, majority green, beat always-long/coin-flip,",
        "not a flat trick) **and** last 6 calendar months from **$100** (profit>0, >=4 green months).",
        "",
        f"- Ledger rows: **{len(df):,}**",
        f"- Unique strategies in ledger: **{df['rule_id'].nunique():,}**",
        f"- Official winners: **{len(s)}**",
        "",
        "## Winners by pair",
        "",
        df.groupby("pair")
        .agg(rows=("rule_id", "count"), strategies=("rule_id", "nunique"), disc_pass=("discovery_pass", "sum"), winners=("survivor", "sum"))
        .to_string(),
        "",
        "## Every winner (best $ first)",
        "",
        "| pair | rule | family | hold | trades (6m) | slots | $100 became | profit $ | profit % | months green | disc folds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in s.iterrows():
        trades = r.get("unseen_n_active", float("nan"))
        slots = r.get("unseen_n", float("nan"))
        try:
            trades_i = int(trades)
        except Exception:
            trades_i = 0
        try:
            slots_i = int(slots)
        except Exception:
            slots_i = 0
        lines.append(
            f"| {r['pair']} | `{r['rule_id']}` | {r.get('family','')} | {int(r['hold'])}h | "
            f"{trades_i} | {slots_i} | {float(r['unseen_end_usd']):.2f} | "
            f"{float(r['unseen_profit_usd']):+.2f} | {float(r.get('unseen_profit_pct', 0) or 0):+.2f}% | "
            f"{int(r.get('unseen_months_green', 0) or 0)}/{int(r.get('unseen_n_months', 6) or 6)} | "
            f"{int(r.get('discovery_folds_pos', 0) or 0)}/{int(r.get('n_folds', 6) or 6)} |"
        )

    lines += [
        "",
        "## Family mix among winners",
        "",
        s.groupby("family").size().sort_values(ascending=False).to_string() if len(s) else "none",
        "",
        "## Read this honestly",
        "",
        "- Nearby RSI/Williams periods are **one idea**, not separate edges.",
        "- Trades = non-overlapping holds that were not flat in Mar-Aug 2026.",
        "- August is a partial month (data ends mid-month).",
        "",
    ]
    out = RES / "ALL_WINNERS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out, "winners", len(s))


if __name__ == "__main__":
    main()
