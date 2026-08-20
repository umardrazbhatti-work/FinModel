#!/usr/bin/env python
"""Write SURVIVOR_BOOK.md + a readable RULE_SCOREBOARD.md from the locked ledger."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.factory import results_dir


def _f(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if pd.isna(v):
        return "nan"
    return f"{v:.{nd}f}"


def main() -> None:
    res = results_dir()
    df = pd.read_csv(res / "RULE_LEDGER.csv")
    surv = df[df["survivor"] == True].copy()
    disc = df[df["discovery_pass"] == True].copy()

    # month packs for survivors
    month_bits = []
    for _, row in surv.iterrows():
        pack = Path(str(row["pack"]))
        mpath = pack / "03_unseen_months.csv"
        if not mpath.exists():
            continue
        m = pd.read_csv(mpath)
        sub = m[m["rule_id"] == row["rule_id"]]
        month_bits.append((row["rule_id"], sub))

    champ = df[df["rule_id"] == "h12_k2_logistic_ohlc"]
    champ_row = champ.iloc[0] if len(champ) else None

    n = len(df)
    n_ids = int(df["rule_id"].nunique())
    lines = [
        "# Survivor book — EURUSD 1h rule factory",
        "",
        "Locked after waves 0-7. Discovery = same 6 folds as S-1/S-2 (2017-08 to 2018-12).",
        "Unseen = last 6 calendar months in the file (2026-03 to 2026-08).",
        "Account = **$100** opening, additive notional (same wealth curve as the research stack).",
        "Cost = 1 pip one-way; 12h holds pay 2-way when in market.",
        "",
        f"- Ledger rows: **{n}**",
        f"- Unique rule ids: **{n_ids}**",
        f"- Discovery PASS: **{len(disc)}**",
        f"- Survivors: **{len(surv)}**",
        "",
        "## Official survivors",
        "",
    ]
    if surv.empty:
        lines.append("None.")
    else:
        for _, r in surv.iterrows():
            lines += [
                f"### `{r['rule_id']}`",
                "",
                f"- Family: **{r['family']}** | hold: **{int(r['hold'])}h**",
                f"- What it does: `{r['rule_id']}` (family {r['family']}). "
                f"Hold **{int(r['hold'])}h** non-overlapping.",
                f"- Discovery mean exp: **{_f(r['discovery_mean_exp'], 8)}** | folds green: "
                f"**{int(r['discovery_folds_pos'])}/{int(r['n_folds'])}** | mean Sharpe {_f(r['discovery_mean_sharpe'], 3)}",
                f"- Discovery $100 mean profit / fold: **${_f(r['discovery_mean_profit_usd'], 3)}**",
                f"- Unseen 6m: **$100 -> ${_f(r['unseen_end_usd'], 2)}** "
                f"(**{_f(r['unseen_profit_usd'], 2)} USD**, {_f(r['unseen_profit_pct'], 2)}%)",
                f"- Unseen months green: **{int(r['unseen_months_green'])}/{int(r['unseen_n_months'])}**",
                f"- Unseen mix long/short/flat: "
                f"{_f(r['unseen_pct_long'], 3)} / {_f(r['unseen_pct_short'], 3)} / {_f(r['unseen_pct_flat'], 3)}",
                f"- Unseen hit: {_f(r['unseen_hit'], 3)} | Sharpe {_f(r['unseen_sharpe'], 3)} | "
                f"max DD ${_f(r['unseen_max_dd_usd'], 2)}",
                f"- Active trades (6m): {int(r['unseen_n_active'])} of {int(r['unseen_n'])} slots",
                "",
            ]

    lines += ["## Month-by-month ($100 fresh each month)", ""]
    for rid, sub in month_bits:
        lines += [f"### {rid}", "", "| month | end USD | profit | exp | active | hit | green |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for _, m in sub.iterrows():
            lines.append(
                f"| {m['month']} | {_f(m['end_usd'], 2)} | {_f(m['profit_usd'], 2)} | "
                f"{_f(m['expectancy'], 6)} | {int(m['n_active'])} | {_f(m['hit_rate'], 3)} | "
                f"{'yes' if bool(m['pass_window']) else 'no'} |"
            )
        lines.append("")

    if champ_row is not None:
        lines += [
            "## S-2 champion on the same unseen 6 months (reference, not a survivor here)",
            "",
            f"- `{champ_row['rule_id']}` discovery PASS (matches S-2: exp {_f(champ_row['discovery_mean_exp'], 8)}, "
            f"{int(champ_row['discovery_folds_pos'])}/6 folds)",
            f"- Unseen: **$100 -> ${_f(champ_row['unseen_end_usd'], 2)}** "
            f"({_f(champ_row['unseen_profit_usd'], 2)} USD, {_f(champ_row['unseen_profit_pct'], 2)}%)",
            f"- Months green: {int(champ_row['unseen_months_green'])}/6 — **failed unseen gate**",
            "",
            "Both survivors beat this champion on the 6-month $100 account.",
            "",
        ]

    lines += [
        "## Discovery PASS that died on unseen (locked, not tradeable)",
        "",
    ]
    dead = disc[disc["survivor"] != True][["rule_id", "family", "discovery_mean_exp", "unseen_profit_usd", "unseen_months_green", "unseen_reason"]]
    if dead.empty:
        lines.append("None besides controls.")
    else:
        lines.append(dead.to_string(index=False))
        lines.append("")

    lines += [
        "## What this is not",
        "",
        "- Not 500 profitable rules. 530 were **tested**. 2 **survived**.",
        "- Rules that made more unseen dollars but failed discovery (Bollinger fade, SMA-fade, etc.) stay FAIL. That is the point of the two-stage gate.",
        "- The two survivors are the same family (RSI 30/70, periods 7 and 9). They are not a diversified book.",
        "- Edge is small (~$0.14 to $0.93 on $100 over 6 months). Do not size this live yet.",
        "- Handler (S-3) is still unattached.",
        "",
        "Packs: `Results/exp_rules_w00_*` ... `w07_*`. Ledger: `Results/RULE_LEDGER.csv`.",
        "",
    ]
    (res / "SURVIVOR_BOOK.md").write_text("\n".join(lines), encoding="utf-8")

    fam = (
        df.groupby("family")
        .agg(n=("rule_id", "count"), disc_pass=("discovery_pass", "sum"), survivors=("survivor", "sum"))
        .reset_index()
    )
    top = df.sort_values("unseen_profit_usd", ascending=False).head(10)
    bot = df.sort_values("unseen_profit_usd", ascending=True).head(10)
    sb = [
        "# Rule Factory scoreboard",
        "",
        "EURUSD 1h | $100 start | unseen = 2026-03 .. 2026-08 | cost 1 pip",
        "",
        f"- Locked rows: **{n}** | unique ids: **{n_ids}**",
        f"- Discovery PASS: **{len(disc)}**",
        f"- Survivors: **{len(surv)}** — `rsi7_30_70_h12` ($100 -> $100.93), `rsi9_30_70_h12` ($100 -> $100.14)",
        f"- S-2 champion unseen: $100 -> $97.48 (not a survivor on this window)",
        "",
        "## By family",
        "",
        fam.to_string(index=False),
        "",
        "## Survivors",
        "",
        surv[["rule_id", "unseen_end_usd", "unseen_profit_usd", "unseen_months_green", "discovery_mean_exp"]].to_string(index=False)
        if len(surv)
        else "none",
        "",
        "## Best 10 unseen $ P&L (many failed discovery — not tradeable)",
        "",
        top[["rule_id", "family", "unseen_profit_usd", "unseen_end_usd", "discovery_pass", "survivor"]].to_string(index=False),
        "",
        "## Worst 10 unseen $ P&L",
        "",
        bot[["rule_id", "family", "unseen_profit_usd", "unseen_end_usd", "discovery_pass"]].to_string(index=False),
        "",
    ]
    (res / "RULE_SCOREBOARD.md").write_text("\n".join(sb), encoding="utf-8")
    print("wrote", res / "SURVIVOR_BOOK.md")
    print("wrote", res / "RULE_SCOREBOARD.md")


if __name__ == "__main__":
    main()
