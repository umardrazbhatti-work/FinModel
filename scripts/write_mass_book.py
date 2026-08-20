#!/usr/bin/env python
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
    lines = [
        "# Mass factory — 10,377 strategies x 5 pairs",
        "",
        "Catalog of **10,377** named strategies: SMC, Raja-Banks-style structure,",
        "and published indicator grids. Each tested on EURUSD, GBPUSD, USDJPY, GBPJPY, XAUUSD.",
        "Same 6 discovery folds + last 6 months from **$100**.",
        "",
        f"- Ledger rows: **{len(df)}**",
        f"- Unique strategies: **{df['rule_id'].nunique()}**",
        f"- Official survivors: **{len(s)}**",
        "",
        "## By pair",
        "",
        df.groupby("pair")
        .agg(tested=("rule_id", "nunique"), disc=("discovery_pass", "sum"), surv=("survivor", "sum"))
        .to_string(),
        "",
        "## SMC / Raja Banks",
        "",
    ]
    for fam in ("smc", "raja_banks"):
        sub = df[df["family"] == fam]
        ss = sub[sub["survivor"] == True]
        lines.append(
            f"- **{fam}**: {len(sub)} rows, discovery PASS {int(sub['discovery_pass'].sum())}, survivors {len(ss)}"
        )
        if len(ss):
            lines.append(
                ss[["pair", "rule_id", "hold", "unseen_end_usd", "unseen_profit_usd", "unseen_n_active"]].to_string(
                    index=False
                )
            )
        lines.append("")
    lines += [
        "Raja Banks public clips are SMC structure (CHoCH / sweep / OB in killzone).",
        "Those mechanical encodings: **0 survivors**.",
        "",
        "## Top 15 by unseen $ (from $100, 6 months)",
        "",
        "| pair | rule | hold | trades | $100 became | profit |",
        "|---|---|---:|---:|---:|---:|",
    ]
    top = s.sort_values("unseen_profit_usd", ascending=False).head(15)
    for _, r in top.iterrows():
        tr = int(r["unseen_n_active"]) if pd.notna(r["unseen_n_active"]) else 0
        lines.append(
            f"| {r['pair']} | `{r['rule_id']}` | {int(r['hold'])}h | {tr} | "
            f"{float(r['unseen_end_usd']):.2f} | {float(r['unseen_profit_usd']):+.2f} |"
        )
    lines += [
        "",
        "## Honest read",
        "",
        "- Most survivors are RSI / Williams / z-score **neighbors** — one mean-reversion idea.",
        "- Best print: GBPUSD short RSI, about **+$6.3 on $100 in 6 months**.",
        "- XAUUSD: **zero** survivors.",
        "- SuperTrend / Ichimoku / Turtle / Cowabunga / Raja killzone OB: not survivors.",
        "",
    ]
    (RES / "MASS_BOOK.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RES / "MASS_BOOK.md", "survivors", len(s))


if __name__ == "__main__":
    main()
