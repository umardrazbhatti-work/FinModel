#!/usr/bin/env python
"""Full rule-system story + every official winner. Run after the 100k factory locks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "Results"


def _i(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _f(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _stem_family(rule_id: str) -> str:
    rid = str(rule_id)
    for pref in ("u2_", "u3_", "u_", "g_"):
        if rid.startswith(pref):
            rid = rid[len(pref) :]
            break
    if rid.startswith("rsi") or "_rsi" in rid:
        return "RSI fade"
    if rid.startswith("wr") or rid.startswith("willr") or "_wr" in rid:
        return "Williams %R"
    if "stoch" in rid:
        return "Stochastic"
    if rid.startswith("z") or "_z" in rid or "zscore" in rid:
        return "z-score fade"
    if "bb" in rid or "smafade" in rid:
        return "band / SMA fade"
    if "smc" in rid or "pd_" in rid:
        return "SMC / premium-discount"
    if "dsma" in rid or "sma" in rid or "ema" in rid:
        return "MA / dual-MA"
    return "other"


def winner_row(r) -> str:
    return (
        f"| {r['pair']} | `{r['rule_id']}` | {r.get('family', '')} | {_i(r['hold'])}h | "
        f"{_i(r.get('unseen_n_active'))} | {_i(r.get('unseen_n'))} | "
        f"{_f(r['unseen_end_usd']):.2f} | {_f(r['unseen_profit_usd']):+.2f} | "
        f"{_f(r.get('unseen_profit_pct')):+.2f}% | "
        f"{_i(r.get('unseen_months_green'))}/{_i(r.get('unseen_n_months'), 6)} | "
        f"{_i(r.get('discovery_folds_pos'))}/{_i(r.get('n_folds'), 6)} | "
        f"{_f(r.get('unseen_sharpe')):.2f} | {_f(r.get('unseen_hit')):.3f} | "
        f"{_f(r.get('unseen_max_dd_usd')):.2f} |"
    )


def write_all_winners(df: pd.DataFrame, s: pd.DataFrame) -> Path:
    lines = [
        "# All official winners — 100k factory",
        "",
        "Gate: discovery 6 folds (mean exp>0, majority green, beat always-long/coin-flip,",
        "not a flat trick) **and** last 6 calendar months from **$100** (profit>0, >=4 green months).",
        "",
        f"- Written: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Ledger rows: **{len(df):,}**",
        f"- Unique strategies in ledger: **{df['rule_id'].nunique():,}**",
        f"- Unique pair x strategy tests: **{df.groupby(['pair','rule_id','hold']).ngroups:,}**",
        f"- Official winners: **{len(s)}**",
        "",
        "## Winners by pair",
        "",
        df.groupby("pair")
        .agg(
            rows=("rule_id", "count"),
            strategies=("rule_id", "nunique"),
            disc_pass=("discovery_pass", "sum"),
            winners=("survivor", "sum"),
        )
        .to_string(),
        "",
        "## Every winner (best $ first)",
        "",
        "| pair | rule | family | hold | trades (6m) | slots | $100 became | profit $ | profit % | months green | disc folds | Sharpe | hit | max DD $ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in s.iterrows():
        lines.append(winner_row(r))
    lines += [
        "",
        "## Family mix among winners",
        "",
        s.groupby("family").size().sort_values(ascending=False).to_string() if len(s) else "none",
        "",
        "## Idea mix among winners (neighbors collapsed)",
        "",
        s.assign(idea=s["rule_id"].map(_stem_family))
        .groupby("idea")
        .size()
        .sort_values(ascending=False)
        .to_string()
        if len(s)
        else "none",
        "",
        "## Read this honestly",
        "",
        "- Nearby RSI / Williams / Stochastic periods are **one idea**, not separate edges.",
        "- `g_*` (mass) and `u_*` (ultra) with the same params are the **same trade**.",
        "- Trades = non-overlapping holds that were not flat in Mar-Aug 2026.",
        "- August is a partial month (data ends mid-month).",
        "",
    ]
    out = RES / "ALL_WINNERS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_story(df: pd.DataFrame, s: pd.DataFrame) -> Path:
    by_pair = (
        df.groupby("pair")
        .agg(
            rows=("rule_id", "count"),
            strategies=("rule_id", "nunique"),
            disc_pass=("discovery_pass", "sum"),
            winners=("survivor", "sum"),
        )
        .sort_index()
    )
    by_fam = (
        df.groupby("family")
        .agg(n=("rule_id", "count"), disc=("discovery_pass", "sum"), surv=("survivor", "sum"))
        .sort_values("n", ascending=False)
    )
    top = s.head(20)
    idea = (
        s.assign(idea=s["rule_id"].map(_stem_family))
        .groupby("idea")
        .agg(n=("rule_id", "count"), best=("$100", "max") if False else ("unseen_end_usd", "max"))
        .sort_values("n", ascending=False)
    )

    best = s.iloc[0] if len(s) else None
    gbp = s[s["pair"] == "GBPUSD"]
    eur = s[s["pair"] == "EURUSD"]
    jpy = s[s["pair"] == "USDJPY"]
    gj = s[s["pair"] == "GBPJPY"]
    xau = s[s["pair"] == "XAUUSD"]

    def pair_best(sub: pd.DataFrame) -> str:
        if sub.empty:
            return "none"
        r = sub.iloc[0]
        return (
            f"`{r['rule_id']}` {_i(r['hold'])}h, {_i(r.get('unseen_n_active'))} trades, "
            f"$100 -> ${_f(r['unseen_end_usd']):.2f} ({_f(r['unseen_profit_usd']):+.2f})"
        )

    n_unique_keys = int(df.groupby(["pair", "rule_id", "hold"]).ngroups)
    n_rules = int(df["rule_id"].nunique())

    lines = [
        "# Rule-based Signal system — full story through the 100k factory",
        "",
        f"Written: **{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}**",
        "Module: **1 — Signal / Alpha**. Pair clock for all factory tests: **1h bars**.",
        "",
        "This is the complete path from the first costed rules to the 100,000-strategy catalog",
        "tested on EURUSD, GBPUSD, USDJPY, GBPJPY, and XAUUSD.",
        "",
        "---",
        "",
        "## 0. What we were looking for",
        "",
        "Super goal: a self-sufficient automated trader. Module 2 (Trade Handler) was already",
        "locked as EURUSD 1h realized-vol + inverse-vol sizing. It **never** picks direction.",
        "Module 1 has to produce a Signal with **positive expectancy after costs**.",
        "",
        "Every later factory row uses the **same official gate**:",
        "",
        "1. **Discovery** — same 6 expanding walk-forward folds as S-1/S-2 (2017-08 to 2018-12).",
        "   Mean expectancy > 0, majority of folds green, beat always-long and coin-flip,",
        "   not a 'always flat' trick.",
        "2. **Unseen** — last **6 calendar months** in the file (**2026-03 to 2026-08**).",
        "   Open **$100**. Profit > 0 and **at least 4 of 6 months** green.",
        "3. Cost: 1 pip one-way on majors; 1.2 pip USDJPY; 1.5 pip GBPJPY; 2.5 pip XAUUSD.",
        "   Holds are **non-overlapping**. A 12h rule can take at most one trade every 12 hours.",
        "",
        "A row that only wins discovery, or only wins the last 6 months, is **not** a winner.",
        "",
        "---",
        "",
        "## 1. S-1 — first costed rules (EURUSD) — FAIL",
        "",
        "Date: 2026-08-17. Pack: `Results/exp_signal_s1_eurusd_1h - 17-08-26 1615Hrs/`.",
        "",
        "12 explicit next-bar rules (sessions, time-of-day, always-long, coin-flip, always-flat)",
        "on the same 6 folds. Cost 1 pip one-way.",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Official S-1 gate | **FAIL** |",
        "| Winning rules | none |",
        "| Best non-control | `tod_train_hours` exp **−5.6e-6** (0/6 folds > 0) |",
        "| always_long | −1.24e-5 (1/6) |",
        "| coin_flip | −1.07e-4 (0/6) |",
        "| always_flat | 0 (best of the set) |",
        "",
        "Decision: these are not Signals. Do not attach the handler. Next = longer-horizon labels (S-2).",
        "",
        "---",
        "",
        "## 2. S-2 — 4h / 12h / 24h direction — PASS (first Signal)",
        "",
        "Date: 2026-08-17. Pack: `Results/exp_signal_s2_eurusd_1h - 17-08-26 1623Hrs/`.",
        "",
        "Horizons {4, 12, 24} × k {0,1,2,3} × persist / logistic-OHLC / logistic+events.",
        "Same 6 folds, 2-way 1-pip cost, non-overlapping holds.",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Official S-2 gate | **PASS** |",
        "| Locked Signal | **`h12_k2_logistic_ohlc`** |",
        "| Mean exp / folds | **+8.87e-5** / **5/6** |",
        "| 12h always-long / coin-flip | −3.1e-4 / −4.5e-4 |",
        "| 4h any model | FAIL |",
        "| Persist any H | FAIL |",
        "| Events at 12h | FAIL (hurt) |",
        "| Oracle 12h | +21.6e-4 (ceiling only — not a Signal) |",
        "",
        "What it does: hold **12h**. Flat if |12h return| < 2 pips. Else a train-only logistic",
        "on lagged OHLC. Small edge, short-heavy on 2017–18. First real Signal.",
        "",
        "**Later, on the 2026-03..08 $100 window, this champion lost:** $100 → **$97.48** (3/6 months).",
        "It stays the 2017-18 lock. It is **not** a 2026 survivor.",
        "",
        "---",
        "",
        "## 3. Rule factory waves 0–7 — 530 named rules, EURUSD",
        "",
        "Same gate, $100 unseen. Ledger started here.",
        "",
        "| Wave | Family | Rules | Disc PASS | Survivors |",
        "|---|---|---:|---:|---:|",
        "| 0 | protocol | 8 | 1 | 0 |",
        "| 1 | time | 126 | 6 | 0 |",
        "| 2 | trend | 72 | 0 | 0 |",
        "| 3 | breakout | 54 | 0 | 0 |",
        "| 4 | mean-reversion | 75 | 4 | **2** |",
        "| 5 | momentum | 43 | 0 | 0 |",
        "| 6 | H=1 control | 10 | 0 | 0 |",
        "| 7 | vol-filter | 163 | 3 | 0 |",
        "",
        "First survivors:",
        "",
        "- `rsi7_30_70_h12` — $100 → **$100.93** (+0.93%), 4/6 months, 113 trades / 235 slots, 5/6 discovery",
        "- `rsi9_30_70_h12` — $100 → **$100.14** (+0.14%), 4/6 months, 95 trades / 235 slots, 4/6 discovery",
        "",
        "Trend, breakout, momentum, and H=1 were dead. Time-of-day and session rules that passed",
        "discovery died on 2026 (many were flat tricks).",
        "",
        "---",
        "",
        "## 4. Waves 8–12 — published retail / pro recipes",
        "",
        "Cowabunga, SuperTrend, Ichimoku, Turtle 20/55, ADX+DI, Stochastic, Williams %R, CCI,",
        "Keltner, Alligator, AO, Connors RSI2, MACD zero-cross, EMA 34/55 pullback, Heikin Ashi,",
        "Aroon, Asian box / London breakout, daily pivots, round numbers, 07:00 UTC follow,",
        "engulfing / pin / inside-bar / NR4/NR7, Hull, TTM squeeze, Vortex, Camarilla,",
        "200SMA+RSI, SuperTrend+RSI, MACDaddy, NY ORB. Wave 12 re-scored prior PASSes at 24h.",
        "",
        "- Catalog then: **609** unique ids",
        "- Famous names (Cowabunga, SuperTrend, Ichimoku, Turtle, Asian box): **not survivors**",
        "- New survivor: **`willr_14_70_30_h12` at 24h** — $100 → **$101.40**, 84 trades / 117 slots, 4/6 months",
        "",
        "Book was still one mean-reversion family (RSI + Williams).",
        "",
        "---",
        "",
        "## 5. S-3 — locked handler sizes the 3 EURUSD survivors",
        "",
        "Pack: `Results/exp_signal_s3_eurusd_1h - 17-08-26 2328Hrs/`.",
        "Handler = EURUSD 1h RV, fold-matched; unseen uses fold_5. **Never sets side.**",
        "It did **not** stand aside on this window (width never hit 1.5). It only scaled size.",
        "",
        "| Rule | slots | trades | stood aside | raw $ | sized $ | Handler? |",
        "|---|---:|---:|---:|---:|---:|---|",
        "| rsi7 12h | 235 | 113 | 0 | 100.93 | 100.51 | hurt |",
        "| rsi9 12h | 235 | 95 | 0 | 100.14 | 99.99 | hurt |",
        "| **willr 24h** | **117** | **84** | **0** | **101.40** | **103.56** | **helped** |",
        "",
        "Best EURUSD combo still on the books: **Williams %R 14, 24h hold, inverse-vol size**,",
        "84 trades / 6 months, $100 → **$103.56**.",
        "",
        "---",
        "",
        "## 6. Mass factory — 10,377 strategies × 5 pairs",
        "",
        "SMC (FVG, OB, sweep, BOS, CHoCH, OTE, PD, killzones, Silver Bullet), Raja-Banks-style",
        "structure, plus dense published grids (SMA/EMA/RSI/Stoch/Donchian/SuperTrend/…).",
        "Book: `Results/MASS_BOOK.md`. **52,744** locked rows at lock time. **87** official survivors.",
        "",
        "| Pair | Tested | Disc PASS | Survivors |",
        "|---|---:|---:|---:|",
        "| EURUSD | 10,377 | 139 | 17 |",
        "| GBPUSD | 10,377 | 438 | **66** |",
        "| USDJPY | 10,377 | 32 | 2 |",
        "| GBPJPY | 10,377 | 692 | 2 |",
        "| XAUUSD | 10,377 | 3 | **0** |",
        "",
        "- Best then: GBPUSD `g_rsi9_40_60_h16` **$100 → $106.30**, 124 trades / 16h",
        "- SMC: **1** survivor (`smc_pd_24_any_h24` on GBPUSD, $105.39, 117 trades)",
        "- Raja Banks encodings: **0**",
        "- XAUUSD: **0** at these costs",
        "",
        "Most of the 87 were RSI / Williams neighbors — one idea, not 87 systems.",
        "",
        "---",
        "",
        "## 7. Ultra / 100,000 catalog × 5 pairs — THIS LEVEL",
        "",
        "Catalog size locked at **100,000** named strategies (`u_*` / `u2_*` / `u3_*` grids:",
        "indicators × holds × sessions × hours × months, plus extra RSI bands).",
        "Already-tested mass ids were skipped. Same $100 / 6-month / 6-fold gate.",
        "",
        f"- Ledger rows now: **{len(df):,}**",
        f"- Unique rule ids: **{n_rules:,}**",
        f"- Unique pair × rule × hold tests: **{n_unique_keys:,}**",
        f"- Discovery PASS rows: **{int(df['discovery_pass'].sum()):,}**",
        f"- Official winners: **{len(s)}**",
        "",
        "### Coverage by pair",
        "",
        by_pair.to_string(),
        "",
        "### Coverage by family",
        "",
        by_fam.to_string(),
        "",
        "### Winners by pair (this lock)",
        "",
        f"- **GBPUSD**: {len(gbp)} winners — best {pair_best(gbp)}",
        f"- **EURUSD**: {len(eur)} winners — best {pair_best(eur)}",
        f"- **USDJPY**: {len(jpy)} winners — best {pair_best(jpy)}",
        f"- **GBPJPY**: {len(gj)} winners — best {pair_best(gj)}",
        f"- **XAUUSD**: {len(xau)} winners — best {pair_best(xau)}",
        "",
        "### Idea mix among official winners",
        "",
        idea.to_string() if len(s) else "none",
        "",
        "---",
        "",
        "## 8. Top 20 official winners (all pairs, best $ first)",
        "",
        "| pair | rule | family | hold | trades | slots | $100 became | profit $ | months | disc | Sharpe | hit | max DD $ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['pair']} | `{r['rule_id']}` | {r.get('family','')} | {_i(r['hold'])}h | "
            f"{_i(r.get('unseen_n_active'))} | {_i(r.get('unseen_n'))} | "
            f"{_f(r['unseen_end_usd']):.2f} | {_f(r['unseen_profit_usd']):+.2f} | "
            f"{_i(r.get('unseen_months_green'))}/{_i(r.get('unseen_n_months'), 6)} | "
            f"{_i(r.get('discovery_folds_pos'))}/{_i(r.get('n_folds'), 6)} | "
            f"{_f(r.get('unseen_sharpe')):.2f} | {_f(r.get('unseen_hit')):.3f} | "
            f"{_f(r.get('unseen_max_dd_usd')):.2f} |"
        )

    if best is not None:
        lines += [
            "",
            f"**Single best print:** {best['pair']} `{best['rule_id']}` "
            f"hold {_i(best['hold'])}h — {_i(best.get('unseen_n_active'))} trades in 6 months, "
            f"$100 → ${_f(best['unseen_end_usd']):.2f} ({_f(best['unseen_profit_usd']):+.2f}).",
        ]

    lines += [
        "",
        "Full table of every winner: `Results/ALL_WINNERS.md`.",
        "",
        "---",
        "",
        "## 9. What actually survived — honest read",
        "",
        "- The live book is still **short-horizon mean reversion** (RSI / Williams / Stochastic /",
        "  a few z-score and band fades), mostly on **GBPUSD**, holds **10–20h**.",
        "- Ultra added names. Many `u_*` winners are the **same trade** as an earlier `g_*` row",
        "  (same period, same bands, same hold, same P&L).",
        "- **SMC** contributed one GBPUSD premium-discount rule. Mechanical FVG / OB / sweep /",
        "  killzone / Raja Banks: essentially zero.",
        "- **Trend following** (SMA/EMA cross, SuperTrend, Ichimoku, Turtle, Alligator): not a survivor.",
        "- **Breakout / price-action / session box / Cowabunga**: not a survivor on this gate.",
        "- **Gold (XAUUSD)** at 2.5 pip cost: still the hardest pair. Do not promote a gold system",
        "  from this factory unless the winner table shows one.",
        "- Best 6-month print is about **+$6 to +$7 on $100**. That is real on the gate, and small.",
        "  It is not a diversified multi-strategy fund.",
        "- S-2 logistic was the first Signal on 2017–18 and **lost money** on 2026-03..08.",
        "  Confirmation on the unseen window is not optional.",
        "",
        "---",
        "",
        "## 10. Where the system stands after this lock",
        "",
        "| Item | Status |",
        "|---|---|",
        "| Module 2 Trade Handler | **LOCKED** — EURUSD 1h RV |",
        "| Module 1 first Signal (2017-18) | `h12_k2_logistic_ohlc` — later failed 2026 unseen |",
        "| Best EURUSD sized combo | WillR 14 @ 24h × handler — $100 → $103.56, 84 trades |",
        "| Best raw factory print | see Top 20 above (almost certainly GBPUSD oscillator) |",
        "| Module 3 Execution | Not started |",
        "| Module 4 Portfolio | Not started |",
        "| Module 5 Monitoring | Not started |",
        "",
        "Do **not** treat 100+ official rows as 100 independent edges. Cluster first.",
        "Do **not** attach the handler to every survivor — S-3 already showed it hurts the RSI pair.",
        "",
    ]
    out = RES / "RULE_SYSTEM_SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    df = pd.read_csv(RES / "RULE_LEDGER.csv", low_memory=False)
    if "pair" not in df.columns:
        df["pair"] = "EURUSD"
    df["pair"] = df["pair"].fillna("EURUSD")
    s = df[df["survivor"] == True].copy()
    s = s.sort_values(["unseen_profit_usd"], ascending=False)
    w = write_all_winners(df, s)
    story = write_story(df, s)
    print("wrote", w, "winners", len(s))
    print("wrote", story, "rows", len(df))


if __name__ == "__main__":
    main()
