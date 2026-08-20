"""Append-only rule ledger. A locked (wave, rule_id, hold) is never rewritten."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd

from src.utils.io import ensure_dir

LEDGER_NAME = "RULE_LEDGER.csv"
SCOREBOARD_NAME = "RULE_SCOREBOARD.md"
KEY_COLS = ("wave", "pair", "rule_id", "hold")


def ledger_path(results_dir: Path) -> Path:
    return Path(results_dir) / LEDGER_NAME


def scoreboard_path(results_dir: Path) -> Path:
    return Path(results_dir) / SCOREBOARD_NAME


def load_ledger(results_dir: Path) -> pd.DataFrame:
    path = ledger_path(results_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def locked_keys(df: pd.DataFrame) -> Set[Tuple[Any, ...]]:
    if df is None or df.empty:
        return set()
    wave = df["wave"].astype(int)
    rid = df["rule_id"].astype(str)
    hold = df["hold"].astype(int)
    if "pair" in df.columns:
        pair = df["pair"].fillna("EURUSD").astype(str)
    else:
        pair = pd.Series(["EURUSD"] * len(df), index=df.index)
    return set(zip(wave, pair, rid, hold))


def _existing_keys(path: Path) -> Set[Tuple[Any, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in KEY_COLS if c in header.columns]
    if "wave" not in cols or "rule_id" not in cols or "hold" not in cols:
        return locked_keys(pd.read_csv(path, low_memory=False))
    keys = pd.read_csv(path, usecols=cols, low_memory=False)
    return locked_keys(keys)


def append_rows(results_dir: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append new keys only. Returns number of rows written."""
    ensure_dir(results_dir)
    path = ledger_path(results_dir)
    have = _existing_keys(path)
    fresh: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        key = (
            int(row["wave"]),
            str(row.get("pair") or "EURUSD"),
            str(row["rule_id"]),
            int(row["hold"]),
        )
        if key in have:
            continue
        rec = dict(row)
        rec.setdefault("locked_at", now)
        fresh.append(rec)
        have.add(key)
    if not fresh:
        return 0
    add = pd.DataFrame(fresh)
    if not path.exists() or path.stat().st_size == 0:
        add.to_csv(path, index=False)
        return int(len(fresh))
    header = list(pd.read_csv(path, nrows=0).columns)
    for col in header:
        if col not in add.columns:
            add[col] = pd.NA
    add = add.reindex(columns=header)
    add.to_csv(path, mode="a", header=False, index=False)
    return int(len(fresh))


def write_scoreboard(results_dir: Path, extra_lines: List[str] | None = None) -> None:
    df = load_ledger(results_dir)
    lines = [
        "# Rule Factory scoreboard",
        "",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    if df.empty:
        lines.append("Ledger empty.")
        if extra_lines:
            lines.extend(["", *extra_lines])
        scoreboard_path(results_dir).write_text("\n".join(lines), encoding="utf-8")
        return

    n = len(df)
    n_pass = int((df.get("discovery_pass") == True).sum()) if "discovery_pass" in df.columns else 0
    n_surv = int((df.get("survivor") == True).sum()) if "survivor" in df.columns else 0
    lines += [
        f"- Locked rows: **{n}**",
        f"- Discovery PASS: **{n_pass}**",
        f"- Survivors (discovery + 6-month unseen): **{n_surv}**",
        "",
        "## By family",
        "",
    ]
    if "family" in df.columns:
        grp = (
            df.groupby("family")
            .agg(
                n=("rule_id", "count"),
                disc_pass=("discovery_pass", "sum") if "discovery_pass" in df.columns else ("rule_id", "count"),
                survivors=("survivor", "sum") if "survivor" in df.columns else ("rule_id", "count"),
            )
            .reset_index()
        )
        lines.append(grp.to_string(index=False))
        lines.append("")

    champ = df[df["rule_id"] == "h12_k2_logistic_ohlc"]
    if len(champ):
        r = champ.iloc[-1]
        lines += [
            "## S-2 champion on this ledger",
            "",
            f"- discovery_pass={r.get('discovery_pass')} mean_exp={r.get('discovery_mean_exp')}",
            f"- unseen 6m start_usd={r.get('unseen_start_usd')} end_usd={r.get('unseen_end_usd')} "
            f"profit_usd={r.get('unseen_profit_usd')} profit_pct={r.get('unseen_profit_pct')}",
            f"- months_green={r.get('unseen_months_green')} survivor={r.get('survivor')}",
            "",
        ]

    if "unseen_profit_usd" in df.columns:
        ranked = df.sort_values("unseen_profit_usd", ascending=False)
        lines += ["## Best 10 by unseen $ P&L (from $100)", "", ranked.head(10).to_string(index=False), "",
                  "## Worst 10 by unseen $ P&L (from $100)", "", ranked.tail(10).to_string(index=False), ""]

    if extra_lines:
        lines.extend(["", *extra_lines])
    scoreboard_path(results_dir).write_text("\n".join(lines), encoding="utf-8")
