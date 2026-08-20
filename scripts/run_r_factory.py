#!/usr/bin/env python
"""Score R-multiple trend/breakout systems. Not the 100k oscillator grid."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.signals.factory import results_dir
from src.signals.r_entries import list_rspecs
from src.signals.r_eval import run_pair
from src.utils.config import load_config
from src.utils.seed import set_seed

PAIRS = [
    ("EURUSD", 0.0001, 0.00005),
    ("GBPUSD", 0.0001, 0.00005),
    ("USDJPY", 0.00012, 0.00006),
    ("GBPJPY", 0.00015, 0.00008),
    ("XAUUSD", 0.00025, 0.00012),
]


def main() -> None:
    t0 = time.time()
    cfg = load_config(ROOT / "configs/r_edge.yaml")
    specs = list_rspecs()
    print(f"R catalog: {len(specs)} systems", flush=True)
    set_seed(42)
    res = results_dir()
    res.mkdir(parents=True, exist_ok=True)
    frames = []
    artifacts = {}
    for pair, cost, slip in PAIRS:
        t1 = time.time()
        print(f"{pair} scoring...", flush=True)
        df = run_pair(pair, cost, slip, cfg, specs)
        keep_sim = df[df["survivor"] == True]
        if keep_sim.empty:
            keep_sim = df.sort_values("unseen_expectancy_r", ascending=False).head(3)
        pair_art = {}
        for _, row in keep_sim.iterrows():
            sim = row.get("_unseen_sim")
            if not isinstance(sim, dict):
                continue
            rid = str(row["rule_id"])
            eq = sim.get("equity")
            rlist = sim.get("r_list")
            trades = sim.get("trades") or []
            pair_art[rid] = {
                "equity": eq.tolist() if eq is not None else [],
                "r_list": rlist.tolist() if rlist is not None else [],
                "reasons": sim.get("reasons") or {},
                "trades": [
                    {
                        "side": t.side,
                        "entry_i": t.entry_i,
                        "exit_i": t.exit_i,
                        "r": t.r,
                        "reason": t.reason,
                    }
                    for t in trades
                ],
            }
        artifacts[pair] = pair_art
        slim = df.drop(columns=["_unseen_sim", "_month_rows"], errors="ignore")
        frames.append(slim)
        n_s = int((df["survivor"] == True).sum())
        print(
            f"{pair} n={len(df)} disc={int(df['discovery_pass'].sum())} "
            f"surv={n_s} {time.time()-t1:.1f}s",
            flush=True,
        )
    out = pd.concat(frames, ignore_index=True)
    path = res / "R_LEDGER.csv"
    out.to_csv(path, index=False)
    art_path = res / "R_ARTIFACTS.json"
    art_path.write_text(json.dumps(artifacts), encoding="utf-8")
    print(f"wrote {path} rows={len(out)} survivors={int(out['survivor'].sum())}", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
