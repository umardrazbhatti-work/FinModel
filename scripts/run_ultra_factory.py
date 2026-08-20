#!/usr/bin/env python
"""Test the 100k catalog. New (ultra) rules on every pair. Ledger-skip already-tested ids."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.catalog import build_catalog, list_rules
from src.signals.factory import lock_wave, results_dir, run_wave, write_wave_pack
from src.signals.ledger import load_ledger, write_scoreboard
from src.utils.config import load_config
from src.utils.seed import set_seed

PAIRS = [
    ("EURUSD", 0.0001),
    ("GBPUSD", 0.0001),
    ("USDJPY", 0.00012),
    ("GBPJPY", 0.00015),
    ("XAUUSD", 0.00025),
]
CHUNK = 500
BASE_CFG = "configs/rules_wave00_protocol.yaml"


def _chunks(items, n):
    for i in range(0, len(items), n):
        yield i // n, items[i : i + n]


def main() -> None:
    t_all = time.time()
    print("Building 100k catalog...", flush=True)
    build_catalog()
    specs = list_rules()
    print(f"Catalog size: {len(specs)}", flush=True)
    proto = load_config(ROOT / BASE_CFG)
    led = load_ledger(results_dir())
    tested = set()
    max_wave = 199
    if not led.empty:
        pair_col = led["pair"] if "pair" in led.columns else "EURUSD"
        for pair_v, rid, h in zip(pair_col.fillna("EURUSD"), led["rule_id"], led["hold"]):
            tested.add((str(pair_v), str(rid), int(h)))
        if "wave" in led.columns:
            max_wave = int(led["wave"].max())

    wave0 = max(200, max_wave + 1)
    del led
    log_path = results_dir() / "ULTRA_PROGRESS.log"
    print(f"Resume wave0={wave0} already_tested={len(tested)}", flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start catalog={len(specs)} wave0={wave0} tested={len(tested)}\n")
    for pair, cost in PAIRS:
        todo = [s for s in specs if (pair, s.rule_id, int(s.hold)) not in tested]
        print(f"{pair}: {len(todo)} not yet locked (of {len(specs)})", flush=True)
        if not todo:
            continue
        for ci, chunk in _chunks(todo, CHUNK):
            wave = wave0 + ci
            cfg = {
                **proto,
                "project": {**proto["project"], "name": f"ultra-{pair}-w{wave}"},
                "data": {**proto["data"], "pair": pair, "cost": cost},
                "signal": {
                    **proto["signal"],
                    "wave": wave,
                    "families": [],
                    "rule_ids": [s.rule_id for s in chunk],
                    "start_usd": 100.0,
                    "unseen_months": 6,
                    "periods_per_year": 6048,
                },
                "output": {
                    "dir": "outputs",
                    "experiment_name": f"exp_ultra_{pair.lower()}_w{wave}",
                },
            }
            set_seed(42)
            t0 = time.time()
            result = run_wave(cfg)
            v = result["verdict"]
            pack_path = results_dir() / f"exp_ultra_{pair.lower()}_w{wave}"
            if v["survivor_count"]:
                pack_path = write_wave_pack(result, cfg)
            else:
                pack_path.mkdir(parents=True, exist_ok=True)
                result["summary"].to_csv(pack_path / "02_summary.csv", index=False)
            n = lock_wave(result, pack_path, scoreboard=False)
            msg = (
                f"{pair} w{wave} n={len(chunk)} disc={v['discovery_pass_count']} "
                f"surv={v['survivor_count']} new={n} {time.time()-t0:.1f}s"
            )
            print(msg, flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        wave0 += (len(todo) + CHUNK - 1) // CHUNK
        write_scoreboard(results_dir(), extra_lines=[f"Finished {pair} ultra."])
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Finished {pair} ultra.\n")
    done = f"ALL DONE in {(time.time()-t_all)/60:.1f} min"
    print(done, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {done}\n")


if __name__ == "__main__":
    main()
