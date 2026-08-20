#!/usr/bin/env python
"""Test the full catalog on every aligned pair. Chunked waves. No prompts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.catalog import build_catalog, list_rules
from src.signals.factory import lock_wave, run_wave, write_wave_pack
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.mass")

PAIRS = [
    ("EURUSD", 0.0001),
    ("GBPUSD", 0.0001),
    ("USDJPY", 0.00012),
    ("GBPJPY", 0.00015),
    ("XAUUSD", 0.00025),
]

CHUNK = 400
BASE_CFG = "configs/rules_wave00_protocol.yaml"


def _chunks(items, n):
    for i in range(0, len(items), n):
        yield i // n, items[i : i + n]


def main() -> None:
    build_catalog()
    specs = list_rules()
    print(f"Catalog size: {len(specs)}", flush=True)
    fam = {}
    for s in specs:
        fam[s.family] = fam.get(s.family, 0) + 1
    print("By family:", fam, flush=True)
    if len(specs) < 10000:
        print(f"WARNING: catalog {len(specs)} < 10000", flush=True)

    proto = load_config(ROOT / BASE_CFG)
    wave0 = 20
    t_all = time.time()
    for pair, cost in PAIRS:
        ids = [s.rule_id for s in specs]
        for ci, chunk in _chunks(ids, CHUNK):
            wave = wave0 + ci
            cfg = {
                **proto,
                "project": {**proto["project"], "name": f"mass-{pair}-w{wave}"},
                "data": {**proto["data"], "pair": pair, "cost": cost},
                "signal": {
                    **proto["signal"],
                    "wave": wave,
                    "families": [],
                    "rule_ids": chunk,
                    "start_usd": 100.0,
                    "unseen_months": 6,
                    "periods_per_year": 6048,
                },
                "output": {
                    "dir": "outputs",
                    "experiment_name": f"exp_mass_{pair.lower()}_w{wave:02d}",
                },
            }
            set_seed(42)
            print(f"===== {pair} wave {wave} rules {len(chunk)} =====", flush=True)
            t0 = time.time()
            result = run_wave(cfg)
            pack = write_wave_pack(result, cfg)
            n = lock_wave(result, pack)
            v = result["verdict"]
            print(
                f"{pair} w{wave} disc_pass={v['discovery_pass_count']} "
                f"surv={v['survivor_count']} new={n} {time.time()-t0:.1f}s -> {pack.name}",
                flush=True,
            )
        wave0 += (len(ids) + CHUNK - 1) // CHUNK
    print(f"ALL PAIRS DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
