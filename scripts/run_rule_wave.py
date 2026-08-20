#!/usr/bin/env python
"""Run one Module 1 rule-factory wave, lock the pack + ledger, move on."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.factory import lock_wave, run_wave, write_wave_pack
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.rule_wave")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg["project"]["seed"]))
    t0 = time.time()
    result = run_wave(cfg, max_folds=args.max_folds)
    pack = write_wave_pack(result, cfg)
    n_new = lock_wave(result, pack)
    v = result["verdict"]
    logger.info(
        "Wave %s locked | rules=%d disc_pass=%d survivors=%d new_ledger=%d | %s | %.1fs",
        v["wave"],
        v["n_rules"],
        v["discovery_pass_count"],
        v["survivor_count"],
        n_new,
        pack,
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
