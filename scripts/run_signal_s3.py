#!/usr/bin/env python
"""S-3 local replay: survivors x locked 1h RV handler. No broker."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals.s3 import run_s3, write_s3_pack
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.signal_s3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/signal_s3_eurusd_1h.yaml")
    args = parser.parse_args()
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg["project"]["seed"]))
    t0 = time.time()
    result = run_s3(cfg)
    pack = write_s3_pack(result, cfg)
    logger.info("S-3 wrote %s in %.1fs", pack, time.time() - t0)
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
