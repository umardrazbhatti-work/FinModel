#!/usr/bin/env python
"""Run every declared wave in order. No prompts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WAVES = [
    "configs/rules_wave00_protocol.yaml",
    "configs/rules_wave01_time.yaml",
    "configs/rules_wave02_trend.yaml",
    "configs/rules_wave03_breakout.yaml",
    "configs/rules_wave04_mr.yaml",
    "configs/rules_wave05_momentum.yaml",
    "configs/rules_wave06_h1.yaml",
    "configs/rules_wave07_vol.yaml",
]


def main() -> None:
    for cfg in WAVES:
        print(f"===== {cfg} =====", flush=True)
        rc = subprocess.call(
            [sys.executable, str(ROOT / "scripts" / "run_rule_wave.py"), "--config", cfg],
            cwd=str(ROOT),
        )
        if rc != 0:
            raise SystemExit(f"wave failed rc={rc} config={cfg}")
    print("All waves complete.", flush=True)


if __name__ == "__main__":
    main()
