#!/usr/bin/env python
"""Module 2 — run the locked Trade Handler on the latest EURUSD 1h bar.

Does not trade. Prints a HandlerDecision (size / stand-aside only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MultiTFDataset, multi_tf_collate
from src.handler import VolatilityTradeHandler
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.handler")


def _data_dir(cfg: dict) -> Path:
    p = Path(cfg["data"]["data_dir"])
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/handler_eurusd_1h.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional ISO timestamp; default = last available 1h bar",
    )
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    set_seed(int(cfg["project"]["seed"]))
    ckpt = args.checkpoint or cfg.get("handler", {}).get("checkpoint")

    handler = VolatilityTradeHandler.from_config(
        cfg_path, checkpoint=ckpt, device=args.device
    )

    data = cfg["data"]
    ds = MultiTFDataset(
        pair=data["pair"],
        data_dir=str(_data_dir(cfg)),
        tfs=data["tfs"],
        primary_tf=data["primary_tf"],
        lookback=data["lookback"],
        horizons=data["horizons"],
        quantiles=data["quantiles"],
        feature_cols=data.get("feature_cols"),
        context_cols=[],
        target_type="realized_vol",
        tradable_tfs=data.get("tradable_tfs"),
        rv_log_transform=bool(data.get("rv_log_transform", True)),
        fold_end=args.end,
        vol_window=data.get("vol_window", 24),
    )
    if len(ds) == 0:
        raise SystemExit("No samples in dataset")
    ds.fit_standardization()

    # Last sample only
    last = len(ds) - 1
    item = ds[last]
    batch = multi_tf_collate([item])
    # Raw closes (not standardized features) for the inverse-vol reference
    tf = handler.primary_tf
    t = ds.timestamps[ds.primary_tf][ds.sample_indices[last]]
    ts = ds.timestamps[tf]
    end_idx = int(np.searchsorted(ts, t, side="right") - 1)
    lb = int(ds.lookback[tf])
    start_idx = max(0, end_idx - lb + 1)
    raw_closes = ds.closes[tf][start_idx : end_idx + 1]
    decision = handler.decide(batch, closes=raw_closes)
    payload = decision.to_dict()
    print(json.dumps(payload, indent=2, default=str))
    logger.info(
        "module=%s version=%s stand_aside=%s size=%.4f reason=%s side=%s",
        decision.module,
        decision.handler_version,
        decision.stand_aside,
        decision.size_multiplier,
        decision.reason,
        decision.side,
    )
    if decision.side is not None:
        raise RuntimeError("Handler must never set side")


if __name__ == "__main__":
    main()
