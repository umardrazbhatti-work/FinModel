#!/usr/bin/env python
"""Evaluate a saved checkpoint on a date range."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MultiTFDataset, multi_tf_collate
from src.evaluation import compute_economic_metrics, compute_statistical_metrics
from src.losses import MultiQuantilePinballLoss
from src.models import MTPTransformer
from src.training import MTPTrainer
from src.utils.config import load_config
from src.utils.io import load_checkpoint, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("mtp.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/eval_metrics.json")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = cfg["data"]
    ds = MultiTFDataset(
        pair=data["pair"],
        data_dir=str(ROOT / data["data_dir"]),
        tfs=data["tfs"],
        primary_tf=data["primary_tf"],
        lookback=data["lookback"],
        horizons=data["horizons"],
        quantiles=data["quantiles"],
        cost_threshold=data["cost_threshold"],
        feature_cols=data.get("feature_cols"),
        context_cols=data.get("context_cols"),
        mode="test",
        fold_start=args.start,
        fold_end=args.end,
        vol_window=data.get("vol_window", 24),
        target_clip=data.get("target_clip", 5.0),
    )
    # Fit standardization on the same range (evaluation-only convenience)
    ds.fit_standardization()
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["context_cols"] = list(ds.context_cols)

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=multi_tf_collate,
    )
    model = MTPTransformer(cfg)
    state = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    loss_fn = MultiQuantilePinballLoss(
        quantiles=cfg["data"]["quantiles"],
        entropy_weight=float(cfg["model"].get("gate_entropy_weight", 0.01)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = MTPTrainer(model, loss_fn, optimizer, device=device)
    out = trainer.evaluate(loader)
    quantiles = cfg["data"]["quantiles"]
    stats = compute_statistical_metrics(
        out["predictions"], out["targets"], out["masks"], quantiles
    )
    econ = compute_economic_metrics(
        out["predictions"],
        out["raw_returns"],
        out["masks"],
        cost=float(cfg["evaluation"]["cost_per_trade"]),
        signal_threshold=float(cfg["evaluation"]["signal_threshold"]),
        quantiles=quantiles,
    )
    result = {"stats": stats, "economic": econ, "n": len(ds)}
    save_json(result, ROOT / args.out)
    logger.info("Saved evaluation to %s", args.out)


if __name__ == "__main__":
    main()
