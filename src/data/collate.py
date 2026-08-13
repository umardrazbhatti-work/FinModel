"""Custom collate for MultiTFDataset batches."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def multi_tf_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack a list of dataset items into a batch dictionary.

    - inputs[tf]  -> [B, seq_len_tf, n_features]
    - targets[tf] -> [B, n_horizons]
    - target_mask[tf] -> [B, n_horizons]
    - raw_returns[tf] -> [B, n_horizons]
    - context -> [B, n_context]
    - timestamps and pairs kept as lists
    """
    if not batch:
        raise ValueError("Empty batch")

    tfs = list(batch[0]["inputs"].keys())
    tradable = list(batch[0]["targets"].keys())

    inputs = {
        tf: torch.stack([item["inputs"][tf] for item in batch], dim=0) for tf in tfs
    }
    targets = {
        tf: torch.stack([item["targets"][tf] for item in batch], dim=0)
        for tf in tradable
    }
    target_mask = {
        tf: torch.stack([item["target_mask"][tf] for item in batch], dim=0)
        for tf in tradable
    }
    raw_returns = {
        tf: torch.stack([item["raw_returns"][tf] for item in batch], dim=0)
        for tf in tradable
    }
    context = torch.stack([item["context"] for item in batch], dim=0)

    return {
        "inputs": inputs,
        "context": context,
        "targets": targets,
        "target_mask": target_mask,
        "raw_returns": raw_returns,
        "timestamp": [item["timestamp"] for item in batch],
        "pair": [item["pair"] for item in batch],
    }
