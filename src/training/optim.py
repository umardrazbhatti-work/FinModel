"""Optimizer + LR scheduler factory from training config."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def build_optimizer_and_scheduler(
    model: nn.Module,
    training_cfg: Dict[str, Any],
    max_epochs: Optional[int] = None,
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """
    Build AdamW + optional LR schedule.

    Config keys (training section):
      lr, weight_decay
      lr_scheduler: "cosine" | "plateau" | "none"  (default cosine)
      cosine_t_max: optional; defaults to max_epochs / training.max_epochs
      cosine_eta_min: floor LR for cosine (default 1e-6)
      plateau_factor, plateau_patience: ReduceLROnPlateau knobs
    """
    tr = training_cfg
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr["lr"]),
        weight_decay=float(tr["weight_decay"]),
    )

    name = str(tr.get("lr_scheduler", "cosine")).lower().strip()
    if name in ("none", "off", "null", ""):
        return optimizer, None

    if name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(tr.get("plateau_factor", 0.5)),
            patience=int(tr.get("plateau_patience", 3)),
        )
        return optimizer, scheduler

    if name in ("cosine", "cosine_annealing", "cosineannealinglr"):
        t_max = tr.get("cosine_t_max")
        if t_max is None:
            t_max = max_epochs if max_epochs is not None else tr.get("max_epochs", 60)
        t_max = max(int(t_max), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=float(tr.get("cosine_eta_min", 1e-6)),
        )
        return optimizer, scheduler

    raise ValueError(
        f"Unknown lr_scheduler={name!r}. Use 'cosine', 'plateau', or 'none'."
    )
