"""Tests for optimizer / scheduler factory."""

from __future__ import annotations

import torch.nn as nn

from src.training.optim import build_optimizer_and_scheduler


def test_cosine_scheduler():
    model = nn.Linear(4, 2)
    opt, sched = build_optimizer_and_scheduler(
        model,
        {"lr": 1e-4, "weight_decay": 5e-4, "lr_scheduler": "cosine", "max_epochs": 60},
        max_epochs=60,
    )
    assert opt.param_groups[0]["lr"] == 1e-4
    assert sched is not None
    assert sched.__class__.__name__ == "CosineAnnealingLR"


def test_plateau_scheduler():
    model = nn.Linear(4, 2)
    opt, sched = build_optimizer_and_scheduler(
        model,
        {
            "lr": 1e-4,
            "weight_decay": 5e-4,
            "lr_scheduler": "plateau",
            "plateau_patience": 3,
        },
    )
    assert opt is not None
    assert sched.__class__.__name__ == "ReduceLROnPlateau"


def test_none_scheduler():
    model = nn.Linear(4, 2)
    opt, sched = build_optimizer_and_scheduler(
        model, {"lr": 1e-4, "weight_decay": 5e-4, "lr_scheduler": "none"}
    )
    assert opt is not None
    assert sched is None
