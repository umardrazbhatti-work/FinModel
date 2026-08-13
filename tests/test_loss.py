"""Unit tests for MultiQuantilePinballLoss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import MultiQuantilePinballLoss


def test_perfect_prediction_near_zero():
    loss_fn = MultiQuantilePinballLoss(entropy_weight=0.0)
    b, h, q = 8, 3, 3
    target = torch.randn(b, h)
    pred = target.unsqueeze(-1).expand(b, h, q).contiguous()
    mask = torch.ones(b, h)
    predictions = {"1h": pred, "30m": pred, "4h": pred}
    targets = {"1h": target, "30m": target, "4h": target}
    masks = {"1h": mask, "30m": mask, "4h": mask}
    out = loss_fn(predictions, targets, masks, gate_weights=None)
    assert float(out["pinball_loss"]) < 1e-5


def test_biased_prediction_positive():
    loss_fn = MultiQuantilePinballLoss(entropy_weight=0.0)
    b, h, q = 8, 3, 3
    target = torch.ones(b, h)
    pred = torch.zeros(b, h, q)
    mask = torch.ones(b, h)
    predictions = {"1h": pred}
    targets = {"1h": target}
    masks = {"1h": mask}
    # only 1h in tradable — override
    loss_fn.tradable_tfs = ["1h"]
    out = loss_fn(predictions, targets, masks)
    assert float(out["pinball_loss"]) > 0


def test_mask_zeros_out():
    loss_fn = MultiQuantilePinballLoss(entropy_weight=0.0)
    loss_fn.tradable_tfs = ["1h"]
    b, h, q = 4, 2, 3
    target = torch.ones(b, h) * 5
    pred = torch.zeros(b, h, q)
    mask = torch.zeros(b, h)
    out = loss_fn({"1h": pred}, {"1h": target}, {"1h": mask})
    assert float(out["pinball_loss"]) == 0.0


def test_gate_entropy_uniform_vs_peaked():
    loss_fn = MultiQuantilePinballLoss(entropy_weight=1.0)
    uniform = torch.ones(6) / 6
    peaked = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # gate_entropy returns -H; uniform has higher H so more negative -H
    e_u = loss_fn.gate_entropy(uniform)
    e_p = loss_fn.gate_entropy(peaked)
    assert float(e_u) < float(e_p)  # -H_uniform < -H_peaked
