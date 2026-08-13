"""Multi-quantile pinball loss with gate entropy regularization."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class MultiQuantilePinballLoss(nn.Module):
    """
    Multi-horizon, multi-TF, multi-quantile pinball loss
    with optional gate entropy regularization.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        tradable_tfs: Optional[List[str]] = None,
        entropy_weight: float = 0.01,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        quantiles = quantiles or [0.1, 0.5, 0.9]
        self.tradable_tfs = list(tradable_tfs or ["30m", "1h", "4h"])
        self.entropy_weight = float(entropy_weight)
        self.reduction = reduction
        self.register_buffer(
            "q_levels", torch.tensor(quantiles, dtype=torch.float32), persistent=True
        )

    def pinball(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Masked pinball loss for one TF.

        pred:   [B, n_horizons, n_quantiles]
        target: [B, n_horizons]
        mask:   [B, n_horizons]
        """
        # errors: target - pred  → [B, H, Q]
        t = target.unsqueeze(-1)
        errors = t - pred
        q = self.q_levels.view(1, 1, -1).to(pred.device)
        loss = torch.maximum(q * errors, (q - 1.0) * errors)
        m = mask.unsqueeze(-1)
        loss = loss * m
        valid = m.sum()
        if valid.item() == 0:
            return pred.new_tensor(0.0)
        if self.reduction == "sum":
            return loss.sum()
        # mean over valid (sample, horizon, quantile)
        return loss.sum() / (valid * pred.shape[-1]).clamp_min(1.0)

    def gate_entropy(self, gate_weights: Tensor) -> Tensor:
        """
        Entropy of gate distribution. Higher when uniform.
        Returned as negative entropy for minimization: -H(g).
        Loss term: entropy_weight * (-H) encourages higher entropy when weight>0.
        """
        g = gate_weights
        if g.dim() == 2:
            # [B, n_tfs] → mean over batch
            g = g.mean(dim=0)
        g = g.clamp_min(1e-8)
        # H = -sum g log g  → we return -H so adding lambda*(-H) maximizes entropy
        # Spec: L_entropy = -λ sum g log g  which equals λ * H when written as L_pinball + L_entropy
        # Wait: L_entropy = -λ sum g_i log(g_i) = λ * H. Maximizing H means this term is larger when uniform.
        # Actually for regularization to prevent collapse, we want to MAXIMIZE entropy, so we MINIMIZE -H.
        # Spec says: L_total = L_pinball + λ_ent * L_entropy where L_entropy = -sum g log g = H.
        # That would maximize pinball + H which encourages higher entropy only if we subtract... 
        # Re-read: L_entropy = -λ sum g_i log(g_i + eps). Since sum g log g is negative, -sum g log g is positive H.
        # L_total = L_pinball + λ * L_entropy with L_entropy = -sum g log g means adding positive entropy.
        # Minimizing L_total would then DECREASE entropy (collapse). That's wrong for anti-collapse.
        #
        # Common practice: L = pinball - λ * H = pinball + λ * sum g log g
        # Spec formula: L_entropy = -λ sum g log g  and L_total = pinball + λ_ent * L_entropy
        # If L_entropy already includes λ: L_total = pinball + L_entropy = pinball - λ sum g log g = pinball + λ H
        # Minimizing pinball + λH encourages large H? No - minimizing H means collapse.
        #
        # Actually: H = -sum g log g >= 0. Minimizing pinball + λH reduces H → collapse. Wrong.
        # Correct anti-collapse: minimize pinball - λH = pinball + λ sum g log g.
        #
        # I'll follow standard practice (anti-collapse): loss += entropy_weight * sum(g * log g)
        # which equals -entropy_weight * H.
        entropy = -torch.sum(g * torch.log(g))
        # Return value used as: total = pinball + entropy_weight * returned
        # We return -H so total = pinball - λH
        return -entropy

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        target_masks: Dict[str, Tensor],
        gate_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        per_tf: Dict[str, Tensor] = {}
        losses = []
        for tf in self.tradable_tfs:
            if tf not in predictions:
                continue
            l_tf = self.pinball(predictions[tf], targets[tf], target_masks[tf])
            per_tf[tf] = l_tf
            losses.append(l_tf)

        if losses:
            pinball_loss = torch.stack(losses).mean()
        else:
            # fallback zero
            any_pred = next(iter(predictions.values()))
            pinball_loss = any_pred.new_tensor(0.0)

        if gate_weights is not None and self.entropy_weight > 0:
            entropy_loss = self.gate_entropy(gate_weights)
        else:
            entropy_loss = pinball_loss.new_tensor(0.0)

        total = pinball_loss + self.entropy_weight * entropy_loss
        return {
            "loss": total,
            "pinball_loss": pinball_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "per_tf_loss": {k: v.detach() for k, v in per_tf.items()},
        }
