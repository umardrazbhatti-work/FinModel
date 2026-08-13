"""Patch embedding with positional encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    """
    Convert a sequence [B, seq_len, n_features] into non-overlapping patches.

    Returns: [B, n_patches, d_model]
    """

    def __init__(
        self,
        n_features: int,
        patch_len: int,
        d_model: int,
        stride: int | None = None,
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride if stride is not None else patch_len
        self.d_model = d_model

        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, seq_len, n_features]
        b, seq_len, n_feat = x.shape
        if n_feat != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_feat}")

        # Ensure sequence length is compatible; right-truncate leftover bars
        if seq_len < self.patch_len:
            pad = self.patch_len - seq_len
            x = torch.nn.functional.pad(x, (0, 0, pad, 0))
            seq_len = x.shape[1]

        n_patches = 1 + (seq_len - self.patch_len) // self.stride
        patches = []
        for i in range(n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :].reshape(b, -1)
            patches.append(patch)
        patches_t = torch.stack(patches, dim=1)  # [B, n_patches, patch_len * n_feat]
        out = self.proj(patches_t)
        out = out + self.pos_embed[:, :n_patches, :]
        return out
