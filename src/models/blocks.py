"""Shared convolutional building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block used by Encoder/Decoder."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, padding=k // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
