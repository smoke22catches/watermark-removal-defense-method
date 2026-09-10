"""Watermark decoder D(x_w') -> m_hat (logits)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNReLU


class Decoder(nn.Module):
    """D(x_w') -> m_hat logits.

    Progressive downsampling builds receptive field before global pooling.
    """

    def __init__(self, msg_len: int = 64, ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(3, ch),
            ConvBNReLU(ch, ch, s=2),
            ConvBNReLU(ch, ch),
            ConvBNReLU(ch, ch, s=2),
            ConvBNReLU(ch, ch),
            ConvBNReLU(ch, ch, s=2),
            ConvBNReLU(ch, ch),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, msg_len),
        )

    def forward(self, x_w_prime: torch.Tensor) -> torch.Tensor:
        return self.net(x_w_prime)
