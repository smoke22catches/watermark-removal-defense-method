"""HiDDeN-style watermark encoder E(x, m) -> x_w."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNReLU


class Encoder(nn.Module):
    """E(x, m) -> x_w.

    Message bits are expanded to full-resolution spatial channels and fused with
    image features. A direct 1x1 linear path from bits → RGB residual provides a
    strong gradient signal; a nonlinear branch refines cover-dependent hiding.
    """

    def __init__(self, msg_len: int = 64, ch: int = 64, strength: float = 0.4) -> None:
        super().__init__()
        self.msg_len = msg_len
        self.strength = float(strength)
        self.msg_conv = nn.Conv2d(msg_len, 3, kernel_size=1, bias=False)
        nn.init.normal_(self.msg_conv.weight, std=0.1)
        self.pre = nn.Sequential(ConvBNReLU(3, ch), ConvBNReLU(ch, ch))
        self.fuse = nn.Sequential(
            ConvBNReLU(ch + msg_len, ch),
            ConvBNReLU(ch, ch),
            nn.Conv2d(ch, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m_map = m.view(b, self.msg_len, 1, 1).expand(b, self.msg_len, h, w)
        residual = torch.tanh(self.msg_conv(m_map) + self.fuse(torch.cat([self.pre(x), m_map], dim=1)))
        return torch.clamp(x + self.strength * residual, -1.0, 1.0)
