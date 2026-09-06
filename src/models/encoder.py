"""HiDDeN-style watermark encoder E(x, m) -> x_w."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNReLU


class Encoder(nn.Module):
    """E(x, m) -> x_w  (HiDDeN-style)."""

    def __init__(self, msg_len: int = 64, ch: int = 64) -> None:
        super().__init__()
        self.msg_len = msg_len
        self.msg_fc = nn.Linear(msg_len, 32 * 32)  # проекція повідомлення у просторову карту
        self.pre = nn.Sequential(ConvBNReLU(3, ch), ConvBNReLU(ch, ch), ConvBNReLU(ch, ch))
        self.fuse = ConvBNReLU(ch + 1, ch)
        self.post = nn.Sequential(ConvBNReLU(ch, ch), nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        feat = self.pre(x)
        m_map = self.msg_fc(m).view(b, 1, 32, 32)
        m_map = F.interpolate(m_map, size=(h, w), mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([feat, m_map], dim=1))
        residual = self.post(fused)
        x_w = torch.clamp(x + residual, -1.0, 1.0)  # адитивне вбудовування, обмежене за LPIPS-бюджетом
        return x_w
