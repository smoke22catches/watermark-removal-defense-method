"""Block-DCT + STE differentiable JPEG compression (watermark-paper style)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard JPEG luminance / chrominance quantization tables (quality=50 baseline).
_Y_QTABLE = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]
_C_QTABLE = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
]


def _quality_scale(quality: int) -> float:
    """IJG-style quality → quantization scale factor."""
    q = int(max(1, min(100, quality)))
    if q < 50:
        return 5000.0 / q
    return 200.0 - q * 2.0


def _scaled_qtable(base: torch.Tensor, quality: int) -> torch.Tensor:
    s = _quality_scale(quality)
    table = torch.floor((base * s + 50.0) / 100.0).clamp(1, 255)
    return table


def _dct_matrix(n: int = 8, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal type-II DCT matrix of size n×n."""
    m = torch.zeros(n, n, dtype=dtype)
    for i in range(n):
        for j in range(n):
            c = math.sqrt(1.0 / n) if i == 0 else math.sqrt(2.0 / n)
            m[i, j] = c * math.cos((math.pi * i * (2 * j + 1)) / (2 * n))
    return m


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for round()."""
    return x + (x.round() - x).detach()


class DiffJPEGReal(nn.Module):
    """Differentiable JPEG: RGB↔YCbCr, 8×8 DCT, STE quantization, optional 4:2:0."""

    def __init__(self, quality: int = 50, chroma_subsample: bool = True) -> None:
        super().__init__()
        self._quality = int(quality)
        self.chroma_subsample = chroma_subsample

        dct = _dct_matrix(8)
        self.register_buffer("dct_mat", dct)
        self.register_buffer("idct_mat", dct.t().contiguous())
        self.register_buffer("y_q_base", torch.tensor(_Y_QTABLE, dtype=torch.float32))
        self.register_buffer("c_q_base", torch.tensor(_C_QTABLE, dtype=torch.float32))
        self.register_buffer("y_q", _scaled_qtable(self.y_q_base, self._quality))
        self.register_buffer("c_q", _scaled_qtable(self.c_q_base, self._quality))

        # BT.601 RGB↔YCbCr (full-range style, channels in [0,1])
        rgb2ycbcr = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ],
            dtype=torch.float32,
        )
        ycbcr2rgb = torch.inverse(rgb2ycbcr)
        self.register_buffer("rgb2ycbcr", rgb2ycbcr)
        self.register_buffer("ycbcr2rgb", ycbcr2rgb)

    @property
    def quality(self) -> int:
        return self._quality

    @quality.setter
    def quality(self, value: int) -> None:
        self._quality = int(value)
        new_y = _scaled_qtable(self.y_q_base, self._quality)
        new_c = _scaled_qtable(self.c_q_base, self._quality)
        with torch.no_grad():
            self.y_q.copy_(new_y.to(device=self.y_q.device, dtype=self.y_q.dtype))
            self.c_q.copy_(new_c.to(device=self.c_q.device, dtype=self.c_q.dtype))

    def _color_transform(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W → apply C×C mix
        b, c, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        out = flat @ matrix.t()
        return out.view(b, h, w, c).permute(0, 3, 1, 2)

    def _pad_to_multiple(self, x: torch.Tensor, m: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)

    def _block_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 8×8 DCT on each spatial block. x: B,1,H,W with H,W % 8 == 0."""
        b, _, h, w = x.shape
        blocks = x.view(b, 1, h // 8, 8, w // 8, 8)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).reshape(-1, 8, 8)
        # DCT: D @ block @ D^T
        d = self.dct_mat.to(dtype=blocks.dtype)
        coeff = d @ blocks @ d.t()
        return coeff.view(b, 1, h // 8, w // 8, 8, 8)

    def _block_idct(self, coeff: torch.Tensor) -> torch.Tensor:
        """Inverse 8×8 DCT. coeff: B,1,nH,nW,8,8 → B,1,H,W."""
        b, _, nh, nw, _, _ = coeff.shape
        blocks = coeff.reshape(-1, 8, 8)
        d_t = self.idct_mat.to(dtype=blocks.dtype)
        spat = d_t @ blocks @ d_t.t()
        spat = spat.view(b, 1, nh, nw, 8, 8).permute(0, 1, 2, 4, 3, 5)
        return spat.reshape(b, 1, nh * 8, nw * 8)

    def _quantize(self, coeff: torch.Tensor, qtable: torch.Tensor) -> torch.Tensor:
        # coeff: B,1,nH,nW,8,8
        q = qtable.to(dtype=coeff.dtype, device=coeff.device).view(1, 1, 1, 1, 8, 8)
        scaled = coeff / q
        return ste_round(scaled) * q

    def _compress_channel(self, ch: torch.Tensor, qtable: torch.Tensor) -> torch.Tensor:
        # ch in roughly [-0.5, 1.5] after color convert; JPEG centers at 128 for [0,255]
        # Work in [0,1]-ish; shift to [-128,127]-like via *255 - 128 for DCT dynamic range
        level = ch * 255.0 - 128.0
        coeff = self._block_dct(level)
        coeff_q = self._quantize(coeff, qtable)
        rec = self._block_idct(coeff_q)
        return (rec + 128.0) / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B,3,H,W]`` in ``[-1, 1]``.
        Returns:
            Compressed image in ``[-1, 1]``, same spatial size as input.
        """
        orig_h, orig_w = x.shape[-2:]
        x01 = (x.clamp(-1, 1) + 1.0) * 0.5
        x01, _ = self._pad_to_multiple(x01, 8)

        ycbcr = self._color_transform(x01, self.rgb2ycbcr)
        # Cb/Cr are centered around 0 with offset; add 0.5 for chroma in [0,1]-like range
        y = ycbcr[:, 0:1]
        cb = ycbcr[:, 1:2] + 0.5
        cr = ycbcr[:, 2:3] + 0.5

        if self.chroma_subsample:
            _, _, h, w = cb.shape
            cb_ds = F.avg_pool2d(cb, 2, stride=2)
            cr_ds = F.avg_pool2d(cr, 2, stride=2)
            cb_ds, _ = self._pad_to_multiple(cb_ds, 8)
            cr_ds, _ = self._pad_to_multiple(cr_ds, 8)
            cb_hat = self._compress_channel(cb_ds, self.c_q)
            cr_hat = self._compress_channel(cr_ds, self.c_q)
            cb_hat = F.interpolate(cb_hat, size=(h, w), mode="bilinear", align_corners=False)
            cr_hat = F.interpolate(cr_hat, size=(h, w), mode="bilinear", align_corners=False)
        else:
            cb_hat = self._compress_channel(cb, self.c_q)
            cr_hat = self._compress_channel(cr, self.c_q)

        y_hat = self._compress_channel(y, self.y_q)
        ycbcr_hat = torch.cat([y_hat, cb_hat - 0.5, cr_hat - 0.5], dim=1)
        rgb01 = self._color_transform(ycbcr_hat, self.ycbcr2rgb).clamp(0, 1)
        rgb01 = rgb01[..., :orig_h, :orig_w]
        return (rgb01 * 2.0 - 1.0).clamp(-1, 1)
