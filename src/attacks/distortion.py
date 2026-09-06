"""Differentiable distortion bank (JPEG approx, noise, downsample-upsample)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffJPEG(nn.Module):
    """Диференційоване наближення JPEG-стиснення (straight-through DCT quantization).

    TODO: When ``use_real_diffjpeg`` is enabled in config, replace this simplified
    quantization-noise approximation with a full block-DCT + STE DiffJPEG
    implementation. The public API (quality, forward) should stay the same.
    """

    def __init__(self, quality: int = 50, use_real_diffjpeg: bool = False) -> None:
        super().__init__()
        self.quality = quality
        self.use_real_diffjpeg = use_real_diffjpeg
        if use_real_diffjpeg:
            # TODO: plug in real DiffJPEG (block DCT + STE quantization) here.
            raise NotImplementedError(
                "Real DiffJPEG is not yet wired; set diffjpeg.use_real_diffjpeg=false "
                "to use the placeholder approximation from start.py."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # апроксимація: додаємо квантизаційний шум замість недиференційованого round()
        # (у реальній імплементації використовується блокова DCT + STE, тут спрощено для ілюстрації)
        noise = (torch.rand_like(x) - 0.5) * (1.0 / max(self.quality, 1))
        return torch.clamp(x + noise, -1.0, 1.0)


class DistortionBank(nn.Module):
    """Random classical distortion: JPEG / Gaussian noise / downsample-upsample."""

    def __init__(
        self,
        jpeg_quality: int = 50,
        gaussian_std: float = 0.03,
        use_real_diffjpeg: bool = False,
    ) -> None:
        super().__init__()
        self.jpeg = DiffJPEG(quality=jpeg_quality, use_real_diffjpeg=use_real_diffjpeg)
        self.gaussian_std = gaussian_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        choice = torch.randint(0, 3, (1,)).item()
        if choice == 0:
            return self.jpeg(x)
        elif choice == 1:
            return torch.clamp(x + torch.randn_like(x) * self.gaussian_std, -1, 1)  # Gaussian noise
        else:
            return F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="bilinear")  # downsample-upsample
