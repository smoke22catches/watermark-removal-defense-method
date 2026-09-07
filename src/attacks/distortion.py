"""Differentiable distortion bank (JPEG approx, noise, downsample-upsample)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffjpeg_real import DiffJPEGReal


class DiffJPEG(nn.Module):
    """Differentiable JPEG-style compression.

    * ``use_real_diffjpeg=False`` (default): simplified quantization-noise
      approximation from the original prototype (``start.py``).
    * ``use_real_diffjpeg=True``: full block-DCT + STE quantization
      (see :class:`DiffJPEGReal`).
    """

    def __init__(
        self,
        quality: int = 50,
        use_real_diffjpeg: bool = False,
        chroma_subsample: bool = True,
    ) -> None:
        super().__init__()
        self.use_real_diffjpeg = use_real_diffjpeg
        self._quality = int(quality)
        self._impl: nn.Module
        if use_real_diffjpeg:
            self._impl = DiffJPEGReal(quality=self._quality, chroma_subsample=chroma_subsample)
        else:
            self._impl = _DiffJPEGPlaceholder(quality=self._quality)

    @property
    def quality(self) -> int:
        return self._quality

    @quality.setter
    def quality(self, value: int) -> None:
        self._quality = int(value)
        if hasattr(self._impl, "quality"):
            self._impl.quality = self._quality  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


class _DiffJPEGPlaceholder(nn.Module):
    """Prototype noise approx (not a real JPEG codec)."""

    def __init__(self, quality: int = 50) -> None:
        super().__init__()
        self.quality = quality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # апроксимація: додаємо квантизаційний шум замість недиференційованого round()
        noise = (torch.rand_like(x) - 0.5) * (1.0 / max(self.quality, 1))
        return torch.clamp(x + noise, -1.0, 1.0)


class DistortionBank(nn.Module):
    """Random classical distortion: JPEG / Gaussian noise / downsample-upsample."""

    def __init__(
        self,
        jpeg_quality: int = 50,
        gaussian_std: float = 0.03,
        use_real_diffjpeg: bool = False,
        chroma_subsample: bool = True,
    ) -> None:
        super().__init__()
        self.jpeg = DiffJPEG(
            quality=jpeg_quality,
            use_real_diffjpeg=use_real_diffjpeg,
            chroma_subsample=chroma_subsample,
        )
        self.gaussian_std = gaussian_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        choice = torch.randint(0, 3, (1,)).item()
        if choice == 0:
            return self.jpeg(x)
        elif choice == 1:
            return torch.clamp(x + torch.randn_like(x) * self.gaussian_std, -1, 1)  # Gaussian noise
        else:
            return F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="bilinear")  # downsample-upsample
