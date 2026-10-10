"""Differentiable distortion bank (JPEG, noise, downsample, geometric)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffjpeg_real import DiffJPEGReal
from .geometric import (
    GeometricCrop,
    GeometricRescale,
    GeometricRotate,
    GeometricTranslate,
)

CLASSICAL_KEYS = ("jpeg", "noise", "downsample")
GEOMETRIC_KEYS = ("rotate", "crop", "rescale", "translate")
ALL_KEYS = CLASSICAL_KEYS + GEOMETRIC_KEYS


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
        noise = (torch.rand_like(x) - 0.5) * (1.0 / max(self.quality, 1))
        return torch.clamp(x + noise, -1.0, 1.0)


def _normalize_weights(weights: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    raw = {k: max(float(weights.get(k, 0.0)), 0.0) for k in keys}
    total = sum(raw.values())
    if total <= 0.0:
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    return {k: v / total for k, v in raw.items()}


def resolve_distortion_probs(
    weights: Optional[Mapping[str, float]] = None,
    geometric_share: float = 0.5,
) -> Dict[str, float]:
    """Return a probability for each transform; geometric keys sum to ``geometric_share``."""
    weights = dict(weights or {})
    g_share = min(max(float(geometric_share), 0.0), 1.0)
    c_share = 1.0 - g_share
    c_norm = _normalize_weights(weights, CLASSICAL_KEYS)
    g_norm = _normalize_weights(weights, GEOMETRIC_KEYS)
    probs = {k: c_norm[k] * c_share for k in CLASSICAL_KEYS}
    probs.update({k: g_norm[k] * g_share for k in GEOMETRIC_KEYS})
    return probs


class DistortionBank(nn.Module):
    """Random classical + geometric distortion bank (differentiable)."""

    def __init__(
        self,
        jpeg_quality: int = 50,
        gaussian_std: float = 0.03,
        use_real_diffjpeg: bool = False,
        chroma_subsample: bool = True,
        geometric_share: float = 0.5,
        max_angle: float = 15.0,
        crop_area: Tuple[float, float] = (0.7, 1.0),
        scale: Tuple[float, float] = (0.8, 1.2),
        translate: float = 0.05,
        padding_mode: str = "border",
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.jpeg = DiffJPEG(
            quality=jpeg_quality,
            use_real_diffjpeg=use_real_diffjpeg,
            chroma_subsample=chroma_subsample,
        )
        self.gaussian_std = float(gaussian_std)
        self.padding_mode = str(padding_mode)
        self.geometric_share = float(geometric_share)
        self.rotate = GeometricRotate(max_angle=max_angle, padding_mode=self.padding_mode)
        self.crop = GeometricCrop(
            area_min=float(crop_area[0]),
            area_max=float(crop_area[1]),
            padding_mode=self.padding_mode,
        )
        self.rescale = GeometricRescale(
            scale_min=float(scale[0]),
            scale_max=float(scale[1]),
            padding_mode=self.padding_mode,
        )
        self.translate = GeometricTranslate(
            max_translate=float(translate),
            padding_mode=self.padding_mode,
        )
        self.probs = resolve_distortion_probs(weights, geometric_share=self.geometric_share)
        self._names = list(ALL_KEYS)
        self._prob_tensor = torch.tensor(
            [self.probs[k] for k in self._names], dtype=torch.double
        )

    @classmethod
    def from_config(
        cls,
        dist_cfg: Mapping[str, Any],
        diffjpeg_cfg: Optional[Mapping[str, Any]] = None,
    ) -> "DistortionBank":
        dj = diffjpeg_cfg or {}
        crop = dist_cfg.get("crop_area", [0.7, 1.0])
        scale = dist_cfg.get("scale", [0.8, 1.2])
        return cls(
            jpeg_quality=int(dj.get("quality", dist_cfg.get("jpeg_quality", 50))),
            gaussian_std=float(dist_cfg.get("gaussian_std", 0.03)),
            use_real_diffjpeg=bool(dj.get("use_real_diffjpeg", False)),
            chroma_subsample=bool(dj.get("chroma_subsample", True)),
            geometric_share=float(dist_cfg.get("geometric_share", 0.5)),
            max_angle=float(dist_cfg.get("max_angle", 15.0)),
            crop_area=(float(crop[0]), float(crop[1])),
            scale=(float(scale[0]), float(scale[1])),
            translate=float(dist_cfg.get("translate", 0.05)),
            padding_mode=str(dist_cfg.get("padding_mode", "border")),
            weights=dist_cfg.get("weights"),
        )

    def sample_transform_name(self) -> str:
        idx = int(torch.multinomial(self._prob_tensor, num_samples=1).item())
        return self._names[idx]

    def apply_named(self, name: str, x: torch.Tensor, **params: Any) -> torch.Tensor:
        if name == "jpeg":
            return self.jpeg(x)
        if name == "noise":
            return torch.clamp(x + torch.randn_like(x) * self.gaussian_std, -1.0, 1.0)
        if name == "downsample":
            return F.interpolate(
                F.avg_pool2d(x, 2), size=x.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(-1.0, 1.0)
        if name == "rotate":
            return self.rotate(x, angles_deg=params.get("angles_deg"))
        if name == "crop":
            return self.crop(
                x, area_frac=params.get("area_frac"), offsets=params.get("offsets")
            )
        if name == "rescale":
            return self.rescale(x, scales=params.get("scales"))
        if name == "translate":
            return self.translate(x, shifts=params.get("shifts"))
        raise ValueError(f"Unknown distortion transform: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        name = self.sample_transform_name()
        return self.apply_named(name, x)
