"""Differentiable geometric distortions via ``affine_grid`` + ``grid_sample``."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_batch_float(
    value: Union[float, torch.Tensor],
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch)
        if t.numel() != batch:
            raise ValueError(f"expected {batch} values, got {t.numel()}")
        return t
    return torch.full((batch,), float(value), device=device, dtype=dtype)


def affine_sample(
    x: torch.Tensor,
    theta: torch.Tensor,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Apply per-sample 2×3 affine ``theta`` with ``align_corners=False``."""
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    y = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    return y.clamp(-1.0, 1.0)


def identity_theta(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch, -1, -1).contiguous()


class GeometricRotate(nn.Module):
    """Rotate by a per-sample angle in ``[-max_angle, max_angle]`` (degrees)."""

    def __init__(self, max_angle: float = 15.0, padding_mode: str = "border") -> None:
        super().__init__()
        self.max_angle = float(max_angle)
        self.padding_mode = padding_mode

    def sample_angles(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * self.max_angle

    def forward(
        self,
        x: torch.Tensor,
        angles_deg: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        b = x.size(0)
        if angles_deg is None:
            angles = self.sample_angles(b, x.device, x.dtype)
        else:
            angles = _as_batch_float(angles_deg, b, x.device, x.dtype)
        rad = angles * (torch.pi / 180.0)
        cos = torch.cos(rad)
        sin = torch.sin(rad)
        zeros = torch.zeros_like(cos)
        theta = torch.stack(
            [
                torch.stack([cos, -sin, zeros], dim=-1),
                torch.stack([sin, cos, zeros], dim=-1),
            ],
            dim=1,
        )
        return affine_sample(x, theta, padding_mode=self.padding_mode)


class GeometricTranslate(nn.Module):
    """Translate by up to ``max_translate`` fraction of each side (per sample)."""

    def __init__(self, max_translate: float = 0.05, padding_mode: str = "border") -> None:
        super().__init__()
        self.max_translate = float(max_translate)
        self.padding_mode = padding_mode

    def sample_shifts(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # [B, 2] fractions of width / height.
        return (torch.rand(batch, 2, device=device, dtype=dtype) * 2 - 1) * self.max_translate

    def forward(
        self,
        x: torch.Tensor,
        shifts: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        b = x.size(0)
        if shifts is None:
            sh = self.sample_shifts(b, x.device, x.dtype)
        else:
            sh = torch.as_tensor(shifts, device=x.device, dtype=x.dtype)
            if sh.ndim == 1:
                sh = sh.view(1, 2).expand(b, -1)
            if sh.shape != (b, 2):
                raise ValueError(f"shifts must be [B,2], got {tuple(sh.shape)}")
        # Image spans [-1, 1]; fraction ``t`` of the side → ``2t`` in normalized coords.
        tx = 2.0 * sh[:, 0]
        ty = 2.0 * sh[:, 1]
        ones = torch.ones(b, device=x.device, dtype=x.dtype)
        zeros = torch.zeros(b, device=x.device, dtype=x.dtype)
        theta = torch.stack(
            [
                torch.stack([ones, zeros, tx], dim=-1),
                torch.stack([zeros, ones, ty], dim=-1),
            ],
            dim=1,
        )
        return affine_sample(x, theta, padding_mode=self.padding_mode)


class GeometricRescale(nn.Module):
    """Zoom in/out with ``scale`` in ``[scale_min, scale_max]`` and border padding."""

    def __init__(
        self,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.padding_mode = padding_mode

    def sample_scales(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = torch.rand(batch, device=device, dtype=dtype)
        return self.scale_min + (self.scale_max - self.scale_min) * u

    def forward(
        self,
        x: torch.Tensor,
        scales: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        b = x.size(0)
        if scales is None:
            s = self.sample_scales(b, x.device, x.dtype)
        else:
            s = _as_batch_float(scales, b, x.device, x.dtype)
        # affine_grid maps output→input; zoom factor s → diagonal 1/s.
        inv = 1.0 / s.clamp(min=1e-6)
        zeros = torch.zeros(b, device=x.device, dtype=x.dtype)
        theta = torch.stack(
            [
                torch.stack([inv, zeros, zeros], dim=-1),
                torch.stack([zeros, inv, zeros], dim=-1),
            ],
            dim=1,
        )
        return affine_sample(x, theta, padding_mode=self.padding_mode)


class GeometricCrop(nn.Module):
    """Random crop of area fraction in ``[area_min, area_max]``, resized back to HxW."""

    def __init__(
        self,
        area_min: float = 0.7,
        area_max: float = 1.0,
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        self.area_min = float(area_min)
        self.area_max = float(area_max)
        self.padding_mode = padding_mode

    def sample_areas(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = torch.rand(batch, device=device, dtype=dtype)
        return self.area_min + (self.area_max - self.area_min) * u

    def forward(
        self,
        x: torch.Tensor,
        area_frac: Optional[Union[float, torch.Tensor]] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Crop+resize via affine zoom into a random window (``align_corners=False``).

        ``offsets`` (optional) is ``[B, 2]`` in ``[0, 1]`` choosing the crop origin
        within the feasible range (0 = top/left-most, 1 = bottom/right-most).
        """
        b = x.size(0)
        if area_frac is None:
            areas = self.sample_areas(b, x.device, x.dtype)
        else:
            areas = _as_batch_float(area_frac, b, x.device, x.dtype)
        # Relative side length of the crop window (area = side^2).
        side = areas.clamp(min=1e-6, max=1.0).sqrt()

        if offsets is None:
            offs = torch.rand(b, 2, device=x.device, dtype=x.dtype)
        else:
            offs = offsets.to(device=x.device, dtype=x.dtype)
            if offs.shape != (b, 2):
                raise ValueError(f"offsets must be [B,2], got {tuple(offs.shape)}")

        # x_in = side * x_out + center; |center| <= 1 - side keeps the window inside.
        slack = (1.0 - side).clamp(min=0.0)
        center_x = -slack + offs[:, 0] * (2.0 * slack)
        center_y = -slack + offs[:, 1] * (2.0 * slack)
        zeros = torch.zeros(b, device=x.device, dtype=x.dtype)
        theta = torch.stack(
            [
                torch.stack([side, zeros, center_x], dim=-1),
                torch.stack([zeros, side, center_y], dim=-1),
            ],
            dim=1,
        )
        return affine_sample(x, theta, padding_mode=self.padding_mode)
