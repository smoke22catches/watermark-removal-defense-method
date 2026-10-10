"""Real (non-differentiable) evaluation transforms — no training DistortionBank."""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.eval.quantize import quantize_uint8


def _to_pil_batch(x: torch.Tensor) -> list[Image.Image]:
    """``x`` in [-1,1], shape [B,3,H,W] → list of RGB PIL images."""
    x01 = ((x.detach().clamp(-1, 1) + 1) * 0.5).cpu()
    u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)
    out = []
    for i in range(u8.size(0)):
        arr = u8[i].permute(1, 2, 0).numpy()
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def _from_pil_batch(images: list[Image.Image], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensors = []
    for im in images:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
        tensors.append(t * 2.0 - 1.0)
    return torch.stack(tensors, dim=0).to(device=device, dtype=dtype)


@torch.no_grad()
def real_jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    """PIL JPEG encode/decode at ``quality`` (after implicit 8-bit quantization)."""
    q = int(max(1, min(100, quality)))
    device, dtype = x.device, x.dtype
    imgs = _to_pil_batch(x)
    out_imgs = []
    for im in imgs:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out_imgs.append(Image.open(buf).convert("RGB"))
    return _from_pil_batch(out_imgs, device, dtype)


def gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum().clamp_min(1e-8)
    return g[:, None] * g[None, :]


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(float(sigma), 1e-3)
    k = int(2 * round(3 * sigma) + 1)
    k = max(k, 3)
    kernel_2d = gaussian_kernel_2d(k, sigma, x.device, x.dtype)
    kernel = kernel_2d.expand(x.size(1), 1, k, k).contiguous()
    pad = k // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.size(1)).clamp(-1, 1)


@torch.no_grad()
def rotate(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    angle = float(angle_deg) * 3.141592653589793 / 180.0
    cos_a = torch.cos(torch.tensor(angle, device=x.device, dtype=x.dtype))
    sin_a = torch.sin(torch.tensor(angle, device=x.device, dtype=x.dtype))
    theta = torch.stack(
        [
            torch.stack([cos_a, -sin_a, torch.zeros((), device=x.device, dtype=x.dtype)]),
            torch.stack([sin_a, cos_a, torch.zeros((), device=x.device, dtype=x.dtype)]),
        ]
    ).unsqueeze(0).expand(x.size(0), -1, -1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border").clamp(-1, 1)


@torch.no_grad()
def center_crop(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    keep_ratio = float(min(max(keep_ratio, 0.1), 1.0))
    _, _, h, w = x.shape
    nh, nw = max(1, int(h * keep_ratio)), max(1, int(w * keep_ratio))
    top, left = (h - nh) // 2, (w - nw) // 2
    cropped = x[:, :, top : top + nh, left : left + nw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False).clamp(-1, 1)


@torch.no_grad()
def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return torch.clamp(x + torch.randn_like(x) * float(std), -1, 1)


@torch.no_grad()
def downsample_upsample(x: torch.Tensor, factor: float) -> torch.Tensor:
    f = max(int(round(float(factor))), 1)
    if f <= 1:
        return x
    return F.interpolate(F.avg_pool2d(x, f), size=x.shape[-2:], mode="bilinear", align_corners=False).clamp(-1, 1)


@torch.no_grad()
def brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    """WAVES-style brightness: scale around mid-grey in [-1,1]."""
    return (x * float(factor)).clamp(-1, 1)


@torch.no_grad()
def contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    return (mean + (x - mean) * float(factor)).clamp(-1, 1)


@torch.no_grad()
def erase(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Random rectangle erase filled with mid-grey (WAVES-style)."""
    frac = float(min(max(frac, 0.05), 0.9))
    y = x.clone()
    b, _, h, w = x.shape
    for i in range(b):
        eh, ew = max(1, int(h * frac**0.5)), max(1, int(w * frac**0.5))
        top = int(torch.randint(0, max(h - eh + 1, 1), (1,)).item())
        left = int(torch.randint(0, max(w - ew + 1, 1), (1,)).item())
        y[i, :, top : top + eh, left : left + ew] = 0.0
    return y.clamp(-1, 1)


def prepare_eval_input(x_w: torch.Tensor) -> torch.Tensor:
    """Always quantize to 8-bit before any evaluation transform."""
    return quantize_uint8(x_w)
