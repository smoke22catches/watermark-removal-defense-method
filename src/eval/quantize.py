"""8-bit quantization for evaluation (published images are always uint8)."""

from __future__ import annotations

import torch


@torch.no_grad()
def quantize_uint8(x: torch.Tensor) -> torch.Tensor:
    """Round-trip ``[-1, 1]`` float tensor through 8-bit storage.

    Maps to ``[0, 255]`` uint8 and back so evaluation matches a published image.
    """
    x01 = (x.detach().clamp(-1.0, 1.0) + 1.0) * 0.5
    u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)
    y01 = u8.to(dtype=x.dtype) / 255.0
    return (y01 * 2.0 - 1.0).clamp(-1.0, 1.0)
