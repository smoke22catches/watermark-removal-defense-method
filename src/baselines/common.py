"""Shared helpers for baseline watermark wrappers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_unit(images: torch.Tensor) -> torch.Tensor:
    """Map ``[-1, 1]`` → ``[0, 1]``."""
    return (images.clamp(-1, 1) + 1.0) * 0.5


def to_signed(images: torch.Tensor) -> torch.Tensor:
    """Map ``[0, 1]`` → ``[-1, 1]``."""
    return images.clamp(0, 1) * 2.0 - 1.0


def resize_nchw(images: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    if images.shape[-1] == size and images.shape[-2] == size:
        return images
    return F.interpolate(
        images,
        size=(int(size), int(size)),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )


def residual_transfer_embed(
    images: torch.Tensor,
    bits: torch.Tensor,
    *,
    native_resolution: int,
    embed_native_fn,
) -> torch.Tensor:
    """Authors' residual-transfer recipe used by TrustMark / VINE / StegaStamp.

    1. Resize cover to the scheme's native resolution.
    2. Embed at native resolution.
    3. Upsample the residual ``x_w_native - x_native`` back to the evaluation size
       and add it to the original cover.
    """
    if images.ndim != 4:
        raise ValueError(f"expected NCHW images, got shape {tuple(images.shape)}")
    h, w = int(images.shape[-2]), int(images.shape[-1])
    native = int(native_resolution)
    x_n = resize_nchw(images, native)
    x_w_n = embed_native_fn(x_n, bits)
    residual = x_w_n - x_n
    if residual.shape[-2:] != (h, w):
        # Bicubic matches VINE / TrustMark author scripts more closely than bilinear.
        residual = F.interpolate(
            residual, size=(h, w), mode="bicubic", align_corners=False
        )
    return (images + residual).clamp(-1, 1)


def bits_to_string(bits: torch.Tensor) -> str:
    """Flatten a bit vector to a ``'01…'`` string (TrustMark binary mode)."""
    flat = bits.detach().float().view(-1).cpu().tolist()
    return "".join("1" if float(b) >= 0.5 else "0" for b in flat)


def string_to_bits(s: str, msg_len: int, device: torch.device) -> torch.Tensor:
    vals = [1.0 if ch == "1" else 0.0 for ch in s[:msg_len]]
    if len(vals) < msg_len:
        vals.extend([0.0] * (msg_len - len(vals)))
    return torch.tensor(vals, device=device, dtype=torch.float32)


@dataclass
class WeightProvenance:
    """Documented weight / code provenance for ``results_meta.json``."""

    scheme: str
    source_url: str
    license: str
    version: str
    weight_files: Dict[str, str] = field(default_factory=dict)  # name -> checksum
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "source_url": self.source_url,
            "license": self.license,
            "version": self.version,
            "weight_files": dict(self.weight_files),
            "notes": self.notes,
        }


class BaselineScheme:
    """Common interface for post-hoc image watermark baselines."""

    name: str = "baseline"
    msg_len: int = 0
    native_resolution: int = 256

    def embed(self, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits or soft bits of shape ``[N, msg_len]``."""
        raise NotImplementedError

    def provenance(self) -> WeightProvenance:
        raise NotImplementedError

    def close(self) -> None:
        return None


class BaselineEncoderAdapter(torch.nn.Module):
    """``encoder(x, m)`` shim so baselines share ``evaluate_protocol``."""

    def __init__(self, scheme: BaselineScheme) -> None:
        super().__init__()
        self.scheme = scheme

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.scheme.embed(x, m)


class BaselineDecoderAdapter(torch.nn.Module):
    """``decoder(x)`` shim returning logits / soft bits ``[N, k]``."""

    def __init__(self, scheme: BaselineScheme) -> None:
        super().__init__()
        self.scheme = scheme

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scheme.decode(x)


def sample_bits(
    batch_size: int,
    msg_len: int,
    *,
    device: torch.device | str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Seeded Bernoulli bits — identical procedure for all schemes.

    Bits are drawn on CPU (so a CPU ``Generator`` works) then moved to ``device``.
    """
    bits = torch.randint(
        0, 2, (int(batch_size), int(msg_len)), generator=generator
    ).float()
    return bits.to(device)
