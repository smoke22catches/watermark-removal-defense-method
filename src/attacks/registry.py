"""Training-time surrogates + re-exports of the evaluation registry.

Evaluation transforms live in ``src.eval`` and must never import this module's
former DiffJPEG path. Training helpers (blur surrogate / identity regen) stay here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Re-export evaluation registry for backward-compatible imports (viz / infer).
from src.eval.registry import (  # noqa: F401
    ATTACK_REGISTRY,
    CLASS_LABEL_UK,
    KNOWLEDGE_UK,
    AttackSpec,
    apply_spec,
    default_strength,
    get_attack,
    list_attack_ids,
    protocol_table_rows,
)
from src.eval.transforms import gaussian_blur


class GaussianBlurSurrogate(nn.Module):
    """VINE-style blur stand-in for the regenerative *training* branch."""

    def __init__(self, sigma: float = 1.2, seq_len: int = 77, embed_dim: int = 768) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.register_buffer(
            "null_text_embeds",
            torch.zeros(1, seq_len, embed_dim),
            persistent=False,
        )

    def expand_text_embeds(self, batch_size: int) -> torch.Tensor:
        return self.null_text_embeds.expand(batch_size, -1, -1)

    def forward(
        self, x_w: torch.Tensor, text_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _ = text_embeds
        return gaussian_blur(x_w, self.sigma)


class IdentityRegen(nn.Module):
    """No-op regen branch (``--regen-branch none``)."""

    def __init__(self, seq_len: int = 77, embed_dim: int = 768) -> None:
        super().__init__()
        self.register_buffer(
            "null_text_embeds",
            torch.zeros(1, seq_len, embed_dim),
            persistent=False,
        )

    def expand_text_embeds(self, batch_size: int) -> torch.Tensor:
        return self.null_text_embeds.expand(batch_size, -1, -1)

    def forward(
        self, x_w: torch.Tensor, text_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _ = text_embeds
        return x_w


@torch.no_grad()
def calibrate_blur_sigma(
    reference: nn.Module,
    image_size: int = 64,
    device: str = "cpu",
    n: int = 4,
) -> float:
    """Binary-search Gaussian sigma so mean L2 matches ``reference`` on noise images."""
    x = torch.rand(n, 3, image_size, image_size, device=device) * 2 - 1
    text = torch.zeros(n, 77, 768, device=device)
    try:
        y = reference(x, text)
        budget = (y - x).flatten(1).norm(dim=1).mean().item()
    except Exception:
        budget = 0.35
    lo, hi = 0.05, 8.0
    best = 1.2
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        z = gaussian_blur(x, mid)
        err = (z - x).flatten(1).norm(dim=1).mean().item()
        best = mid
        if err < budget:
            lo = mid
        else:
            hi = mid
    return float(best)


SCHEME_INFO: Dict[str, Dict[str, object]] = {
    "ours": {
        "label_uk": "Запропонований метод",
        "payload_bits": None,
        "train_resolution": None,
        "weights": "local",
        "available": True,
    },
    "stegastamp": {
        "label_uk": "StegaStamp",
        "payload_bits": 100,
        "train_resolution": 400,
        "weights": "official",
        "available": False,
    },
    "trustmark": {
        "label_uk": "TrustMark",
        "payload_bits": 100,
        "train_resolution": 256,
        "weights": "official",
        "available": False,
    },
    "vine": {
        "label_uk": "VINE",
        "payload_bits": 100,
        "train_resolution": 256,
        "weights": "official",
        "available": False,
    },
    "stable_signature": {
        "label_uk": "Stable Signature",
        "payload_bits": 48,
        "train_resolution": 512,
        "weights": "official",
        "available": False,
    },
}

SCHEME_ORDER: Tuple[str, ...] = (
    "ours",
    "stegastamp",
    "trustmark",
    "vine",
    "stable_signature",
)
