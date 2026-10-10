"""Perturbation metrics between watermarked and transformed images."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from src.utils.metrics import LPIPSMeter


@torch.no_grad()
def perturbation_norms(x_w: torch.Tensor, x_t: torch.Tensor) -> Dict[str, List[float]]:
    """Per-image ``||T(x_w)-x_w||_2`` and ``||.||_inf`` (over all pixels/channels)."""
    diff = (x_t - x_w).detach()
    flat = diff.flatten(1)
    l2 = flat.norm(p=2, dim=1)
    linf = flat.abs().amax(dim=1)
    return {
        "pert_l2": l2.cpu().tolist(),
        "pert_linf": linf.cpu().tolist(),
    }


@torch.no_grad()
def perturbation_lpips(
    x_w: torch.Tensor,
    x_t: torch.Tensor,
    meter: Optional[LPIPSMeter] = None,
) -> List[float]:
    """LPIPS between watermarked and transformed images."""
    if meter is None:
        meter = LPIPSMeter().to(x_w.device)
    return meter(x_w, x_t)
