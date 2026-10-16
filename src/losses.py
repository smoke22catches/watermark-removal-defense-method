"""Decode and perceptual losses (BCEWithLogits + LPIPS + MSE)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """MSE + LPIPS between cover ``x`` and watermarked ``x_w`` (range [-1, 1])."""

    def __init__(self, net: str = "alex", use_lpips: bool = True) -> None:
        super().__init__()
        self.use_lpips = use_lpips
        if use_lpips:
            try:
                import lpips

                net_mod = lpips.LPIPS(net=net)
                for p in net_mod.parameters():
                    p.requires_grad_(False)
                # Register so ``.to(device)`` moves LPIPS with the parent module.
                self.add_module("_lpips", net_mod)
            except Exception:
                # Fallback if lpips weights are unavailable (e.g. offline smoke test)
                self.use_lpips = False

    def forward(self, x_w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss_mse = F.mse_loss(x_w, x)
        lpips_net = getattr(self, "_lpips", None)
        if lpips_net is not None:
            # LPIPS expects [-1, 1]; models already operate in that range.
            loss_lpips = lpips_net(x_w, x).mean()
            return loss_mse + loss_lpips
        return loss_mse


def decode_loss(logits: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits between predicted and target message bits."""
    return F.binary_cross_entropy_with_logits(logits, m)


def combined_loss(
    logits: torch.Tensor,
    m: torch.Tensor,
    x_w: torch.Tensor,
    x: torch.Tensor,
    perc_fn: PerceptualLoss,
    lambda_perc: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, decode, perceptual) matching the training objective in start.py."""
    loss_dec = decode_loss(logits, m)
    loss_perc = perc_fn(x_w, x)
    loss = loss_dec + lambda_perc * loss_perc
    return loss, loss_dec, loss_perc
