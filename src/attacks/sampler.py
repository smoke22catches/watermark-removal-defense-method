"""Stochastic attack sampler over distortion / regen / PGD branches."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class AttackSampler(nn.Module):
    """Sample an attack from A_dist / A_regen / A_adv according to ``probs``."""

    def __init__(
        self,
        distortion_bank: nn.Module,
        regen_proxy: nn.Module,
        decoder: nn.Module,
        probs: Sequence[float] = (0.4, 0.4, 0.2),
        pgd_eps: float = 0.02,
        pgd_alpha: float = 0.005,
        pgd_steps: int = 5,
    ) -> None:
        super().__init__()
        self.distortion_bank = distortion_bank
        self.regen_proxy = regen_proxy
        self.decoder = decoder
        self.probs = tuple(probs)  # p(dist), p(regen), p(adv) — відповідає 𝒜_dist, 𝒜_regen, 𝒜_adv з моделі
        self.pgd_eps = pgd_eps
        self.pgd_alpha = pgd_alpha
        self.pgd_steps = pgd_steps

    def forward(
        self,
        x_w: torch.Tensor,
        m: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, str]:
        r = torch.rand(1).item()
        if r < self.probs[0]:
            return self.distortion_bank(x_w), "dist"
        elif r < self.probs[0] + self.probs[1]:
            return self.regen_proxy(x_w, text_embeds), "regen"
        else:
            # Marker only: train_step re-runs PGD outside no_grad so grads reach x_w.
            # (Matching start.py comment: "PGD рахується окремо ... на живому графі".)
            # Running PGD here under torch.no_grad() would raise RuntimeError.
            _ = (m, self.decoder, self.pgd_eps, self.pgd_alpha, self.pgd_steps)
            return x_w.detach(), "adv"
