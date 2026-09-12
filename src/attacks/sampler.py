"""Stochastic attack sampler over clean / distortion / regen / PGD branches."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class AttackSampler(nn.Module):
    """Sample an attack from A_dist / A_regen / A_adv; remaining mass is clean.

    ``probs`` is ``(p_dist, p_regen, p_adv)``. If the three sum to less than 1,
    the residual probability is the clean (identity) branch — used for warm-start.
    """

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
        self.probs = tuple(probs)  # p(dist), p(regen), p(adv); residual → clean
        self.pgd_eps = pgd_eps
        self.pgd_alpha = pgd_alpha
        self.pgd_steps = pgd_steps

    @property
    def p_clean(self) -> float:
        return max(0.0, 1.0 - sum(self.probs[:3]))

    def forward(
        self,
        x_w: torch.Tensor,
        m: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, str]:
        r = torch.rand(1).item()
        p_dist, p_regen, p_adv = self.probs[0], self.probs[1], self.probs[2]
        if r < p_dist:
            return self.distortion_bank(x_w), "dist"
        if r < p_dist + p_regen:
            return self.regen_proxy(x_w, text_embeds), "regen"
        if r < p_dist + p_regen + p_adv:
            # Marker only: train_step re-runs PGD, then applies STE to x_w.
            _ = (m, self.decoder, self.pgd_eps, self.pgd_alpha, self.pgd_steps)
            return x_w.detach(), "adv"
        return x_w, "clean"
