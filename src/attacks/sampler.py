"""Stochastic attack sampler over clean / distortion / regen / PGD branches."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


def curriculum_probs(epoch: int, cfg: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Return ``(p_dist, p_regen, p_adv)``; residual ``1-sum`` is clean.

    Shared by the training loop, Algorithm 1 export, and figure B1 so the
    published probabilities cannot drift from the code.
    """
    cur = cfg.get("curriculum", {})
    p_adv_default = float(cfg.get("attack_probs", [0.4, 0.4, 0.2])[2])
    p_adv = float(cur.get("p_adv", 0.2)) if cur else p_adv_default
    regen_branch = str(cfg.get("regen_branch", "ddim_proxy"))

    if regen_branch == "none":
        return max(0.0, 1.0 - p_adv), 0.0, p_adv

    if not cur.get("enabled", True):
        probs = cfg.get("attack_probs", [0.4, 0.4, 0.2])
        return float(probs[0]), float(probs[1]), float(probs[2])

    warm = int(cur.get("warm_start_epochs", 0))
    if epoch < warm:
        return 0.0, 0.0, 0.0

    t = epoch - warm
    p_regen = min(
        float(cur.get("p_regen_start", 0.1)) + float(cur.get("p_regen_step", 0.01)) * t,
        float(cur.get("p_regen_max", 0.4)),
    )
    p_adv = float(cur.get("p_adv", 0.15))
    p_dist_floor = float(cur.get("p_dist_floor", 0.2))
    p_clean_floor = float(cur.get("p_clean_floor", 0.2))

    p_dist = max(p_dist_floor, 1.0 - p_regen - p_adv - p_clean_floor)
    total = p_dist + p_regen + p_adv
    max_attack = 1.0 - p_clean_floor
    if total > max_attack and total > 0:
        scale = max_attack / total
        p_dist, p_regen, p_adv = p_dist * scale, p_regen * scale, p_adv * scale
    return p_dist, p_regen, p_adv


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
