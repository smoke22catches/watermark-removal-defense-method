"""PGD adversarial attack maximizing decoder bit error."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def straight_through(x_w: torch.Tensor, x_attacked: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward uses ``x_attacked``, backward is identity on ``x_w``.

    Lets decode loss update the encoder even when the attack itself is non-differentiable
    or intentionally detached (PGD inner loop, no_grad distortion/regen).
    """
    return x_w + (x_attacked.detach() - x_w.detach())


def pgd_attack_on_decoder(
    x_w: torch.Tensor,
    m: torch.Tensor,
    decoder: nn.Module,
    eps: float = 0.02,
    alpha: float = 0.005,
    steps: int = 5,
) -> torch.Tensor:
    """Projected gradient ascent on BCEWithLogits to fool the decoder.

    Returns a detached adversarial image. Callers that need encoder gradients should
    wrap with :func:`straight_through`.
    """
    x_adv = x_w.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = decoder(x_adv)
        loss = F.binary_cross_entropy_with_logits(logits, m)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # Bound relative to the live watermarked image (no grad through the box).
        x_adv = torch.min(torch.max(x_adv, (x_w - eps).detach()), (x_w + eps).detach())
        x_adv = x_adv.clamp(-1, 1)
        x_adv.requires_grad_(True)
    return x_adv.detach()
