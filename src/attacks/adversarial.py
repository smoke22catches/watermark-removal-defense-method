"""PGD adversarial attack maximizing decoder bit error."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def straight_through(x_w: torch.Tensor, x_attacked: torch.Tensor) -> torch.Tensor:
    """Treat the attack residual as constant: forward ``x_attacked``, backward through ``x_w``.

    Used for the PGD branch: ``x_adv = x_w + (x_pgd - x_w).detach()`` so the encoder
    receives gradient through ``x_w`` while the inner-loop perturbation stays fixed.
    Distortion / regeneration keep a live autograd path and do not use this helper.
    """
    return x_w + (x_attacked - x_w).detach()


def pgd_attack_on_decoder(
    x_center: torch.Tensor,
    m: torch.Tensor,
    decoder: nn.Module,
    eps: float = 0.02,
    alpha: float = 0.005,
    steps: int = 5,
) -> torch.Tensor:
    """Projected gradient ascent on BCEWithLogits to fool the decoder.

    Starts from ``x_center`` and projects onto the L-infinity ball of radius ``eps``
    centered on ``x_center`` (watermarked ``x_w`` for the adv branch, regenerated
    ``x_r`` for guided regeneration). Inner loop uses ``torch.autograd.grad`` w.r.t.
    the input only (no decoder ``.grad`` accumulation). Decoder is briefly set to
    ``eval()`` so BatchNorm running stats are not updated by adversarial samples.

    Returns a detached adversarial image. Callers that need encoder gradients should
    wrap with :func:`straight_through` (center = live tensor that should carry grad).
    """
    was_training = decoder.training
    decoder.eval()
    try:
        x_adv = x_center.detach().clone().requires_grad_(True)
        for _ in range(steps):
            logits = decoder(x_adv)
            loss = F.binary_cross_entropy_with_logits(logits, m)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            # Bound relative to the projection center (no grad through the box).
            x_adv = torch.min(
                torch.max(x_adv, (x_center - eps).detach()),
                (x_center + eps).detach(),
            )
            x_adv = x_adv.clamp(-1, 1)
            x_adv.requires_grad_(True)
        return x_adv.detach()
    finally:
        decoder.train(was_training)
