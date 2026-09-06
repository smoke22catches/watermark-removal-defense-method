"""PGD adversarial attack maximizing decoder bit error."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack_on_decoder(
    x_w: torch.Tensor,
    m: torch.Tensor,
    decoder: nn.Module,
    eps: float = 0.02,
    alpha: float = 0.005,
    steps: int = 5,
) -> torch.Tensor:
    """Projected gradient ascent on BCEWithLogits to fool the decoder."""
    x_adv = x_w.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = decoder(x_adv)
        loss = F.binary_cross_entropy_with_logits(logits, m)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x_adv, x_w - eps, x_w + eps).clamp(-1, 1)
        x_adv.requires_grad_(True)
    return x_adv.detach()
