"""Acceptance tests: guided PGD on regeneration output in late training."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.adversarial import pgd_attack_on_decoder
from src.attacks.distortion import DistortionBank
from src.attacks.regeneration import (
    apply_guided_pgd_ste,
    resolve_guided_start_epoch,
    should_apply_guided,
)
from src.attacks.sampler import AttackSampler
from src.engine.train import train_step
from src.losses import PerceptualLoss
from src.models import Decoder, Encoder

MSG_LEN = 16
IMAGE_SIZE = 32
BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS_G = 0.05
BETA_G = 0.01
J_G = 5


class TinyFrozenRegen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, padding=1),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        _ = text_embeds
        return torch.clamp(self.net(x_w), -1.0, 1.0)


def test_guided_share_schedule() -> None:
    """Before start_epoch: never; afterwards share within 0.05 of guided_prob."""
    start = 10
    guided_prob = 0.5
    n = 1000
    rng = torch.Generator().manual_seed(123)

    before = sum(
        should_apply_guided(epoch=start - 1, guided_start_epoch=start, guided_prob=guided_prob, generator=rng)
        for _ in range(n)
    )
    assert before == 0, f"guided applied {before}/{n} times before start_epoch"

    rng2 = torch.Generator().manual_seed(123)
    after = sum(
        should_apply_guided(epoch=start, guided_start_epoch=start, guided_prob=guided_prob, generator=rng2)
        for _ in range(n)
    )
    share = after / n
    assert abs(share - guided_prob) <= 0.05, (
        f"observed guided share {share:.3f} not within 0.05 of {guided_prob}"
    )
    print(f"[ok] schedule: before={before} after_share={share:.3f} (target={guided_prob})")


def test_guided_increases_decode_loss_and_ball() -> None:
    """On a fixed decoder, mean decode loss rises; ||x_g - x_r||_inf <= eps."""
    torch.manual_seed(0)
    decoder = Decoder(msg_len=MSG_LEN, ch=16).to(DEVICE).eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    losses_r = []
    losses_g = []
    max_linf = 0.0
    for i in range(8):
        torch.manual_seed(100 + i)
        x_r = (torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 2 - 1).requires_grad_(True)
        m = torch.randint(0, 2, (BATCH, MSG_LEN), device=DEVICE).float()
        with torch.no_grad():
            loss_r = F.binary_cross_entropy_with_logits(decoder(x_r), m).item()
        x_g = apply_guided_pgd_ste(
            x_r, m, decoder, eps_guided=EPS_G, beta_guided=BETA_G, J_guided=J_G
        )
        with torch.no_grad():
            loss_g = F.binary_cross_entropy_with_logits(decoder(x_g), m).item()
            linf = (x_g - x_r).abs().max().item()
        losses_r.append(loss_r)
        losses_g.append(loss_g)
        max_linf = max(max_linf, linf)
        assert linf <= EPS_G + 1e-6, f"||x_g - x_r||_inf={linf} > eps_guided={EPS_G}"

    mean_r = sum(losses_r) / len(losses_r)
    mean_g = sum(losses_g) / len(losses_g)
    assert mean_g >= mean_r, f"mean decode loss guided {mean_g:.6f} < regen {mean_r:.6f}"
    print(f"[ok] loss_r={mean_r:.4f} loss_g={mean_g:.4f} max_linf={max_linf:.6f}")


def test_guided_path_encoder_grad() -> None:
    """Encoder receives non-zero gradient when the guided regen path is taken."""
    torch.manual_seed(7)
    encoder = Encoder(msg_len=MSG_LEN, ch=16, strength=0.4).to(DEVICE)
    decoder = Decoder(msg_len=MSG_LEN, ch=16).to(DEVICE)
    regen = TinyFrozenRegen().to(DEVICE)
    sampler = AttackSampler(
        DistortionBank().to(DEVICE),
        regen,
        decoder,
        probs=(0.0, 1.0, 0.0),
        seed=0,
    )
    perc = PerceptualLoss(use_lpips=False).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    x = torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 2 - 1
    text_embeds = torch.zeros(BATCH, 77, 768, device=DEVICE)

    # Force guided: start_epoch=0, guided_prob=1.0
    stats = train_step(
        x,
        encoder,
        decoder,
        sampler,
        text_embeds,
        opt,
        perc,
        msg_len=MSG_LEN,
        lambda_perc=0.0,
        device=DEVICE,
        epoch=0,
        guided_start_epoch=0,
        guided_prob=1.0,
        eps_guided=EPS_G,
        beta_guided=BETA_G,
        J_guided=J_G,
    )
    assert stats["attack_type"] == "regen"
    assert stats["guided_applied"] is True

    total = 0.0
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"encoder.{name}.grad is None on guided path"
        g = float(p.grad.detach().float().norm().item())
        assert math.isfinite(g) and g > 0.0, f"encoder.{name} bad grad norm {g}"
        total += g * g
    total = math.sqrt(total)
    print(f"[ok] guided encoder_grad_norm_total={total:.6e}")


def test_resolve_guided_start_default_half() -> None:
    assert resolve_guided_start_epoch({"epochs": 100, "regen": {}}) == 50
    assert resolve_guided_start_epoch({"epochs": 100, "regen": {"guided_start_epoch": 70}}) == 70
    print("[ok] guided_start_epoch defaults to epochs//2")


def test_pgd_center_is_x_r_not_x_w() -> None:
    """Projection ball must be around x_r; a far x_w must not pull the result back."""
    torch.manual_seed(3)
    decoder = Decoder(msg_len=MSG_LEN, ch=8).to(DEVICE).eval()
    x_r = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    m = torch.ones(1, MSG_LEN, device=DEVICE)
    x_pgd = pgd_attack_on_decoder(x_r, m, decoder, eps=EPS_G, alpha=BETA_G, steps=J_G)
    assert (x_pgd - x_r).abs().max().item() <= EPS_G + 1e-6
    print("[ok] PGD ball centered on x_r")


if __name__ == "__main__":
    print(f"device={DEVICE}")
    test_resolve_guided_start_default_half()
    test_guided_share_schedule()
    test_pgd_center_is_x_r_not_x_w()
    test_guided_increases_decode_loss_and_ball()
    test_guided_path_encoder_grad()
    print("=== all guided-regen acceptance tests passed ===")
