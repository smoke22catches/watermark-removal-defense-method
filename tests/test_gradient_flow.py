"""Acceptance tests: encoder receives gradient through attack branches."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.adversarial import pgd_attack_on_decoder
from src.attacks.distortion import DistortionBank
from src.attacks.sampler import AttackSampler
from src.engine.train import train_step
from src.losses import PerceptualLoss
from src.models import Decoder, Encoder


MSG_LEN = 16
IMAGE_SIZE = 32
BATCH = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TinyFrozenRegen(nn.Module):
    """Differentiable frozen VAE/UNet stand-in for gradient-flow checks."""

    def __init__(self) -> None:
        super().__init__()
        self.vae = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, padding=1),
        )
        self.unet = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, padding=1),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        _ = text_embeds
        z = self.vae(x_w)
        z = self.unet(z)
        return torch.clamp(z, -1.0, 1.0)


def _bn_buffers(module: nn.Module) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, m in module.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            out.append((name, m.running_mean.detach().clone(), m.running_var.detach().clone()))
    return out


def _assert_bn_unchanged(
    before: List[Tuple[str, torch.Tensor, torch.Tensor]],
    module: nn.Module,
) -> None:
    after = {name: (m.running_mean, m.running_var) for name, m in module.named_modules() if isinstance(m, nn.BatchNorm2d)}
    for name, mean0, var0 in before:
        mean1, var1 = after[name]
        assert torch.equal(mean0, mean1), f"BN running_mean changed at {name}"
        assert torch.equal(var0, var1), f"BN running_var changed at {name}"


def _encoder_grad_norms(encoder: nn.Module) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"encoder.{name}.grad is None"
        g = float(p.grad.detach().float().norm().item())
        assert math.isfinite(g), f"encoder.{name} grad norm not finite: {g}"
        assert g > 0.0, f"encoder.{name} grad norm is zero"
        norms[name] = g
    return norms


def _assert_frozen_grads_none(modules: Iterable[nn.Module]) -> None:
    for mod in modules:
        for name, p in mod.named_parameters():
            assert p.grad is None, f"frozen {type(mod).__name__}.{name}.grad is not None"


def _assert_decoder_grads_none_or_zero(decoder: nn.Module) -> None:
    for name, p in decoder.named_parameters():
        if p.grad is None:
            continue
        assert float(p.grad.detach().abs().sum().item()) == 0.0, (
            f"decoder.{name}.grad populated after PGD inner loop"
        )


def _build_stack(force_branch: str):
    encoder = Encoder(msg_len=MSG_LEN, ch=16, strength=0.4).to(DEVICE)
    decoder = Decoder(msg_len=MSG_LEN, ch=16).to(DEVICE)
    distortion_bank = DistortionBank(jpeg_quality=50, gaussian_std=0.03).to(DEVICE)
    regen_proxy = TinyFrozenRegen().to(DEVICE)
    text_encoder = nn.Linear(4, 4).to(DEVICE)
    text_encoder.requires_grad_(False)

    probs = {"dist": (1.0, 0.0, 0.0), "regen": (0.0, 1.0, 0.0), "adv": (0.0, 0.0, 1.0)}[force_branch]
    sampler = AttackSampler(distortion_bank, regen_proxy, decoder, probs=probs)
    perc = PerceptualLoss(use_lpips=False).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    return encoder, decoder, sampler, perc, opt, regen_proxy, text_encoder


def _run_forced_branch(force_branch: str) -> Dict[str, float]:
    torch.manual_seed(0)
    encoder, decoder, sampler, perc, opt, regen_proxy, text_encoder = _build_stack(force_branch)
    x = torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 2 - 1
    text_embeds = torch.zeros(BATCH, 77, 768, device=DEVICE)

    if force_branch == "adv":
        encoder.train()
        decoder.train()
        m = torch.randint(0, 2, (BATCH, MSG_LEN), device=DEVICE).float()
        x_w = encoder(x, m)
        bn_before = _bn_buffers(decoder)
        opt.zero_grad(set_to_none=True)
        x_pgd = pgd_attack_on_decoder(x_w, m, decoder, eps=0.02, alpha=0.005, steps=5)
        _assert_decoder_grads_none_or_zero(decoder)
        _assert_bn_unchanged(bn_before, decoder)
        assert x_pgd.shape == x_w.shape

    stats = train_step(
        x,
        encoder,
        decoder,
        sampler,
        text_embeds,
        opt,
        perc,
        msg_len=MSG_LEN,
        lambda_perc=0.0,  # decode path only — isolates attack→encoder gradient
        device=DEVICE,
        pgd_eps=0.02,
        pgd_alpha=0.005,
        pgd_steps=5,
    )
    assert stats["attack_type"] == force_branch, stats

    norms = _encoder_grad_norms(encoder)
    _assert_frozen_grads_none([regen_proxy.vae, regen_proxy.unet, text_encoder])
    return norms


def test_gradient_flow_each_branch() -> Dict[str, Dict[str, float]]:
    """Encoder grads finite and >0 for dist / regen / adv; frozen modules stay clean."""
    all_norms: Dict[str, Dict[str, float]] = {}
    for branch in ("dist", "regen", "adv"):
        norms = _run_forced_branch(branch)
        total = math.sqrt(sum(v * v for v in norms.values()))
        all_norms[branch] = {"__total__": total, **norms}
        print(f"[grad-flow] branch={branch} encoder_grad_norm_total={total:.6e}")
        for name, g in norms.items():
            print(f"  {name}: {g:.6e}")
    return all_norms


def test_smoke_two_train_steps() -> None:
    """Existing smoke: two training iterations complete without error."""
    torch.manual_seed(1)
    encoder = Encoder(msg_len=MSG_LEN, ch=16).to(DEVICE)
    decoder = Decoder(msg_len=MSG_LEN, ch=16).to(DEVICE)
    sampler = AttackSampler(
        DistortionBank().to(DEVICE),
        TinyFrozenRegen().to(DEVICE),
        decoder,
        probs=(0.4, 0.4, 0.2),
    )
    perc = PerceptualLoss(use_lpips=False).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    text_embeds = torch.zeros(BATCH, 77, 768, device=DEVICE)
    for _ in range(2):
        x = torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 2 - 1
        stats = train_step(
            x,
            encoder,
            decoder,
            sampler,
            text_embeds,
            opt,
            perc,
            msg_len=MSG_LEN,
            lambda_perc=0.05,
            device=DEVICE,
        )
        assert math.isfinite(stats["loss"])
        assert stats["attack_type"] in {"dist", "regen", "adv"}


if __name__ == "__main__":
    print(f"device={DEVICE}")
    norms = test_gradient_flow_each_branch()
    test_smoke_two_train_steps()
    print("=== all gradient-flow acceptance tests passed ===")
    print("Measured encoder gradient norms (total L2 per branch):")
    for branch, d in norms.items():
        print(f"  {branch}: {d['__total__']:.6e}")
