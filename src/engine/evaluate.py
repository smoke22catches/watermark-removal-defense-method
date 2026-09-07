"""Evaluation: attack sweep including UNSEEN attacks held out from training."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.attacks.distortion import DiffJPEG
from src.attacks.regeneration import TextConditioner, guided_regen_attack, make_text_embeds


def _unseen_gaussian_blur(
    x: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.5,
) -> torch.Tensor:
    """UNSEEN attack: Gaussian blur not present in DistortionBank training set."""
    # Separable approx via avg-pool stacks is avoided; use conv with fixed Gaussian kernel.
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel = kernel_2d.expand(x.size(1), 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.size(1)).clamp(-1, 1)


def apply_attack(
    name: str,
    x_w: torch.Tensor,
    m: torch.Tensor,
    decoder: nn.Module,
    regen_proxy: Optional[nn.Module],
    text_embeds: Optional[torch.Tensor],
    config: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Apply a named attack for evaluation."""
    config = config or {}
    dj = config.get("diffjpeg", {})
    regen_cfg = config.get("regen", {})
    eval_cfg = config.get("eval", {})

    if name == "clean":
        return x_w
    if name == "jpeg":
        return DiffJPEG(
            quality=int(dj.get("quality", 50)),
            use_real_diffjpeg=bool(dj.get("use_real_diffjpeg", False)),
            chroma_subsample=bool(dj.get("chroma_subsample", True)),
        )(x_w)
    if name == "regen":
        if regen_proxy is None or text_embeds is None:
            raise ValueError("regen attack requires regen_proxy and text_embeds")
        return regen_proxy(x_w, text_embeds)
    if name == "guided_regen":
        if regen_proxy is None or text_embeds is None:
            raise ValueError("guided_regen requires regen_proxy and text_embeds")
        return guided_regen_attack(
            x_w,
            decoder,
            m,
            regen_proxy,
            text_embeds,
            guidance_scale=float(regen_cfg.get("guidance_scale", 2.0)),
        )
    if name in ("unseen_blur", "unseen"):
        return _unseen_gaussian_blur(
            x_w,
            kernel_size=int(eval_cfg.get("unseen_blur_kernel", 5)),
            sigma=float(eval_cfg.get("unseen_blur_sigma", 1.5)),
        )
    if name == "noise":
        return torch.clamp(x_w + torch.randn_like(x_w) * 0.05, -1, 1)
    raise ValueError(f"Unknown attack: {name}")


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    attacks: Sequence[str] = ("clean", "jpeg", "regen", "guided_regen", "unseen_blur"),
    msg_len: int = 64,
    regen_proxy: Optional[nn.Module] = None,
    config: Optional[Mapping[str, Any]] = None,
    text_conditioner: Optional[TextConditioner] = None,
) -> Dict[str, List[float]]:
    """Attack sweep over the validation loader; returns per-attack bit-accuracy lists."""
    config = config or {}
    regen_cfg = config.get("regen", {})
    encoder.eval()
    decoder.eval()

    # guided_regen needs grad on the attacked image; disable no_grad selectively below
    results: Dict[str, List[float]] = {a: [] for a in attacks}

    for batch in val_loader:
        x = batch[0].to(device)
        m = torch.randint(0, 2, (x.size(0), msg_len), device=device).float()
        x_w = encoder(x, m)
        text_embeds = make_text_embeds(
            x.size(0),
            device,
            seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
            dim=int(regen_cfg.get("text_embed_dim", 768)),
            conditioner=text_conditioner,
        )

        for a in attacks:
            if a == "guided_regen":
                # Needs autograd inside guided_regen_attack
                with torch.enable_grad():
                    x_test = apply_attack(
                        a, x_w, m, decoder, regen_proxy, text_embeds, config
                    )
            else:
                x_test = apply_attack(a, x_w, m, decoder, regen_proxy, text_embeds, config)

            logits = decoder(x_test)
            bit_acc = ((torch.sigmoid(logits) > 0.5).float() == m).float().mean().item()
            results[a].append(bit_acc)

    for a in attacks:
        if results[a]:
            mean_acc = sum(results[a]) / len(results[a])
            print(f"{a}: bit-accuracy = {mean_acc:.4f}")
        else:
            print(f"{a}: bit-accuracy = n/a")
    return results


def evaluate_single_image(
    encoder: nn.Module,
    decoder: nn.Module,
    image: torch.Tensor,
    device: str = "cuda",
    msg: Optional[torch.Tensor] = None,
    msg_len: int = 64,
    attack: Optional[str] = None,
    regen_proxy: Optional[nn.Module] = None,
    config: Optional[Mapping[str, Any]] = None,
    text_conditioner: Optional[TextConditioner] = None,
) -> Dict[str, Any]:
    """Embed / optionally attack / decode a single image; report bits + PSNR/SSIM."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    config = config or {}
    regen_cfg = config.get("regen", {})
    encoder.eval()
    decoder.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0)
    x = image.to(device)
    if msg is None:
        m = torch.randint(0, 2, (1, msg_len), device=device).float()
    else:
        m = msg.to(device).float()
        if m.ndim == 1:
            m = m.unsqueeze(0)

    with torch.no_grad():
        x_w = encoder(x, m)

    text_embeds = make_text_embeds(
        x.size(0),
        device,
        seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
        dim=int(regen_cfg.get("text_embed_dim", 768)),
        conditioner=text_conditioner,
    )

    x_test = x_w
    if attack and attack != "clean":
        if attack == "guided_regen":
            with torch.enable_grad():
                x_test = apply_attack(
                    attack, x_w, m, decoder, regen_proxy, text_embeds, config
                )
        else:
            with torch.no_grad():
                x_test = apply_attack(
                    attack, x_w, m, decoder, regen_proxy, text_embeds, config
                )

    with torch.no_grad():
        logits = decoder(x_test)
        bits = (torch.sigmoid(logits) > 0.5).float()
        bit_acc = (bits == m).float().mean().item()

    def _to01(t: torch.Tensor) -> torch.Tensor:
        return ((t.detach().cpu().clamp(-1, 1) + 1) / 2).squeeze(0)

    x01 = _to01(x).permute(1, 2, 0).numpy()
    xw01 = _to01(x_w).permute(1, 2, 0).numpy()
    psnr = float(peak_signal_noise_ratio(x01, xw01, data_range=1.0))
    ssim = float(structural_similarity(x01, xw01, channel_axis=2, data_range=1.0))

    return {
        "message": m.detach().cpu(),
        "recovered": bits.detach().cpu(),
        "bit_accuracy": bit_acc,
        "psnr": psnr,
        "ssim": ssim,
        "x": x.detach().cpu(),
        "x_w": x_w.detach().cpu(),
        "x_test": x_test.detach().cpu(),
    }
