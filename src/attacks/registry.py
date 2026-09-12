"""Evaluation attack registry: protocol metadata + apply implementations.

Training-time DistortionBank / RegenerationProxy / PGD are unchanged. This module
is the single source of truth for the *evaluation* sweep and the C3 protocol table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attacks.adversarial import pgd_attack_on_decoder
from src.attacks.distortion import DiffJPEG
from src.attacks.regeneration import guided_regen_attack


@dataclass(frozen=True)
class AttackSpec:
    id: str
    label_uk: str
    cls: str  # dist_geometric | dist_photometric | dist_degradation | regen | adv | sem | clean
    implementation: str
    strengths: Tuple[float, ...]
    epsilon: float
    delta: float
    knowledge_level: str  # black | grey | white
    seen_in_training: bool
    strength_key: str = "strength"


# Ukrainian labels for attack *classes* (glossary, verbatim).
CLASS_LABEL_UK: Dict[str, str] = {
    "clean": "Без атаки",
    "dist_geometric": "Дисторсійні (геометричні)",
    "dist_photometric": "Дисторсійні (фотометричні)",
    "dist_degradation": "Дисторсійні (деградаційні)",
    "regen": "Регенераційні",
    "adv": "Змагальні",
    "sem": "Семантично-керовані",
}

KNOWLEDGE_UK: Dict[str, str] = {
    "black": "чорний",
    "grey": "сірий",
    "white": "білий",
}

# Default evaluation protocol. Strengths are ordered weak → strong.
ATTACK_REGISTRY: Tuple[AttackSpec, ...] = (
    AttackSpec(
        id="clean",
        label_uk="Без атаки",
        cls="clean",
        implementation="identity",
        strengths=(0.0,),
        epsilon=0.0,
        delta=0.0,
        knowledge_level="black",
        seen_in_training=True,
    ),
    AttackSpec(
        id="rotate",
        label_uk="Поворот",
        cls="dist_geometric",
        implementation="src.attacks.registry.rotate_attack",
        strengths=(5.0, 15.0, 30.0),
        epsilon=30.0,
        delta=0.15,
        knowledge_level="black",
        seen_in_training=False,
        strength_key="angle_deg",
    ),
    AttackSpec(
        id="crop",
        label_uk="Обрізання",
        cls="dist_geometric",
        implementation="src.attacks.registry.center_crop_attack",
        strengths=(0.9, 0.75, 0.6),
        epsilon=0.4,
        delta=0.2,
        knowledge_level="black",
        seen_in_training=False,
        strength_key="keep_ratio",
    ),
    AttackSpec(
        id="noise",
        label_uk="Гаусів шум",
        cls="dist_photometric",
        implementation="src.attacks.registry.gaussian_noise_attack",
        strengths=(0.02, 0.05, 0.10),
        epsilon=0.10,
        delta=0.08,
        knowledge_level="black",
        seen_in_training=True,
        strength_key="std",
    ),
    AttackSpec(
        id="unseen_blur",
        label_uk="Гаусів розмиття (відкладене)",
        cls="dist_photometric",
        implementation="src.attacks.registry.gaussian_blur_attack",
        strengths=(0.8, 1.5, 3.0),
        epsilon=3.0,
        delta=0.12,
        knowledge_level="black",
        seen_in_training=False,
        strength_key="sigma",
    ),
    AttackSpec(
        id="jpeg",
        label_uk="JPEG",
        cls="dist_degradation",
        implementation="src.attacks.distortion.DiffJPEG",
        strengths=(80.0, 50.0, 30.0),
        epsilon=0.70,
        delta=0.20,
        knowledge_level="black",
        seen_in_training=True,
        strength_key="quality",
    ),
    AttackSpec(
        id="downsample",
        label_uk="Даунсемпл–апсемпл",
        cls="dist_degradation",
        implementation="src.attacks.registry.downsample_attack",
        strengths=(2.0, 4.0),
        epsilon=4.0,
        delta=0.18,
        knowledge_level="black",
        seen_in_training=True,
        strength_key="factor",
    ),
    AttackSpec(
        id="regen",
        label_uk="DDIM-проксі (навчальний)",
        cls="regen",
        implementation="src.attacks.regeneration.RegenerationProxy",
        strengths=(0.2, 0.3, 0.5),
        epsilon=0.5,
        delta=0.25,
        knowledge_level="grey",
        seen_in_training=True,
        strength_key="t_start",
    ),
    AttackSpec(
        id="guided_regen",
        label_uk="Керована регенерація",
        cls="regen",
        implementation="src.attacks.regeneration.guided_regen_attack",
        strengths=(1.0, 2.0, 4.0),
        epsilon=0.5,
        delta=0.30,
        knowledge_level="white",
        seen_in_training=False,
        strength_key="guidance_scale",
    ),
    AttackSpec(
        id="diffpure",
        label_uk="DiffPure (повний крок)",
        cls="regen",
        implementation="src.attacks.registry.diffpure_attack",
        strengths=(20.0, 50.0),
        epsilon=1.0,
        delta=0.40,
        knowledge_level="grey",
        seen_in_training=False,
        strength_key="n_steps",
    ),
    AttackSpec(
        id="rinse_2",
        label_uk="Промивання ×2",
        cls="regen",
        implementation="src.attacks.registry.rinse_attack",
        strengths=(2.0,),
        epsilon=0.6,
        delta=0.35,
        knowledge_level="grey",
        seen_in_training=False,
        strength_key="repeats",
    ),
    AttackSpec(
        id="rinse_4",
        label_uk="Промивання ×4",
        cls="regen",
        implementation="src.attacks.registry.rinse_attack",
        strengths=(4.0,),
        epsilon=0.8,
        delta=0.45,
        knowledge_level="grey",
        seen_in_training=False,
        strength_key="repeats",
    ),
    AttackSpec(
        id="vae_alt",
        label_uk="VAE іншої архітектури",
        cls="regen",
        implementation="src.attacks.registry.alt_vae_attack",
        strengths=(4.0, 8.0),
        epsilon=0.7,
        delta=0.30,
        knowledge_level="grey",
        seen_in_training=False,
        strength_key="bottleneck",
    ),
    AttackSpec(
        id="diffusion_alt",
        label_uk="Дифузійний каркас (інший)",
        cls="regen",
        implementation="src.attacks.registry.alt_diffusion_attack",
        strengths=(3.0, 5.0),
        epsilon=0.7,
        delta=0.35,
        knowledge_level="grey",
        seen_in_training=False,
        strength_key="scales",
    ),
    AttackSpec(
        id="adv",
        label_uk="PGD на декодер",
        cls="adv",
        implementation="src.attacks.adversarial.pgd_attack_on_decoder",
        strengths=(0.01, 0.02, 0.04),
        epsilon=0.04,
        delta=0.10,
        knowledge_level="white",
        seen_in_training=True,
        strength_key="eps",
    ),
    AttackSpec(
        id="semantic_inpaint",
        label_uk="Семантична інпейнтинг-регенерація",
        cls="sem",
        implementation="src.attacks.registry.semantic_inpaint_attack",
        strengths=(0.25, 0.40),
        epsilon=0.40,
        delta=0.50,
        knowledge_level="grey",
        seen_in_training=False,
        strength_key="mask_frac",
    ),
)

_BY_ID: Dict[str, AttackSpec] = {s.id: s for s in ATTACK_REGISTRY}


def get_attack(attack_id: str) -> AttackSpec:
    if attack_id not in _BY_ID:
        raise KeyError(f"Unknown attack id: {attack_id}")
    return _BY_ID[attack_id]


def list_attack_ids() -> List[str]:
    return [s.id for s in ATTACK_REGISTRY]


def default_strength(spec: AttackSpec) -> float:
    return float(spec.strengths[len(spec.strengths) // 2])


def gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum().clamp_min(1e-8)
    return g[:, None] * g[None, :]


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(float(sigma), 1e-3)
    k = int(2 * round(3 * sigma) + 1)
    k = max(k, 3)
    kernel_2d = gaussian_kernel_2d(k, sigma, x.device, x.dtype)
    kernel = kernel_2d.expand(x.size(1), 1, k, k).contiguous()
    pad = k // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.size(1)).clamp(-1, 1)


def rotate_attack(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    angle = float(angle_deg) * math_pi() / 180.0
    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0).repeat(x.size(0), 1, 1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border").clamp(-1, 1)


def math_pi() -> float:
    return 3.141592653589793


def center_crop_attack(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    keep_ratio = float(min(max(keep_ratio, 0.1), 1.0))
    _, _, h, w = x.shape
    nh, nw = max(1, int(h * keep_ratio)), max(1, int(w * keep_ratio))
    top, left = (h - nh) // 2, (w - nw) // 2
    cropped = x[:, :, top : top + nh, left : left + nw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False).clamp(-1, 1)


def gaussian_noise_attack(x: torch.Tensor, std: float) -> torch.Tensor:
    return torch.clamp(x + torch.randn_like(x) * float(std), -1, 1)


def gaussian_blur_attack(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian_blur(x, sigma)


def downsample_attack(x: torch.Tensor, factor: float) -> torch.Tensor:
    f = max(int(round(float(factor))), 1)
    if f <= 1:
        return x
    return F.interpolate(F.avg_pool2d(x, f), size=x.shape[-2:], mode="bilinear").clamp(-1, 1)


def alt_vae_attack(x: torch.Tensor, bottleneck: float) -> torch.Tensor:
    """Non-SD VAE stand-in: strided spatial bottleneck + bilinear decode (different architecture)."""
    factor = max(int(round(float(bottleneck))), 2)
    z = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    # Channel mixing that SD-VAE does not use (1x1 conv-like random-but-fixed via mean/chroma split).
    y = 0.5 * z + 0.5 * z.mean(dim=1, keepdim=True)
    return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False).clamp(-1, 1)


def alt_diffusion_attack(x: torch.Tensor, scales: float) -> torch.Tensor:
    """Alternate diffusion-like surrogate (not the training DDIM/SD backbone)."""
    n = max(int(round(float(scales))), 1)
    y = x
    for i in range(n):
        sigma = 0.6 + 0.4 * i
        y = gaussian_blur(y, sigma)
        y = torch.clamp(y + torch.randn_like(y) * 0.03, -1, 1)
        y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    return y.clamp(-1, 1)


def semantic_inpaint_attack(x: torch.Tensor, mask_frac: float) -> torch.Tensor:
    """Centre-mask inpainting filled from a heavily blurred surround (semantic-regen stand-in)."""
    frac = float(min(max(mask_frac, 0.05), 0.9))
    _, _, h, w = x.shape
    mh, mw = max(1, int(h * frac)), max(1, int(w * frac))
    top, left = (h - mh) // 2, (w - mw) // 2
    fill = gaussian_blur(x, sigma=8.0)
    # Mix in spatial mean so the hole is not a copy of the watermarked region.
    fill = 0.5 * fill + 0.5 * x.mean(dim=(2, 3), keepdim=True)
    y = x.clone()
    y[:, :, top : top + mh, left : left + mw] = fill[:, :, top : top + mh, left : left + mw]
    return y.clamp(-1, 1)


def rinse_attack(
    x: torch.Tensor,
    repeats: float,
    regen_proxy: Optional[nn.Module],
    text_embeds: Optional[torch.Tensor],
) -> torch.Tensor:
    n = max(int(round(float(repeats))), 1)
    y = x
    if regen_proxy is None or text_embeds is None:
        for _ in range(n):
            y = alt_diffusion_attack(y, scales=2.0)
        return y
    for _ in range(n):
        y = regen_proxy(y, text_embeds)
    return y.clamp(-1, 1)


def diffpure_attack(
    x: torch.Tensor,
    n_steps: float,
    regen_proxy: Optional[nn.Module],
    text_embeds: Optional[torch.Tensor],
) -> torch.Tensor:
    """Full-step DiffPure-style purification (many DDIM steps / strong surrogate)."""
    steps = max(int(round(float(n_steps))), 1)
    if regen_proxy is not None and hasattr(regen_proxy, "n_steps") and text_embeds is not None:
        old = regen_proxy.n_steps
        old_t = getattr(regen_proxy, "t_start", None)
        try:
            regen_proxy.n_steps = steps
            if old_t is not None:
                regen_proxy.t_start = min(0.9, max(float(old_t), 0.6))
            return regen_proxy(x, text_embeds).clamp(-1, 1)
        finally:
            regen_proxy.n_steps = old
            if old_t is not None:
                regen_proxy.t_start = old_t
    # Placeholder path: stacked blur+noise approximating a long reverse process.
    y = x
    for i in range(min(steps, 16)):
        y = gaussian_blur(y, sigma=1.0 + 0.15 * i)
        y = torch.clamp(y + torch.randn_like(y) * 0.04, -1, 1)
    return y


class GaussianBlurSurrogate(nn.Module):
    """VINE-style blur stand-in for the regenerative training branch.

    ``sigma`` is calibrated so the mean L2 perturbation matches a reference proxy.
    """

    def __init__(self, sigma: float = 1.2) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        _ = text_embeds
        return gaussian_blur(x_w, self.sigma)


class IdentityRegen(nn.Module):
    """No-op regen branch (``--regen-branch none``)."""

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        _ = text_embeds
        return x_w


@torch.no_grad()
def calibrate_blur_sigma(
    reference: nn.Module,
    image_size: int = 64,
    device: str = "cpu",
    n: int = 4,
) -> float:
    """Binary-search Gaussian sigma so mean L2 matches ``reference`` on noise images."""
    x = torch.rand(n, 3, image_size, image_size, device=device) * 2 - 1
    text = torch.zeros(n, 77, 768, device=device)
    try:
        y = reference(x, text)
        budget = (y - x).flatten(1).norm(dim=1).mean().item()
    except Exception:
        budget = 0.35
    lo, hi = 0.05, 8.0
    best = 1.2
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        z = gaussian_blur(x, mid)
        err = (z - x).flatten(1).norm(dim=1).mean().item()
        best = mid
        if err < budget:
            lo = mid
        else:
            hi = mid
    return float(best)


def apply_spec(
    spec: AttackSpec,
    x_w: torch.Tensor,
    *,
    strength: Optional[float] = None,
    message: Optional[torch.Tensor] = None,
    decoder: Optional[nn.Module] = None,
    regen_proxy: Optional[nn.Module] = None,
    text_embeds: Optional[torch.Tensor] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Apply a registry attack at ``strength`` (default = mid-level)."""
    config = config or {}
    s = default_strength(spec) if strength is None else float(strength)
    dj = config.get("diffjpeg", {})
    regen_cfg = config.get("regen", {})
    pgd = config.get("pgd", {})
    aid = spec.id

    if aid == "clean":
        return x_w
    if aid == "rotate":
        return rotate_attack(x_w, s)
    if aid == "crop":
        return center_crop_attack(x_w, s)
    if aid == "noise":
        return gaussian_noise_attack(x_w, s)
    if aid == "unseen_blur":
        return gaussian_blur_attack(x_w, s)
    if aid == "jpeg":
        return DiffJPEG(
            quality=int(s),
            use_real_diffjpeg=bool(dj.get("use_real_diffjpeg", False)),
            chroma_subsample=bool(dj.get("chroma_subsample", True)),
        )(x_w)
    if aid == "downsample":
        return downsample_attack(x_w, s)
    if aid == "regen":
        if regen_proxy is None or text_embeds is None:
            return alt_diffusion_attack(x_w, scales=3.0)
        old_t = getattr(regen_proxy, "t_start", None)
        try:
            if old_t is not None:
                regen_proxy.t_start = float(s)
            return regen_proxy(x_w, text_embeds)
        finally:
            if old_t is not None:
                regen_proxy.t_start = old_t
    if aid == "guided_regen":
        if regen_proxy is None or text_embeds is None or decoder is None or message is None:
            return alt_diffusion_attack(x_w, scales=3.0)
        return guided_regen_attack(
            x_w,
            decoder,
            message,
            regen_proxy,
            text_embeds,
            guidance_scale=float(s),
        )
    if aid == "diffpure":
        return diffpure_attack(x_w, s, regen_proxy, text_embeds)
    if aid in ("rinse_2", "rinse_4"):
        repeats = 2.0 if aid == "rinse_2" else 4.0
        return rinse_attack(x_w, repeats, regen_proxy, text_embeds)
    if aid == "vae_alt":
        return alt_vae_attack(x_w, s)
    if aid == "diffusion_alt":
        return alt_diffusion_attack(x_w, s)
    if aid == "adv":
        if decoder is None or message is None:
            raise ValueError("adv attack requires decoder and message")
        return pgd_attack_on_decoder(
            x_w,
            message,
            decoder,
            eps=float(s),
            alpha=float(pgd.get("alpha", 0.005)),
            steps=int(pgd.get("steps", 5)),
        )
    if aid == "semantic_inpaint":
        return semantic_inpaint_attack(x_w, s)
    raise ValueError(f"No implementation for attack id={aid}")


# Baseline scheme metadata (payload / resolution / weight provenance).
SCHEME_INFO: Dict[str, Dict[str, Any]] = {
    "ours": {
        "label_uk": "Запропонований метод",
        "payload_bits": None,  # filled from config msg_len
        "train_resolution": None,
        "weights": "local",
        "available": True,
    },
    "stegastamp": {
        "label_uk": "StegaStamp",
        "payload_bits": 100,
        "train_resolution": 400,
        "weights": "official",
        "available": False,
    },
    "trustmark": {
        "label_uk": "TrustMark",
        "payload_bits": 100,
        "train_resolution": 256,
        "weights": "official",
        "available": False,
    },
    "vine": {
        "label_uk": "VINE",
        "payload_bits": 100,
        "train_resolution": 256,
        "weights": "official",
        "available": False,
    },
    "stable_signature": {
        "label_uk": "Stable Signature",
        "payload_bits": 48,
        "train_resolution": 512,
        "weights": "official",
        "available": False,
    },
}

SCHEME_ORDER: Tuple[str, ...] = (
    "ours",
    "stegastamp",
    "trustmark",
    "vine",
    "stable_signature",
)
