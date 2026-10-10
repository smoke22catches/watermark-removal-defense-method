"""Evaluation transform registry — single source of truth for sweeps and protocol tables.

Does **not** import ``src.attacks.distortion`` (training approximations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.eval.generative import (
    DEFAULT_EVAL_DIFFUSION_ID,
    DEFAULT_TRAIN_DIFFUSION_ID,
    EvalDiffPure,
    assert_eval_diffusion_differs,
    build_eval_purifier,
    compressai_codec,
    rinse,
)
from src.eval.transforms import (
    brightness,
    center_crop,
    contrast,
    downsample_upsample,
    erase,
    gaussian_blur,
    gaussian_noise,
    prepare_eval_input,
    real_jpeg,
    rotate,
)


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
    source: str = "own"  # WAVES | own | compressai | diffusers | semanticregen
    strength_key: str = "strength"
    not_evaluated: bool = False
    not_evaluated_reason: str = ""


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

# WAVES-aligned strengths where the paper defines them; see docs/evaluation_protocol.md.
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
        source="own",
    ),
    AttackSpec(
        id="rotate",
        label_uk="Поворот",
        cls="dist_geometric",
        implementation="src.eval.transforms.rotate",
        strengths=(5.0, 15.0, 30.0, 45.0, 90.0),
        epsilon=90.0,
        delta=0.15,
        knowledge_level="black",
        seen_in_training=True,
        source="WAVES",
        strength_key="angle_deg",
    ),
    AttackSpec(
        id="crop",
        label_uk="Обрізання (resized-crop)",
        cls="dist_geometric",
        implementation="src.eval.transforms.center_crop",
        strengths=(0.9, 0.75, 0.6, 0.5),
        epsilon=0.5,
        delta=0.2,
        knowledge_level="black",
        seen_in_training=True,
        source="WAVES",
        strength_key="keep_ratio",
    ),
    AttackSpec(
        id="erase",
        label_uk="Витирання",
        cls="dist_geometric",
        implementation="src.eval.transforms.erase",
        strengths=(0.05, 0.10, 0.20),
        epsilon=0.20,
        delta=0.15,
        knowledge_level="black",
        seen_in_training=False,
        source="WAVES",
        strength_key="frac",
    ),
    AttackSpec(
        id="brightness",
        label_uk="Яскравість",
        cls="dist_photometric",
        implementation="src.eval.transforms.brightness",
        strengths=(0.5, 0.75, 1.25, 1.5),
        epsilon=1.5,
        delta=0.12,
        knowledge_level="black",
        seen_in_training=False,
        source="WAVES",
        strength_key="factor",
    ),
    AttackSpec(
        id="contrast",
        label_uk="Контраст",
        cls="dist_photometric",
        implementation="src.eval.transforms.contrast",
        strengths=(0.5, 0.75, 1.25, 1.5),
        epsilon=1.5,
        delta=0.12,
        knowledge_level="black",
        seen_in_training=False,
        source="WAVES",
        strength_key="factor",
    ),
    AttackSpec(
        id="noise",
        label_uk="Гаусів шум",
        cls="dist_photometric",
        implementation="src.eval.transforms.gaussian_noise",
        strengths=(0.02, 0.05, 0.10),
        epsilon=0.10,
        delta=0.08,
        knowledge_level="black",
        seen_in_training=True,
        source="WAVES",
        strength_key="std",
    ),
    AttackSpec(
        id="unseen_blur",
        label_uk="Гаусів розмиття",
        cls="dist_photometric",
        implementation="src.eval.transforms.gaussian_blur",
        strengths=(0.8, 1.5, 3.0),
        epsilon=3.0,
        delta=0.12,
        knowledge_level="black",
        seen_in_training=False,
        source="WAVES",
        strength_key="sigma",
    ),
    AttackSpec(
        id="jpeg",
        label_uk="JPEG (PIL)",
        cls="dist_degradation",
        implementation="src.eval.transforms.real_jpeg",
        strengths=(90.0, 70.0, 50.0, 30.0, 10.0),
        epsilon=0.70,
        delta=0.20,
        knowledge_level="black",
        seen_in_training=True,
        source="WAVES",
        strength_key="quality",
    ),
    AttackSpec(
        id="downsample",
        label_uk="Даунсемпл–апсемпл",
        cls="dist_degradation",
        implementation="src.eval.transforms.downsample_upsample",
        strengths=(2.0, 4.0),
        epsilon=4.0,
        delta=0.18,
        knowledge_level="black",
        seen_in_training=True,
        source="own",
        strength_key="factor",
    ),
    AttackSpec(
        id="diffpure",
        label_uk="DiffPure (інша SD-модель)",
        cls="regen",
        implementation="src.eval.generative.EvalDiffPure",
        strengths=(0.2, 0.4, 0.6),
        epsilon=0.6,
        delta=0.40,
        knowledge_level="grey",
        seen_in_training=False,
        source="diffusers",
        strength_key="t_start",
    ),
    AttackSpec(
        id="rinse_2",
        label_uk="Промивання ×2",
        cls="regen",
        implementation="src.eval.generative.rinse",
        strengths=(2.0,),
        epsilon=0.6,
        delta=0.35,
        knowledge_level="grey",
        seen_in_training=False,
        source="WAVES",
        strength_key="repeats",
    ),
    AttackSpec(
        id="rinse_4",
        label_uk="Промивання ×4",
        cls="regen",
        implementation="src.eval.generative.rinse",
        strengths=(4.0,),
        epsilon=0.8,
        delta=0.45,
        knowledge_level="grey",
        seen_in_training=False,
        source="WAVES",
        strength_key="repeats",
    ),
    AttackSpec(
        id="bmshj2018",
        label_uk="CompressAI bmshj2018",
        cls="regen",
        implementation="src.eval.generative.compressai_codec",
        strengths=(1.0, 3.0, 5.0, 8.0),
        epsilon=0.7,
        delta=0.30,
        knowledge_level="grey",
        seen_in_training=False,
        source="compressai",
        strength_key="quality",
    ),
    AttackSpec(
        id="cheng2020",
        label_uk="CompressAI cheng2020",
        cls="regen",
        implementation="src.eval.generative.compressai_codec",
        strengths=(1.0, 3.0, 5.0, 6.0),
        epsilon=0.7,
        delta=0.30,
        knowledge_level="grey",
        seen_in_training=False,
        source="compressai",
        strength_key="quality",
    ),
    AttackSpec(
        id="adv",
        label_uk="PGD на декодер (сурогат)",
        cls="adv",
        implementation="src.attacks.adversarial.pgd_attack_on_decoder",
        strengths=(2 / 255, 4 / 255, 6 / 255, 8 / 255),
        epsilon=8 / 255,
        delta=0.10,
        knowledge_level="white",
        seen_in_training=True,
        source="WAVES",
        strength_key="eps",
    ),
    AttackSpec(
        id="semantic_regen",
        label_uk="SemanticRegen (Tallam et al.)",
        cls="sem",
        implementation="https://github.com/KrtiT/semanticRegen",
        strengths=(1.0,),
        epsilon=0.40,
        delta=0.50,
        knowledge_level="grey",
        seen_in_training=False,
        source="semanticregen",
        not_evaluated=True,
        not_evaluated_reason=(
            "Official code at https://github.com/KrtiT/semanticRegen (MIT) is a Jupyter "
            "notebook pipeline (VQA + segmentation + SD inpainting), not an importable "
            "library API. We do not invent a reimplementation; mark as not_evaluated."
        ),
    ),
)

_BY_ID: Dict[str, AttackSpec] = {s.id: s for s in ATTACK_REGISTRY}

# Lazily constructed eval DiffPure (different model than training).
_PURIFIER: Optional[EvalDiffPure] = None


def get_attack(attack_id: str) -> AttackSpec:
    if attack_id not in _BY_ID:
        raise KeyError(f"Unknown attack id: {attack_id}")
    return _BY_ID[attack_id]


def list_attack_ids() -> List[str]:
    return [s.id for s in ATTACK_REGISTRY]


def default_strength(spec: AttackSpec) -> float:
    return float(spec.strengths[len(spec.strengths) // 2])


def protocol_table_rows() -> List[Dict[str, Any]]:
    """One row per registry entry (for the paper protocol table)."""
    rows = []
    for spec in ATTACK_REGISTRY:
        rows.append(
            {
                "id": spec.id,
                "class": spec.cls,
                "knowledge_level": spec.knowledge_level,
                "strengths": list(spec.strengths),
                "seen_in_training": spec.seen_in_training,
                "source": spec.source,
                "not_evaluated": spec.not_evaluated,
                "not_evaluated_reason": spec.not_evaluated_reason,
                "implementation": spec.implementation,
                "epsilon": spec.epsilon,
                "delta": spec.delta,
                "label_uk": spec.label_uk,
            }
        )
    return rows


def _get_purifier(config: Mapping[str, Any], device: str) -> EvalDiffPure:
    global _PURIFIER
    if _PURIFIER is None:
        _PURIFIER = build_eval_purifier(config, device)
        regen_cfg = config.get("regen", {})
        train_id = str(regen_cfg.get("sd_model_id", DEFAULT_TRAIN_DIFFUSION_ID))
        assert_eval_diffusion_differs(train_id, _PURIFIER.model_id)
    return _PURIFIER


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
    """Apply a registry evaluation transform (always after 8-bit quantization)."""
    _ = (regen_proxy, text_embeds)  # training proxy must not be used for held-out regen
    config = config or {}
    if spec.not_evaluated:
        raise RuntimeError(
            f"Attack '{spec.id}' is marked not_evaluated: {spec.not_evaluated_reason}"
        )

    x = prepare_eval_input(x_w)
    s = default_strength(spec) if strength is None else float(strength)
    pgd = config.get("pgd", {})
    aid = spec.id
    device = str(x.device)

    if aid == "clean":
        return x
    if aid == "rotate":
        return rotate(x, s)
    if aid == "crop":
        return center_crop(x, s)
    if aid == "erase":
        return erase(x, s)
    if aid == "brightness":
        return brightness(x, s)
    if aid == "contrast":
        return contrast(x, s)
    if aid == "noise":
        return gaussian_noise(x, s)
    if aid == "unseen_blur":
        return gaussian_blur(x, s)
    if aid == "jpeg":
        return real_jpeg(x, int(round(s)))
    if aid == "downsample":
        return downsample_upsample(x, s)
    if aid == "diffpure":
        purifier = _get_purifier(config, device)
        return purifier(x, t_start=s)
    if aid == "rinse_2":
        return rinse(x, repeats=2, purifier=_get_purifier(config, device))
    if aid == "rinse_4":
        return rinse(x, repeats=4, purifier=_get_purifier(config, device))
    if aid == "bmshj2018":
        return compressai_codec(x, model="bmshj2018", quality=int(round(s)), device=device)
    if aid == "cheng2020":
        return compressai_codec(x, model="cheng2020", quality=int(round(s)), device=device)
    if aid == "adv":
        if decoder is None or message is None:
            raise ValueError("adv attack requires decoder and message")
        from src.attacks.adversarial import pgd_attack_on_decoder

        return pgd_attack_on_decoder(
            x,
            message,
            decoder,
            eps=float(s),
            alpha=float(pgd.get("alpha", 0.005)),
            steps=int(pgd.get("steps", 5)),
        )
    raise ValueError(f"No implementation for attack id={aid}")
