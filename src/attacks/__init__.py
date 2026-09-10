"""Attack modules: distortion, regeneration, PGD, and sampling."""

from .adversarial import pgd_attack_on_decoder, straight_through
from .distortion import DiffJPEG, DistortionBank
from .regeneration import (
    RegenerationProxy,
    TextConditioner,
    build_regen_proxy,
    build_text_conditioner,
    guided_regen_attack,
    make_text_embeds,
)
from .sampler import AttackSampler

__all__ = [
    "DiffJPEG",
    "DistortionBank",
    "RegenerationProxy",
    "TextConditioner",
    "guided_regen_attack",
    "build_regen_proxy",
    "build_text_conditioner",
    "make_text_embeds",
    "pgd_attack_on_decoder",
    "straight_through",
    "AttackSampler",
]
