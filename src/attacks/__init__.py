"""Attack modules: distortion, regeneration, PGD, and sampling."""

from .adversarial import pgd_attack_on_decoder, straight_through
from .distortion import DiffJPEG, DistortionBank
from .geometric import (
    GeometricCrop,
    GeometricRescale,
    GeometricRotate,
    GeometricTranslate,
)
from .regeneration import (
    RegenerationProxy,
    TextConditioner,
    build_regen_proxy,
    build_text_conditioner,
    encode_unconditional_text_embeds,
    apply_guided_pgd_ste,
    guided_hparams,
    guided_regen_attack,
    resolve_guided_start_epoch,
    should_apply_guided,
    make_text_embeds,
)
from .registry import (
    ATTACK_REGISTRY,
    GaussianBlurSurrogate,
    apply_spec,
    get_attack,
)
from .sampler import AttackSampler

__all__ = [
    "DiffJPEG",
    "DistortionBank",
    "GeometricCrop",
    "GeometricRescale",
    "GeometricRotate",
    "GeometricTranslate",
    "RegenerationProxy",
    "TextConditioner",
    "guided_regen_attack",
    "apply_guided_pgd_ste",
    "guided_hparams",
    "resolve_guided_start_epoch",
    "should_apply_guided",
    "build_regen_proxy",
    "build_text_conditioner",
    "encode_unconditional_text_embeds",
    "make_text_embeds",
    "pgd_attack_on_decoder",
    "straight_through",
    "AttackSampler",
    "ATTACK_REGISTRY",
    "GaussianBlurSurrogate",
    "apply_spec",
    "get_attack",
]
