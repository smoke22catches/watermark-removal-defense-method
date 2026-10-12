"""Post-hoc baseline watermark schemes for comparative evaluation."""

from src.baselines.common import (
    BaselineDecoderAdapter,
    BaselineEncoderAdapter,
    BaselineScheme,
    WeightProvenance,
    sample_bits,
)
from src.baselines.factory import (
    SchemeLoadResult,
    as_encoder_decoder,
    load_baseline,
    probe_availability,
)

__all__ = [
    "BaselineScheme",
    "BaselineEncoderAdapter",
    "BaselineDecoderAdapter",
    "WeightProvenance",
    "sample_bits",
    "SchemeLoadResult",
    "load_baseline",
    "probe_availability",
    "as_encoder_decoder",
]
