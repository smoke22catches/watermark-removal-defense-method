"""Load baseline watermark schemes and report availability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.baselines.common import BaselineScheme


@dataclass
class SchemeLoadResult:
    scheme_id: str
    available: bool
    instance: Optional[BaselineScheme] = None
    reason: str = ""
    not_applicable: bool = False


def load_baseline(
    scheme_id: str,
    *,
    device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
) -> SchemeLoadResult:
    """Attempt to construct a baseline; never approximate with unofficial weights."""
    config = config or {}
    bl_cfg = config.get("baselines") or {}
    sid = scheme_id.lower().strip()

    if sid == "ours":
        return SchemeLoadResult(sid, True, reason="local Encoder/Decoder checkpoint")

    if sid in ("stable_signature", "tree_ring", "treering"):
        return SchemeLoadResult(
            sid if sid != "treering" else "tree_ring",
            available=False,
            not_applicable=True,
            reason=(
                "Not applicable to this comparison: the method watermarks images "
                "during generation and cannot embed a message into an arbitrary "
                "existing image (post-hoc)."
            ),
        )

    try:
        if sid == "stegastamp":
            from src.baselines.stegastamp_wm import StegaStampBaseline

            weights_dir = bl_cfg.get("stegastamp_weights_dir", "weights/stegastamp")
            inst = StegaStampBaseline(device=device, weights_dir=weights_dir)
            return SchemeLoadResult(sid, True, instance=inst)
        if sid == "trustmark":
            from src.baselines.trustmark_wm import TrustMarkBaseline

            inst = TrustMarkBaseline(
                device=device,
                model_type=str(bl_cfg.get("trustmark_variant", "Q")),
                use_ecc=bool(bl_cfg.get("trustmark_use_ecc", False)),
            )
            return SchemeLoadResult(sid, True, instance=inst)
        if sid == "vine":
            from src.baselines.vine_wm import VINEBaseline

            inst = VINEBaseline(
                device=device,
                encoder_id=str(bl_cfg.get("vine_encoder_id", "Shilin-LU/VINE-R-Enc")),
                decoder_id=str(bl_cfg.get("vine_decoder_id", "Shilin-LU/VINE-R-Dec")),
            )
            return SchemeLoadResult(sid, True, instance=inst)
    except Exception as e:
        return SchemeLoadResult(
            sid,
            available=False,
            reason=f"wrapper/weights failed to load: {e}",
        )

    return SchemeLoadResult(sid, False, reason=f"unknown scheme id '{sid}'")


def probe_availability(
    scheme_id: str,
    *,
    device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
) -> SchemeLoadResult:
    """Load then immediately close heavy resources (availability probe)."""
    res = load_baseline(scheme_id, device=device, config=config)
    if res.instance is not None:
        try:
            res.instance.close()
        except Exception:
            pass
        # Keep available=True but drop the live instance for the probe.
        return SchemeLoadResult(
            res.scheme_id,
            True,
            instance=None,
            reason=res.reason or "loaded successfully",
            not_applicable=res.not_applicable,
        )
    return res


def as_encoder_decoder(
    result: SchemeLoadResult,
) -> Tuple[Any, Any, int]:
    """Return ``(encoder, decoder, msg_len)`` adapters for ``evaluate_protocol``."""
    if not result.available or result.instance is None:
        raise RuntimeError(f"scheme '{result.scheme_id}' is not available: {result.reason}")
    from src.baselines.common import BaselineDecoderAdapter, BaselineEncoderAdapter

    enc = BaselineEncoderAdapter(result.instance)
    dec = BaselineDecoderAdapter(result.instance)
    return enc, dec, int(result.instance.msg_len)
