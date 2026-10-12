"""VINE (Lu et al., ICLR 2025) wrapper — official Hugging Face weights."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.baselines.common import (
    BaselineScheme,
    WeightProvenance,
    residual_transfer_embed,
    resize_nchw,
    to_unit,
)

VINE_SOURCE = "https://github.com/Shilin-LU/VINE"
VINE_LICENSE = "See repository LICENSE (authors' official release)"
VINE_ENC_ID = "Shilin-LU/VINE-R-Enc"
VINE_DEC_ID = "Shilin-LU/VINE-R-Dec"


def try_import_vine() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    try:
        from vine_turbo import VINE_Turbo  # type: ignore
        from stega_encoder_decoder import CustomConvNeXt  # type: ignore

        return VINE_Turbo, CustomConvNeXt, None
    except Exception as e:
        try:
            # Some installs expose the package as ``vine``.
            from vine.vine_turbo import VINE_Turbo  # type: ignore
            from vine.stega_encoder_decoder import CustomConvNeXt  # type: ignore

            return VINE_Turbo, CustomConvNeXt, None
        except Exception as e2:
            return None, None, f"{e}; {e2}"


class VINEBaseline(BaselineScheme):
    """VINE-R post-hoc watermark (100 bits, encode at 256, residual to eval size)."""

    name = "vine"
    msg_len = 100
    native_resolution = 256

    def __init__(
        self,
        device: str = "cpu",
        *,
        encoder_id: str = VINE_ENC_ID,
        decoder_id: str = VINE_DEC_ID,
    ) -> None:
        VINE_Turbo, CustomConvNeXt, err = try_import_vine()
        if VINE_Turbo is None or CustomConvNeXt is None:
            raise ImportError(
                "VINE package is not installed. Install the official release:\n"
                "  pip install 'git+https://github.com/Shilin-LU/VINE'\n"
                f"Import error: {err}"
            )
        self.device = device
        self.encoder_id = encoder_id
        self.decoder_id = decoder_id
        self._encoder = VINE_Turbo.from_pretrained(encoder_id).to(device).eval()
        self._decoder = CustomConvNeXt.from_pretrained(decoder_id).to(device).eval()

    def embed(self, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        if bits.shape[-1] != self.msg_len:
            raise ValueError(f"VINE expects msg_len={self.msg_len}, got {bits.shape[-1]}")

        def _native(x_n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return self._encoder(x_n, m.float()).clamp(-1, 1)

        return residual_transfer_embed(
            images,
            bits,
            native_resolution=self.native_resolution,
            embed_native_fn=_native,
        )

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        # Authors decode after resizing to 256 and mapping to [0, 1].
        x01 = to_unit(images)
        x256 = resize_nchw(x01, self.native_resolution, mode="bicubic")
        with torch.no_grad():
            pred = self._decoder(x256)
        pred = pred.view(pred.size(0), -1)[:, : self.msg_len].float()
        # Decoder outputs soft bits in ~[0, 1]; convert to logits for metrics.
        return (pred.clamp(0, 1) * 2.0 - 1.0) * 10.0

    def provenance(self) -> WeightProvenance:
        return WeightProvenance(
            scheme=self.name,
            source_url=VINE_SOURCE,
            license=VINE_LICENSE,
            version=f"encoder={self.encoder_id}; decoder={self.decoder_id}",
            weight_files={
                "encoder": self.encoder_id,
                "decoder": self.decoder_id,
            },
            notes=(
                "Official ICLR 2025 weights on Hugging Face. "
                "Embed at 256² then upsample residual to evaluation resolution "
                "(authors' watermark_encoding.py recipe)."
            ),
        )
