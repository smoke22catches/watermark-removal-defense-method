"""VINE (Lu et al., ICLR 2025) wrapper — official Hugging Face weights."""

from __future__ import annotations

import types
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
    """Import VINE encoder/decoder classes from the official package layout.

    Official ``pip install git+https://github.com/Shilin-LU/VINE`` installs a
    ``vine`` package whose modules live under ``vine.src.*``. Older layouts (flat
    modules on ``sys.path``) are still tried for local checkouts.
    """
    errors: list[str] = []
    candidates = (
        ("vine.src.vine_turbo", "vine.src.stega_encoder_decoder"),
        ("vine.vine_turbo", "vine.stega_encoder_decoder"),
        ("vine_turbo", "stega_encoder_decoder"),
    )
    for enc_mod, dec_mod in candidates:
        try:
            import importlib

            VINE_Turbo = getattr(importlib.import_module(enc_mod), "VINE_Turbo")
            CustomConvNeXt = getattr(importlib.import_module(dec_mod), "CustomConvNeXt")
            return VINE_Turbo, CustomConvNeXt, None
        except Exception as e:
            errors.append(f"{enc_mod}/{dec_mod}: {e}")
    return None, None, "; ".join(errors)


def _patch_vine_turbo_batched_forward(encoder: torch.nn.Module) -> None:
    """Expand VINE's batch-1 timestep / text-embed caches to the input batch.

    Upstream ``VINE_Turbo.forward`` keeps ``timesteps`` and ``fixed_a2b_emb_base``
    at batch size 1. That works for single-image demos but breaks UNet attention
    when evaluation uses ``batch_size > 1`` (default 8): cross-attn / residual
    shapes disagree (e.g. 8192 vs 1024). Expand both tensors to ``B`` before the
    UNet call; keep the authors' per-sample scheduler loop otherwise unchanged.
    """

    def forward(self, x, secret=None, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps
        B = int(x.shape[0])
        if int(timesteps.shape[0]) != B:
            timesteps = timesteps[:1].expand(B)
        emb = self.fixed_a2b_emb_base
        if int(emb.shape[0]) != B:
            emb = emb[:1].expand(B, -1, -1)
        x_sec = self.sec_encoder(secret, x)
        x_enc = self.vae_enc(x_sec, direction="a2b").to(x.dtype)
        model_pred = self.unet(
            x_enc, timesteps, encoder_hidden_states=emb
        ).sample.to(x.dtype)
        x_out = torch.stack(
            [
                self.sched.step(
                    model_pred[i], timesteps[i], x_enc[i], return_dict=True
                ).prev_sample
                for i in range(B)
            ]
        )
        return self.vae_dec(x_out, direction="a2b").to(x.dtype)

    encoder.forward = types.MethodType(forward, encoder)


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
                "VINE package is not installed. Install via the project helper "
                "(avoids VINE's pinned torch==2.0.1 clobbering this env):\n"
                "  python scripts/install_baselines.py --scheme vine\n"
                "  # or manually: pip install --no-deps "
                "'git+https://github.com/Shilin-LU/VINE' && pip install peft\n"
                f"Import error: {err}"
            )
        self.device = device
        self.encoder_id = encoder_id
        self.decoder_id = decoder_id
        # HF config defaults device=cuda; override so CPU probing works.
        self._encoder = VINE_Turbo.from_pretrained(encoder_id, device=device).to(device).eval()
        _patch_vine_turbo_batched_forward(self._encoder)
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
