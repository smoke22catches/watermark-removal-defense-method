"""Adobe TrustMark wrapper (official PyPI / GitHub release)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.baselines.common import (
    BaselineScheme,
    WeightProvenance,
    residual_transfer_embed,
    resize_nchw,
)

# Documented in adobe/trustmark python/trustmark/trustmark.py (MIT).
TRUSTMARK_SOURCE = "https://github.com/adobe/trustmark"
TRUSTMARK_LICENSE = "MIT"
TRUSTMARK_HOST = "https://cai-watermark.adobe.net/watermarking/trustmark-models/"
TRUSTMARK_CHECKSUMS_Q = {
    "encoder_Q.ckpt": "700328b8754db934b2f6cb5e5185d81f",
    "decoder_Q.ckpt": "4ced90e9cfe13e3295ad082887fe9187",
    "trustmark_Q.yaml": "fe40df84a7feeebfceb7a7678d7e6ec6",
}


def try_import_trustmark() -> Tuple[Optional[Any], Optional[str]]:
    try:
        from trustmark import TrustMark  # type: ignore

        return TrustMark, None
    except Exception as e:
        return None, str(e)


class TrustMarkBaseline(BaselineScheme):
    """Post-hoc TrustMark-Q (100-bit payload, native encode 256).

    Uses the official Adobe neural encoder/decoder tensors directly (no BCH),
    so evaluation messages are raw 100-bit strings matching the shared sampler.
    Residual transfer follows ``TrustMark.encode`` (native residual → eval size).
    """

    name = "trustmark"
    msg_len = 100
    native_resolution = 256

    def __init__(
        self,
        device: str = "cpu",
        model_type: str = "Q",
        *,
        use_ecc: bool = False,
    ) -> None:
        _ = use_ecc  # ECC is intentionally bypassed for raw-bit protocol fairness
        TrustMark, err = try_import_trustmark()
        if TrustMark is None:
            raise ImportError(
                "TrustMark package is not installed. Install via:\n"
                "  python scripts/install_baselines.py --scheme trustmark\n"
                "  # or: pip install trustmark\n"
                "  # or: pip install 'git+https://github.com/adobe/trustmark.git#subdirectory=python'\n"
                f"Import error: {err}"
            )
        self.device = device
        self.model_type = model_type
        self._tm = TrustMark(
            verbose=False,
            model_type=model_type,
            use_ECC=False,
            secret_len=self.msg_len,
            device=device,
            loadRemover=False,
            loadBBoxDetector=False,
        )
        self._enc_res = int(self._tm.model_resolution_enc)
        self._dec_res = int(self._tm.model_resolution_dec)

    def embed(self, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        if bits.shape[-1] != self.msg_len:
            raise ValueError(f"TrustMark expects msg_len={self.msg_len}, got {bits.shape[-1]}")

        def _native(x_n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            # Official path: cover/secret → encoder → residual (see TrustMark.encode).
            cover = resize_nchw(x_n, self._enc_res).to(self.device)
            secret = m.float().to(self.device)
            with torch.no_grad():
                stego, _ = self._tm.encoder(cover, secret)
                stego = stego.clamp(-1, 1)
                residual = stego - cover
                residual = residual - residual.mean(dim=(2, 3), keepdim=True)
            if residual.shape[-2:] != x_n.shape[-2:]:
                residual = F.interpolate(
                    residual,
                    size=x_n.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return (x_n.to(self.device) + residual).clamp(-1, 1).to(dtype=x_n.dtype)

        return residual_transfer_embed(
            images,
            bits,
            native_resolution=self.native_resolution,
            embed_native_fn=_native,
        )

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        # Official subimage_decode resizes to model_resolution_dec then runs decoder.
        x = resize_nchw(images, self._dec_res).to(self.device)
        with torch.no_grad():
            logits = self._tm.decoder.decoder(x)
        logits = logits.view(logits.size(0), -1)[:, : self.msg_len].float()
        return logits.to(device=images.device, dtype=images.dtype)

    def provenance(self) -> WeightProvenance:
        weight_files: Dict[str, str] = dict(TRUSTMARK_CHECKSUMS_Q)
        try:
            import trustmark as tm_pkg  # type: ignore

            root = Path(tm_pkg.__file__).resolve().parent / "models"
            for name in list(weight_files):
                p = root / name
                if p.is_file():
                    from src.baselines.common import file_md5

                    weight_files[name] = file_md5(p)
        except Exception:
            pass
        ver = "unknown"
        try:
            import importlib.metadata as md

            ver = md.version("trustmark")
        except Exception:
            ver = "git+https://github.com/adobe/trustmark (python/)"
        return WeightProvenance(
            scheme=self.name,
            source_url=TRUSTMARK_SOURCE,
            license=TRUSTMARK_LICENSE,
            version=ver,
            weight_files=weight_files,
            notes=(
                f"Variant {self.model_type}; weights auto-downloaded from {TRUSTMARK_HOST} "
                "with Adobe-published MD5 checksums. Neural encode/decode without BCH so "
                "payload is raw 100 bits. Residual upsampled from 256² to eval size."
            ),
        )
