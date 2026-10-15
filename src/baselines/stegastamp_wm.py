"""StegaStamp (Tancik et al., CVPR 2020) — official TensorFlow SavedModel weights."""

from __future__ import annotations

import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.baselines.common import (
    BaselineScheme,
    WeightProvenance,
    file_sha256,
    residual_transfer_embed,
    resize_nchw,
    to_unit,
)

STEGA_SOURCE = "https://github.com/tancik/StegaStamp"
STEGA_LICENSE = "MIT"
# Authors' README historically pointed here; the Berkeley host now returns 404.
STEGA_WEIGHTS_URL = "http://people.eecs.berkeley.edu/~tancik/stegastamp/saved_models.tar.xz"
# Community mirror of the same TF SavedModel (not a PyTorch re-train).
STEGA_HF_REPO = "KingTechnician/stegastamp"
STEGA_HF_ALLOW = ["stegastamp_pretrained/**"]
# Checksum filled after first successful archive download (also verified in docs/baselines.md).
STEGA_ARCHIVE_SHA256: Optional[str] = None
DEFAULT_WEIGHTS_DIR = Path("weights/stegastamp")


def try_import_tensorflow() -> Tuple[Optional[Any], Optional[str]]:
    try:
        import tensorflow as tf  # type: ignore

        return tf, None
    except Exception as e:
        return None, str(e)


def _find_stegastamp_model_dir(weights_dir: Path) -> Optional[Path]:
    candidates = [
        weights_dir / "stegastamp_pretrained",
        weights_dir / "saved_models" / "stegastamp_pretrained",
    ]
    for c in candidates:
        if (c / "saved_model.pb").is_file() or list(c.glob("*.pb")):
            return c
    return None


def _download_stegastamp_from_hf(weights_dir: Path) -> Path:
    """Fetch the TF SavedModel folder from the Hugging Face community mirror."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download StegaStamp weights from the "
            f"HF mirror ({STEGA_HF_REPO}). Install it or place stegastamp_pretrained "
            f"under {weights_dir}."
        ) from e

    print(
        f"[stegastamp] Berkeley archive unavailable; downloading TF SavedModel "
        f"from Hugging Face ({STEGA_HF_REPO})"
    )
    snapshot_download(
        repo_id=STEGA_HF_REPO,
        allow_patterns=list(STEGA_HF_ALLOW),
        local_dir=str(weights_dir),
    )
    found = _find_stegastamp_model_dir(weights_dir)
    if found is None:
        raise FileNotFoundError(
            f"HF snapshot of {STEGA_HF_REPO} did not contain stegastamp_pretrained "
            f"under {weights_dir}"
        )
    return found


def ensure_stegastamp_weights(
    weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
    *,
    expected_sha256: Optional[str] = STEGA_ARCHIVE_SHA256,
) -> Path:
    """Download official StegaStamp TF SavedModel weights if needed.

    Prefers the authors' Berkeley ``saved_models.tar.xz``. When that host is down
    (404), falls back to the Hugging Face community mirror of the same SavedModel.
    """
    weights_dir = Path(weights_dir)
    existing = _find_stegastamp_model_dir(weights_dir)
    if existing is not None:
        return existing

    weights_dir.mkdir(parents=True, exist_ok=True)
    archive = weights_dir / "saved_models.tar.xz"
    berkeley_err: Optional[BaseException] = None

    if not archive.is_file():
        try:
            print(f"[stegastamp] downloading official weights from {STEGA_WEIGHTS_URL}")
            urllib.request.urlretrieve(STEGA_WEIGHTS_URL, filename=str(archive))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            berkeley_err = e
            if archive.is_file():
                archive.unlink()
            print(f"[stegastamp] Berkeley download failed: {e}")

    if archive.is_file():
        digest = file_sha256(archive)
        if expected_sha256 and digest != expected_sha256:
            raise RuntimeError(
                f"StegaStamp archive sha256 mismatch: got {digest}, expected {expected_sha256}"
            )
        print(f"[stegastamp] archive sha256={digest}")
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(path=weights_dir)
        found = _find_stegastamp_model_dir(weights_dir)
        if found is not None:
            return found
        print(
            "[stegastamp] archive extracted but stegastamp_pretrained not found; "
            "trying HF mirror"
        )

    try:
        return _download_stegastamp_from_hf(weights_dir)
    except Exception as hf_err:
        detail = f"Berkeley={berkeley_err!r}; HF={hf_err!r}" if berkeley_err else repr(hf_err)
        raise FileNotFoundError(
            "Could not obtain StegaStamp SavedModel weights. "
            f"Place stegastamp_pretrained/ under {weights_dir} manually. ({detail})"
        ) from hf_err


class StegaStampBaseline(BaselineScheme):
    """Official StegaStamp SavedModel (100-bit secret, native 400×400)."""

    name = "stegastamp"
    msg_len = 100
    native_resolution = 400

    def __init__(
        self,
        device: str = "cpu",
        *,
        weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
    ) -> None:
        tf, err = try_import_tensorflow()
        if tf is None:
            raise ImportError(
                "StegaStamp official weights are a TensorFlow SavedModel. "
                "Install TensorFlow to evaluate this baseline:\n"
                "  python scripts/install_baselines.py --scheme stegastamp\n"
                "  # or: pip install 'tensorflow>=2.12'\n"
                f"Import error: {err}"
            )
        self.tf = tf
        self.device = device
        self.model_dir = ensure_stegastamp_weights(weights_dir)
        self._archive_sha256 = (
            file_sha256(Path(weights_dir) / "saved_models.tar.xz")
            if (Path(weights_dir) / "saved_models.tar.xz").is_file()
            else ""
        )
        # TF1 SavedModel via compat.v1 session (as in authors' encode_image.py).
        # Use string tags/keys — TF2 removed tf.saved_model.signature_constants.
        tf.compat.v1.disable_eager_execution()
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            meta = tf.compat.v1.saved_model.loader.load(
                self._sess,
                ["serve"],
                str(self.model_dir),
            )
            sig = meta.signature_def["serving_default"]
            self._secret_name = sig.inputs["secret"].name
            self._image_name = sig.inputs["image"].name
            self._stego_name = sig.outputs["stegastamp"].name
            self._decoded_name = sig.outputs["decoded"].name
            self._secret_t = self._graph.get_tensor_by_name(self._secret_name)
            self._image_t = self._graph.get_tensor_by_name(self._image_name)
            self._stego_t = self._graph.get_tensor_by_name(self._stego_name)
            self._decoded_t = self._graph.get_tensor_by_name(self._decoded_name)

    def close(self) -> None:
        try:
            self._sess.close()
        except Exception:
            pass

    def _embed_numpy(self, images01: np.ndarray, secrets: np.ndarray) -> np.ndarray:
        # images01: NHWC float32 in [0, 1]; secrets: N×100 float32
        feed = {self._secret_t: secrets, self._image_t: images01}
        hidden = self._sess.run(self._stego_t, feed_dict=feed)
        return np.asarray(hidden, dtype=np.float32)

    def _decode_numpy(self, images01: np.ndarray) -> np.ndarray:
        feed = {self._image_t: images01}
        # Some exports still require a dummy secret placeholder — try without first.
        try:
            decoded = self._sess.run(self._decoded_t, feed_dict=feed)
        except Exception:
            secrets = np.zeros((images01.shape[0], self.msg_len), dtype=np.float32)
            feed[self._secret_t] = secrets
            decoded = self._sess.run(self._decoded_t, feed_dict=feed)
        return np.asarray(decoded, dtype=np.float32)

    def embed(self, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        if bits.shape[-1] != self.msg_len:
            raise ValueError(f"StegaStamp expects msg_len={self.msg_len}, got {bits.shape[-1]}")

        def _native(x_n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            x01 = to_unit(x_n).detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            secrets = m.detach().cpu().numpy().astype(np.float32)
            hidden = self._embed_numpy(x01, secrets)
            t = torch.from_numpy(hidden).permute(0, 3, 1, 2).to(device=x_n.device, dtype=x_n.dtype)
            return t.clamp(0, 1) * 2.0 - 1.0

        return residual_transfer_embed(
            images,
            bits,
            native_resolution=self.native_resolution,
            embed_native_fn=_native,
        )

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        x_n = resize_nchw(images, self.native_resolution)
        x01 = to_unit(x_n).detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
        decoded = self._decode_numpy(x01)
        soft = torch.from_numpy(decoded).to(device=images.device, dtype=torch.float32)
        soft = soft.view(soft.size(0), -1)[:, : self.msg_len]
        return (soft.clamp(0, 1) * 2.0 - 1.0) * 10.0

    def provenance(self) -> WeightProvenance:
        return WeightProvenance(
            scheme=self.name,
            source_url=STEGA_SOURCE,
            license=STEGA_LICENSE,
            version="official saved_models/stegastamp_pretrained (Berkeley)",
            weight_files={
                "saved_models.tar.xz": self._archive_sha256 or "(compute on download)",
                "model_dir": str(self.model_dir).replace("\\", "/"),
            },
            notes=(
                f"Primary URL {STEGA_WEIGHTS_URL} (authors' README); "
                f"fallback HF mirror {STEGA_HF_REPO} (TF SavedModel only). "
                "Embed at 400×400 then residual-upsample to evaluation resolution "
                "(authors' encode_image.py saves residual for the same purpose)."
            ),
        )
