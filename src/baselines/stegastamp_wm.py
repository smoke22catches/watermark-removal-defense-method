"""StegaStamp (Tancik et al., CVPR 2020) — official TensorFlow SavedModel weights."""

from __future__ import annotations

import tarfile
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
STEGA_WEIGHTS_URL = "http://people.eecs.berkeley.edu/~tancik/stegastamp/saved_models.tar.xz"
# Checksum filled after first successful download (also verified in docs/baselines.md).
STEGA_ARCHIVE_SHA256: Optional[str] = None
DEFAULT_WEIGHTS_DIR = Path("weights/stegastamp")


def try_import_tensorflow() -> Tuple[Optional[Any], Optional[str]]:
    try:
        import tensorflow as tf  # type: ignore

        return tf, None
    except Exception as e:
        return None, str(e)


def ensure_stegastamp_weights(
    weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
    *,
    expected_sha256: Optional[str] = STEGA_ARCHIVE_SHA256,
) -> Path:
    """Download and extract the official Berkeley ``saved_models.tar.xz`` if needed."""
    weights_dir = Path(weights_dir)
    model_dir = weights_dir / "stegastamp_pretrained"
    if (model_dir / "saved_model.pb").is_file() or list(model_dir.glob("*.pb")):
        return model_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    archive = weights_dir / "saved_models.tar.xz"
    if not archive.is_file():
        print(f"[stegastamp] downloading official weights from {STEGA_WEIGHTS_URL}")
        urllib.request.urlretrieve(STEGA_WEIGHTS_URL, filename=str(archive))
    digest = file_sha256(archive)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            f"StegaStamp archive sha256 mismatch: got {digest}, expected {expected_sha256}"
        )
    print(f"[stegastamp] archive sha256={digest}")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(path=weights_dir)
    # Official archive lays out ``saved_models/stegastamp_pretrained``.
    candidates = [
        weights_dir / "stegastamp_pretrained",
        weights_dir / "saved_models" / "stegastamp_pretrained",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"Could not locate stegastamp_pretrained under {weights_dir} after extraction"
    )


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
                "  pip install 'tensorflow>=2.12'\n"
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
        tf.compat.v1.disable_eager_execution()
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            meta = tf.compat.v1.saved_model.loader.load(
                self._sess,
                [tf.saved_model.SERVING],
                str(self.model_dir),
            )
            sig = meta.signature_def[
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            ]
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
                f"Weights from {STEGA_WEIGHTS_URL} (authors' README). "
                "Embed at 400×400 then residual-upsample to evaluation resolution "
                "(authors' encode_image.py saves residual for the same purpose)."
            ),
        )
