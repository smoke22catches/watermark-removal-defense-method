"""Acceptance tests for unconditional null-text conditioning of the regen proxy."""

from __future__ import annotations

import gc
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.regeneration import (
    RegenerationProxy,
    encode_unconditional_text_embeds,
    make_text_embeds,
)


CROSS_DIM = 8
SEQ_LEN = 4
MODEL_ID = "test-org/test-sd"
REVISION = "test-rev"


class _Cfg:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Cfg(scaling_factor=1.0)
        self.enc = nn.Conv2d(3, 4, 1)
        self.dec = nn.Conv2d(4, 3, 1)

    class _Dist:
        def __init__(self, t: torch.Tensor) -> None:
            self._t = t

        def sample(self) -> torch.Tensor:
            return self._t

    class _EncOut:
        def __init__(self, t: torch.Tensor) -> None:
            self.latent_dist = FakeVAE._Dist(t)

    def encode(self, x: torch.Tensor) -> "FakeVAE._EncOut":
        return FakeVAE._EncOut(self.enc(x))

    class _DecOut:
        def __init__(self, t: torch.Tensor) -> None:
            self.sample = t

    def decode(self, z: torch.Tensor) -> "FakeVAE._DecOut":
        return FakeVAE._DecOut(self.dec(z))


class FakeUNet(nn.Module):
    def __init__(self, cross_dim: int = CROSS_DIM) -> None:
        super().__init__()
        self.config = _Cfg(cross_attention_dim=cross_dim)
        self.proj = nn.Conv2d(4, 4, 1)

    def forward(self, lat: torch.Tensor, t: Any, encoder_hidden_states: torch.Tensor = None):
        assert encoder_hidden_states is not None
        assert encoder_hidden_states.shape[0] == lat.shape[0]
        assert encoder_hidden_states.shape[-1] == self.config.cross_attention_dim
        return types.SimpleNamespace(sample=self.proj(lat))


class FakeScheduler:
    def __init__(self) -> None:
        self.config = _Cfg(num_train_timesteps=1000)
        self.timesteps = torch.tensor([500, 250])

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        _ = timesteps
        return latents + 0.01 * noise

    def set_timesteps(self, n_steps: int, device: Any = None) -> None:
        self.timesteps = torch.linspace(800, 200, n_steps, device=device).long()

    def step(self, noise_pred: torch.Tensor, t: Any, lat: torch.Tensor) -> types.SimpleNamespace:
        _ = (noise_pred, t)
        return types.SimpleNamespace(prev_sample=lat)


class FakeTokenizer:
    model_max_length = SEQ_LEN

    def __init__(self, *a: Any, **k: Any) -> None:
        self.load_kwargs = k

    @classmethod
    def from_pretrained(cls, *a: Any, **k: Any) -> "FakeTokenizer":
        return cls(*a, **k)

    def __call__(self, text: str, **kwargs: Any) -> Any:
        assert text == ""
        _ = kwargs
        return types.SimpleNamespace(input_ids=torch.zeros(1, SEQ_LEN, dtype=torch.long))


class FakeTextEncoder(nn.Module):
    """Large-ish module so GPU memory drop is observable after free."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__()
        self.load_kwargs = k
        # ~8 MB at fp32 — enough to see a drop on CUDA after delete + empty_cache.
        self.weight = nn.Parameter(torch.randn(1024, 2048))

    @classmethod
    def from_pretrained(cls, *a: Any, **k: Any) -> "FakeTextEncoder":
        return cls(*a, **k)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        b, s = input_ids.shape
        out = torch.zeros(b, s, CROSS_DIM, device=input_ids.device, dtype=self.weight.dtype)
        out = out + self.weight.mean() * 0.0  # touch params so they stay allocated
        return (out,)


def _build_proxy(null_embeds: torch.Tensor) -> RegenerationProxy:
    return RegenerationProxy(
        FakeVAE(),
        FakeUNet(CROSS_DIM),
        FakeScheduler(),
        null_text_embeds=null_embeds,
        n_steps=2,
        t_start=0.3,
        sd_model_id=MODEL_ID,
        sd_revision=REVISION,
    )


def test_cached_last_dim_matches_unet_cross_attention_dim() -> None:
    null = torch.randn(1, SEQ_LEN, CROSS_DIM)
    proxy = _build_proxy(null)
    assert proxy.null_text_embeds.shape[-1] == proxy.unet.config.cross_attention_dim
    expanded = proxy.expand_text_embeds(4)
    assert expanded.shape == (4, SEQ_LEN, CROSS_DIM)
    # .expand is a view (no copy).
    assert expanded.data_ptr() == proxy.null_text_embeds.data_ptr()
    assert not hasattr(proxy, "text_encoder")


def test_proxy_forward_batch_sizes_1_and_4() -> None:
    null = torch.randn(1, SEQ_LEN, CROSS_DIM)
    proxy = _build_proxy(null)
    for b in (1, 4):
        x = torch.randn(b, 3, 16, 16, requires_grad=True)
        y = proxy(x)  # expands cached embeds internally
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None


def test_text_encoder_freed_from_gpu_memory() -> None:
    if not torch.cuda.is_available():
        print("[skip] CUDA unavailable for text-encoder memory test")
        return

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated(device)

    load_log: List[Dict[str, Any]] = []

    class TrackingTE(FakeTextEncoder):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "TrackingTE":
            load_log.append({"kind": "text_encoder", "model_id": model_id, **kwargs})
            m = cls()
            return m.to(device)

    class TrackingTok(FakeTokenizer):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "TrackingTok":
            load_log.append({"kind": "tokenizer", "model_id": model_id, **kwargs})
            return cls()

    with mock.patch.dict(
        sys.modules,
        {
            "transformers": types.SimpleNamespace(
                CLIPTextModel=TrackingTE,
                CLIPTokenizer=TrackingTok,
            )
        },
    ):
        # Force re-import path inside encode_* to see our fake transformers.
        embeds = encode_unconditional_text_embeds(
            sd_model_id=MODEL_ID,
            sd_revision=REVISION,
            device=device,
            torch_dtype=torch.float32,
            cache_dir=None,  # no disk — force encode + free
        )

    after = torch.cuda.memory_allocated(device)
    # Embeds remain (~small); text encoder weights must be gone.
    embed_bytes = embeds.numel() * embeds.element_size()
    # Residual after free should be near embed size, far below encoder (~8MB).
    assert after - before < 2 * 1024 * 1024, (
        f"GPU memory still elevated after freeing text encoder: "
        f"before={before}, after={after}, embeds={embed_bytes}"
    )
    assert embeds.shape[-1] == CROSS_DIM
    assert any(e["kind"] == "text_encoder" for e in load_log)


def test_tokenizer_text_encoder_unet_same_model_id_revision() -> None:
    """Tokenizer / text encoder / UNet must load from the same id + revision."""
    calls: List[Dict[str, Any]] = []

    class SpyTE(FakeTextEncoder):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "SpyTE":
            calls.append({"component": "text_encoder", "model_id": model_id, "kwargs": kwargs})
            return cls()

    class SpyTok(FakeTokenizer):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "SpyTok":
            calls.append({"component": "tokenizer", "model_id": model_id, "kwargs": kwargs})
            return cls()

    class SpyUNet(FakeUNet):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "SpyUNet":
            calls.append({"component": "unet", "model_id": model_id, "kwargs": kwargs})
            return cls()

    class SpyVAE(FakeVAE):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "SpyVAE":
            calls.append({"component": "vae", "model_id": model_id, "kwargs": kwargs})
            return cls()

    class SpySched:
        config = _Cfg(num_train_timesteps=1000)

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "SpySched":
            calls.append({"component": "scheduler", "model_id": model_id, "kwargs": kwargs})
            return cls()

    fake_diffusers = types.SimpleNamespace(
        AutoencoderKL=SpyVAE,
        UNet2DConditionModel=SpyUNet,
        DDIMScheduler=SpySched,
    )
    fake_transformers = types.SimpleNamespace(
        CLIPTextModel=SpyTE,
        CLIPTokenizer=SpyTok,
    )

    with mock.patch.dict(
        sys.modules,
        {"diffusers": fake_diffusers, "transformers": fake_transformers},
    ):
        from src.attacks.regeneration import build_regen_proxy

        proxy = build_regen_proxy(
            use_placeholder=False,
            sd_model_id=MODEL_ID,
            sd_revision=REVISION,
            n_steps=2,
            device="cpu",
            torch_dtype="float32",
            null_text_cache_dir=None,
        )

    by_comp = {c["component"]: c for c in calls}
    for name in ("vae", "unet", "tokenizer", "text_encoder"):
        assert name in by_comp, f"missing load for {name}: {calls}"
        assert by_comp[name]["model_id"] == MODEL_ID
        assert by_comp[name]["kwargs"].get("revision") == REVISION

    assert proxy.sd_model_id == MODEL_ID
    assert proxy.sd_revision == REVISION
    assert proxy.null_text_embeds.shape[-1] == CROSS_DIM

    # Shared helper used by train/eval expands the same cache.
    te = make_text_embeds(3, "cpu", regen_proxy=proxy)
    assert te.shape == (3, SEQ_LEN, CROSS_DIM)


if __name__ == "__main__":
    test_cached_last_dim_matches_unet_cross_attention_dim()
    test_proxy_forward_batch_sizes_1_and_4()
    test_text_encoder_freed_from_gpu_memory()
    test_tokenizer_text_encoder_unet_same_model_id_revision()
    print("=== all null-text embed acceptance tests passed ===")
