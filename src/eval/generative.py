"""Held-out generative / codec evaluation transforms (not training approximations)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.transforms import gaussian_blur

# Must differ from the default training RegenerationProxy checkpoint.
DEFAULT_EVAL_DIFFUSION_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_TRAIN_DIFFUSION_ID = "runwayml/stable-diffusion-v1-5"


def assert_eval_diffusion_differs(train_id: str, eval_id: str) -> None:
    if str(train_id).strip() == str(eval_id).strip():
        raise AssertionError(
            f"Eval diffusion model must differ from training proxy: both are {train_id!r}"
        )


class EvalDiffPure(nn.Module):
    """Full-step diffusion purification with a *different* SD checkpoint than training."""

    def __init__(
        self,
        model_id: str = DEFAULT_EVAL_DIFFUSION_ID,
        train_model_id: str = DEFAULT_TRAIN_DIFFUSION_ID,
        n_steps: int = 50,
        t_start: float = 0.5,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        assert_eval_diffusion_differs(train_model_id, model_id)
        self.model_id = model_id
        self.train_model_id = train_model_id
        self.n_steps = int(n_steps)
        self.t_start = float(t_start)
        self.device = device
        self._proxy: Optional[nn.Module] = None
        self._local_files_only = local_files_only
        self._available = True
        self._error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._proxy is not None or not self._available:
            return
        try:
            from src.attacks.regeneration import build_regen_proxy

            self._proxy = build_regen_proxy(
                use_placeholder=False,
                sd_model_id=self.model_id,
                n_steps=self.n_steps,
                t_start=self.t_start,
                device=self.device,
                local_files_only=self._local_files_only,
                null_text_cache_dir=".cache/null_text_embeds_eval",
            )
        except Exception as e:
            self._available = False
            self._error = str(e)
            self._proxy = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, t_start: Optional[float] = None) -> torch.Tensor:
        self._ensure_loaded()
        if self._proxy is None:
            # Documented CPU/offline stand-in when eval SD weights are unavailable.
            return _offline_diffpure(x, n_steps=self.n_steps, t_start=t_start or self.t_start)
        old_t = getattr(self._proxy, "t_start", None)
        old_n = getattr(self._proxy, "n_steps", None)
        try:
            if t_start is not None and old_t is not None:
                self._proxy.t_start = float(t_start)
            if old_n is not None:
                self._proxy.n_steps = self.n_steps
            embeds = self._proxy.expand_text_embeds(x.size(0)).to(x.device)
            return self._proxy(x, embeds).clamp(-1, 1)
        finally:
            if old_t is not None:
                self._proxy.t_start = old_t
            if old_n is not None:
                self._proxy.n_steps = old_n


def _offline_diffpure(x: torch.Tensor, n_steps: int, t_start: float) -> torch.Tensor:
    """Blur+noise stack approximating long reverse diffusion (offline / no weights)."""
    y = x
    steps = max(int(n_steps), 1)
    strength = float(min(max(t_start, 0.05), 0.95))
    for i in range(min(steps, 24)):
        y = gaussian_blur(y, sigma=0.8 + 1.5 * strength + 0.1 * i)
        y = torch.clamp(y + torch.randn_like(y) * (0.02 + 0.05 * strength), -1, 1)
    return y.clamp(-1, 1)


@torch.no_grad()
def rinse(
    x: torch.Tensor,
    repeats: int,
    purifier: Optional[EvalDiffPure] = None,
    t_start: float = 0.4,
) -> torch.Tensor:
    """Apply diffusion purification ``repeats`` times (WAVES Regen-2x / 4x)."""
    n = max(int(repeats), 1)
    y = x
    for _ in range(n):
        if purifier is None:
            y = _offline_diffpure(y, n_steps=20, t_start=t_start)
        else:
            y = purifier(y, t_start=t_start)
    return y.clamp(-1, 1)


_COMPRESSAI_CACHE: dict[tuple[str, int], Any] = {}


def _load_compressai(model: str, quality: int, device: str):
    key = (model, int(quality))
    if key in _COMPRESSAI_CACHE:
        return _COMPRESSAI_CACHE[key]
    try:
        import compressai.zoo as zoo
    except ImportError as e:
        raise ImportError(
            "compressai is required for learned-codec eval attacks. "
            "Install with: pip install compressai"
        ) from e
    builders = {
        "bmshj2018": getattr(zoo, "bmshj2018_factorized", None) or getattr(zoo, "bmshj2018_hyperprior", None),
        "cheng2020": getattr(zoo, "cheng2020_anchor", None) or getattr(zoo, "cheng2020_attn", None),
    }
    builder = builders.get(model)
    if builder is None:
        raise ValueError(f"Unknown compressai model key: {model}")
    net = builder(quality=int(quality), pretrained=True).eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    _COMPRESSAI_CACHE[key] = net
    return net


@torch.no_grad()
def compressai_codec(
    x: torch.Tensor,
    model: str,
    quality: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """CompressAI autoencoder round-trip (architecture ≠ SD VAE)."""
    device = device or str(x.device)
    try:
        net = _load_compressai(model, quality, device)
    except Exception:
        # Offline stand-in: spatial bottleneck ≠ SD VAE latent path.
        return _offline_codec(x, quality=quality)

    # CompressAI expects [0, 1].
    x01 = (x.clamp(-1, 1) + 1) * 0.5
    out = net(x01.to(device))
    if isinstance(out, dict):
        y01 = out["x_hat"]
    else:
        y01 = out[0] if isinstance(out, (tuple, list)) else out
    return (y01 * 2 - 1).clamp(-1, 1).to(dtype=x.dtype, device=x.device)


def _offline_codec(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Shape-preserving stand-in when compressai weights are unavailable."""
    q = max(int(quality), 1)
    factor = max(2, 10 - min(q, 8))  # lower quality → coarser bottleneck
    z = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    y = F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)
    # Mild channel mix distinct from SD VAE.
    y = 0.7 * y + 0.3 * y.mean(dim=1, keepdim=True)
    return y.clamp(-1, 1)


def build_eval_purifier(config: Mapping[str, Any], device: str) -> EvalDiffPure:
    eval_cfg = config.get("eval", {}) if config else {}
    regen_cfg = config.get("regen", {}) if config else {}
    train_id = str(regen_cfg.get("sd_model_id", DEFAULT_TRAIN_DIFFUSION_ID))
    eval_id = str(eval_cfg.get("diffpure_model_id", DEFAULT_EVAL_DIFFUSION_ID))
    return EvalDiffPure(
        model_id=eval_id,
        train_model_id=train_id,
        n_steps=int(eval_cfg.get("diffpure_steps", 50)),
        t_start=float(eval_cfg.get("diffpure_t_start", 0.5)),
        device=device,
        local_files_only=bool(regen_cfg.get("local_files_only", False)),
    )
