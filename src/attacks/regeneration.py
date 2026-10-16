"""Diffusion-based regeneration attack proxy and unconditional null-text conditioning."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_NULL_TEXT_CACHE_DIR = ".cache/null_text_embeds"


def _resolve_torch_dtype(name: Optional[str], device: str) -> torch.dtype:
    if name is not None:
        key = str(name).lower()
        if key in ("fp16", "float16", "half"):
            return torch.float16
        if key in ("fp32", "float32", "float"):
            return torch.float32
        if key in ("bf16", "bfloat16"):
            return torch.bfloat16
    if str(device).startswith("cuda"):
        return torch.float16
    return torch.float32


def _revision_key(sd_revision: Optional[str]) -> str:
    return str(sd_revision) if sd_revision not in (None, "") else "default"


def _dtype_key(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def null_text_cache_path(
    cache_dir: Union[str, Path],
    sd_model_id: str,
    sd_revision: Optional[str],
    dtype: torch.dtype,
) -> Path:
    """Disk path keyed by (model id, revision, dtype)."""
    digest = hashlib.sha1(
        f"{sd_model_id}|{_revision_key(sd_revision)}|{_dtype_key(dtype)}".encode("utf-8")
    ).hexdigest()[:16]
    safe = sd_model_id.replace("/", "__").replace("\\", "__")
    return Path(cache_dir) / f"null_text__{safe}__{_revision_key(sd_revision)}__{_dtype_key(dtype)}__{digest}.pt"


def encode_unconditional_text_embeds(
    sd_model_id: str = DEFAULT_SD_MODEL_ID,
    sd_revision: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    local_files_only: bool = False,
    cache_dir: Optional[Union[str, Path]] = DEFAULT_NULL_TEXT_CACHE_DIR,
) -> torch.Tensor:
    """Tokenize/encode empty prompt ``\"\"`` once; free the text encoder afterwards.

    Returns a ``[1, seq, dim]`` tensor on ``device`` in ``torch_dtype`` (UNet dtype).
    Optionally loads/saves a disk cache keyed by ``(sd_model_id, sd_revision, dtype)``.
    """
    device = torch.device(device)
    if torch_dtype is None:
        torch_dtype = _resolve_torch_dtype(None, str(device))

    rev_kw: dict[str, Any] = {}
    if sd_revision not in (None, ""):
        rev_kw["revision"] = str(sd_revision)

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = null_text_cache_path(cache_dir, sd_model_id, sd_revision, torch_dtype)
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(cached, torch.Tensor):
                raise RuntimeError(f"Corrupt null-text cache at {cache_path}")
            return cached.to(device=device, dtype=torch_dtype)

    try:
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as e:
        raise ImportError(
            "Encoding null text embeds requires the 'transformers' package. "
            "Install the project env or set regen.use_placeholder=true."
        ) from e

    tokenizer = CLIPTokenizer.from_pretrained(
        sd_model_id,
        subfolder="tokenizer",
        local_files_only=local_files_only,
        **rev_kw,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        sd_model_id,
        subfolder="text_encoder",
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        **rev_kw,
    )
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    text_encoder.to(device)

    try:
        with torch.no_grad():
            tokens = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            embeds = text_encoder(input_ids)[0].detach().to(dtype=torch_dtype).contiguous()
    finally:
        # Free text encoder (and tokenizer refs) — not needed after caching.
        del text_encoder
        del tokenizer
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeds.detach().cpu(), cache_path)

    return embeds


class PlaceholderRegenProxy(nn.Module):
    """Lightweight stand-in for RegenerationProxy used in dry-runs / CPU smoke tests.

    Applies mild noise + blur to approximate a regeneration-style degradation without
    loading Stable Diffusion weights. Used when ``regen.use_placeholder=true``.
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        seq_len: int = 77,
        embed_dim: int = 768,
        sd_model_id: str = DEFAULT_SD_MODEL_ID,
        sd_revision: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.sd_model_id = sd_model_id
        self.sd_revision = _revision_key(sd_revision)
        self.register_buffer(
            "null_text_embeds",
            torch.zeros(1, seq_len, embed_dim),
            persistent=False,
        )

    def expand_text_embeds(self, batch_size: int) -> torch.Tensor:
        return self.null_text_embeds.expand(batch_size, -1, -1)

    def forward(self, x_w: torch.Tensor, text_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = text_embeds
        noisy = x_w + torch.randn_like(x_w) * self.noise_std
        blurred = F.avg_pool2d(noisy, kernel_size=3, stride=1, padding=1)
        return torch.clamp(blurred, -1.0, 1.0)


class RegenerationProxy(nn.Module):
    """
    Диференційований проксі регенераційної атаки:
    x_w -> latent -> +noise(t) -> few-step DDIM denoise -> x_regen
    Це навчальний сурогат реальних атак (Zhao et al. regeneration attack; DiffPure).

    Conditioning is unconditional null-text: the empty prompt ``\"\"`` encoded once at
    startup and expanded per batch inside :meth:`forward`.
    """

    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        scheduler: Any,
        null_text_embeds: torch.Tensor,
        n_steps: int = 4,
        t_start: float = 0.3,
        scaling_factor: Optional[float] = None,
        sd_model_id: str = DEFAULT_SD_MODEL_ID,
        sd_revision: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.vae = vae.eval()
        self.unet = unet.eval()
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.t_start = t_start
        self.sd_model_id = sd_model_id
        self.sd_revision = _revision_key(sd_revision)
        if scaling_factor is None:
            scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))
        self.scaling_factor = float(scaling_factor)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if null_text_embeds.ndim != 3 or null_text_embeds.size(0) != 1:
            raise ValueError(
                f"null_text_embeds must have shape [1, seq, dim], got {tuple(null_text_embeds.shape)}"
            )
        cross_dim = int(getattr(getattr(unet, "config", None), "cross_attention_dim", null_text_embeds.size(-1)))
        if int(null_text_embeds.size(-1)) != cross_dim:
            raise ValueError(
                f"null_text_embeds last dim {null_text_embeds.size(-1)} != "
                f"unet.config.cross_attention_dim {cross_dim}"
            )
        self.register_buffer("null_text_embeds", null_text_embeds.detach().contiguous(), persistent=False)

    def expand_text_embeds(self, batch_size: int) -> torch.Tensor:
        """Expand cached empty-prompt embeds to ``batch_size`` (view, no copy)."""
        return self.null_text_embeds.expand(batch_size, -1, -1)

    def forward(self, x_w: torch.Tensor, text_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Prefer the proxy's cached unconditional embeds; allow override for tests.
        unet_dtype = next(self.unet.parameters()).dtype
        x_in = x_w.to(dtype=unet_dtype)
        if text_embeds is None:
            cond = self.expand_text_embeds(x_w.size(0)).to(dtype=unet_dtype, device=x_w.device)
        else:
            cond = text_embeds.to(dtype=unet_dtype, device=x_w.device)
            if cond.size(0) == 1 and x_w.size(0) != 1:
                cond = cond.expand(x_w.size(0), -1, -1)

        # Training needs grads through x_w; eval DiffPure/rinse must not force
        # enable_grad() (a 50-step × rinse_4 chain looked wedged / blew memory).
        step_ctx = torch.enable_grad() if x_w.requires_grad else torch.inference_mode()
        with step_ctx:
            latents = self.vae.encode(x_in).latent_dist.sample() * self.scaling_factor
            t_idx = int(self.t_start * self.scheduler.config.num_train_timesteps)
            noise = torch.randn_like(latents)
            timesteps = torch.tensor([t_idx], device=x_w.device, dtype=torch.long)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            self.scheduler.set_timesteps(self.n_steps, device=x_w.device)
            lat = noisy_latents
            for t in self.scheduler.timesteps:
                noise_pred = self.unet(lat, t, encoder_hidden_states=cond).sample
                lat = self.scheduler.step(noise_pred, t, lat).prev_sample

            x_regen = self.vae.decode(lat / self.scaling_factor).sample
        return torch.clamp(x_regen.float(), -1.0, 1.0)


def build_regen_proxy(
    use_placeholder: bool = True,
    sd_model_id: str = DEFAULT_SD_MODEL_ID,
    sd_revision: Optional[str] = None,
    n_steps: int = 4,
    t_start: float = 0.3,
    device: str = "cpu",
    torch_dtype: Optional[str] = None,
    local_files_only: bool = False,
    null_text_cache_dir: Optional[Union[str, Path]] = DEFAULT_NULL_TEXT_CACHE_DIR,
    text_embed_seq_len: int = 77,
    text_embed_dim: int = 768,
) -> nn.Module:
    """Construct RegenerationProxy or its placeholder.

    When ``use_placeholder=False``, loads AutoencoderKL / UNet2DConditionModel /
    DDIMScheduler and the empty-prompt embedding from the same ``sd_model_id`` /
    ``sd_revision`` checkpoint. The text encoder is freed after encoding.
    """
    if use_placeholder:
        return PlaceholderRegenProxy(
            seq_len=text_embed_seq_len,
            embed_dim=text_embed_dim,
            sd_model_id=sd_model_id,
            sd_revision=sd_revision,
        )

    try:
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    except ImportError as e:
        raise ImportError(
            "Real RegenerationProxy requires the 'diffusers' package. "
            "Install the project env (environment.yml) or set regen.use_placeholder=true."
        ) from e

    dtype = _resolve_torch_dtype(torch_dtype, device)
    load_kw: dict[str, Any] = {"local_files_only": local_files_only, "torch_dtype": dtype}
    if sd_revision not in (None, ""):
        load_kw["revision"] = str(sd_revision)

    try:
        vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", **load_kw)
        unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet", **load_kw)
        sched_kw = {"local_files_only": local_files_only}
        if sd_revision not in (None, ""):
            sched_kw["revision"] = str(sd_revision)
        scheduler = DDIMScheduler.from_pretrained(
            sd_model_id, subfolder="scheduler", **sched_kw
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Stable Diffusion components from '{sd_model_id}' "
            f"(revision={sd_revision!r}). "
            "Ensure network access / HF cache, run `huggingface-cli login` if needed, "
            "or set regen.use_placeholder=true for offline dry-runs. "
            f"Underlying error: {e}"
        ) from e

    null_embeds = encode_unconditional_text_embeds(
        sd_model_id=sd_model_id,
        sd_revision=sd_revision,
        device=device,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        cache_dir=null_text_cache_dir,
    )

    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    proxy = RegenerationProxy(
        vae,
        unet,
        scheduler,
        null_text_embeds=null_embeds,
        n_steps=n_steps,
        t_start=t_start,
        scaling_factor=scaling,
        sd_model_id=sd_model_id,
        sd_revision=sd_revision,
    )
    return proxy.to(device)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (text encoder is no longer kept resident).
# ---------------------------------------------------------------------------


class TextConditioner(nn.Module):
    """Thin wrapper around a cached null-text tensor (no resident text encoder)."""

    def __init__(
        self,
        null_text_embeds: torch.Tensor,
        sd_model_id: str = DEFAULT_SD_MODEL_ID,
        sd_revision: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sd_model_id = sd_model_id
        self.sd_revision = _revision_key(sd_revision)
        self.register_buffer("_cached", null_text_embeds.detach().contiguous(), persistent=False)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self._cached.expand(batch_size, -1, -1)

    @property
    def seq_len(self) -> int:
        return int(self._cached.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self._cached.shape[2])


def build_text_conditioner(
    use_real_text_embeds: bool = False,
    use_placeholder_regen: bool = True,
    sd_model_id: str = DEFAULT_SD_MODEL_ID,
    sd_revision: Optional[str] = None,
    prompt: str = "",
    device: str = "cpu",
    torch_dtype: Optional[str] = None,
    local_files_only: bool = False,
    text_embed_seq_len: int = 77,
    text_embed_dim: int = 768,
    null_text_cache_dir: Optional[Union[str, Path]] = DEFAULT_NULL_TEXT_CACHE_DIR,
) -> Optional[TextConditioner]:
    """Build a null-text conditioner, or ``None`` when placeholder zeros suffice.

    Prefer :func:`build_regen_proxy`, which attaches the same cache to the proxy.
    Empty prompt only; non-empty ``prompt`` is ignored (unconditional reconstruction).
    """
    _ = prompt
    need = bool(use_real_text_embeds) or (not use_placeholder_regen)
    if not need:
        return None
    dtype = _resolve_torch_dtype(torch_dtype, device)
    embeds = encode_unconditional_text_embeds(
        sd_model_id=sd_model_id,
        sd_revision=sd_revision,
        device=device,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        cache_dir=null_text_cache_dir,
    )
    _ = (text_embed_seq_len, text_embed_dim)
    return TextConditioner(embeds, sd_model_id=sd_model_id, sd_revision=sd_revision)


def make_text_embeds(
    batch_size: int,
    device: torch.device | str,
    seq_len: int = 77,
    dim: int = 768,
    conditioner: Optional[TextConditioner] = None,
    regen_proxy: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Create UNet conditioning embeds from the proxy/conditioner cache (or zeros)."""
    if regen_proxy is not None and hasattr(regen_proxy, "expand_text_embeds"):
        return regen_proxy.expand_text_embeds(batch_size).to(device)
    if conditioner is not None:
        return conditioner(batch_size).to(device)
    return torch.zeros(batch_size, seq_len, dim, device=device)


def resolve_guided_start_epoch(config: Mapping[str, Any]) -> int:
    """Epoch (0-based) when guided regen may start; default = ``epochs // 2``."""
    regen = config.get("regen", {}) or {}
    raw = regen.get("guided_start_epoch", None)
    if raw is None:
        return int(config.get("epochs", 100)) // 2
    return int(raw)


def guided_hparams(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolved guided-regeneration hyperparameters from config."""
    regen = config.get("regen", {}) or {}
    return {
        "guided_start_epoch": resolve_guided_start_epoch(config),
        "guided_prob": float(regen.get("guided_prob", 0.5)),
        "eps_guided": float(regen.get("eps_guided", 0.02)),
        "beta_guided": float(regen.get("beta_guided", 0.005)),
        "J_guided": int(regen.get("J_guided", 5)),
    }


def should_apply_guided(
    epoch: int,
    guided_start_epoch: int,
    guided_prob: float,
    generator: Optional[torch.Generator] = None,
) -> bool:
    """Bernoulli draw for guided PGD; always False before ``guided_start_epoch``."""
    if int(epoch) < int(guided_start_epoch):
        return False
    p = float(guided_prob)
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    if generator is None:
        return bool(torch.rand((), dtype=torch.float64).item() < p)
    return bool(torch.rand((), dtype=torch.float64, generator=generator).item() < p)


def apply_guided_pgd_ste(
    x_r: torch.Tensor,
    m: torch.Tensor,
    decoder: nn.Module,
    *,
    eps_guided: float = 0.02,
    beta_guided: float = 0.005,
    J_guided: int = 5,
) -> torch.Tensor:
    """PGD on decode loss around regenerated ``x_r``, with STE residual.

    Inner loop: signed-gradient ascent (``J_guided`` steps, size ``beta_guided``),
    project onto ``||· - x_r||_∞ ≤ eps_guided`` and ``[-1, 1]``. Forward uses the
    PGD image; backward flows through ``x_r`` via
    ``x_guided = x_r + (x_pgd - x_r).detach()``.
    """
    from src.attacks.adversarial import pgd_attack_on_decoder, straight_through

    x_pgd = pgd_attack_on_decoder(
        x_r,
        m,
        decoder,
        eps=float(eps_guided),
        alpha=float(beta_guided),
        steps=int(J_guided),
    )
    return straight_through(x_r, x_pgd)


def guided_regen_attack(
    x_w: torch.Tensor,
    decoder: nn.Module,
    m: torch.Tensor,
    regen_proxy: nn.Module,
    text_embeds: Optional[torch.Tensor] = None,
    guidance_scale: float = 2.0,
    eps_guided: float = 0.02,
    beta_guided: float = 0.005,
    J_guided: int = 5,
) -> torch.Tensor:
    """Eval-time guided regeneration: regen proxy then PGD around ``x_r``.

    ``guidance_scale`` is retained for CLI/config compatibility; when ``J_guided > 0``
    the multi-step PGD path is used (preferred). Returns a detached tensor.
    """
    if text_embeds is None and hasattr(regen_proxy, "expand_text_embeds"):
        text_embeds = regen_proxy.expand_text_embeds(x_w.size(0))
    x_r = regen_proxy(x_w, text_embeds)
    if int(J_guided) > 0:
        from src.attacks.adversarial import pgd_attack_on_decoder

        return pgd_attack_on_decoder(
            x_r.detach(),
            m,
            decoder,
            eps=float(eps_guided),
            alpha=float(beta_guided),
            steps=int(J_guided),
        )
    # Legacy single-step fallback.
    x_regen = x_r.detach().clone().requires_grad_(True)
    logits = decoder(x_regen)
    loss = F.binary_cross_entropy_with_logits(logits, m)
    grad = torch.autograd.grad(loss, x_regen)[0]
    step = float(guidance_scale) * 0.01
    x_regen = torch.clamp(x_regen + step * grad.sign(), -1, 1)
    return x_regen.detach()
