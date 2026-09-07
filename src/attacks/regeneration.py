"""Diffusion-based regeneration attack proxy, CLIP conditioning, and guided variant."""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceholderRegenProxy(nn.Module):
    """Lightweight stand-in for RegenerationProxy used in dry-runs / CPU smoke tests.

    Applies mild noise + blur to approximate a regeneration-style degradation without
    loading Stable Diffusion weights. Used when ``regen.use_placeholder=true``.
    """

    def __init__(self, noise_std: float = 0.05) -> None:
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        _ = text_embeds
        noisy = x_w + torch.randn_like(x_w) * self.noise_std
        blurred = F.avg_pool2d(noisy, kernel_size=3, stride=1, padding=1)
        return torch.clamp(blurred, -1.0, 1.0)


class RegenerationProxy(nn.Module):
    """
    Диференційований проксі регенераційної атаки:
    x_w -> latent -> +noise(t) -> few-step DDIM denoise -> x_regen
    Це навчальний сурогат реальних атак (Zhao et al. regeneration attack; DiffPure).
    """

    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        scheduler: Any,
        n_steps: int = 4,
        t_start: float = 0.3,
        scaling_factor: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.vae = vae.eval()
        self.unet = unet.eval()
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.t_start = t_start
        if scaling_factor is None:
            scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))
        self.scaling_factor = float(scaling_factor)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        # Match UNet / VAE dtype (often fp16 on CUDA).
        unet_dtype = next(self.unet.parameters()).dtype
        x_in = x_w.to(dtype=unet_dtype)
        text_embeds = text_embeds.to(dtype=unet_dtype, device=x_w.device)

        latents = self.vae.encode(x_in).latent_dist.sample() * self.scaling_factor
        t_idx = int(self.t_start * self.scheduler.config.num_train_timesteps)
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([t_idx], device=x_w.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        self.scheduler.set_timesteps(self.n_steps, device=x_w.device)
        lat = noisy_latents
        for t in self.scheduler.timesteps:
            with torch.enable_grad():  # свідомо не no_grad — щоб градієнт міг протікати до E під час adv-тренування
                noise_pred = self.unet(lat, t, encoder_hidden_states=text_embeds).sample
                lat = self.scheduler.step(noise_pred, t, lat).prev_sample

        x_regen = self.vae.decode(lat / self.scaling_factor).sample
        return torch.clamp(x_regen.float(), -1.0, 1.0)


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


def build_regen_proxy(
    use_placeholder: bool = True,
    sd_model_id: str = "runwayml/stable-diffusion-v1-5",
    n_steps: int = 4,
    t_start: float = 0.3,
    device: str = "cpu",
    torch_dtype: Optional[str] = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Construct RegenerationProxy or its placeholder.

    When ``use_placeholder=False``, loads AutoencoderKL / UNet2DConditionModel /
    DDIMScheduler from ``sd_model_id`` (requires ``diffusers`` + Hugging Face cache).
    """
    if use_placeholder:
        return PlaceholderRegenProxy()

    try:
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    except ImportError as e:
        raise ImportError(
            "Real RegenerationProxy requires the 'diffusers' package. "
            "Install the project env (environment.yml) or set regen.use_placeholder=true."
        ) from e

    dtype = _resolve_torch_dtype(torch_dtype, device)
    load_kw: dict[str, Any] = {"local_files_only": local_files_only, "torch_dtype": dtype}

    try:
        vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", **load_kw)
        unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet", **load_kw)
        scheduler = DDIMScheduler.from_pretrained(
            sd_model_id, subfolder="scheduler", local_files_only=local_files_only
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Stable Diffusion components from '{sd_model_id}'. "
            "Ensure network access / HF cache, run `huggingface-cli login` if needed, "
            "or set regen.use_placeholder=true for offline dry-runs. "
            f"Underlying error: {e}"
        ) from e

    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    proxy = RegenerationProxy(
        vae, unet, scheduler, n_steps=n_steps, t_start=t_start, scaling_factor=scaling
    )
    return proxy.to(device)


class TextConditioner(nn.Module):
    """Frozen CLIP text encoder producing UNet ``encoder_hidden_states``."""

    def __init__(
        self,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        prompt: str = "",
        device: Union[str, torch.device] = "cpu",
        torch_dtype: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as e:
            raise ImportError(
                "Real text embeds require 'transformers'. "
                "Install the project env or set regen.use_real_text_embeds=false."
            ) from e

        dtype = _resolve_torch_dtype(torch_dtype, str(device))
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                sd_model_id, subfolder="tokenizer", local_files_only=local_files_only
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                sd_model_id,
                subfolder="text_encoder",
                local_files_only=local_files_only,
                torch_dtype=dtype,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLIP text encoder from '{sd_model_id}'. "
                f"Underlying error: {e}"
            ) from e

        self.prompt = prompt
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder.to(device)

        # Cache unconditional / fixed-prompt embedding (1, 77, dim)
        self.register_buffer("_cached", self._encode_prompt(prompt), persistent=False)

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.text_encoder.device)
        out = self.text_encoder(input_ids)[0]
        return out

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return ``[B, seq, dim]`` embeddings (expanded from the cached prompt)."""
        assert self._cached is not None
        return self._cached.expand(batch_size, -1, -1).contiguous()

    @property
    def seq_len(self) -> int:
        return int(self._cached.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self._cached.shape[2])


def build_text_conditioner(
    use_real_text_embeds: bool = False,
    use_placeholder_regen: bool = True,
    sd_model_id: str = "runwayml/stable-diffusion-v1-5",
    prompt: str = "",
    device: str = "cpu",
    torch_dtype: Optional[str] = None,
    local_files_only: bool = False,
    text_embed_seq_len: int = 77,
    text_embed_dim: int = 768,
) -> Optional[TextConditioner]:
    """Build CLIP conditioner, or ``None`` to use zero placeholder embeds.

    CLIP is loaded when ``use_real_text_embeds=true``, or automatically whenever
    the real RegenerationProxy is used (``use_placeholder_regen=false``), since
    the SD UNet requires real ``encoder_hidden_states``.
    """
    _ = (text_embed_seq_len, text_embed_dim)
    need_clip = bool(use_real_text_embeds) or (not use_placeholder_regen)
    if not need_clip:
        return None
    return TextConditioner(
        sd_model_id=sd_model_id,
        prompt=prompt,
        device=device,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )


def make_text_embeds(
    batch_size: int,
    device: torch.device | str,
    seq_len: int = 77,
    dim: int = 768,
    conditioner: Optional[TextConditioner] = None,
) -> torch.Tensor:
    """Create UNet conditioning embeds (CLIP if ``conditioner`` else zeros)."""
    if conditioner is not None:
        embeds = conditioner(batch_size)
        return embeds.to(device)
    return torch.zeros(batch_size, seq_len, dim, device=device)


def guided_regen_attack(
    x_w: torch.Tensor,
    decoder: nn.Module,
    m: torch.Tensor,
    regen_proxy: nn.Module,
    text_embeds: torch.Tensor,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """
    Керована регенераційна атака: денойзинг ведеться не лише за prior дифузійної моделі,
    а й у напрямку, що явно псує вихід декодувальника (аналог guided diffusion removal).
    """
    x_regen = regen_proxy(x_w, text_embeds)
    x_regen = x_regen.clone().detach().requires_grad_(True)
    logits = decoder(x_regen)
    loss = -F.binary_cross_entropy_with_logits(logits, m)  # максимізуємо похибку
    grad = torch.autograd.grad(loss, x_regen)[0]
    x_regen = torch.clamp(x_regen - guidance_scale * 0.01 * grad.sign(), -1, 1)
    return x_regen.detach()
