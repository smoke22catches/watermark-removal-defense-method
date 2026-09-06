"""Diffusion-based regeneration attack proxy and guided variant."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceholderRegenProxy(nn.Module):
    """Lightweight stand-in for RegenerationProxy used in dry-runs / CPU smoke tests.

    Applies mild noise + blur to approximate a regeneration-style degradation without
    loading Stable Diffusion weights.

    TODO: Replace with real RegenerationProxy (VAE + UNet + DDIM) when
    ``regen.use_placeholder=false`` and ``sd_model_id`` is available.
    """

    def __init__(self, noise_std: float = 0.05) -> None:
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        # text_embeds unused intentionally (placeholder API parity with RegenerationProxy)
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
    ) -> None:
        super().__init__()
        self.vae = vae.eval()
        self.unet = unet.eval()
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.t_start = t_start
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

    def forward(self, x_w: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(x_w).latent_dist.sample() * 0.18215
        t_idx = int(self.t_start * self.scheduler.config.num_train_timesteps)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(
            latents, noise, torch.tensor([t_idx], device=x_w.device)
        )

        self.scheduler.set_timesteps(self.n_steps, device=x_w.device)
        lat = noisy_latents
        for t in self.scheduler.timesteps:
            with torch.enable_grad():  # свідомо не no_grad — щоб градієнт міг протікати до E під час adv-тренування
                noise_pred = self.unet(lat, t, encoder_hidden_states=text_embeds).sample
                lat = self.scheduler.step(noise_pred, t, lat).prev_sample

        x_regen = self.vae.decode(lat / 0.18215).sample
        return torch.clamp(x_regen, -1.0, 1.0)


def build_regen_proxy(
    use_placeholder: bool = True,
    sd_model_id: str = "runwayml/stable-diffusion-v1-5",
    n_steps: int = 4,
    t_start: float = 0.3,
    device: str = "cpu",
) -> nn.Module:
    """Construct RegenerationProxy or its placeholder.

    TODO: When ``use_placeholder=False``, loads AutoencoderKL / UNet2DConditionModel /
    DDIMScheduler from ``sd_model_id`` (requires Hugging Face cache / network).
    """
    if use_placeholder:
        return PlaceholderRegenProxy()

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(sd_model_id, subfolder="scheduler")
    proxy = RegenerationProxy(vae, unet, scheduler, n_steps=n_steps, t_start=t_start)
    return proxy.to(device)


def make_text_embeds(
    batch_size: int,
    device: torch.device | str,
    seq_len: int = 77,
    dim: int = 768,
) -> torch.Tensor:
    """Create placeholder conditioning for the UNet / regen proxy.

    TODO: Replace with real CLIP / text-encoder embeddings from the SD pipeline.
    """
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
