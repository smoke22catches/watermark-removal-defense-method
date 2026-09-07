#!/usr/bin/env python3
"""Smoke checks for DiffJPEG STE path and optional real SD / CLIP loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def check_diffjpeg_ste() -> None:
    from src.attacks.distortion import DiffJPEG

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    jpeg = DiffJPEG(quality=50, use_real_diffjpeg=True, chroma_subsample=True)
    y = jpeg(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "non-finite DiffJPEG output"
    grad = torch.autograd.grad(y.sum(), x)[0]
    assert grad is not None and torch.isfinite(grad).all(), "STE grads missing/non-finite"
    assert grad.abs().sum() > 0, "STE grads are all zero"
    print("[ok] DiffJPEG real path: shape, finite, STE grads")


def check_placeholder_regen() -> None:
    from src.attacks.regeneration import build_regen_proxy, make_text_embeds

    proxy = build_regen_proxy(use_placeholder=True)
    x = torch.randn(1, 3, 64, 64)
    te = make_text_embeds(1, "cpu")
    y = proxy(x, te)
    assert y.shape == x.shape
    print("[ok] Placeholder RegenerationProxy forward")


def check_real_sd(device: str, model_id: str, local_files_only: bool) -> None:
    from src.attacks.regeneration import (
        build_regen_proxy,
        build_text_conditioner,
        make_text_embeds,
    )

    print(f"[..] loading real regen + CLIP on {device} ({model_id}) …")
    proxy = build_regen_proxy(
        use_placeholder=False,
        sd_model_id=model_id,
        n_steps=2,
        t_start=0.3,
        device=device,
        local_files_only=local_files_only,
    )
    conditioner = build_text_conditioner(
        use_real_text_embeds=True,
        use_placeholder_regen=False,
        sd_model_id=model_id,
        prompt="",
        device=device,
        local_files_only=local_files_only,
    )
    x = torch.randn(1, 3, 128, 128, device=device)
    te = make_text_embeds(1, device, conditioner=conditioner)
    with torch.no_grad():
        y = proxy(x, te)
    assert y.shape == x.shape
    assert y.min() >= -1.0 - 1e-3 and y.max() <= 1.0 + 1e-3
    print("[ok] Real RegenerationProxy + CLIP text embeds forward")


def main() -> int:
    p = argparse.ArgumentParser(description="Validate placeholder implementations")
    p.add_argument("--skip-sd", action="store_true", help="Skip real SD/CLIP load test")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sd-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    check_diffjpeg_ste()
    check_placeholder_regen()

    if args.skip_sd:
        print("[skip] real SD/CLIP (--skip-sd)")
        return 0

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[skip] real SD/CLIP (CUDA unavailable)")
        return 0

    try:
        check_real_sd(device, args.sd_model_id, args.local_files_only)
    except Exception as e:
        print(f"[skip] real SD/CLIP unavailable: {e}")
        return 0

    print("=== all placeholder checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
