#!/usr/bin/env python3
"""Validate the conda / CUDA / PyTorch environment for this project."""

from __future__ import annotations

import sys


def main() -> int:
    print("=== watermark-removal env check ===")
    print(f"Python: {sys.version}")

    try:
        import torch
    except ImportError as e:
        print(f"FAIL: cannot import torch ({e})")
        print("Create the env: conda env create -f environment.yml")
        return 1

    print(f"torch:          {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda compiled:  {torch.version.cuda}")
    print(f"cudnn:          {torch.backends.cudnn.version() if torch.cuda.is_available() else 'n/a'}")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"device count:   {torch.cuda.device_count()}")
        print(f"device name:    {torch.cuda.get_device_name(idx)}")
        cap = torch.cuda.get_device_capability(idx)
        print(f"capability:     sm_{cap[0]}{cap[1]}")
        # Ada RTX 4000-series is sm_89
        x = torch.randn(2, 3, device="cuda")
        y = x * 2
        print(f"cuda op test:   ok (sum={float(y.sum()):.3f})")
    else:
        print("device name:    cpu (no CUDA device visible)")
        print("TIP: ensure NVIDIA drivers + a CUDA 12.x PyTorch wheel are installed.")

    for pkg in ("torchvision", "torchaudio", "diffusers", "lpips", "skimage", "cv2", "yaml"):
        try:
            mod = __import__(pkg if pkg != "skimage" else "skimage")
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg}: {ver}")
        except Exception as e:
            print(f"  {pkg}: MISSING ({e})")

    print("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
