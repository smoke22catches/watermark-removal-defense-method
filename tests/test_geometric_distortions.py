"""Acceptance tests for differentiable geometric distortions."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.distortion import DistortionBank, resolve_distortion_probs
from src.attacks.geometric import (
    GeometricCrop,
    GeometricRescale,
    GeometricRotate,
    GeometricTranslate,
)

OUT_DIR = ROOT / "runs" / "geometric_distortion_viz"
TOL = 1e-5


def _checkerboard(batch: int = 2, size: int = 64) -> torch.Tensor:
    """Structured image so geometric changes are visible and grads are informative."""
    ys = torch.linspace(-1, 1, size).view(1, 1, size, 1)
    xs = torch.linspace(-1, 1, size).view(1, 1, 1, size)
    r = torch.sin(6.0 * math.pi * xs) * torch.cos(6.0 * math.pi * ys)
    g = torch.sin(4.0 * math.pi * (xs + ys))
    b = torch.cos(5.0 * math.pi * (xs - ys))
    img = torch.cat([r, g, b], dim=1).expand(batch, -1, -1, -1).contiguous()
    return img.clamp(-1, 1)


def test_identity_parameters() -> None:
    x = _checkerboard(batch=2, size=64)
    rotate = GeometricRotate(max_angle=15.0)
    crop = GeometricCrop(area_min=0.7, area_max=1.0)
    rescale = GeometricRescale(scale_min=0.8, scale_max=1.2)
    translate = GeometricTranslate(max_translate=0.05)

    y_rot = rotate(x, angles_deg=0.0)
    y_crop = crop(x, area_frac=1.0, offsets=torch.zeros(2, 2))
    y_scale = rescale(x, scales=1.0)
    y_tr = translate(x, shifts=torch.zeros(2, 2))

    for name, y in (
        ("rotate", y_rot),
        ("crop", y_crop),
        ("rescale", y_scale),
        ("translate", y_tr),
    ):
        err = (y - x).abs().max().item()
        assert err < TOL, f"{name} identity error {err} >= {TOL}"


def test_gradient_flows() -> None:
    x = _checkerboard(batch=2, size=32).requires_grad_(True)
    transforms = {
        "rotate": GeometricRotate(),
        "crop": GeometricCrop(),
        "rescale": GeometricRescale(),
        "translate": GeometricTranslate(),
    }
    for name, module in transforms.items():
        if x.grad is not None:
            x.grad = None
        y = module(x)
        y.sum().backward(retain_graph=True)
        assert x.grad is not None, f"{name}: grad is None"
        g = x.grad.detach()
        assert torch.isfinite(g).all(), f"{name}: non-finite grad"
        assert g.abs().sum().item() > 0.0, f"{name}: zero grad"


def test_per_sample_randomness() -> None:
    torch.manual_seed(0)
    x = _checkerboard(batch=2, size=64)
    # Force a non-identity regime so distinct params produce distinct outputs.
    rotate = GeometricRotate(max_angle=30.0)
    angles = rotate.sample_angles(2, x.device, x.dtype)
    assert angles[0].item() != angles[1].item()
    y = rotate(x, angles_deg=angles)
    assert not torch.allclose(y[0], y[1]), "rotate: batch samples identical"

    crop = GeometricCrop(area_min=0.5, area_max=0.9)
    areas = torch.tensor([0.5, 0.9], dtype=x.dtype)
    offsets = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=x.dtype)
    y_c = crop(x, area_frac=areas, offsets=offsets)
    assert not torch.allclose(y_c[0], y_c[1]), "crop: batch samples identical"

    rescale = GeometricRescale(scale_min=0.7, scale_max=1.3)
    scales = torch.tensor([0.7, 1.3], dtype=x.dtype)
    y_s = rescale(x, scales=scales)
    assert not torch.allclose(y_s[0], y_s[1]), "rescale: batch samples identical"

    translate = GeometricTranslate(max_translate=0.1)
    shifts = torch.tensor([[-0.1, -0.1], [0.1, 0.1]], dtype=x.dtype)
    y_t = translate(x, shifts=shifts)
    assert not torch.allclose(y_t[0], y_t[1]), "translate: batch samples identical"


def test_geometric_share_default_half() -> None:
    probs = resolve_distortion_probs(geometric_share=0.5)
    g = sum(probs[k] for k in ("rotate", "crop", "rescale", "translate"))
    c = sum(probs[k] for k in ("jpeg", "noise", "downsample"))
    assert abs(g - 0.5) < 1e-9
    assert abs(c - 0.5) < 1e-9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_save_visual_grid() -> None:
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = _checkerboard(batch=1, size=128)
    bank = DistortionBank(geometric_share=0.5)
    panels = [("original", x[0])]
    specs = [
        ("rotate", {"angles_deg": 12.0}),
        ("crop", {"area_frac": 0.75, "offsets": torch.tensor([[0.2, 0.3]])}),
        ("rescale", {"scales": 0.85}),
        ("translate", {"shifts": torch.tensor([[0.04, -0.03]])}),
        ("jpeg", {}),
        ("noise", {}),
        ("downsample", {}),
    ]
    torch.manual_seed(1)
    for name, params in specs:
        y = bank.apply_named(name, x, **params)
        assert y.min() >= -1.0 - 1e-5 and y.max() <= 1.0 + 1e-5
        panels.append((name, y[0].detach()))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4))
    for ax, (title, img) in zip(axes, panels):
        vis = ((img.clamp(-1, 1) + 1) * 0.5).permute(1, 2, 0).cpu().numpy()
        ax.imshow(vis)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("DistortionBank: original vs transforms", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "geometric_distortion_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.is_file(), f"missing visual grid at {out}"
    print(f"[ok] wrote visual grid → {out}")


def test_gradient_flow_dist_branch_still_passes() -> None:
    """Existing forced-distortion gradient-flow check still passes."""
    from tests.test_gradient_flow import _run_forced_branch

    norms = _run_forced_branch("dist")
    total = math.sqrt(sum(v * v for v in norms.values()))
    assert total > 0.0 and math.isfinite(total)
    print(f"[ok] forced dist-branch encoder grad norm={total:.6e}")


if __name__ == "__main__":
    test_identity_parameters()
    test_gradient_flows()
    test_per_sample_randomness()
    test_geometric_share_default_half()
    test_save_visual_grid()
    test_gradient_flow_dist_branch_still_passes()
    print("=== all geometric distortion acceptance tests passed ===")
