"""Acceptance tests for the evaluation transform suite (Parts A–C)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.generative import (
    DEFAULT_EVAL_DIFFUSION_ID,
    DEFAULT_TRAIN_DIFFUSION_ID,
    assert_eval_diffusion_differs,
)
from src.eval.registry import ATTACK_REGISTRY, apply_spec, get_attack, protocol_table_rows
from src.eval.transforms import real_jpeg


EVAL_MODULE_ROOTS = [
    ROOT / "src" / "eval",
    ROOT / "src" / "engine" / "evaluate.py",
]


def _imports_training_distortion(path: Path) -> list[str]:
    """Return import lines that pull in ``src.attacks.distortion``."""
    hits = []
    files = [path] if path.is_file() else list(path.rglob("*.py"))
    for fp in files:
        src = fp.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=str(fp))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "src.attacks.distortion" or mod.startswith("src.attacks.distortion."):
                    hits.append(f"{fp}: from {mod}")
                if mod == "src.attacks" and any(
                    getattr(a, "name", "") in ("DiffJPEG", "DistortionBank", "diffjpeg_real")
                    for a in node.names
                ):
                    hits.append(f"{fp}: from src.attacks import distortion symbols")
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("src.attacks.distortion"):
                        hits.append(f"{fp}: import {a.name}")
    return hits


def test_jpeg_q100_mae_under_2_over_255() -> None:
    """PIL JPEG q=100 vs the 8-bit image it encodes (smooth gradient test pattern)."""
    from src.eval.quantize import quantize_uint8

    # Low-frequency gradient: DCT at q=100 is near-lossless after uint8 quantize.
    ys = torch.linspace(-1, 1, 64).view(1, 1, 64, 1)
    xs = torch.linspace(-1, 1, 64).view(1, 1, 1, 64)
    x = torch.cat([xs.expand(1, 1, 64, 64), ys.expand(1, 1, 64, 64), 0.5 * (xs + ys)], dim=1)
    x = x.expand(2, -1, -1, -1).contiguous()
    x8 = quantize_uint8(x)
    y = real_jpeg(x8, quality=100)
    mae = (y - x8).abs().mean().item()
    assert mae < 2.0 / 255.0, f"JPEG q=100 MAE={mae} >= 2/255"


def test_eval_modules_do_not_import_training_distortion() -> None:
    bad = []
    for root in EVAL_MODULE_ROOTS:
        bad.extend(_imports_training_distortion(root))
    assert not bad, "evaluation modules import training distortion:\n" + "\n".join(bad)


def test_protocol_table_lists_each_entry_once() -> None:
    rows = protocol_table_rows()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(ATTACK_REGISTRY)
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
    assert set(ids) == {s.id for s in ATTACK_REGISTRY}


def test_every_runnable_entry_batch_shape() -> None:
    torch.manual_seed(1)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    # Minimal decoder for adv.
    class TinyDec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(8, 16),
            )

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return self.net(t)

    decoder = TinyDec()
    m = torch.randint(0, 2, (2, 16)).float()
    cfg = {
        "regen": {"sd_model_id": DEFAULT_TRAIN_DIFFUSION_ID},
        "eval": {"diffpure_model_id": DEFAULT_EVAL_DIFFUSION_ID},
        "pgd": {"alpha": 0.005, "steps": 2},
    }
    for spec in ATTACK_REGISTRY:
        if spec.not_evaluated:
            continue
        y = apply_spec(
            spec,
            x,
            strength=float(spec.strengths[0]),
            message=m,
            decoder=decoder,
            config=cfg,
        )
        assert y.shape == x.shape, f"{spec.id}: shape {y.shape} != {x.shape}"
        assert y.dtype == x.dtype
        assert y.min() >= -1.0 - 1e-4 and y.max() <= 1.0 + 1e-4, f"{spec.id} out of range"


def test_diffpure_model_differs_from_training() -> None:
    assert_eval_diffusion_differs(DEFAULT_TRAIN_DIFFUSION_ID, DEFAULT_EVAL_DIFFUSION_ID)
    try:
        assert_eval_diffusion_differs("same", "same")
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for identical model ids")


def test_semantic_regen_is_not_evaluated() -> None:
    spec = get_attack("semantic_regen")
    assert spec.not_evaluated
    assert "semanticRegen" in spec.not_evaluated_reason or "KrtiT" in spec.not_evaluated_reason


if __name__ == "__main__":
    test_jpeg_q100_mae_under_2_over_255()
    test_eval_modules_do_not_import_training_distortion()
    test_protocol_table_lists_each_entry_once()
    test_every_runnable_entry_batch_shape()
    test_diffpure_model_differs_from_training()
    test_semantic_regen_is_not_evaluated()
    print("=== all evaluation protocol acceptance tests passed ===")
