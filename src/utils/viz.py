"""Plotting and qualitative visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [-1, 1] (or [0, 1]) to HWC uint8 RGB."""
    x = t.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def plot_training_curves(
    metrics_csv: Path | str,
    out_dir: Path | str,
    loss_keys: Sequence[str] = ("loss", "loss_decode", "loss_perc"),
    bitacc_prefix: str = "bit_acc_",
) -> List[Path]:
    """Plot loss curves and per-attack bit-accuracy vs epoch from metrics.csv."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    saved: List[Path] = []

    if "epoch" not in df.columns:
        return saved

    # Loss curves
    present = [k for k in loss_keys if k in df.columns]
    if present:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in present:
            ax.plot(df["epoch"], df[k], label=k)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training losses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / "losses.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    # Bit-accuracy per attack
    bit_cols = [c for c in df.columns if c.startswith(bitacc_prefix)]
    if bit_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in bit_cols:
            ax.plot(df["epoch"], df[c], label=c.replace(bitacc_prefix, ""))
        ax.set_xlabel("epoch")
        ax.set_ylabel("bit accuracy")
        ax.set_title("Bit accuracy vs epoch")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / "bit_accuracy.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_eval_bit_accuracy(
    results: Mapping[str, float],
    out_path: Path | str,
    title: str = "Bit accuracy per attack",
) -> Path:
    """Bar chart of mean bit-accuracy per attack name."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    vals = [results[k] for k in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color="#2a6f97")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("bit accuracy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_qualitative_grid(
    original: torch.Tensor,
    watermarked: torch.Tensor,
    attacked: Mapping[str, torch.Tensor],
    out_path: Path | str,
    max_attacks: int = 4,
) -> Path:
    """Save a small grid: original | watermarked | attacked... | residual(wm-orig)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    residual = (watermarked - original).abs()
    residual = residual / residual.amax().clamp_min(1e-8)

    cols: List[tuple[str, torch.Tensor]] = [
        ("original", original),
        ("watermarked", watermarked),
    ]
    for name, t in list(attacked.items())[:max_attacks]:
        cols.append((name, t))
    cols.append(("|wm-orig|", residual))

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, (title, tensor) in zip(axes, cols):
        ax.imshow(_to_uint8_image(tensor))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
