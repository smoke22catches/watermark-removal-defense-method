"""Atomic training checkpoints with RNG / sampler state for interruptible runs."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.utils.metrics import config_hash

ITER_CKPT_RE = re.compile(r"^iter_(\d+)\.pt$")
COMPLETED_MARKER = "COMPLETED"


def get_rng_state() -> Dict[str, Any]:
    """Capture python / NumPy / Torch CPU / CUDA RNG states."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None
    return state


def set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG states saved by :func:`get_rng_state`."""
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def atomic_torch_save(obj: Any, path: Path | str) -> None:
    """Write ``obj`` via a temporary file then ``os.replace`` (atomic on POSIX/Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def is_valid_checkpoint(path: Path | str) -> bool:
    """Return True if ``path`` loads as a dict with required training keys."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 64:
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    required = ("encoder", "decoder", "optimizer", "epoch", "global_step", "config_hash")
    return all(k in ckpt for k in required)


def list_iter_checkpoints(ckpt_dir: Path | str) -> List[Tuple[int, Path]]:
    """Return ``(global_step, path)`` pairs sorted ascending."""
    ckpt_dir = Path(ckpt_dir)
    out: List[Tuple[int, Path]] = []
    if not ckpt_dir.is_dir():
        return out
    for p in ckpt_dir.iterdir():
        m = ITER_CKPT_RE.match(p.name)
        if m and p.is_file():
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def find_latest_valid_checkpoint(run_dir: Path | str) -> Path:
    """Find the newest valid ``iter_*.pt`` (or ``last.pt``) under ``run_dir/checkpoints``.

    Corrupted / truncated newest files are skipped in favor of older ones.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints" if (run_dir / "checkpoints").is_dir() else run_dir
    candidates: List[Path] = [p for _, p in reversed(list_iter_checkpoints(ckpt_dir))]
    last = ckpt_dir / "last.pt"
    if last.is_file():
        # Prefer numbered iters; use last.pt only as fallback if no numbered exist.
        if not candidates:
            candidates = [last]
        elif last.resolve() not in {c.resolve() for c in candidates}:
            # Probe last.pt after numbered ones if they all fail.
            candidates.append(last)
    for path in candidates:
        if is_valid_checkpoint(path):
            return path
    raise FileNotFoundError(
        f"No valid checkpoint found under {ckpt_dir}. "
        "All iter_*.pt / last.pt files were missing or corrupted."
    )


def prune_iter_checkpoints(ckpt_dir: Path | str, keep: int = 2) -> None:
    """Keep the newest ``keep`` ``iter_*.pt`` files; never deletes ``best.pt``."""
    ckpt_dir = Path(ckpt_dir)
    files = list_iter_checkpoints(ckpt_dir)
    for _, path in files[:-keep] if keep > 0 else files:
        if path.name == "best.pt":
            continue
        try:
            path.unlink()
        except OSError:
            pass


def build_checkpoint_payload(
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    iter_in_epoch: int,
    global_step: int,
    lambda_perc: float,
    attack_sampler: Any,
    best_metric: float,
    config: Mapping[str, Any],
    cfg_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the full resumable training state dict."""
    sampler_state = None
    if attack_sampler is not None and hasattr(attack_sampler, "state_dict_resume"):
        sampler_state = attack_sampler.state_dict_resume()
    elif attack_sampler is not None:
        sampler_state = {
            "probs": tuple(getattr(attack_sampler, "probs", ())),
            "rng": attack_sampler._rng.get_state() if hasattr(attack_sampler, "_rng") else None,
        }
    payload: Dict[str, Any] = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "iter_in_epoch": int(iter_in_epoch),
        "global_step": int(global_step),
        "lambda_perc": float(lambda_perc),
        "attack_sampler": sampler_state,
        "rng": get_rng_state(),
        "best_metric": float(best_metric),
        "config_hash": cfg_hash if cfg_hash is not None else config_hash(config),
        "config": dict(config),
    }
    return payload


def save_training_checkpoint(
    run_dir: Path | str,
    payload: Mapping[str, Any],
    *,
    also_best: bool = False,
    keep_last: int = 2,
) -> Path:
    """Atomically save ``iter_{global_step}.pt`` and ``last.pt``; prune old iters."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = int(payload["global_step"])
    path = ckpt_dir / f"iter_{step:08d}.pt"
    atomic_torch_save(dict(payload), path)
    atomic_torch_save(dict(payload), ckpt_dir / "last.pt")
    if also_best:
        atomic_torch_save(dict(payload), ckpt_dir / "best.pt")
    prune_iter_checkpoints(ckpt_dir, keep=keep_last)
    return path


def load_training_checkpoint(
    path: Path | str,
    *,
    expected_config_hash: Optional[str] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint; raise if corrupt or config hash mismatches."""
    path = Path(path)
    if not is_valid_checkpoint(path):
        raise RuntimeError(f"Checkpoint is missing, truncated, or invalid: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if expected_config_hash is not None:
        got = str(ckpt.get("config_hash", ""))
        if got != str(expected_config_hash):
            raise RuntimeError(
                f"Config hash mismatch for resume: checkpoint has {got!r}, "
                f"current config has {expected_config_hash!r}. "
                "Refuse to resume with a modified config."
            )
    return ckpt


def write_completed_marker(run_dir: Path | str, payload: Optional[Mapping[str, Any]] = None) -> Path:
    """Write the ``COMPLETED`` marker file for external job runners."""
    run_dir = Path(run_dir)
    path = run_dir / COMPLETED_MARKER
    if payload:
        from src.utils.metrics import dump_json

        dump_json(path.with_suffix(".json"), payload)
    path.write_text("ok\n", encoding="utf-8")
    return path


def resolve_resume_checkpoint(
    resume: Any,
    run_dir: Optional[Path | str] = None,
) -> Optional[Path]:
    """Resolve ``resume`` CLI/config value to a checkpoint path (or None).

    ``auto`` → latest valid under ``run_dir``.
    Directory → latest valid under that directory.
    File path → that file (must be valid).
    """
    if resume in (None, "", False):
        return None
    resume_s = str(resume)
    if resume_s.lower() == "auto":
        if run_dir is None:
            raise ValueError("--resume auto requires a run directory (--run-dir)")
        return find_latest_valid_checkpoint(run_dir)
    path = Path(resume_s)
    if path.is_dir():
        return find_latest_valid_checkpoint(path)
    if path.is_file():
        if not is_valid_checkpoint(path):
            raise RuntimeError(f"Resume checkpoint is invalid or corrupted: {path}")
        return path
    raise FileNotFoundError(f"Resume path not found: {path}")
