"""Acceptance tests: iteration-level interruptible resume."""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders
from src.engine.train import train
from src.models import Decoder, Encoder
from src.utils.checkpoint import (
    find_latest_valid_checkpoint,
    is_valid_checkpoint,
    load_training_checkpoint,
)
from src.utils.logging import RunLogger
from src.utils.metrics import config_hash
from src.utils.seed import set_seed

DEVICE = "cpu"
MSG_LEN = 8
IMAGE_SIZE = 16
BATCH = 4
SEED = 123


def _base_cfg(run_dir: Path, **overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "seed": SEED,
        "device": DEVICE,
        "epochs": 2,
        "msg_len": MSG_LEN,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH,
        "num_workers": 0,
        "num_samples": 32,
        "val_fraction": 0.25,
        "lr": 1e-3,
        "lambda_perc": 0.0,
        "use_lpips": False,
        "val_every": 0,
        "ckpt_every_iters": 2,
        "amp": False,
        "regen_branch": "ddim_proxy",
        "dataset": "synthetic",
        "data_root": "./data",
        "encoder_ch": 16,
        "decoder_ch": 16,
        "encoder_strength": 0.4,
        "run_dir": str(run_dir),
        "resume": None,
        "curriculum": {"p0": 0.1, "kappa": 0.0, "p_max": 0.1, "p_adv": 0.2},
        "regen": {
            "use_placeholder": True,
            "guided_start_epoch": 999,
            "guided_prob": 0.0,
        },
        "pgd": {"eps": 0.02, "alpha": 0.005, "steps": 2},
        "eval": {"attacks": ["clean"]},
        "log": {"save_best_metric": "loss"},
    }
    cfg.update(overrides)
    return cfg


def _make_models(cfg: Dict[str, Any]):
    enc = Encoder(
        msg_len=int(cfg["msg_len"]),
        ch=int(cfg["encoder_ch"]),
        strength=float(cfg["encoder_strength"]),
    )
    dec = Decoder(msg_len=int(cfg["msg_len"]), ch=int(cfg["decoder_ch"]))
    return enc, dec


def _train_once(run_dir: Path, cfg: Dict[str, Any]):
    set_seed(int(cfg["seed"]))
    train_loader, val_loader = build_dataloaders(
        dataset_name=str(cfg["dataset"]),
        data_root=str(cfg["data_root"]),
        image_size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=0,
        num_samples=cfg.get("num_samples"),
        val_fraction=float(cfg["val_fraction"]),
        seed=int(cfg["seed"]),
    )
    logger = RunLogger(run_dir, cfg, append_metrics=True)
    enc, dec = _make_models(cfg)
    try:
        result = train(enc, dec, train_loader, val_loader, cfg, run_dir=run_dir, logger=logger)
    finally:
        logger.close()
    return result, enc, dec


def _state_close(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], tol: float = 1e-6) -> None:
    assert a.keys() == b.keys()
    for k in a:
        diff = (a[k].float() - b[k].float()).abs().max().item()
        assert diff <= tol, f"{k}: max abs diff {diff} > {tol}"


def test_interrupt_resume_matches_uninterrupted() -> None:
    """SIGTERM mid-epoch + --resume auto matches an uninterrupted run (weights)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        cfg_ref = _base_cfg(ref_dir)
        _, enc_ref, dec_ref = _train_once(ref_dir, cfg_ref)
        ref_e = {k: v.detach().cpu().clone() for k, v in enc_ref.state_dict().items()}
        ref_d = {k: v.detach().cpu().clone() for k, v in dec_ref.state_dict().items()}

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # ~8 train batches/epoch with 32 samples, 25% val → 24 train / 4 = 6 batches.
        # Interrupt in the middle of epoch 1 (0-based epoch 0).
        cfg_part = _base_cfg(run_dir, interrupt_after_step=3)
        result, _, _ = _train_once(run_dir, cfg_part)
        assert result["interrupted"] is True
        assert not (run_dir / "COMPLETED").exists()

        cfg_resume = _base_cfg(run_dir, resume="auto")
        # Drop test-only interrupt key so the second segment runs to completion.
        result2, enc2, dec2 = _train_once(run_dir, cfg_resume)
        assert result2["interrupted"] is False
        assert (run_dir / "COMPLETED").exists()
        _state_close(ref_e, enc2.state_dict())
        _state_close(ref_d, dec2.state_dict())
        print("[ok] interrupt+resume weights match uninterrupted within 1e-6")


def test_sigterm_handler_saves_and_resumes() -> None:
    """Real SIGTERM to a child trainer; resume auto completes; weights finite."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "sig_run"
        run_dir.mkdir()
        worker = ROOT / "tests" / "_resume_sigterm_worker.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [sys.executable, str(worker), str(run_dir)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Wait until at least one checkpoint exists, then SIGTERM.
        deadline = time.time() + 120
        ckpt_dir = run_dir / "checkpoints"
        while time.time() < deadline:
            if ckpt_dir.is_dir() and any(ckpt_dir.glob("iter_*.pt")):
                break
            if proc.poll() is not None:
                out = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
                raise RuntimeError(f"worker exited early ({proc.returncode}):\n{out}")
            time.sleep(0.2)
        else:
            proc.kill()
            raise RuntimeError("timeout waiting for first checkpoint")

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        assert any(ckpt_dir.glob("iter_*.pt")), "no checkpoint after SIGTERM"

        # Match the worker config (hash must agree); only flip resume.
        cfg = _base_cfg(
            run_dir,
            resume="auto",
            epochs=4,
            ckpt_every_iters=1,
        )
        result, enc, dec = _train_once(run_dir, cfg)
        assert result["interrupted"] is False
        assert (run_dir / "COMPLETED").exists()
        for p in list(enc.parameters()) + list(dec.parameters()):
            assert torch.isfinite(p).all()
        print("[ok] SIGTERM → resume auto completed")


def test_corrupted_latest_falls_back() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "corrupt"
        run_dir.mkdir()
        cfg = _base_cfg(run_dir, interrupt_after_step=4, ckpt_every_iters=2)
        _train_once(run_dir, cfg)
        latest = find_latest_valid_checkpoint(run_dir)
        # Truncate the newest numbered checkpoint.
        iters = sorted((run_dir / "checkpoints").glob("iter_*.pt"))
        assert len(iters) >= 1
        newest = iters[-1]
        with open(newest, "wb") as f:
            f.write(b"truncated")
        assert not is_valid_checkpoint(newest)
        fallback = find_latest_valid_checkpoint(run_dir)
        assert fallback != newest or len(iters) == 1
        if len(iters) >= 2:
            assert fallback == iters[-2] or is_valid_checkpoint(fallback)
            assert is_valid_checkpoint(fallback)
        # Resume should succeed via fallback.
        cfg2 = _base_cfg(run_dir, resume="auto")
        result, _, _ = _train_once(run_dir, cfg2)
        assert result["interrupted"] is False
        print(f"[ok] corrupted {newest.name} → fell back to {fallback.name}")


def test_config_hash_mismatch_refuses_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "hash"
        run_dir.mkdir()
        cfg = _base_cfg(run_dir, interrupt_after_step=2)
        _train_once(run_dir, cfg)
        ckpt_path = find_latest_valid_checkpoint(run_dir)
        bad_hash = "deadbeef0000"
        try:
            load_training_checkpoint(ckpt_path, expected_config_hash=bad_hash)
            raise AssertionError("expected RuntimeError for config hash mismatch")
        except RuntimeError as e:
            assert "Config hash mismatch" in str(e)
        # End-to-end via train()
        cfg_bad = _base_cfg(run_dir, resume="auto", lr=9.9e-3)
        try:
            _train_once(run_dir, cfg_bad)
            raise AssertionError("expected train() to refuse modified config")
        except RuntimeError as e:
            assert "Config hash mismatch" in str(e)
        print("[ok] modified config refuses resume")


def test_metrics_global_step_unique_after_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "metrics"
        run_dir.mkdir()
        cfg = _base_cfg(run_dir, interrupt_after_step=3, ckpt_every_iters=1)
        _train_once(run_dir, cfg)
        cfg2 = _base_cfg(run_dir, resume="auto", ckpt_every_iters=1)
        _train_once(run_dir, cfg2)
        path = run_dir / "metrics.csv"
        assert path.exists()
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        keys = [
            (r.get("global_step"), r.get("metric_name"), r.get("attack"), r.get("epoch"))
            for r in rows
            if r.get("global_step") not in (None, "")
        ]
        assert len(keys) == len(set(keys)), f"duplicate metric rows: {keys}"
        # Each global step should appear for loss exactly once.
        loss_steps = [
            r["global_step"]
            for r in rows
            if r.get("metric_name") == "loss" and r.get("attack") == "train"
        ]
        assert len(loss_steps) == len(set(loss_steps))
        print(f"[ok] metrics.csv unique global_steps ({len(set(loss_steps))} loss rows)")


if __name__ == "__main__":
    test_corrupted_latest_falls_back()
    test_config_hash_mismatch_refuses_resume()
    test_metrics_global_step_unique_after_resume()
    test_interrupt_resume_matches_uninterrupted()
    test_sigterm_handler_saves_and_resumes()
    print("=== all resume-checkpoint acceptance tests passed ===")
