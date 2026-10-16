#!/usr/bin/env python3
"""Train regeneration-robust watermark Encoder/Decoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running as `python scripts/train.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders
from src.data.splits import DEFAULT_SPLIT_SEED, DEFAULT_SPLITS_DIR
from src.engine.train import export_algorithm_tex, train
from src.models import Decoder, Encoder
from src.utils.checkpoint import COMPLETED_MARKER
from src.utils.config import add_common_train_args, config_from_args, load_yaml
from src.utils.logging import RunLogger, create_run_dir
from src.utils.metrics import PROTOCOL_VERSION, config_hash, dump_json, git_commit_hash, utc_now_iso
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train watermark Encoder/Decoder")
    add_common_train_args(parser)
    parser.add_argument("--runs-root", type=str, default="runs")
    return parser.parse_args()


def _resolve_run_dir(args: argparse.Namespace, cfg: dict) -> Path:
    """Pick an existing run dir for resume, or create a new one."""
    resume = cfg.get("resume")
    run_dir_cfg = cfg.get("run_dir") or getattr(args, "run_dir", None)
    if run_dir_cfg:
        path = Path(run_dir_cfg)
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (path / "plots").mkdir(parents=True, exist_ok=True)
        return path
    if resume not in (None, "", False):
        resume_s = str(resume)
        if resume_s.lower() != "auto":
            p = Path(resume_s)
            if p.is_dir():
                return p
            if p.is_file():
                # checkpoints/iter_x.pt → run dir
                parent = p.parent
                return parent.parent if parent.name == "checkpoints" else parent
        raise SystemExit(
            "--resume auto (or a bare checkpoint path without --run-dir) requires "
            "--run-dir pointing at the existing run directory."
        )
    return create_run_dir(
        args.runs_root, str(cfg.get("run_name", "default")), seed=int(cfg.get("seed", 42))
    )


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    seed = int(cfg.get("seed", 42))
    cfg["seed"] = seed
    cfg.setdefault("regen_branch", "ddim_proxy")

    run_dir = _resolve_run_dir(args, cfg)
    cfg["run_dir"] = str(run_dir)

    # On resume, prefer the snapshot saved with the run so config hashes match.
    resume = cfg.get("resume")
    saved_cfg_path = run_dir / "config.yaml"
    if resume not in (None, "", False) and saved_cfg_path.is_file():
        saved = load_yaml(saved_cfg_path)
        keep_resume = cfg.get("resume")
        keep_device = cfg.get("device")
        saved["resume"] = keep_resume
        if keep_device is not None:
            saved["device"] = keep_device
        saved["run_dir"] = str(run_dir)
        saved["seed"] = seed
        cfg = saved

    if (run_dir / COMPLETED_MARKER).exists() and resume not in (None, "", False):
        print(f"Run already completed: {run_dir / COMPLETED_MARKER}")
        print("Skipping training.")
        return

    set_seed(seed)

    logger = RunLogger(
        run_dir,
        cfg,
        append_metrics=True,
        metrics_flush_every=int(cfg.get("log_every_iters", 50)),
    )
    logger.log(f"Run directory: {run_dir}")
    logger.log(f"seed={seed} regen_branch={cfg.get('regen_branch')} config_hash={config_hash(cfg)}")

    export_path = cfg.get("export_algorithm") or getattr(args, "export_algorithm", None)
    if export_path:
        out = export_algorithm_tex(cfg, export_path)
        logger.log(f"Wrote Algorithm 1 to {out}")
    export_algorithm_tex(cfg, run_dir / "algorithm1.tex")

    meta = {
        "seed": seed,
        "git_commit": git_commit_hash(ROOT),
        "config_hash": config_hash(cfg),
        "dataset": str(cfg.get("dataset", "synthetic")),
        "regen_branch": str(cfg.get("regen_branch", "ddim_proxy")),
        "start_ts": utc_now_iso(),
        "end_ts": None,
        "status": "running",
        "image_size": int(cfg.get("image_size", 128)),
        "msg_len": int(cfg.get("msg_len", 64)),
        "epochs": int(cfg.get("epochs", 100)),
        "scheme": "ours",
        "protocol_version": PROTOCOL_VERSION,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    dump_json(run_dir / "run_meta.json", meta)

    device = str(cfg.get("device", "cuda"))
    train_loader, val_loader = build_dataloaders(
        dataset_name=str(cfg.get("dataset", "synthetic")),
        data_root=str(cfg.get("data_root", "./data")),
        image_size=int(cfg.get("image_size", 128)),
        batch_size=int(cfg.get("batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 0)),
        num_samples=cfg.get("num_samples"),
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=seed,
        splits_dir=str(cfg.get("splits_dir", DEFAULT_SPLITS_DIR)),
        split_seed=int(cfg.get("split_seed", DEFAULT_SPLIT_SEED)),
        pin_memory=cfg.get("pin_memory"),
        persistent_workers=cfg.get("persistent_workers"),
        prefetch_factor=cfg.get("prefetch_factor"),
        device=device,
    )
    split_info = getattr(train_loader, "split_info", {}) or {}
    meta["splits"] = {
        "splits_dir": split_info.get("splits_dir"),
        "split_seed": split_info.get("split_seed"),
        "split_files": split_info.get("split_files"),
        "split_hashes": split_info.get("split_hashes"),
        "counts": split_info.get("counts"),
        "train_list": split_info.get("train_list", []),
        "test_list": split_info.get("test_list", []),
    }
    dump_json(run_dir / "run_meta.json", meta)
    logger.log(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}  "
        f"split_hashes={split_info.get('split_hashes')}"
    )

    msg_len = int(cfg.get("msg_len", 16))
    encoder = Encoder(
        msg_len=msg_len,
        ch=int(cfg.get("encoder_ch", 64)),
        strength=float(cfg.get("encoder_strength", 0.4)),
    )
    decoder = Decoder(msg_len=msg_len, ch=int(cfg.get("decoder_ch", 64)))

    try:
        result = train(
            encoder,
            decoder,
            train_loader,
            val_loader,
            cfg,
            run_dir=run_dir,
            logger=logger,
        )
        if result.get("interrupted"):
            meta["status"] = "interrupted"
        else:
            meta["status"] = "done"
    except Exception:
        meta["status"] = "failed"
        meta["end_ts"] = utc_now_iso()
        dump_json(run_dir / "run_meta.json", meta)
        raise
    finally:
        meta["end_ts"] = utc_now_iso()
        dump_json(run_dir / "run_meta.json", meta)
        logger.close()

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
