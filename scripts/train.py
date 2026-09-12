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
from src.engine.train import export_algorithm_tex, train
from src.models import Decoder, Encoder
from src.utils.config import add_common_train_args, config_from_args
from src.utils.logging import RunLogger, create_run_dir
from src.utils.metrics import config_hash, dump_json, git_commit_hash, utc_now_iso
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train watermark Encoder/Decoder")
    add_common_train_args(parser)
    parser.add_argument("--runs-root", type=str, default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    seed = int(cfg.get("seed", 42))
    cfg["seed"] = seed
    cfg.setdefault("regen_branch", "ddim_proxy")
    set_seed(seed)

    run_dir = create_run_dir(
        args.runs_root, str(cfg.get("run_name", "default")), seed=seed
    )
    logger = RunLogger(run_dir, cfg)
    logger.log(f"Run directory: {run_dir}")
    logger.log(f"seed={seed} regen_branch={cfg.get('regen_branch')} config_hash={config_hash(cfg)}")

    export_path = cfg.get("export_algorithm") or getattr(args, "export_algorithm", None)
    if export_path:
        out = export_algorithm_tex(cfg, export_path)
        logger.log(f"Wrote Algorithm 1 to {out}")
    # Always snapshot the algorithm next to the run as well.
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
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    dump_json(run_dir / "run_meta.json", meta)

    train_loader, val_loader = build_dataloaders(
        dataset_name=str(cfg.get("dataset", "synthetic")),
        data_root=str(cfg.get("data_root", "./data")),
        image_size=int(cfg.get("image_size", 128)),
        batch_size=int(cfg.get("batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 0)),
        num_samples=cfg.get("num_samples"),
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=seed,
    )
    logger.log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    msg_len = int(cfg.get("msg_len", 16))
    encoder = Encoder(
        msg_len=msg_len,
        ch=int(cfg.get("encoder_ch", 64)),
        strength=float(cfg.get("encoder_strength", 0.4)),
    )
    decoder = Decoder(msg_len=msg_len, ch=int(cfg.get("decoder_ch", 64)))

    try:
        train(
            encoder,
            decoder,
            train_loader,
            val_loader,
            cfg,
            run_dir=run_dir,
            logger=logger,
        )
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
