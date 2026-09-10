#!/usr/bin/env python3
"""Train regeneration-robust watermark Encoder/Decoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders
from src.engine.train import train
from src.models import Decoder, Encoder
from src.utils.config import add_common_train_args, config_from_args
from src.utils.logging import RunLogger, create_run_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train watermark Encoder/Decoder")
    add_common_train_args(parser)
    parser.add_argument("--runs-root", type=str, default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    set_seed(int(cfg.get("seed", 42)))

    run_dir = create_run_dir(args.runs_root, str(cfg.get("run_name", "default")))
    logger = RunLogger(run_dir, cfg)
    logger.log(f"Run directory: {run_dir}")

    train_loader, val_loader = build_dataloaders(
        dataset_name=str(cfg.get("dataset", "synthetic")),
        data_root=str(cfg.get("data_root", "./data")),
        image_size=int(cfg.get("image_size", 128)),
        batch_size=int(cfg.get("batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 0)),
        num_samples=cfg.get("num_samples"),
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", 42)),
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
    finally:
        logger.close()

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
