"""Child process for SIGTERM resume tests. Args: <run_dir>."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders
from src.engine.train import train
from src.models import Decoder, Encoder
from src.utils.logging import RunLogger
from src.utils.seed import set_seed


def main() -> None:
    run_dir = Path(sys.argv[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    cfg = {
        "seed": 123,
        "device": "cpu",
        "epochs": 4,
        "msg_len": 8,
        "image_size": 16,
        "batch_size": 4,
        "num_workers": 0,
        "num_samples": 32,
        "val_fraction": 0.25,
        "lr": 1e-3,
        "lambda_perc": 0.0,
        "use_lpips": False,
        "val_every": 0,
        "ckpt_every_iters": 1,
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
        "regen": {"use_placeholder": True, "guided_start_epoch": 999, "guided_prob": 0.0},
        "pgd": {"eps": 0.02, "alpha": 0.005, "steps": 2},
        "eval": {"attacks": ["clean"]},
        "log": {"save_best_metric": "loss"},
    }
    set_seed(123)
    train_loader, val_loader = build_dataloaders(
        "synthetic",
        "./data",
        image_size=16,
        batch_size=4,
        num_workers=0,
        num_samples=32,
        val_fraction=0.25,
        seed=123,
        splits_dir=str(run_dir / "splits"),
        split_seed=20241010,
    )
    logger = RunLogger(run_dir, cfg, append_metrics=True)
    enc = Encoder(msg_len=8, ch=16, strength=0.4)
    dec = Decoder(msg_len=8, ch=16)
    try:
        train(enc, dec, train_loader, val_loader, cfg, run_dir=run_dir, logger=logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
