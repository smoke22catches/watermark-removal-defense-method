#!/usr/bin/env python3
"""Inference / evaluation entrypoint for watermark Encoder/Decoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.regeneration import build_regen_proxy, build_text_conditioner, make_text_embeds
from src.data.datasets import build_dataloaders
from src.engine.evaluate import apply_attack, evaluate, evaluate_single_image
from src.models import Decoder, Encoder
from src.utils.config import apply_set_overrides, load_yaml
from src.utils.logging import RunLogger, create_run_dir
from src.utils.seed import set_seed
from src.utils.viz import plot_eval_bit_accuracy, save_qualitative_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate / infer watermark model")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--attacks",
        type=str,
        default=None,
        help="Comma-separated attack names (default: from config)",
    )
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--run-name", type=str, default="eval")
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--image", type=str, default=None, help="Single-image mode path")
    p.add_argument("--attack", type=str, default="clean", help="Attack for single-image mode")
    p.add_argument("--msg-len", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", dest="set_overrides", action="append", default=[])
    return p.parse_args()


def _load_models(ckpt_path: str, cfg: dict, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    msg_len = int(cfg.get("msg_len", ckpt.get("config", {}).get("msg_len", 64)))
    encoder = Encoder(msg_len=msg_len, ch=int(cfg.get("encoder_ch", 64)))
    decoder = Decoder(msg_len=msg_len, ch=int(cfg.get("decoder_ch", 64)))
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    return encoder.to(device), decoder.to(device), msg_len


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.device is not None:
        cfg["device"] = args.device
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.num_samples is not None:
        cfg["num_samples"] = args.num_samples
    if args.msg_len is not None:
        cfg["msg_len"] = args.msg_len
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.seed is not None:
        cfg["seed"] = args.seed
    apply_set_overrides(cfg, args.set_overrides)

    set_seed(int(cfg.get("seed", 42)))
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] CUDA unavailable; using CPU.")
        device = "cpu"
        cfg["device"] = device

    run_dir = create_run_dir(args.runs_root, str(cfg.get("run_name", "eval")))
    logger = RunLogger(run_dir, cfg)
    logger.log(f"Run directory: {run_dir}")

    encoder, decoder, msg_len = _load_models(args.checkpoint, cfg, device)
    regen_cfg = cfg.get("regen", {})
    use_placeholder = bool(regen_cfg.get("use_placeholder", True))
    sd_model_id = str(regen_cfg.get("sd_model_id", "runwayml/stable-diffusion-v1-5"))
    torch_dtype = regen_cfg.get("torch_dtype")
    local_files_only = bool(regen_cfg.get("local_files_only", False))

    regen_proxy = build_regen_proxy(
        use_placeholder=use_placeholder,
        sd_model_id=sd_model_id,
        n_steps=int(regen_cfg.get("n_steps", 4)),
        t_start=float(regen_cfg.get("t_start", 0.3)),
        device=device,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    ).to(device)

    text_conditioner = build_text_conditioner(
        use_real_text_embeds=bool(regen_cfg.get("use_real_text_embeds", False)),
        use_placeholder_regen=use_placeholder,
        sd_model_id=sd_model_id,
        prompt=str(regen_cfg.get("prompt", "")),
        device=device,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        text_embed_seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
        text_embed_dim=int(regen_cfg.get("text_embed_dim", 768)),
    )

    try:
        if args.image:
            tfm = transforms.Compose(
                [
                    transforms.Resize((int(cfg.get("image_size", 128)),) * 2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            img = tfm(Image.open(args.image).convert("RGB"))
            out = evaluate_single_image(
                encoder,
                decoder,
                img,
                device=device,
                msg_len=msg_len,
                attack=args.attack,
                regen_proxy=regen_proxy,
                config=cfg,
                text_conditioner=text_conditioner,
            )
            logger.log(
                f"bit_accuracy={out['bit_accuracy']:.4f}  "
                f"PSNR={out['psnr']:.2f}  SSIM={out['ssim']:.4f}"
            )
            logger.log(f"message  : {out['message'].int().tolist()}")
            logger.log(f"recovered: {out['recovered'].int().tolist()}")
            logger.log_metrics(
                {
                    "mode": "single_image",
                    "attack": args.attack,
                    "bit_accuracy": out["bit_accuracy"],
                    "psnr": out["psnr"],
                    "ssim": out["ssim"],
                }
            )
            save_qualitative_grid(
                out["x"],
                out["x_w"],
                {args.attack: out["x_test"]},
                run_dir / "plots" / "qualitative.png",
            )
        else:
            attacks = (
                [a.strip() for a in args.attacks.split(",") if a.strip()]
                if args.attacks
                else list(
                    cfg.get("eval", {}).get(
                        "attacks",
                        ["clean", "jpeg", "regen", "guided_regen", "unseen_blur"],
                    )
                )
            )
            _, val_loader = build_dataloaders(
                dataset_name=str(cfg.get("dataset", "synthetic")),
                data_root=str(cfg.get("data_root", "./data")),
                image_size=int(cfg.get("image_size", 128)),
                batch_size=int(cfg.get("batch_size", 4)),
                num_workers=int(cfg.get("num_workers", 0)),
                num_samples=cfg.get("num_samples"),
                val_fraction=float(cfg.get("val_fraction", 0.1)),
                seed=int(cfg.get("seed", 42)),
            )
            results = evaluate(
                encoder,
                decoder,
                val_loader,
                device=device,
                attacks=attacks,
                msg_len=msg_len,
                regen_proxy=regen_proxy,
                config=cfg,
                text_conditioner=text_conditioner,
            )
            means = {a: (sum(v) / len(v) if v else float("nan")) for a, v in results.items()}
            for a, m in means.items():
                logger.log_metrics({"attack": a, "bit_accuracy": m})
            plot_eval_bit_accuracy(means, run_dir / "plots" / "bit_accuracy_bars.png")

            # Qualitative grid from first batch
            batch = next(iter(val_loader))
            x = batch[0][:1].to(device)

            m = torch.randint(0, 2, (1, msg_len), device=device).float()
            with torch.no_grad():
                x_w = encoder(x, m)
            text_embeds = make_text_embeds(
                1,
                device,
                seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
                dim=int(regen_cfg.get("text_embed_dim", 768)),
                conditioner=text_conditioner,
            )
            attacked = {}
            for a in attacks[:4]:
                if a == "guided_regen":
                    with torch.enable_grad():
                        attacked[a] = apply_attack(
                            a, x_w, m, decoder, regen_proxy, text_embeds, cfg
                        ).detach().cpu()
                else:
                    with torch.no_grad():
                        attacked[a] = apply_attack(
                            a, x_w, m, decoder, regen_proxy, text_embeds, cfg
                        ).cpu()
            save_qualitative_grid(
                x.cpu(), x_w.cpu(), attacked, run_dir / "plots" / "qualitative.png"
            )
    finally:
        logger.close()

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
