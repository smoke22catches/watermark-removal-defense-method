#!/usr/bin/env python3
"""Inference / evaluation entrypoint for watermark Encoder/Decoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.registry import (
    ATTACK_REGISTRY,
    SCHEME_INFO,
    apply_spec,
    default_strength,
    get_attack,
)
from src.attacks.regeneration import build_regen_proxy, build_text_conditioner, make_text_embeds
from src.data.datasets import build_dataloaders
from src.engine.evaluate import apply_attack, evaluate_protocol, evaluate_single_image
from src.engine.train import _build_regen_branch
from src.models import Decoder, Encoder
from src.utils.config import apply_set_overrides, load_yaml
from src.utils.logging import RunLogger, create_run_dir
from src.utils.metrics import (
    NOT_EVALUATED,
    PROTOCOL_VERSION,
    config_hash,
    dump_json,
    git_commit_hash,
    long_row,
    utc_now_iso,
)
from src.utils.seed import set_seed
from src.utils.viz import plot_eval_bit_accuracy, save_qualitative_grid

EVAL_METRICS = (
    "bit_accuracy",
    "tpr_at_0.1pct_fpr",
    "tpr_at_1pct_fpr",
    "psnr",
    "ssim",
    "lpips",
    "fid",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate / infer watermark model")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--attacks",
        type=str,
        default=None,
        help="Comma-separated attack ids (default: full registry)",
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
    p.add_argument(
        "--scheme",
        type=str,
        default="ours",
        choices=list(SCHEME_INFO.keys()),
        help="Watermark scheme evaluated under the same protocol",
    )
    p.add_argument(
        "--sweep-strengths",
        action="store_true",
        help="Evaluate every registry strength level with paired quality metrics",
    )
    p.add_argument(
        "--quality-only",
        action="store_true",
        help="Measure PSNR/SSIM/LPIPS/FID of watermarked vs cover (no attack)",
    )
    p.add_argument("--set", dest="set_overrides", action="append", default=[])
    return p.parse_args()


def _load_models(ckpt_path: str, cfg: dict, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    msg_len = int(cfg.get("msg_len", ckpt.get("config", {}).get("msg_len", 64)))
    encoder = Encoder(msg_len=msg_len, ch=int(cfg.get("encoder_ch", 64)))
    decoder = Decoder(msg_len=msg_len, ch=int(cfg.get("decoder_ch", 64)))
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    ckpt_cfg = ckpt.get("config") or {}
    if isinstance(ckpt_cfg, dict):
        # Training-run identity (regen-branch / architecture) wins over eval yaml defaults.
        for k in ("regen_branch", "msg_len", "image_size", "encoder_ch", "decoder_ch"):
            if k in ckpt_cfg:
                cfg[k] = ckpt_cfg[k]
        if "regen" in ckpt_cfg and isinstance(ckpt_cfg["regen"], dict):
            merged = dict(ckpt_cfg["regen"])
            merged.update(cfg.get("regen") or {})
            cfg["regen"] = merged
    return encoder.to(device), decoder.to(device), msg_len


def _not_evaluated_rows(
    seed: int,
    sweep: bool,
    quality_only: bool,
) -> List[Dict[str, Any]]:
    specs = [get_attack("clean")] if quality_only else list(ATTACK_REGISTRY)
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        strengths = spec.strengths if sweep else (default_strength(spec),)
        for strength in strengths:
            for metric in EVAL_METRICS:
                rows.append(
                    long_row(
                        epoch="",
                        seed=seed,
                        attack=spec.id,
                        strength=float(strength),
                        epsilon=spec.epsilon,
                        delta=spec.delta,
                        knowledge_level=spec.knowledge_level,
                        seen_flag="true" if spec.seen_in_training else "false",
                        metric_name=metric,
                        metric_value=NOT_EVALUATED,
                    )
                )
    return rows


def _write_meta(run_dir: Path, cfg: dict, args: argparse.Namespace, extra: Dict[str, Any]) -> None:
    info = SCHEME_INFO[args.scheme]
    payload = {
        "dataset": str(cfg.get("dataset", "synthetic")),
        "scheme": args.scheme,
        "scheme_label_uk": info["label_uk"],
        "checkpoint": args.checkpoint,
        "seed": int(cfg.get("seed", 42)),
        "protocol_version": PROTOCOL_VERSION,
        "payload_bits": int(cfg.get("msg_len", info.get("payload_bits") or 0)),
        "train_resolution": int(cfg.get("image_size", info.get("train_resolution") or 0)),
        "weights": info.get("weights"),
        "sweep_strengths": bool(args.sweep_strengths),
        "quality_only": bool(args.quality_only),
        "config_hash": config_hash(cfg),
        "git_commit": git_commit_hash(ROOT),
        "timestamp": utc_now_iso(),
        "regen_branch": str(cfg.get("regen_branch", "ddim_proxy")),
    }
    payload.update(extra)
    dump_json(run_dir / "results_meta.json", payload)
    dump_json(run_dir / "run_meta.json", payload)


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

    seed = int(cfg.get("seed", 42))
    cfg["seed"] = seed
    set_seed(seed)
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] CUDA unavailable; using CPU.")
        device = "cpu"
        cfg["device"] = device

    run_dir = create_run_dir(args.runs_root, str(cfg.get("run_name", "eval")), seed=seed)
    logger = RunLogger(run_dir, cfg, metrics_filename="results.csv")
    logger.log(f"Run directory: {run_dir}")

    scheme = args.scheme
    info = SCHEME_INFO[scheme]
    available = bool(info.get("available", False)) and scheme == "ours"
    if scheme != "ours":
        available = False  # official weights are not bundled

    if not available:
        logger.log(
            f"Scheme '{scheme}' is not available in this checkout "
            f"(weights={info.get('weights')}). Writing {NOT_EVALUATED} markers."
        )
        rows = _not_evaluated_rows(seed, args.sweep_strengths, args.quality_only)
        logger.log_metric_rows(rows)
        _write_meta(
            run_dir,
            cfg,
            args,
            {"status": NOT_EVALUATED, "reason": "weights_not_bundled"},
        )
        logger.close()
        print(f"Done. Results in {run_dir}")
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for --scheme ours")

    encoder, decoder, msg_len = _load_models(args.checkpoint, cfg, device)
    regen_cfg = cfg.get("regen", {})
    use_placeholder = bool(regen_cfg.get("use_placeholder", True))
    sd_model_id = str(regen_cfg.get("sd_model_id", "runwayml/stable-diffusion-v1-5"))
    torch_dtype = regen_cfg.get("torch_dtype")
    local_files_only = bool(regen_cfg.get("local_files_only", False))

    regen_proxy = _build_regen_branch(cfg, device)
    # Real DDIM eval proxy is still needed for regen/diffpure/rinse even if
    # the run was trained with blur_surrogate — fall back to the configured proxy.
    if str(cfg.get("regen_branch", "ddim_proxy")) != "ddim_proxy":
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
            try:
                spec = get_attack("unseen_blur" if args.attack == "unseen" else args.attack)
                seen = "true" if spec.seen_in_training else "false"
                eps, delta, k = spec.epsilon, spec.delta, spec.knowledge_level
            except KeyError:
                spec = None
                seen, eps, delta, k = "", "", "", ""
            logger.log_metric_rows(
                [
                    long_row(
                        seed=seed,
                        attack=args.attack,
                        strength=default_strength(spec) if spec else "",
                        epsilon=eps,
                        delta=delta,
                        knowledge_level=k,
                        seen_flag=seen,
                        metric_name="bit_accuracy",
                        metric_value=out["bit_accuracy"],
                    ),
                    long_row(
                        seed=seed,
                        attack=args.attack,
                        metric_name="psnr",
                        metric_value=out["psnr"],
                        epsilon=eps,
                        delta=delta,
                        knowledge_level=k,
                        seen_flag=seen,
                    ),
                    long_row(
                        seed=seed,
                        attack=args.attack,
                        metric_name="ssim",
                        metric_value=out["ssim"],
                        epsilon=eps,
                        delta=delta,
                        knowledge_level=k,
                        seen_flag=seen,
                    ),
                ]
            )
            save_qualitative_grid(
                out["x"],
                out["x_w"],
                {args.attack: out["x_test"]},
                run_dir / "plots" / "qualitative.png",
            )
            _write_meta(run_dir, cfg, args, {"mode": "single_image", "status": "done"})
        else:
            if args.attacks:
                attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
            elif args.quality_only:
                attacks = ["clean"]
            else:
                attacks = [s.id for s in ATTACK_REGISTRY]
            _, val_loader = build_dataloaders(
                dataset_name=str(cfg.get("dataset", "synthetic")),
                data_root=str(cfg.get("data_root", "./data")),
                image_size=int(cfg.get("image_size", 128)),
                batch_size=int(cfg.get("batch_size", 4)),
                num_workers=int(cfg.get("num_workers", 0)),
                num_samples=cfg.get("num_samples"),
                val_fraction=float(cfg.get("val_fraction", 0.1)),
                seed=seed,
            )
            rows = evaluate_protocol(
                encoder,
                decoder,
                val_loader,
                device=device,
                attacks=attacks,
                msg_len=msg_len,
                regen_proxy=regen_proxy,
                config=cfg,
                text_conditioner=text_conditioner,
                seed=seed,
                epoch="",
                sweep_strengths=bool(args.sweep_strengths) and not args.quality_only,
                compute_detection=not args.quality_only,
                compute_fid=True,
                compute_quality=True,
                quality_only=bool(args.quality_only),
            )
            logger.log_metric_rows(rows)
            means = {}
            for r in rows:
                if r.get("metric_name") == "bit_accuracy":
                    try:
                        means[str(r["attack"])] = float(r["metric_value"])
                    except (TypeError, ValueError):
                        pass
            if means:
                plot_eval_bit_accuracy(means, run_dir / "plots" / "bit_accuracy_bars.png")

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
                try:
                    spec = get_attack("unseen_blur" if a == "unseen" else a)
                except KeyError:
                    continue
                if spec.id in ("guided_regen", "adv"):
                    with torch.enable_grad():
                        attacked[a] = apply_spec(
                            spec, x_w, message=m, decoder=decoder,
                            regen_proxy=regen_proxy, text_embeds=text_embeds, config=cfg,
                        ).detach().cpu()
                else:
                    with torch.no_grad():
                        attacked[a] = apply_spec(
                            spec, x_w, message=m, decoder=decoder,
                            regen_proxy=regen_proxy, text_embeds=text_embeds, config=cfg,
                        ).cpu()
            save_qualitative_grid(
                x.cpu(), x_w.cpu(), attacked, run_dir / "plots" / "qualitative.png"
            )
            _write_meta(
                run_dir,
                cfg,
                args,
                {
                    "mode": "sweep",
                    "status": "done",
                    "attacks": attacks,
                    "n_rows": len(rows),
                },
            )
    finally:
        logger.close()

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
