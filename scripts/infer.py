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
from src.attacks.regeneration import build_regen_proxy, make_text_embeds
from src.baselines.factory import as_encoder_decoder, load_baseline
from src.data.datasets import build_test_loader
from src.data.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLITS_DIR,
    verify_infer_no_train_overlap,
)
from src.engine.evaluate import (
    apply_attack,
    configure_eval_cuda,
    estimate_eval_workload,
    evaluate_protocol,
    evaluate_single_image,
)
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
    load_json,
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
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap test batches (overrides eval.max_batches). Full paper runs omit this.",
    )
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
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Smoke-eval preset: mid strengths only, skip DiffPure/rinse/compressai, "
            "cap at 32 batches (overridden by --max-batches / --attacks)"
        ),
    )
    p.add_argument("--set", dest="set_overrides", action="append", default=[])
    return p.parse_args()


def _checkpoint_run_dir(ckpt_path: str | Path) -> Optional[Path]:
    """``.../checkpoints/best.pt`` → run directory containing ``run_meta.json``."""
    p = Path(ckpt_path).resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent
    return p.parent if (p.parent / "run_meta.json").is_file() else None


def _load_models(ckpt_path: str, cfg: dict, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    msg_len = int(cfg.get("msg_len", ckpt.get("config", {}).get("msg_len", 64)))
    encoder = Encoder(
        msg_len=msg_len,
        ch=int(cfg.get("encoder_ch", 64)),
        strength=float(cfg.get("encoder_strength", 0.4)),
    )
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
    return encoder.to(device), decoder.to(device), msg_len, ckpt


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
        "msg_len": int(
            extra.get("msg_len")
            or (
                cfg.get("msg_len", info.get("payload_bits") or 0)
                if args.scheme == "ours"
                else info.get("payload_bits") or cfg.get("msg_len") or 0
            )
        ),
        "payload_bits": (
            int(cfg.get("msg_len", info.get("payload_bits") or 0))
            if args.scheme == "ours"
            else int(info.get("payload_bits") or cfg.get("msg_len") or 0)
        ),
        "native_resolution": int(
            extra.get("native_resolution")
            or info.get("train_resolution")
            or cfg.get("image_size")
            or 0
        ),
        "train_resolution": (
            int(cfg.get("image_size", info.get("train_resolution") or 0))
            if args.scheme == "ours"
            else int(info.get("train_resolution") or cfg.get("image_size") or 0)
        ),
        "weights": info.get("weights"),
        "weight_source": extra.get("weight_source"),
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

    eval_cfg = cfg.setdefault("eval", {})
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}
        cfg["eval"] = eval_cfg
    # Infer-only larger default batch (diffusion under-fills the GPU at train batch_size).
    if args.batch_size is None and eval_cfg.get("batch_size") is not None:
        cfg["batch_size"] = int(eval_cfg["batch_size"])
    if args.max_batches is not None:
        eval_cfg["max_batches"] = int(args.max_batches)
    if args.fast:
        # Keep protocol knobs intact unless the user did not override them.
        if args.attacks is None and not args.quality_only:
            args.attacks = (
                "clean,jpeg,rotate,crop,noise,unseen_blur,"
                "regen_proxy,guided_regen,adv"
            )
        if args.max_batches is None and eval_cfg.get("max_batches") is None:
            eval_cfg["max_batches"] = 32
        args.sweep_strengths = False
        print(
            "[infer] --fast: mid strengths only, max_batches="
            f"{eval_cfg.get('max_batches')}, attacks={args.attacks}",
            flush=True,
        )

    seed = int(cfg.get("seed", 42))
    cfg["seed"] = seed
    set_seed(seed)
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] CUDA unavailable; using CPU.")
        device = "cpu"
        cfg["device"] = device
    if device.startswith("cuda"):
        configure_eval_cuda()

    run_dir = create_run_dir(args.runs_root, str(cfg.get("run_name", "eval")), seed=seed)
    logger = RunLogger(run_dir, cfg, metrics_filename="results.csv")
    logger.log(f"Run directory: {run_dir}")

    scheme = args.scheme
    info = dict(SCHEME_INFO[scheme])
    cfg["scheme"] = scheme
    baseline_handle = None
    weight_meta: Dict[str, Any] = {}
    train_list: List[str] = []

    if scheme == "ours":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --scheme ours")
        encoder, decoder, msg_len, _ckpt = _load_models(args.checkpoint, cfg, device)
        train_run_dir = _checkpoint_run_dir(args.checkpoint)
        train_meta: Dict[str, Any] = {}
        if train_run_dir is not None and (train_run_dir / "run_meta.json").is_file():
            train_meta = load_json(train_run_dir / "run_meta.json")
        train_splits = train_meta.get("splits") or {}
        train_list = list(train_splits.get("train_list") or [])
        available = True
        reason = "local checkpoint"
        weight_meta = {
            "scheme": "ours",
            "source_url": "local",
            "license": "project",
            "version": "checkpoint",
            "weight_files": {"checkpoint": str(args.checkpoint)},
            "notes": "Proposed Encoder/Decoder",
        }
    else:
        load_res = load_baseline(scheme, device=device, config=cfg)
        available = bool(load_res.available)
        reason = load_res.reason
        if load_res.not_applicable:
            reason = load_res.reason or str(info.get("reason", "not_applicable"))
        if available and load_res.instance is not None:
            baseline_handle = load_res.instance
            encoder, decoder, msg_len = as_encoder_decoder(load_res)
            weight_meta = load_res.instance.provenance().as_dict()
            # Overlap check only applies when a training run_meta is provided.
            train_list = []
        else:
            encoder = decoder = None  # type: ignore[assignment]
            msg_len = int(info.get("payload_bits") or cfg.get("msg_len") or 0)

    if not available:
        logger.log(
            f"Scheme '{scheme}' is not available ({reason}). "
            f"Writing {NOT_EVALUATED} markers. See docs/baselines.md."
        )
        rows = _not_evaluated_rows(seed, args.sweep_strengths, args.quality_only)
        logger.log_metric_rows(rows)
        _write_meta(
            run_dir,
            cfg,
            args,
            {
                "status": NOT_EVALUATED,
                "reason": reason,
                "scheme": scheme,
                "msg_len": int(info.get("payload_bits") or 0),
                "native_resolution": info.get("train_resolution"),
                "weight_source": weight_meta or {"scheme": scheme},
            },
        )
        logger.close()
        print(f"Done. Results in {run_dir}")
        return

    regen_cfg = cfg.get("regen", {})
    use_placeholder = bool(regen_cfg.get("use_placeholder", True))
    sd_model_id = str(regen_cfg.get("sd_model_id", "runwayml/stable-diffusion-v1-5"))
    sd_revision = regen_cfg.get("sd_revision")
    torch_dtype = regen_cfg.get("torch_dtype")
    local_files_only = bool(regen_cfg.get("local_files_only", False))

    regen_proxy = _build_regen_branch(cfg, device)
    # Real DDIM eval proxy is still needed for regen/diffpure/rinse even if
    # the run was trained with blur_surrogate — fall back to the configured proxy.
    if str(cfg.get("regen_branch", "ddim_proxy")) != "ddim_proxy":
        regen_proxy = build_regen_proxy(
            use_placeholder=use_placeholder,
            sd_model_id=sd_model_id,
            sd_revision=sd_revision,
            n_steps=int(regen_cfg.get("n_steps", 4)),
            t_start=float(regen_cfg.get("t_start", 0.3)),
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            null_text_cache_dir=regen_cfg.get("null_text_cache_dir", ".cache/null_text_embeds"),
        ).to(device)

    # Null-text embeds live on the proxy (same checkpoint as VAE/UNet).
    text_conditioner = None
    if isinstance(cfg, dict) and hasattr(regen_proxy, "sd_model_id"):
        regen_out = cfg.setdefault("regen", {})
        regen_out["sd_model_id"] = str(regen_proxy.sd_model_id)
        regen_out["sd_revision"] = getattr(regen_proxy, "sd_revision", None)

    native_res = (
        int(getattr(baseline_handle, "native_resolution", 0) or 0)
        if baseline_handle is not None
        else int(cfg.get("image_size", info.get("train_resolution") or 0) or 0)
    )
    if baseline_handle is not None:
        native_res = int(baseline_handle.native_resolution)
        msg_len = int(baseline_handle.msg_len)
    logger.log(
        f"Scheme={scheme} available msg_len={msg_len} native_resolution={native_res}"
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
            test_loader, split_info = build_test_loader(
                dataset_name=str(cfg.get("dataset", "synthetic")),
                data_root=str(cfg.get("data_root", "./data")),
                image_size=int(cfg.get("image_size", 128)),
                batch_size=int(cfg.get("batch_size", 4)),
                num_workers=int(cfg.get("num_workers", 0)),
                splits_dir=str(cfg.get("splits_dir", DEFAULT_SPLITS_DIR)),
                split_seed=int(cfg.get("split_seed", DEFAULT_SPLIT_SEED)),
                num_samples=cfg.get("num_samples"),
            )
            test_list = list(split_info.get("test_list") or [])
            if train_list:
                try:
                    verify_infer_no_train_overlap(
                        test_list,
                        train_list,
                        data_root=str(cfg.get("data_root", "./data")),
                    )
                except RuntimeError as e:
                    raise SystemExit(
                        f"Inference aborted: test set overlaps training set from "
                        f"checkpoint run metadata.\n{e}"
                    ) from e
            else:
                logger.log(
                    "[warn] no train_list in checkpoint run_meta; "
                    "skipping train/test overlap safety check"
                )
            max_batches = eval_cfg.get("max_batches")
            max_batches_i = int(max_batches) if max_batches is not None else None
            n_loader = len(test_loader)
            n_eff = n_loader if max_batches_i is None else min(n_loader, max_batches_i)
            workload = estimate_eval_workload(
                attacks,
                n_batches=n_eff,
                batch_size=int(cfg.get("batch_size", 4)),
                sweep_strengths=bool(args.sweep_strengths) and not args.quality_only,
                config=cfg,
            )
            logger.log(
                f"Test batches: {n_eff}/{n_loader} (max_batches={max_batches_i})  "
                f"batch_size={cfg.get('batch_size')}  "
                f"split_hashes={split_info.get('split_hashes')}  "
                f"(seed={seed} does not change test membership)"
            )
            logger.log(
                f"Estimated diffusion UNet forwards≈"
                f"{workload['approx_diffusion_unet_forwards']:,} across "
                f"{workload['attack_strength_combos']} attack/strength combos "
                f"({workload['heavy_combos']} heavy). "
                f"Tip: omit --sweep-strengths, raise --batch-size, or use --fast "
                f"for iteration; full paper runs need the uncapped test split."
            )
            rows = evaluate_protocol(
                encoder,
                decoder,
                test_loader,
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
                max_batches=max_batches_i,
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

            batch = next(iter(test_loader))
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
                regen_proxy=regen_proxy,
            )
            attacked = {}
            for a in attacks[:4]:
                try:
                    spec = get_attack("unseen_blur" if a == "unseen" else a)
                except KeyError:
                    continue
                if spec.id == "adv":
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
                    "scheme": scheme,
                    "msg_len": int(msg_len),
                    "native_resolution": int(native_res),
                    "weight_source": weight_meta,
                    "attacks": attacks,
                    "n_rows": len(rows),
                    "splits": {
                        "splits_dir": split_info.get("splits_dir"),
                        "split_seed": split_info.get("split_seed"),
                        "split_files": split_info.get("split_files"),
                        "split_hashes": split_info.get("split_hashes"),
                        "counts": split_info.get("counts"),
                        "test_list": test_list,
                        "train_list_from_checkpoint": train_list,
                    },
                },
            )
    finally:
        if baseline_handle is not None:
            try:
                baseline_handle.close()
            except Exception:
                pass
        logger.close()

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
