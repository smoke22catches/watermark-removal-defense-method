"""Training engine: train_step and curriculum training loop."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks.adversarial import pgd_attack_on_decoder, straight_through
from src.attacks.distortion import DistortionBank
from src.attacks.regeneration import (
    build_regen_proxy,
    build_text_conditioner,
    make_text_embeds,
)
from src.attacks.sampler import AttackSampler
from src.engine.evaluate import evaluate
from src.losses import PerceptualLoss, combined_loss
from src.models import Decoder, Encoder
from src.utils.viz import plot_training_curves


def train_step(
    x: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    attack_sampler: AttackSampler,
    text_embeds: torch.Tensor,
    opt_ED: torch.optim.Optimizer,
    perc_loss: PerceptualLoss,
    msg_len: int = 64,
    lambda_perc: float = 1.0,
    device: str = "cuda",
    pgd_eps: float = 0.02,
    pgd_alpha: float = 0.005,
    pgd_steps: int = 5,
) -> Dict[str, Any]:
    """One Stackelberg min-max training step (leader embed → follower attack → decode).

    Attacks may be non-differentiable; a straight-through estimator routes decode
    gradients back to the encoder through ``x_w``.
    """
    b = x.size(0)
    m = torch.randint(0, 2, (b, msg_len), device=device).float()

    # --- Leader: embed ---
    x_w = encoder(x, m)

    # --- Follower: sample attack (no_grad for speed; STE restores encoder path) ---
    with torch.no_grad():
        x_attacked, attack_type = attack_sampler(x_w, m, text_embeds)

    if attack_type == "adv":
        x_attacked = pgd_attack_on_decoder(
            x_w, m, decoder, eps=pgd_eps, alpha=pgd_alpha, steps=pgd_steps
        )
    elif attack_type == "clean":
        x_attacked = x_w

    # STE: forward = attacked image; backward = identity on x_w (except clean).
    if attack_type != "clean":
        x_attacked = straight_through(x_w, x_attacked)

    logits = decoder(x_attacked)
    loss, loss_decode, loss_perc = combined_loss(
        logits, m, x_w, x, perc_loss, lambda_perc=lambda_perc
    )

    opt_ED.zero_grad()
    loss.backward()
    opt_ED.step()

    return {
        "loss": loss.item(),
        "loss_decode": loss_decode.item(),
        "loss_perc": loss_perc.item(),
        "attack_type": attack_type,
    }


def _curriculum_probs(epoch: int, cfg: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Return (p_dist, p_regen, p_adv); residual 1-sum is clean."""
    cur = cfg.get("curriculum", {})
    if not cur.get("enabled", True):
        probs = cfg.get("attack_probs", [0.4, 0.4, 0.2])
        return float(probs[0]), float(probs[1]), float(probs[2])

    warm = int(cur.get("warm_start_epochs", 0))
    if epoch < warm:
        return 0.0, 0.0, 0.0

    # Epochs after warm-start: ramp regen while keeping a clean floor.
    t = epoch - warm
    p_regen = min(
        float(cur.get("p_regen_start", 0.1)) + float(cur.get("p_regen_step", 0.01)) * t,
        float(cur.get("p_regen_max", 0.4)),
    )
    p_adv = float(cur.get("p_adv", 0.15))
    p_dist_floor = float(cur.get("p_dist_floor", 0.2))
    p_clean_floor = float(cur.get("p_clean_floor", 0.2))

    # Allocate: clean floor + adv + regen, remainder → dist (at least p_dist_floor).
    p_dist = max(p_dist_floor, 1.0 - p_regen - p_adv - p_clean_floor)
    # Renormalize if over-allocated.
    total = p_dist + p_regen + p_adv
    max_attack = 1.0 - p_clean_floor
    if total > max_attack and total > 0:
        scale = max_attack / total
        p_dist, p_regen, p_adv = p_dist * scale, p_regen * scale, p_adv * scale
    return p_dist, p_regen, p_adv


def _lambda_perc_for_epoch(epoch: int, cfg: Mapping[str, Any]) -> float:
    """Optional lower perceptual weight during warm-start."""
    base = float(cfg.get("lambda_perc", 0.1))
    cur = cfg.get("curriculum", {})
    warm = int(cur.get("warm_start_epochs", 0))
    warm_lp = cur.get("warm_lambda_perc")
    if warm_lp is not None and epoch < warm:
        return float(warm_lp)
    return base


def _update_curriculum(attack_sampler: AttackSampler, epoch: int, cfg: Mapping[str, Any]) -> None:
    """Curriculum: clean warm-start, then increase attack share with a clean floor."""
    attack_sampler.probs = _curriculum_probs(epoch, cfg)


def train(
    encoder: Encoder,
    decoder: Decoder,
    dataloader: DataLoader,
    val_loader: DataLoader,
    config: Mapping[str, Any],
    run_dir: Any = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Full curriculum training loop with periodic evaluation and checkpointing."""
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    encoder, decoder = encoder.to(device), decoder.to(device)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(config.get("lr", 1e-4)),
    )

    dj = config.get("diffjpeg", {})
    dist_cfg = config.get("distortion", {})
    distortion_bank = DistortionBank(
        jpeg_quality=int(dj.get("quality", 50)),
        gaussian_std=float(dist_cfg.get("gaussian_std", 0.03)),
        use_real_diffjpeg=bool(dj.get("use_real_diffjpeg", False)),
        chroma_subsample=bool(dj.get("chroma_subsample", True)),
    ).to(device)

    regen_cfg = config.get("regen", {})
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

    pgd = config.get("pgd", {})
    probs = tuple(config.get("attack_probs", [0.4, 0.4, 0.2]))
    attack_sampler = AttackSampler(
        distortion_bank,
        regen_proxy,
        decoder,
        probs=probs,
        pgd_eps=float(pgd.get("eps", 0.02)),
        pgd_alpha=float(pgd.get("alpha", 0.005)),
        pgd_steps=int(pgd.get("steps", 5)),
    )

    perc_loss = PerceptualLoss(use_lpips=True).to(device)

    start_epoch = 0
    best_metric = -1.0
    resume = config.get("resume")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", -1.0))
        if logger:
            logger.log(f"Resumed from {resume} at epoch {start_epoch}")

    epochs = int(config.get("epochs", 100))
    msg_len = int(config.get("msg_len", 64))
    val_every = int(config.get("val_every", 5))
    eval_attacks = list(
        config.get("eval", {}).get(
            "attacks", ["clean", "jpeg", "regen", "guided_regen", "unseen_blur"]
        )
    )

    if logger:
        warm = int(config.get("curriculum", {}).get("warm_start_epochs", 0))
        logger.log(
            f"Curriculum: warm_start_epochs={warm}, "
            f"lambda_perc={config.get('lambda_perc')}, "
            f"STE enabled for attacked steps"
        )

    for epoch in range(start_epoch, epochs):
        encoder.train()
        decoder.train()
        _update_curriculum(attack_sampler, epoch, config)
        lambda_perc = _lambda_perc_for_epoch(epoch, config)

        epoch_stats = {"loss": 0.0, "loss_decode": 0.0, "loss_perc": 0.0, "n": 0}
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device)
            text_embeds = make_text_embeds(
                x.size(0),
                device,
                seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
                dim=int(regen_cfg.get("text_embed_dim", 768)),
                conditioner=text_conditioner,
            )
            stats = train_step(
                x,
                encoder,
                decoder,
                attack_sampler,
                text_embeds,
                opt,
                perc_loss,
                msg_len=msg_len,
                lambda_perc=lambda_perc,
                device=device,
                pgd_eps=float(pgd.get("eps", 0.02)),
                pgd_alpha=float(pgd.get("alpha", 0.005)),
                pgd_steps=int(pgd.get("steps", 5)),
            )
            epoch_stats["loss"] += stats["loss"]
            epoch_stats["loss_decode"] += stats["loss_decode"]
            epoch_stats["loss_perc"] += stats["loss_perc"]
            epoch_stats["n"] += 1
            pbar.set_postfix(loss=stats["loss"], atk=stats["attack_type"])

        n = max(epoch_stats["n"], 1)
        p_dist, p_regen, p_adv = attack_sampler.probs
        row: Dict[str, Any] = {
            "epoch": epoch,
            "loss": epoch_stats["loss"] / n,
            "loss_decode": epoch_stats["loss_decode"] / n,
            "loss_perc": epoch_stats["loss_perc"] / n,
            "lambda_perc": lambda_perc,
            "p_clean": attack_sampler.p_clean,
            "p_regen": p_regen,
            "p_dist": p_dist,
            "p_adv": p_adv,
        }

        # Periodic validation
        if epoch % val_every == 0 or epoch == epochs - 1:
            eval_out = evaluate(
                encoder,
                decoder,
                val_loader,
                device=device,
                attacks=eval_attacks,
                msg_len=msg_len,
                regen_proxy=regen_proxy,
                config=config,
                text_conditioner=text_conditioner,
            )
            for a, vals in eval_out.items():
                if vals:
                    row[f"bit_acc_{a}"] = sum(vals) / len(vals)

        if logger:
            logger.log_metrics(row)
            bit = row.get("bit_acc_clean")
            bit_s = f" bit_clean={bit:.4f}" if bit is not None else ""
            logger.log(
                f"epoch {epoch}: loss={row['loss']:.4f} "
                f"decode={row['loss_decode']:.4f} perc={row['loss_perc']:.4f} "
                f"λ={lambda_perc:.3f} p_clean={row['p_clean']:.2f}{bit_s}"
            )

        if run_dir is not None:
            ckpt_dir = run_dir / "checkpoints"
            payload = {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": opt.state_dict(),
                "config": dict(config),
                "best_metric": best_metric,
            }
            torch.save(payload, ckpt_dir / "last.pt")

            metric_key = config.get("log", {}).get("save_best_metric", "bit_acc_clean")
            if metric_key in row:
                if row[metric_key] >= best_metric:
                    best_metric = float(row[metric_key])
                    payload["best_metric"] = best_metric
                    torch.save(payload, ckpt_dir / "best.pt")
            elif "loss" in row and (best_metric < 0 or row["loss"] < -best_metric):
                best_metric = -float(row["loss"])
                payload["best_metric"] = best_metric
                torch.save(payload, ckpt_dir / "best.pt")

    if run_dir is not None and (run_dir / "metrics.csv").exists():
        plot_training_curves(run_dir / "metrics.csv", run_dir / "plots")

    return {"best_metric": best_metric, "encoder": encoder, "decoder": decoder}
