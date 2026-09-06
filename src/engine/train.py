"""Training engine: train_step and curriculum training loop."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks.adversarial import pgd_attack_on_decoder
from src.attacks.distortion import DistortionBank
from src.attacks.regeneration import build_regen_proxy, make_text_embeds
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
    """One Stackelberg min-max training step (leader embed → follower attack → decode)."""
    b = x.size(0)
    m = torch.randint(0, 2, (b, msg_len), device=device).float()

    # --- Крок "лідера": вбудовування ---
    x_w = encoder(x, m)

    # --- Крок "послідовника": семплована атака з поточного простору 𝒜 ---
    with torch.no_grad():
        x_attacked, attack_type = attack_sampler(x_w, m, text_embeds)
        # для adv-гілки потрібен градієнт через x_w -> тому PGD рахується окремо вище на живому графі

    if attack_type == "adv":
        x_attacked = pgd_attack_on_decoder(
            x_w, m, decoder, eps=pgd_eps, alpha=pgd_alpha, steps=pgd_steps
        )  # перерахунок із градієнтом до x_w

    # --- декодування та функція втрат (відповідає формулі з Кроку 4 моделі) ---
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


def _update_curriculum(attack_sampler: AttackSampler, epoch: int, cfg: Mapping[str, Any]) -> None:
    """Curriculum: increase A_regen share with epochs (matches start.py train())."""
    cur = cfg.get("curriculum", {})
    if not cur.get("enabled", True):
        return
    p_regen = min(
        float(cur.get("p_regen_start", 0.1)) + float(cur.get("p_regen_step", 0.01)) * epoch,
        float(cur.get("p_regen_max", 0.5)),
    )
    p_dist_floor = float(cur.get("p_dist_floor", 0.2))
    p_adv = float(cur.get("p_adv", 0.2))
    attack_sampler.probs = (max(0.5 - p_regen, p_dist_floor), p_regen, p_adv)


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
    ).to(device)

    regen_cfg = config.get("regen", {})
    regen_proxy = build_regen_proxy(
        use_placeholder=bool(regen_cfg.get("use_placeholder", True)),
        sd_model_id=str(regen_cfg.get("sd_model_id", "runwayml/stable-diffusion-v1-5")),
        n_steps=int(regen_cfg.get("n_steps", 4)),
        t_start=float(regen_cfg.get("t_start", 0.3)),
        device=device,
    ).to(device)

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
    lambda_perc = float(config.get("lambda_perc", 1.0))
    val_every = int(config.get("val_every", 5))
    eval_attacks = list(
        config.get("eval", {}).get(
            "attacks", ["clean", "jpeg", "regen", "guided_regen", "unseen_blur"]
        )
    )

    for epoch in range(start_epoch, epochs):
        encoder.train()
        decoder.train()
        _update_curriculum(attack_sampler, epoch, config)

        epoch_stats = {"loss": 0.0, "loss_decode": 0.0, "loss_perc": 0.0, "n": 0}
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device)
            text_embeds = make_text_embeds(
                x.size(0),
                device,
                seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
                dim=int(regen_cfg.get("text_embed_dim", 768)),
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
        row: Dict[str, Any] = {
            "epoch": epoch,
            "loss": epoch_stats["loss"] / n,
            "loss_decode": epoch_stats["loss_decode"] / n,
            "loss_perc": epoch_stats["loss_perc"] / n,
            "p_regen": attack_sampler.probs[1],
        }

        # Periodic validation (Крок 9)
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
            )
            for a, vals in eval_out.items():
                if vals:
                    row[f"bit_acc_{a}"] = sum(vals) / len(vals)

        if logger:
            logger.log_metrics(row)
            logger.log(
                f"epoch {epoch}: loss={row['loss']:.4f} "
                f"decode={row['loss_decode']:.4f} perc={row['loss_perc']:.4f}"
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
