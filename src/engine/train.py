"""Training engine: train_step and curriculum training loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks.adversarial import pgd_attack_on_decoder
from src.attacks.distortion import DistortionBank
from src.attacks.registry import (
    GaussianBlurSurrogate,
    IdentityRegen,
    calibrate_blur_sigma,
)
from src.attacks.regeneration import (
    PlaceholderRegenProxy,
    build_regen_proxy,
    build_text_conditioner,
    make_text_embeds,
)
from src.attacks.sampler import AttackSampler
from src.engine.evaluate import evaluate_protocol
from src.losses import PerceptualLoss, combined_loss
from src.models import Decoder, Encoder
from src.utils.metrics import (
    count_parameters,
    dump_json,
    long_row,
    measure_embed_decode_latency,
    peak_gpu_memory_bytes,
    reset_peak_gpu_memory,
)
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
    pgd = cfg.get("pgd", {})
    p_adv = float(cur.get("p_adv", 0.2)) if cur else float(cfg.get("attack_probs", [0.4, 0.4, 0.2])[2])
    regen_branch = str(cfg.get("regen_branch", "ddim_proxy"))

    if regen_branch == "none":
        attack_sampler.probs = (max(0.0, 1.0 - p_adv), 0.0, p_adv)
        return

    if not cur.get("enabled", True):
        return
    p_regen = min(
        float(cur.get("p_regen_start", 0.1)) + float(cur.get("p_regen_step", 0.01)) * epoch,
        float(cur.get("p_regen_max", 0.5)),
    )
    p_dist_floor = float(cur.get("p_dist_floor", 0.2))
    p_adv = float(cur.get("p_adv", 0.2))
    attack_sampler.probs = (max(0.5 - p_regen, p_dist_floor), p_regen, p_adv)
    _ = pgd


def _build_regen_branch(
    config: Mapping[str, Any],
    device: str,
) -> nn.Module:
    """Construct the regenerative training branch (DDIM proxy, blur, or identity)."""
    branch = str(config.get("regen_branch", "ddim_proxy"))
    regen_cfg = config.get("regen", {})
    if branch == "none":
        return IdentityRegen()
    if branch == "blur_surrogate":
        ref = PlaceholderRegenProxy()
        sigma = float(regen_cfg.get("blur_sigma") or 0.0)
        if sigma <= 0:
            sigma = calibrate_blur_sigma(
                ref,
                image_size=int(config.get("image_size", 128)),
                device="cpu",
            )
        print(f"[train] blur_surrogate sigma={sigma:.4f} (matched L2 budget vs placeholder regen)")
        return GaussianBlurSurrogate(sigma=sigma).to(device)

    use_placeholder = bool(regen_cfg.get("use_placeholder", True))
    return build_regen_proxy(
        use_placeholder=use_placeholder,
        sd_model_id=str(regen_cfg.get("sd_model_id", "runwayml/stable-diffusion-v1-5")),
        n_steps=int(regen_cfg.get("n_steps", 4)),
        t_start=float(regen_cfg.get("t_start", 0.3)),
        device=device,
        torch_dtype=regen_cfg.get("torch_dtype"),
        local_files_only=bool(regen_cfg.get("local_files_only", False)),
    ).to(device)


def export_algorithm_tex(config: Mapping[str, Any], path: str | Path) -> Path:
    """Emit a LaTeX algorithm/algorithmic block from the *resolved* config and loop."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(config.get("epochs", 100))
    msg_len = int(config.get("msg_len", 64))
    lr = float(config.get("lr", 1e-4))
    lam = float(config.get("lambda_perc", 1.0))
    batch = int(config.get("batch_size", 4))
    val_every = int(config.get("val_every", 5))
    probs = list(config.get("attack_probs", [0.4, 0.4, 0.2]))
    cur = config.get("curriculum", {})
    pgd = config.get("pgd", {})
    regen = config.get("regen", {})
    branch = str(config.get("regen_branch", "ddim_proxy"))
    cur_on = bool(cur.get("enabled", True))
    p0 = float(cur.get("p_regen_start", 0.1))
    dp = float(cur.get("p_regen_step", 0.01))
    pmax = float(cur.get("p_regen_max", 0.5))
    p_floor = float(cur.get("p_dist_floor", 0.2))
    p_adv = float(cur.get("p_adv", probs[2] if len(probs) > 2 else 0.2))
    n_steps = int(regen.get("n_steps", 4))
    t_start = float(regen.get("t_start", 0.3))
    eps = float(pgd.get("eps", 0.02))
    alpha = float(pgd.get("alpha", 0.005))
    pgd_steps = int(pgd.get("steps", 5))
    image_size = int(config.get("image_size", 128))

    if branch == "ddim_proxy":
        regen_line = (
            r"        \STATE $x' \gets \mathrm{RegenerationProxy}_{\mathrm{DDIM}}"
            f"(x_w; n_{{\\mathrm{{steps}}}}={n_steps}, t_{{\\mathrm{{start}}}}={t_start})$"
        )
    elif branch == "blur_surrogate":
        regen_line = r"        \STATE $x' \gets \mathrm{GaussianBlur}(x_w)$ \COMMENT{matched L2 budget}"
    else:
        regen_line = r"        \STATE $x' \gets x_w$ \COMMENT{regen branch disabled}"

    if cur_on and branch != "none":
        curr_block = f"""  \\STATE Update curriculum:
  \\STATE \\quad $p_{{\\mathrm{{regen}}}} \\gets \\min({p0} + {dp}\\cdot e,\\ {pmax})$
  \\STATE \\quad $p_{{\\mathrm{{dist}}}} \\gets \\max(0.5 - p_{{\\mathrm{{regen}}}},\\ {p_floor})$
  \\STATE \\quad $p_{{\\mathrm{{adv}}}} \\gets {p_adv}$"""
    elif branch == "none":
        curr_block = f"""  \\STATE Regen branch disabled: $p_{{\\mathrm{{regen}}}} \\gets 0$,
  \\STATE \\quad $p_{{\\mathrm{{adv}}}} \\gets {p_adv}$,\\ $p_{{\\mathrm{{dist}}}} \\gets 1-p_{{\\mathrm{{adv}}}}$"""
    else:
        curr_block = (
            f"  \\STATE Fixed attack probabilities "
            f"$({probs[0]}, {probs[1]}, {probs[2]})$ "
            r"for $(\mathcal{A}_{\mathrm{dist}}, \mathcal{A}_{\mathrm{regen}}, \mathcal{A}_{\mathrm{adv}})$"
        )

    tex = f"""% Auto-generated from the resolved training config and loop. Do not edit by hand.
\\begin{{algorithm}}[t]
\\caption{{Навчання запропонованого методу (мінімакс із навчальним планом)}}
\\label{{alg:train}}
\\begin{{algorithmic}}[1]
\\REQUIRE Cover dataset, epochs $E={epochs}$, payload $L={msg_len}$\\,bit, image size ${image_size}$,
         batch ${batch}$, Adam $\\eta={lr}$, $\\lambda_{{\\mathrm{{perc}}}}={lam}$,
         regen-branch $\\mathtt{{{branch.replace("_", r"\\_")}}}$, PGD $(\\varepsilon={eps},\\ \\alpha={alpha},\\ T={pgd_steps})$
\\ENSURE Encoder $E_\\theta$, Decoder $D_\\phi$
\\FOR{{$e = 0$ \\TO ${epochs - 1}$}}
{curr_block}
  \\FOR{{each minibatch $x$}}
    \\STATE Sample message $m \\sim \\mathrm{{Bernoulli}}(1/2)^{{L}}$
    \\STATE $x_w \\gets E_\\theta(x, m)$ \\COMMENT{{leader: embed}}
    \\STATE Sample attack type from $\\{{dist, regen, adv\\}}$ with $(p_{{\\mathrm{{dist}}}}, p_{{\\mathrm{{regen}}}}, p_{{\\mathrm{{adv}}}})$
    \\IF{{type $=$ dist}}
      \\STATE $x' \\gets \\mathrm{{DistortionBank}}(x_w)$ \\COMMENT{{JPEG / noise / downsample}}
    \\ELSIF{{type $=$ regen}}
{regen_line}
    \\ELSE
      \\STATE $x' \\gets \\mathrm{{PGD}}_{{D_\\phi}}(x_w, m; \\varepsilon={eps}, \\alpha={alpha}, T={pgd_steps})$
      \\COMMENT{{follower; gradient reaches $x_w$}}
    \\ENDIF
    \\STATE $\\hat{{z}} \\gets D_\\phi(x')$
    \\STATE $\\mathcal{{L}} \\gets \\mathcal{{L}}_{{\\mathrm{{dec}}}}(\\hat{{z}}, m) + {lam}\\,\\mathcal{{L}}_{{\\mathrm{{perc}}}}(x_w, x)$
    \\STATE Adam update of $\\theta, \\phi$ on $\\nabla \\mathcal{{L}}$
    \\COMMENT{{minimax cadence: one follower attack then one leader step per batch}}
  \\ENDFOR
  \\IF{{$e \\bmod {val_every} = 0$}}
    \\STATE Validate bit accuracy and PSNR/SSIM/LPIPS per attack
  \\ENDIF
\\ENDFOR
\\end{{algorithmic}}
\\end{{algorithm}}
"""
    path.write_text(tex, encoding="utf-8")
    return path


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

    seed = int(config.get("seed", 42))
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

    regen_proxy = _build_regen_branch(config, device)

    text_conditioner = build_text_conditioner(
        use_real_text_embeds=bool(regen_cfg.get("use_real_text_embeds", False)),
        use_placeholder_regen=use_placeholder or str(config.get("regen_branch", "ddim_proxy")) != "ddim_proxy",
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
    lambda_perc = float(config.get("lambda_perc", 1.0))
    val_every = int(config.get("val_every", 5))
    eval_attacks = list(
        config.get("eval", {}).get(
            "attacks", ["clean", "jpeg", "regen", "guided_regen", "unseen_blur"]
        )
    )

    enc_params = count_parameters(encoder)
    dec_params = count_parameters(decoder)
    epoch_times: list[float] = []
    epoch_mem: list[int] = []
    train_t0 = time.perf_counter()
    latency_images: Optional[torch.Tensor] = None

    for epoch in range(start_epoch, epochs):
        encoder.train()
        decoder.train()
        _update_curriculum(attack_sampler, epoch, config)
        reset_peak_gpu_memory()
        t_epoch = time.perf_counter()

        epoch_stats = {"loss": 0.0, "loss_decode": 0.0, "loss_perc": 0.0, "n": 0}
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device)
            if latency_images is None:
                latency_images = x.detach().cpu()
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
        elapsed = time.perf_counter() - t_epoch
        peak_mem = peak_gpu_memory_bytes()
        epoch_times.append(elapsed)
        epoch_mem.append(peak_mem)

        wide: Dict[str, Any] = {
            "epoch": epoch,
            "loss": epoch_stats["loss"] / n,
            "loss_decode": epoch_stats["loss_decode"] / n,
            "loss_perc": epoch_stats["loss_perc"] / n,
            "p_regen": attack_sampler.probs[1],
            "wall_clock_s": elapsed,
            "peak_gpu_memory_bytes": peak_mem,
        }

        long_rows = [
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="loss", metric_value=wide["loss"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="loss_decode", metric_value=wide["loss_decode"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="loss_perc", metric_value=wide["loss_perc"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="p_regen", metric_value=wide["p_regen"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="wall_clock_s", metric_value=elapsed),
            long_row(
                epoch=epoch,
                seed=seed,
                attack="train",
                metric_name="peak_gpu_memory_gb",
                metric_value=peak_mem / (1024**3),
            ),
        ]

        if epoch % val_every == 0 or epoch == epochs - 1:
            eval_rows = evaluate_protocol(
                encoder,
                decoder,
                val_loader,
                device=device,
                attacks=eval_attacks,
                msg_len=msg_len,
                regen_proxy=regen_proxy,
                config=config,
                text_conditioner=text_conditioner,
                seed=seed,
                epoch=epoch,
                sweep_strengths=False,
                compute_detection=False,
                compute_fid=False,
                compute_quality=True,
            )
            long_rows.extend(eval_rows)
            for row in eval_rows:
                if row.get("metric_name") == "bit_accuracy":
                    try:
                        wide[f"bit_acc_{row['attack']}"] = float(row["metric_value"])
                    except (TypeError, ValueError):
                        pass

        if logger:
            logger.log_metric_rows(long_rows)
            logger.log(
                f"epoch {epoch}: loss={wide['loss']:.4f} "
                f"decode={wide['loss_decode']:.4f} perc={wide['loss_perc']:.4f} "
                f"time={elapsed:.1f}s peak_mem={peak_mem / (1024**3):.3f}GB"
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
            if metric_key in wide:
                if wide[metric_key] >= best_metric:
                    best_metric = float(wide[metric_key])
                    payload["best_metric"] = best_metric
                    torch.save(payload, ckpt_dir / "best.pt")
            elif "loss" in wide and (best_metric < 0 or wide["loss"] < -best_metric):
                best_metric = -float(wide["loss"])
                payload["best_metric"] = best_metric
                torch.save(payload, ckpt_dir / "best.pt")

    total_s = time.perf_counter() - train_t0
    if latency_images is None:
        try:
            latency_images = next(iter(dataloader))[0]
        except Exception:
            latency_images = torch.zeros(1, 3, int(config.get("image_size", 128)), int(config.get("image_size", 128)))

    lat = measure_embed_decode_latency(
        encoder,
        decoder,
        latency_images.to(device),
        msg_len=msg_len,
        device=device,
        n_images=100,
        warmup=10,
    )
    cost = {
        "seconds_per_epoch": epoch_times,
        "seconds_per_epoch_mean": float(sum(epoch_times) / max(len(epoch_times), 1)),
        "peak_gpu_memory_bytes": epoch_mem,
        "peak_gpu_memory_gb_max": float(max(epoch_mem) / (1024**3)) if epoch_mem else 0.0,
        "total_training_wall_clock_s": float(total_s),
        "encoder_params": enc_params,
        "decoder_params": dec_params,
        "params_total": enc_params + dec_params,
        "embed_latency_ms": lat.embed_ms,
        "decode_latency_ms": lat.decode_ms,
        "latency_n_images": lat.n_images,
        "scheme": "ours",
        "regen_branch": str(config.get("regen_branch", "ddim_proxy")),
        "seed": seed,
    }
    if run_dir is not None:
        dump_json(Path(run_dir) / "cost.json", cost)
        if (Path(run_dir) / "metrics.csv").exists():
            plot_training_curves(Path(run_dir) / "metrics.csv", Path(run_dir) / "plots")

    if logger:
        logger.log(
            f"cost: {cost['seconds_per_epoch_mean']:.2f}s/epoch  "
            f"params={cost['params_total']}  "
            f"embed={lat.embed_ms:.2f}ms decode={lat.decode_ms:.2f}ms"
        )

    return {"best_metric": best_metric, "encoder": encoder, "decoder": decoder, "cost": cost}
