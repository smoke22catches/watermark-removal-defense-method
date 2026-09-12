"""Training engine: train_step and curriculum training loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks.adversarial import pgd_attack_on_decoder, straight_through
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
from src.attacks.sampler import AttackSampler, curriculum_probs
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
    """Thin wrapper so Algorithm 1 export and the loop share one implementation."""
    return curriculum_probs(epoch, cfg)


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
    """Emit a LaTeX algorithm/algorithmic block from the *resolved* config and loop.

    Curriculum arithmetic is taken from :func:`_curriculum_probs` (same defaults,
    same warm-start / clean-floor / rescale) so the published pseudocode cannot
    drift from ``train()`` / ``train_step``.
    """
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
    pmax = float(cur.get("p_regen_max", 0.4))
    p_floor = float(cur.get("p_dist_floor", 0.2))
    p_clean_floor = float(cur.get("p_clean_floor", 0.2))
    p_adv = float(cur.get("p_adv", 0.15 if cur else (probs[2] if len(probs) > 2 else 0.2)))
    warm = int(cur.get("warm_start_epochs", 0))
    n_steps = int(regen.get("n_steps", 4))
    t_start = float(regen.get("t_start", 0.3))
    eps = float(pgd.get("eps", 0.02))
    alpha = float(pgd.get("alpha", 0.005))
    pgd_steps = int(pgd.get("steps", 5))
    image_size = int(config.get("image_size", 128))
    warm_lp = cur.get("warm_lambda_perc")
    lam_warm = float(warm_lp) if warm_lp is not None else lam

    # Numeric snapshot from the same function the loop calls.
    p_e0 = _curriculum_probs(0, config)
    p_end = _curriculum_probs(max(epochs - 1, 0), config)
    p_post = _curriculum_probs(warm, config) if warm > 0 else p_e0

    if branch == "ddim_proxy":
        regen_line = (
            r"        \STATE $x' \gets \mathrm{RegenerationProxy}_{\mathrm{DDIM}}"
            f"(x_w; n_{{\\mathrm{{steps}}}}={n_steps}, t_{{\\mathrm{{start}}}}={t_start})$"
        )
    elif branch == "blur_surrogate":
        sigma = regen.get("blur_sigma")
        sigma_note = f", $\\sigma={float(sigma):.3f}$" if sigma else ", matched L2 budget"
        regen_line = (
            r"        \STATE $x' \gets \mathrm{GaussianBlur}(x_w)$ "
            f"\\COMMENT{{VINE-style surrogate{sigma_note}}}"
        )
    else:
        regen_line = r"        \STATE $x' \gets x_w$ \COMMENT{regen branch disabled}"

    if branch == "none":
        curr_block = (
            f"  \\STATE Regen branch disabled: "
            f"$p_{{\\mathrm{{regen}}}}\\gets 0$, "
            f"$p_{{\\mathrm{{adv}}}}\\gets {p_adv}$, "
            f"$p_{{\\mathrm{{dist}}}}\\gets 1-p_{{\\mathrm{{adv}}}}$"
        )
    elif not cur_on:
        curr_block = (
            f"  \\STATE Fixed attack probabilities "
            f"$({probs[0]}, {probs[1]}, {probs[2]})$ "
            r"for $(\mathcal{A}_{\mathrm{dist}}, \mathcal{A}_{\mathrm{regen}}, \mathcal{A}_{\mathrm{adv}})$; "
            f"residual $p_{{\\mathrm{{clean}}}}=1-\\sum p$"
        )
    else:
        curr_block = f"""  \\IF{{$e < {warm}$}}
    \\STATE $p_{{\\mathrm{{dist}}}}, p_{{\\mathrm{{regen}}}}, p_{{\\mathrm{{adv}}}} \\gets 0$
    \\COMMENT{{clean warm-start; $\\lambda_{{\\mathrm{{perc}}}}={lam_warm}$}}
  \\ELSE
    \\STATE $t \\gets e - {warm}$
    \\STATE $p_{{\\mathrm{{regen}}}} \\gets \\min({p0} + {dp}\\cdot t,\\ {pmax})$
    \\STATE $p_{{\\mathrm{{adv}}}} \\gets {p_adv}$
    \\STATE $p_{{\\mathrm{{dist}}}} \\gets \\max({p_floor},\\ 1 - p_{{\\mathrm{{regen}}}} - p_{{\\mathrm{{adv}}}} - {p_clean_floor})$
    \\STATE $p_{{\\mathrm{{clean}}}} \\gets {p_clean_floor}$ \\COMMENT{{clean floor}}
    \\IF{{$p_{{\\mathrm{{dist}}}}+p_{{\\mathrm{{regen}}}}+p_{{\\mathrm{{adv}}}} > 1-{p_clean_floor}$}}
      \\STATE rescale $(p_{{\\mathrm{{dist}}}}, p_{{\\mathrm{{regen}}}}, p_{{\\mathrm{{adv}}}})$ to sum $1-{p_clean_floor}$
    \\ENDIF
    \\STATE $\\lambda_{{\\mathrm{{perc}}}} \\gets {lam}$
  \\ENDIF"""

    branch_tex = branch.replace("_", r"\_")
    p0s = ", ".join(f"{v:.3f}" for v in p_e0)
    pends = ", ".join(f"{v:.3f}" for v in p_end)
    pposts = ", ".join(f"{v:.3f}" for v in p_post)

    tex = f"""% Auto-generated from src.engine.train.train / train_step / _curriculum_probs.
% Do not edit by hand — re-run with --export-algorithm.
% Resolved (p_dist, p_regen, p_adv): e=0 -> ({p0s}); after warm-start e={warm} -> ({pposts}); e={epochs - 1} -> ({pends}).
\\begin{{algorithm}}[t]
\\caption{{Навчання запропонованого методу (мінімакс із навчальним планом)}}
\\label{{alg:train}}
\\begin{{algorithmic}}[1]
\\REQUIRE Cover dataset, epochs $E={epochs}$, payload $L={msg_len}$\\,bit, image size ${image_size}$,
         batch ${batch}$, Adam $\\eta={lr}$, $\\lambda_{{\\mathrm{{perc}}}}={lam}$,
         regen-branch $\\mathtt{{{branch_tex}}}$, PGD $(\\varepsilon={eps},\\ \\alpha={alpha},\\ T={pgd_steps})$
\\ENSURE Encoder $E_\\theta$, Decoder $D_\\phi$
\\FOR{{$e = 0$ \\TO ${epochs - 1}$}}
{curr_block}
  \\FOR{{each minibatch $x$}}
    \\STATE Sample message $m \\sim \\mathrm{{Bernoulli}}(1/2)^{{L}}$
    \\STATE $x_w \\gets E_\\theta(x, m)$ \\COMMENT{{leader: embed}}
    \\STATE Sample type $\\in\\{{dist, regen, adv, clean\\}}$ with $(p_{{\\mathrm{{dist}}}}, p_{{\\mathrm{{regen}}}}, p_{{\\mathrm{{adv}}}})$; residual $\\to$ clean
    \\IF{{type $=$ dist}}
      \\STATE $x' \\gets \\mathrm{{DistortionBank}}(x_w)$ \\COMMENT{{JPEG / noise / downsample}}
    \\ELSIF{{type $=$ regen}}
{regen_line}
    \\ELSIF{{type $=$ adv}}
      \\STATE $x' \\gets \\mathrm{{PGD}}_{{D_\\phi}}(x_w, m; \\varepsilon={eps}, \\alpha={alpha}, T={pgd_steps})$
      \\COMMENT{{follower}}
    \\ELSE
      \\STATE $x' \\gets x_w$
    \\ENDIF
    \\IF{{type $\\neq$ clean}}
      \\STATE $x' \\gets x' + \\mathrm{{sg}}(x_w) - \\mathrm{{sg}}(x')$ \\COMMENT{{straight-through: forward $x'$, backward $\\partial/\\partial x_w$}}
    \\ENDIF
    \\STATE $\\hat{{z}} \\gets D_\\phi(x')$
    \\STATE $\\mathcal{{L}} \\gets \\mathcal{{L}}_{{\\mathrm{{dec}}}}(\\hat{{z}}, m) + \\lambda_{{\\mathrm{{perc}}}}\\,\\mathcal{{L}}_{{\\mathrm{{perc}}}}(x_w, x)$
    \\STATE Adam update of $\\theta, \\phi$ on $\\nabla \\mathcal{{L}}$
    \\COMMENT{{minimax cadence: one follower attack then one leader step per batch}}
  \\ENDFOR
  \\IF{{$e \\bmod {val_every} = 0$ \\OR $e = {epochs - 1}$}}
    \\STATE Validate bit accuracy and PSNR/SSIM/LPIPS per attack (attacked vs cover)
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
    if isinstance(regen_proxy, GaussianBlurSurrogate) and isinstance(config, dict):
        config.setdefault("regen", {})
        config["regen"]["blur_sigma"] = float(regen_proxy.sigma)
        if run_dir is not None:
            from src.utils.logging import save_config

            save_config(config, Path(run_dir) / "config.yaml")

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

        p_dist, p_regen, p_adv = attack_sampler.probs
        wide: Dict[str, Any] = {
            "epoch": epoch,
            "loss": epoch_stats["loss"] / n,
            "loss_decode": epoch_stats["loss_decode"] / n,
            "loss_perc": epoch_stats["loss_perc"] / n,
            "lambda_perc": lambda_perc,
            "p_clean": attack_sampler.p_clean,
            "p_regen": p_regen,
            "p_dist": p_dist,
            "p_adv": p_adv,
            "wall_clock_s": elapsed,
            "peak_gpu_memory_bytes": peak_mem,
        }

        long_rows = [
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="loss", metric_value=wide["loss"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="loss_decode", metric_value=wide["loss_decode"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="loss_perc", metric_value=wide["loss_perc"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="lambda_perc", metric_value=lambda_perc),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="p_clean", metric_value=wide["p_clean"]),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="p_regen", metric_value=p_regen),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="p_dist", metric_value=p_dist),
            long_row(epoch=epoch, seed=seed, attack="train", metric_name="p_adv", metric_value=p_adv),
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
            bit = wide.get("bit_acc_clean")
            bit_s = f" bit_clean={bit:.4f}" if bit is not None else ""
            logger.log(
                f"epoch {epoch}: loss={wide['loss']:.4f} "
                f"decode={wide['loss_decode']:.4f} perc={wide['loss_perc']:.4f} "
                f"λ={lambda_perc:.3f} p_clean={wide['p_clean']:.2f}{bit_s} "
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
