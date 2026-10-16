"""Training engine: train_step and curriculum training loop."""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    apply_guided_pgd_ste,
    build_regen_proxy,
    guided_hparams,
    make_text_embeds,
    should_apply_guided,
)
from src.attacks.sampler import (
    AttackSampler,
    curriculum_params,
    curriculum_probs,
    validate_curriculum_config,
)
from src.data.datasets import make_reusable_train_loader, reseed_train_loader, skip_batches
from src.engine.evaluate import evaluate_protocol
from src.losses import PerceptualLoss, combined_loss
from src.models import Decoder, Encoder
from src.utils.checkpoint import (
    build_checkpoint_payload,
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_training_checkpoint,
    set_rng_state,
    write_completed_marker,
)
from src.utils.metrics import (
    config_hash,
    count_parameters,
    dump_json,
    long_row,
    measure_embed_decode_latency,
    peak_gpu_memory_bytes,
    reset_peak_gpu_memory,
)
from src.utils.viz import plot_training_curves


class _InterruptState:
    """Process-wide flag set by SIGINT/SIGTERM; finish current iter then exit."""

    requested: bool = False


def _install_interrupt_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        _InterruptState.requested = True
        print(f"[train] received signal {signum}; will checkpoint after current iteration")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not in main thread, or unsupported on this platform.
            pass


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
    epoch: int = 0,
    guided_start_epoch: Optional[int] = None,
    guided_prob: float = 0.5,
    eps_guided: float = 0.02,
    beta_guided: float = 0.005,
    J_guided: int = 5,
    scaler: Any = None,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """One Stackelberg min-max training step (leader embed → follower attack → decode).

    Distortion / regeneration keep a live autograd path so decode loss reaches the
    encoder through the (frozen) attack modules. PGD builds a constant residual via
    :func:`straight_through` so gradient reaches the encoder through ``x_w``.
    After ``guided_start_epoch``, a share ``guided_prob`` of regen steps apply PGD
    centered on the regenerated image ``x_r`` with STE through ``x_r``.
    """
    b = x.size(0)
    m = torch.randint(0, 2, (b, msg_len), device=device).float()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_ctx = (
        torch.autocast(device_type=device_type, enabled=bool(use_amp))
        if use_amp
        else torch.autocast(device_type=device_type, enabled=False)
    )

    with amp_ctx:
        # --- Leader: embed ---
        x_w = encoder(x, m)

        # --- Follower: sample attack (autograd on for dist/regen graph through E) ---
        x_attacked, attack_type = attack_sampler(x_w, m, text_embeds)
        guided_applied = False

        if attack_type == "adv":
            # Inner loop on detached input; STE keeps encoder path through x_w.
            x_pgd = pgd_attack_on_decoder(
                x_w, m, decoder, eps=pgd_eps, alpha=pgd_alpha, steps=pgd_steps
            )
            x_attacked = straight_through(x_w, x_pgd)
        elif (
            attack_type == "regen"
            and guided_start_epoch is not None
            and should_apply_guided(
                epoch,
                int(guided_start_epoch),
                float(guided_prob),
                generator=attack_sampler._rng,
            )
        ):
            # PGD ball centered on x_r (not x_w); STE residual keeps grad through x_r.
            x_attacked = apply_guided_pgd_ste(
                x_attacked,
                m,
                decoder,
                eps_guided=eps_guided,
                beta_guided=beta_guided,
                J_guided=J_guided,
            )
            guided_applied = True
        # dist / plain regen: keep the live graph from attack_sampler (no STE).

        logits = decoder(x_attacked)
        loss, loss_decode, loss_perc = combined_loss(
            logits, m, x_w, x, perc_loss, lambda_perc=lambda_perc
        )

    opt_ED.zero_grad(set_to_none=True)
    if scaler is not None and use_amp:
        scaler.scale(loss).backward()
        scaler.step(opt_ED)
        scaler.update()
    else:
        loss.backward()
        opt_ED.step()

    return {
        "loss": float(loss.detach().item()),
        "loss_decode": float(loss_decode.detach().item()),
        "loss_perc": float(loss_perc.detach().item()),
        "attack_type": attack_type,
        "guided_applied": guided_applied,
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
    """Set sampler probabilities from the single curriculum schedule."""
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
        sd_revision=regen_cfg.get("sd_revision"),
        n_steps=int(regen_cfg.get("n_steps", 4)),
        t_start=float(regen_cfg.get("t_start", 0.3)),
        device=device,
        torch_dtype=regen_cfg.get("torch_dtype"),
        local_files_only=bool(regen_cfg.get("local_files_only", False)),
        null_text_cache_dir=regen_cfg.get("null_text_cache_dir", ".cache/null_text_embeds"),
        text_embed_seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
        text_embed_dim=int(regen_cfg.get("text_embed_dim", 768)),
    ).to(device)


def export_algorithm_tex(config: Mapping[str, Any], path: str | Path) -> Path:
    """Emit a LaTeX algorithm/algorithmic block from the *resolved* config and loop.

    Curriculum arithmetic is taken from :func:`_curriculum_probs` so the published
    pseudocode cannot drift from ``train()`` / ``train_step``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(config.get("epochs", 100))
    msg_len = int(config.get("msg_len", 64))
    lr = float(config.get("lr", 1e-4))
    lam = float(config.get("lambda_perc", 1.0))
    batch = int(config.get("batch_size", 4))
    val_every = int(config.get("val_every", 5))
    p0, kappa, pmax, p_adv = curriculum_params(config)
    pgd = config.get("pgd", {})
    regen = config.get("regen", {})
    branch = str(config.get("regen_branch", "ddim_proxy"))
    n_steps = int(regen.get("n_steps", 4))
    t_start = float(regen.get("t_start", 0.3))
    eps = float(pgd.get("eps", 0.02))
    alpha = float(pgd.get("alpha", 0.005))
    pgd_steps = int(pgd.get("steps", 5))
    gh = guided_hparams(config)
    image_size = int(config.get("image_size", 128))

    # Numeric snapshot from the same function the loop calls.
    p_e0 = _curriculum_probs(0, config)
    p_end = _curriculum_probs(max(epochs - 1, 0), config)

    guided_comment = (
        f"optional guided PGD on $x_r$ when $e \\ge {gh['guided_start_epoch']}$ "
        f"w.p. {gh['guided_prob']}: "
        f"$\\varepsilon_g={gh['eps_guided']},\\ \\beta_g={gh['beta_guided']},\\ J_g={gh['J_guided']}$; "
        r"ball centered on $x_r$; STE through $x_r$"
    )
    if branch == "ddim_proxy":
        regen_line = (
            r"        \STATE $x_r \gets \mathrm{RegenerationProxy}_{\mathrm{DDIM}}"
            f"(x_w; n_{{\\mathrm{{steps}}}}={n_steps}, t_{{\\mathrm{{start}}}}={t_start})$ "
            r"\COMMENT{frozen VAE/UNet; autograd through $x_w$}"
            "\n"
            f"        \\STATE $x' \\gets x_r$ \\COMMENT{{{guided_comment}}}"
        )
    elif branch == "blur_surrogate":
        sigma = regen.get("blur_sigma")
        sigma_note = f", $\\sigma={float(sigma):.3f}$" if sigma else ", matched L2 budget"
        regen_line = (
            r"        \STATE $x_r \gets \mathrm{GaussianBlur}(x_w)$ "
            f"\\COMMENT{{VINE-style surrogate{sigma_note}; autograd through $x_w$}}"
            "\n"
            f"        \\STATE $x' \\gets x_r$ \\COMMENT{{{guided_comment}}}"
        )
    else:
        regen_line = r"        \STATE $x' \gets x_w$ \COMMENT{regen branch disabled}"

    curr_block = f"""  \\STATE $p_{{\\mathrm{{regen}}}} \\gets \\min({p0} + {kappa}\\cdot e,\\ {pmax})$
  \\STATE $p_{{\\mathrm{{adv}}}} \\gets {p_adv}$
  \\STATE $p_{{\\mathrm{{dist}}}} \\gets 1 - p_{{\\mathrm{{regen}}}} - p_{{\\mathrm{{adv}}}}$
  \\STATE $\\lambda_{{\\mathrm{{perc}}}} \\gets {lam}$"""

    branch_tex = branch.replace("_", r"\_")
    p0s = ", ".join(f"{v:.3f}" for v in p_e0)
    pends = ", ".join(f"{v:.3f}" for v in p_end)

    tex = f"""% Auto-generated from src.engine.train.train / train_step / _curriculum_probs.
% Do not edit by hand — re-run with --export-algorithm.
% Resolved (p_dist, p_regen, p_adv): e=0 -> ({p0s}); e={epochs - 1} -> ({pends}).
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
    \\STATE Sample type $\\sim \\mathrm{{Categorical}}(p_{{\\mathrm{{dist}}}}, p_{{\\mathrm{{regen}}}}, p_{{\\mathrm{{adv}}}})$
    \\IF{{type $=$ dist}}
      \\STATE $x' \\gets \\mathrm{{DistortionBank}}(x_w)$ \\COMMENT{{JPEG / noise / downsample / geometric; autograd through $x_w$}}
    \\ELSIF{{type $=$ regen}}
{regen_line}
    \\ELSE
      \\STATE $x_{{\\mathrm{{pgd}}}} \\gets \\mathrm{{PGD}}_{{D_\\phi}}(\\mathrm{{sg}}(x_w), m; \\varepsilon={eps}, \\alpha={alpha}, T={pgd_steps})$
      \\COMMENT{{follower; decoder in eval; grads w.r.t. input only}}
      \\STATE $x' \\gets x_w + \\mathrm{{sg}}(x_{{\\mathrm{{pgd}}}} - x_w)$ \\COMMENT{{constant residual; grad through $x_w$}}
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
    validate_curriculum_config(config)

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
    distortion_bank = DistortionBank.from_config(dist_cfg, dj).to(device)

    regen_cfg = config.get("regen", {})

    regen_proxy = _build_regen_branch(config, device)
    if isinstance(config, dict):
        regen_out = config.setdefault("regen", {})
        if isinstance(regen_proxy, GaussianBlurSurrogate):
            regen_out["blur_sigma"] = float(regen_proxy.sigma)
        # Record checkpoint identity used for VAE/UNet/null-text (same source).
        if hasattr(regen_proxy, "sd_model_id"):
            regen_out["sd_model_id"] = str(regen_proxy.sd_model_id)
        if hasattr(regen_proxy, "sd_revision"):
            regen_out["sd_revision"] = getattr(regen_proxy, "sd_revision", None)
        if run_dir is not None:
            from src.utils.logging import save_config

            save_config(config, Path(run_dir) / "config.yaml")

    # Null-text embeds live on the proxy; no resident text encoder.
    text_conditioner = None

    pgd = config.get("pgd", {})
    attack_sampler = AttackSampler(
        distortion_bank,
        regen_proxy,
        decoder,
        probs=_curriculum_probs(0, config),
        pgd_eps=float(pgd.get("eps", 0.02)),
        pgd_alpha=float(pgd.get("alpha", 0.005)),
        pgd_steps=int(pgd.get("steps", 5)),
        seed=seed,
    )

    use_lpips = bool(config.get("use_lpips", True))
    perc_loss = PerceptualLoss(use_lpips=use_lpips).to(device)

    # Limit PyTorch intra-op CPU threads so they do not fight DataLoader workers.
    # High htop %CPU on the main process is often OpenMP/MKL using all cores.
    torch_threads = config.get("torch_num_threads")
    if torch_threads is not None:
        torch.set_num_threads(max(1, int(torch_threads)))
    elif str(device).startswith("cuda"):
        # Sensible default on GPU servers: leave headroom for image decode workers.
        try:
            torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        except Exception:
            pass

    epochs = int(config.get("epochs", 100))
    msg_len = int(config.get("msg_len", 64))
    val_every = int(config.get("val_every", 5))
    ckpt_every = int(config.get("ckpt_every_iters", 500))
    log_every = int(config.get("log_every_iters", 50))
    use_amp = bool(config.get("amp", False)) and str(device).startswith("cuda")
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    interrupt_after = config.get("interrupt_after_step")
    interrupt_after_step = int(interrupt_after) if interrupt_after is not None else None
    batch_size = int(config.get("batch_size", getattr(dataloader, "batch_size", 4) or 4))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = config.get("pin_memory")
    if pin_memory is None:
        pin_memory = str(device).startswith("cuda")
    prefetch_factor = config.get("prefetch_factor")
    persistent_workers = config.get("persistent_workers")
    train_dataset = dataloader.dataset
    # Prefer the caller's loader when it was built with make_*_train_loader so we
    # do not spawn a second worker pool. Otherwise build one long-lived loader.
    if hasattr(dataloader, "_shuffle_generator"):
        train_loader = dataloader
    else:
        train_loader = make_reusable_train_loader(
            train_dataset,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
            persistent_workers=persistent_workers,
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else None,
            device=device,
        )
    pin_memory_transfer = bool(getattr(train_loader, "pin_memory", False))

    sched_name = config.get("lr_scheduler", None)
    if sched_name == "cosine":
        scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    else:
        # Constant LR, but still checkpointed for resume parity.
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _step: 1.0)

    cfg_hash = config_hash(config)
    start_epoch = 0
    iter_in_epoch = 0
    global_step = 0
    best_metric = -1.0
    interrupted = False
    # Stashed until the resume epoch's DataLoader skip is done. DataLoader iteration
    # advances the global torch RNG even with a seeded RandomSampler, so mid-epoch
    # resume must restore RNG *after* skip_batches, not before.
    pending_rng: Optional[Dict[str, Any]] = None

    resume_path = resolve_resume_checkpoint(config.get("resume"), run_dir)
    if resume_path is not None:
        ckpt = load_training_checkpoint(
            resume_path,
            expected_config_hash=cfg_hash,
            map_location=device,
        )
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("attack_sampler") is not None:
            attack_sampler.load_state_dict_resume(ckpt["attack_sampler"])
        if ckpt.get("rng") is not None:
            pending_rng = ckpt["rng"]
        start_epoch = int(ckpt.get("epoch", 0))
        iter_in_epoch = int(ckpt.get("iter_in_epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_metric = float(ckpt.get("best_metric", -1.0))
        if logger:
            logger.log(
                f"Resumed from {resume_path} at epoch={start_epoch} "
                f"iter_in_epoch={iter_in_epoch} global_step={global_step}"
            )

    # Prefer a lightweight train-time subset. Full WAVES suite (rinse_4 = 4×50-step
    # DiffPure, compressai downloads, …) belongs in scripts/infer.py, not every val.
    eval_cfg = config.get("eval", {}) or {}
    _default_val = [
        "clean",
        "jpeg",
        "rotate",
        "crop",
        "noise",
        "unseen_blur",
        "regen_proxy",
        "guided_regen",
        "adv",
    ]
    _default_quick = ["clean", "jpeg"]
    _heavy_val = {"diffpure", "rinse_2", "rinse_4", "bmshj2018", "cheng2020"}

    def _resolve_val_attacks(raw: Sequence[str]) -> List[str]:
        attacks = list(raw)
        if any(a in _heavy_val for a in attacks):
            stripped = [a for a in attacks if a not in _heavy_val] or list(_default_val)
            print(
                "[train] train-time val omits heavy attacks "
                f"{sorted(_heavy_val & set(attacks))}; "
                "use scripts/infer.py or set eval.val_attacks to override",
                flush=True,
            )
            return stripped
        return attacks

    if eval_cfg.get("val_attacks") is not None:
        eval_attacks_full = list(eval_cfg["val_attacks"])
    else:
        eval_attacks_full = _resolve_val_attacks(eval_cfg.get("attacks") or _default_val)
    eval_attacks_quick = list(eval_cfg.get("val_attacks_quick") or _default_quick)
    val_full_every = int(eval_cfg.get("val_full_every", 0) or 0)
    val_max_batches = eval_cfg.get("val_max_batches")
    val_max_batches_i = int(val_max_batches) if val_max_batches is not None else None
    val_compute_quality = bool(eval_cfg.get("val_compute_quality", False))

    enc_params = count_parameters(encoder)
    dec_params = count_parameters(decoder)
    epoch_times: list[float] = []
    epoch_mem: list[int] = []
    train_t0 = time.perf_counter()
    latency_images: Optional[torch.Tensor] = None

    gh = guided_hparams(config)
    _install_interrupt_handlers()
    _InterruptState.requested = False

    if logger:
        p0, kappa, p_max, p_adv0 = curriculum_params(config)
        logger.log(
            f"Curriculum: p0={p0}, kappa={kappa}, p_max={p_max}, p_adv={p_adv0}, "
            f"lambda_perc={config.get('lambda_perc')}, "
            f"autograd through dist/regen; STE residual for PGD; "
            f"guided_regen start_epoch={gh['guided_start_epoch']} "
            f"prob={gh['guided_prob']} eps={gh['eps_guided']} "
            f"beta={gh['beta_guided']} J={gh['J_guided']}; "
            f"ckpt_every_iters={ckpt_every} amp={use_amp} "
            f"num_workers={num_workers} pin_memory={bool(pin_memory)} "
            f"log_every_iters={log_every} "
            f"val_every={val_every} val_full_every={val_full_every} "
            f"val_max_batches={val_max_batches_i} "
            f"val_quick={eval_attacks_quick} val_full={eval_attacks_full} "
            f"config_hash={cfg_hash}"
        )

    def _save_ckpt(*, also_best: bool = False) -> None:
        if run_dir is None:
            return
        payload = build_checkpoint_payload(
            encoder=encoder,
            decoder=decoder,
            optimizer=opt,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            iter_in_epoch=iter_in_epoch,
            global_step=global_step,
            lambda_perc=lambda_perc,
            attack_sampler=attack_sampler,
            best_metric=best_metric,
            config=config,
            cfg_hash=cfg_hash,
        )
        save_training_checkpoint(run_dir, payload, also_best=also_best, keep_last=2)

    epoch = start_epoch
    while epoch < epochs:
        encoder.train()
        decoder.train()
        _update_curriculum(attack_sampler, epoch, config)
        lambda_perc = _lambda_perc_for_epoch(epoch, config)
        reset_peak_gpu_memory()
        t_epoch = time.perf_counter()

        epoch_stats = {"loss": 0.0, "loss_decode": 0.0, "loss_perc": 0.0, "n": 0}
        branch_counts = {"dist": 0, "regen": 0, "adv": 0}
        guided_regen_count = 0

        n_skip = iter_in_epoch if epoch == start_epoch else 0
        # Epoch-boundary resume: restore RNG before the loader so creation burn matches
        # an uninterrupted run. Mid-epoch resume: restore after skip (see pending_rng).
        if pending_rng is not None and n_skip == 0:
            set_rng_state(pending_rng)
            pending_rng = None

        reseed_train_loader(train_loader, seed, epoch)
        if pending_rng is not None and n_skip > 0:
            # DataLoader.__iter__ advances the global RNG; skip first, then restore.
            data_iter = iter(train_loader)
            for _ in range(n_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    break
            set_rng_state(pending_rng)
            pending_rng = None
            batch_iter = data_iter
        else:
            batch_iter = skip_batches(train_loader, n_skip)

        pbar = tqdm(
            batch_iter,
            desc=f"epoch {epoch + 1}/{epochs}",
            leave=False,
            initial=n_skip,
            total=len(train_loader),
            mininterval=1.0,
        )
        for batch in pbar:
            x = batch[0].to(device, non_blocking=pin_memory_transfer)
            if latency_images is None:
                latency_images = x.detach().cpu()
            text_embeds = make_text_embeds(
                x.size(0),
                device,
                seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
                dim=int(regen_cfg.get("text_embed_dim", 768)),
                conditioner=text_conditioner,
                regen_proxy=regen_proxy,
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
                epoch=epoch,
                guided_start_epoch=int(gh["guided_start_epoch"]),
                guided_prob=float(gh["guided_prob"]),
                eps_guided=float(gh["eps_guided"]),
                beta_guided=float(gh["beta_guided"]),
                J_guided=int(gh["J_guided"]),
                scaler=scaler,
                use_amp=use_amp,
            )
            iter_in_epoch += 1
            global_step += 1
            epoch_stats["loss"] += stats["loss"]
            epoch_stats["loss_decode"] += stats["loss_decode"]
            epoch_stats["loss_perc"] += stats["loss_perc"]
            epoch_stats["n"] += 1
            atk = str(stats["attack_type"])
            if atk in branch_counts:
                branch_counts[atk] += 1
            if stats.get("guided_applied"):
                guided_regen_count += 1
            if global_step % max(log_every, 1) == 0 or iter_in_epoch == 1:
                pbar.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    atk=atk,
                    step=global_step,
                    guided=int(bool(stats.get("guided_applied"))),
                )

            if logger and (global_step % max(log_every, 1) == 0 or iter_in_epoch == 1):
                logger.log_metric_rows(
                    [
                        long_row(
                            epoch=epoch,
                            global_step=global_step,
                            seed=seed,
                            attack="train",
                            metric_name="loss",
                            metric_value=stats["loss"],
                        ),
                        long_row(
                            epoch=epoch,
                            global_step=global_step,
                            seed=seed,
                            attack="train",
                            metric_name="loss_decode",
                            metric_value=stats["loss_decode"],
                        ),
                    ]
                )

            if run_dir is not None and ckpt_every > 0 and global_step % ckpt_every == 0:
                _save_ckpt()

            if interrupt_after_step is not None and global_step >= interrupt_after_step:
                _InterruptState.requested = True

            if _InterruptState.requested:
                if logger:
                    logger.flush_metrics()
                _save_ckpt()
                interrupted = True
                if logger:
                    logger.log(
                        f"Interrupted after global_step={global_step}; checkpoint saved."
                    )
                break

        if logger:
            logger.flush_metrics()

        if interrupted:
            break

        n = max(epoch_stats["n"], 1)
        elapsed = time.perf_counter() - t_epoch
        peak_mem = peak_gpu_memory_bytes()
        epoch_times.append(elapsed)
        epoch_mem.append(peak_mem)
        scheduler.step()

        p_dist, p_regen, p_adv = attack_sampler.probs
        freq_dist = branch_counts["dist"] / n
        freq_regen = branch_counts["regen"] / n
        freq_adv = branch_counts["adv"] / n
        guided_share = (
            guided_regen_count / branch_counts["regen"]
            if branch_counts["regen"] > 0
            else 0.0
        )
        wide: Dict[str, Any] = {
            "epoch": epoch,
            "loss": epoch_stats["loss"] / n,
            "loss_decode": epoch_stats["loss_decode"] / n,
            "loss_perc": epoch_stats["loss_perc"] / n,
            "lambda_perc": lambda_perc,
            "p_regen": p_regen,
            "p_dist": p_dist,
            "p_adv": p_adv,
            "freq_dist": freq_dist,
            "freq_regen": freq_regen,
            "freq_adv": freq_adv,
            "guided_regen_count": guided_regen_count,
            "guided_regen_share": guided_share,
            "wall_clock_s": elapsed,
            "peak_gpu_memory_bytes": peak_mem,
        }

        long_rows = [
            long_row(
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                attack="train",
                metric_name="epoch_loss",
                metric_value=wide["loss"],
            ),
            long_row(
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                attack="train",
                metric_name="lambda_perc",
                metric_value=lambda_perc,
            ),
            long_row(
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                attack="train",
                metric_name="p_regen",
                metric_value=p_regen,
            ),
            long_row(
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                attack="train",
                metric_name="freq_regen",
                metric_value=freq_regen,
            ),
            long_row(
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                attack="train",
                metric_name="guided_regen_count",
                metric_value=float(guided_regen_count),
            ),
            long_row(
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                attack="train",
                metric_name="wall_clock_s",
                metric_value=elapsed,
            ),
        ]

        if val_every > 0 and (epoch % val_every == 0 or epoch == epochs - 1):
            # Quick val most epochs; full train-time attack suite sparsely (and last).
            is_last = epoch == epochs - 1
            use_full = (
                val_full_every <= 0
                or is_last
                or (epoch % val_full_every == 0)
            )
            eval_attacks = eval_attacks_full if use_full else eval_attacks_quick
            mode = "full" if use_full else "quick"
            if logger:
                logger.log(
                    f"[val] epoch={epoch} mode={mode} attacks={eval_attacks} "
                    f"max_batches={val_max_batches_i} quality={val_compute_quality}"
                )
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
                compute_quality=val_compute_quality,
                max_batches=val_max_batches_i,
            )
            # Stamp global_step for dedup on resume.
            for row in eval_rows:
                row["global_step"] = global_step
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
                f"lambda={lambda_perc:.3f}{bit_s} "
                f"p_cfg=(dist={p_dist:.3f}, regen={p_regen:.3f}, adv={p_adv:.3f}) "
                f"p_emp=(dist={freq_dist:.3f}, regen={freq_regen:.3f}, adv={freq_adv:.3f}) "
                f"guided_regen={guided_regen_count}"
                f"(share={guided_share:.3f}) "
                f"step={global_step} "
                f"time={elapsed:.1f}s peak_mem={peak_mem / (1024**3):.3f}GB"
            )

        also_best = False
        metric_key = config.get("log", {}).get("save_best_metric", "bit_acc_clean")
        if metric_key in wide:
            if wide[metric_key] >= best_metric:
                best_metric = float(wide[metric_key])
                also_best = True
        elif "loss" in wide and (best_metric < 0 or wide["loss"] < -best_metric):
            best_metric = -float(wide["loss"])
            also_best = True

        # End-of-epoch checkpoint: next epoch starts at iter 0.
        epoch += 1
        iter_in_epoch = 0
        start_epoch = epoch  # subsequent epochs must not re-skip
        _save_ckpt(also_best=also_best)

    total_s = time.perf_counter() - train_t0
    if latency_images is None:
        try:
            latency_images = next(iter(dataloader))[0]
        except Exception:
            latency_images = torch.zeros(
                1, 3, int(config.get("image_size", 128)), int(config.get("image_size", 128))
            )

    lat = measure_embed_decode_latency(
        encoder,
        decoder,
        latency_images.to(device),
        msg_len=msg_len,
        device=device,
        n_images=min(100, max(1, latency_images.size(0) * 10)),
        warmup=2,
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
        "global_step": global_step,
        "interrupted": interrupted,
    }
    if run_dir is not None:
        dump_json(Path(run_dir) / "cost.json", cost)
        if (Path(run_dir) / "metrics.csv").exists():
            plot_training_curves(Path(run_dir) / "metrics.csv", Path(run_dir) / "plots")
        if not interrupted:
            write_completed_marker(
                run_dir,
                {"global_step": global_step, "epochs": epochs, "config_hash": cfg_hash},
            )

    if logger:
        logger.log(
            f"cost: {cost['seconds_per_epoch_mean']:.2f}s/epoch  "
            f"params={cost['params_total']}  "
            f"embed={lat.embed_ms:.2f}ms decode={lat.decode_ms:.2f}ms"
            + ("  [interrupted]" if interrupted else "  [completed]")
        )

    return {
        "best_metric": best_metric,
        "encoder": encoder,
        "decoder": decoder,
        "cost": cost,
        "interrupted": interrupted,
        "global_step": global_step,
    }
