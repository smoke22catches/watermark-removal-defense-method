"""Evaluation: real-transform sweep (never uses training DistortionBank approximations)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.attacks.regeneration import TextConditioner, make_text_embeds
from src.baselines.common import sample_bits
from src.eval.metrics import perturbation_lpips, perturbation_norms
from src.eval.registry import (
    ATTACK_REGISTRY,
    apply_spec,
    get_attack,
    resolve_strengths,
)
from src.utils.metrics import (
    FIDMeter,
    LPIPSMeter,
    NOT_EVALUATED,
    bit_accuracy_per_image,
    compute_detection_report,
    detection_applicable,
    long_row,
    matched_bit_counts,
    psnr_batch,
    ssim_batch,
)


def apply_attack(
    name: str,
    x_w: torch.Tensor,
    m: torch.Tensor,
    decoder: nn.Module,
    regen_proxy: Optional[nn.Module],
    text_embeds: Optional[torch.Tensor],
    config: Optional[Mapping[str, Any]] = None,
    strength: Optional[float] = None,
) -> torch.Tensor:
    """Apply a named evaluation transform (registry-backed, 8-bit quantized)."""
    config = config or {}
    alias = "unseen_blur" if name == "unseen" else name
    try:
        spec = get_attack(alias)
    except KeyError as e:
        raise ValueError(f"Unknown attack: {name}") from e
    return apply_spec(
        spec,
        x_w,
        strength=strength,
        message=m,
        decoder=decoder,
        regen_proxy=regen_proxy,
        text_embeds=text_embeds,
        config=config,
    )


def _needs_grad(attack_id: str) -> bool:
    return attack_id in ("adv", "guided_regen")


def _iter_batches(loader: DataLoader, max_batches: Optional[int] = None):
    """Yield loader batches, optionally capped for fast train-time validation."""
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        yield batch


def _batch_count(loader: DataLoader, max_batches: Optional[int] = None) -> int:
    n = len(loader)
    if max_batches is None:
        return n
    return max(0, min(n, int(max_batches)))


def evaluate_protocol(
    encoder: nn.Module,
    decoder: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    attacks: Optional[Sequence[str]] = None,
    msg_len: int = 64,
    regen_proxy: Optional[nn.Module] = None,
    config: Optional[Mapping[str, Any]] = None,
    text_conditioner: Optional[TextConditioner] = None,
    seed: int = 0,
    epoch: Any = "",
    sweep_strengths: bool = False,
    compute_detection: bool = True,
    compute_fid: bool = True,
    compute_quality: bool = True,
    quality_only: bool = False,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Full evaluation protocol → long-format rows (bit-acc, pert norms, LPIPS, …)."""
    config = config or {}
    regen_cfg = config.get("regen", {})
    encoder.eval()
    decoder.eval()

    if attacks is None:
        attacks = [s.id for s in ATTACK_REGISTRY if not s.not_evaluated]
    if quality_only:
        attacks = ["clean"]

    lpips_meter = LPIPSMeter().to(device) if compute_quality else None
    rows: List[Dict[str, Any]] = []
    k = int(msg_len)
    det_ok = detection_applicable(k)
    # Identical seeded bit sampling for every scheme (reproducible across baselines).
    msg_gen = torch.Generator(device="cpu")
    msg_gen.manual_seed(int(seed) + 17_389)
    n_batches = _batch_count(val_loader, max_batches)

    # Clean, never-watermarked detection statistics (shared across attacks).
    clean_stats: List[float] = []
    if compute_detection and det_ok:
        with torch.no_grad():
            for batch in _iter_batches(val_loader, max_batches):
                x = batch[0].to(device)
                m_ref = sample_bits(x.size(0), k, device=device, generator=msg_gen)
                logits_clean = decoder(x)
                clean_stats.extend(matched_bit_counts(logits_clean, m_ref).cpu().tolist())
        if len(clean_stats) < 5000:
            print(
                f"[WARNING] clean detection set has {len(clean_stats)} < 5000 images; "
                "0.1% FPR cannot be estimated reliably"
            )
    elif compute_detection and not det_ok:
        print(
            f"[eval] detection metrics not applicable for msg_len={k} (zero-bit scheme)"
        )

    for attack_id in attacks:
        try:
            spec = get_attack("unseen_blur" if attack_id == "unseen" else attack_id)
        except KeyError:
            print(f"[eval] skipping unknown attack '{attack_id}'")
            continue
        if spec.not_evaluated:
            print(f"[eval] skipping not_evaluated '{spec.id}': {spec.not_evaluated_reason}")
            continue

        strengths: Sequence[float]
        grid = resolve_strengths(spec, config)
        if sweep_strengths:
            strengths = grid
        else:
            strengths = (float(grid[len(grid) // 2]),)

        for strength in strengths:
            # Heavy regen attacks can take many minutes per strength with no other
            # output; announce start so the run does not look wedged after the
            # previous attack's summary line (e.g. after rinse_2 → rinse_4).
            print(
                f"[eval] {spec.id} strength={strength}: "
                f"starting ({n_batches} batches)...",
                flush=True,
            )
            fid = FIDMeter(device=device) if compute_fid and compute_quality else None
            bit_accs: List[float] = []
            pos_stats: List[float] = []
            psnrs: List[float] = []
            ssims: List[float] = []
            lpips_vals: List[float] = []
            pert_l2: List[float] = []
            pert_linf: List[float] = []
            pert_lpips: List[float] = []

            # Reset message RNG per (attack, strength) so every transform sees
            # the same cover→message pairing (fair across schemes).
            msg_gen.manual_seed(int(seed) + 17_389)
            heavy = spec.id in (
                "diffpure",
                "rinse_2",
                "rinse_4",
                "regen_proxy",
                "guided_regen",
                "bmshj2018",
                "cheng2020",
            )
            for batch_idx, batch in enumerate(_iter_batches(val_loader, max_batches)):
                x = batch[0].to(device)
                m = sample_bits(x.size(0), k, device=device, generator=msg_gen)
                with torch.no_grad():
                    x_w = encoder(x, m)
                text_embeds = make_text_embeds(
                    x.size(0),
                    device,
                    seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
                    dim=int(regen_cfg.get("text_embed_dim", 768)),
                    conditioner=text_conditioner,
                    regen_proxy=regen_proxy,
                )

                def _run_attack() -> torch.Tensor:
                    return apply_spec(
                        spec,
                        x_w,
                        strength=float(strength),
                        message=m,
                        decoder=decoder,
                        regen_proxy=regen_proxy,
                        text_embeds=text_embeds,
                        config=config,
                    )

                if _needs_grad(spec.id):
                    with torch.enable_grad():
                        x_test = _run_attack()
                else:
                    with torch.no_grad():
                        x_test = _run_attack()

                with torch.no_grad():
                    logits = decoder(x_test)
                    acc = bit_accuracy_per_image(logits, m)
                    bit_accs.extend(acc.detach().cpu().tolist())
                    if compute_detection and det_ok:
                        pos_stats.extend(matched_bit_counts(logits, m).cpu().tolist())
                    # Perturbation vs watermarked image (required by protocol).
                    norms = perturbation_norms(x_w, x_test)
                    pert_l2.extend(norms["pert_l2"])
                    pert_linf.extend(norms["pert_linf"])
                    if lpips_meter is not None:
                        pert_lpips.extend(perturbation_lpips(x_w, x_test, lpips_meter))
                    if compute_quality:
                        psnrs.extend(psnr_batch(x, x_test))
                        ssims.extend(ssim_batch(x, x_test))
                        if lpips_meter is not None:
                            lpips_vals.extend(lpips_meter(x, x_test))
                        if fid is not None:
                            fid.update(x, x_test)

                if heavy and (
                    batch_idx == 0
                    or (batch_idx + 1) % max(1, n_batches // 5) == 0
                    or batch_idx + 1 == n_batches
                ):
                    print(
                        f"[eval] {spec.id} strength={strength}: "
                        f"batch {batch_idx + 1}/{n_batches}",
                        flush=True,
                    )

            def _mean(vals: List[float]) -> float:
                return float(sum(vals) / len(vals)) if vals else float("nan")

            seen = "true" if spec.seen_in_training else "false"
            common = dict(
                epoch=epoch,
                seed=seed,
                attack=spec.id,
                strength=float(strength),
                epsilon=spec.epsilon,
                delta=spec.delta,
                knowledge_level=spec.knowledge_level,
                seen_flag=seen,
                source=spec.source,
            )

            def _emit(name: str, value: Any) -> None:
                rows.append(long_row(metric_name=name, metric_value=value, **common))

            _emit("bit_accuracy", _mean(bit_accs))
            _emit("pert_l2", _mean(pert_l2))
            _emit("pert_linf", _mean(pert_linf))
            _emit("pert_lpips", _mean(pert_lpips))
            if compute_detection:
                for target, tag in ((1e-3, "0.1pct"), (1e-2, "1pct")):
                    if not det_ok:
                        _emit(f"tpr_at_{tag}_fpr", NOT_EVALUATED)
                        _emit(f"empirical_fpr_at_{tag}", NOT_EVALUATED)
                        _emit(f"tpr_at_empirical_{tag}_fpr", NOT_EVALUATED)
                        _emit(f"analytic_threshold_{tag}", NOT_EVALUATED)
                        _emit(f"empirical_threshold_{tag}", NOT_EVALUATED)
                        continue
                    report = compute_detection_report(
                        pos_stats, clean_stats, msg_len=k, target_fpr=target
                    )
                    _emit(f"tpr_at_{tag}_fpr", report.tpr_analytic)
                    _emit(f"empirical_fpr_at_{tag}", report.empirical_fpr)
                    _emit(f"tpr_at_empirical_{tag}_fpr", report.tpr_empirical)
                    _emit(f"analytic_threshold_{tag}", report.analytic_threshold)
                    _emit(f"empirical_threshold_{tag}", report.empirical_threshold)
                    if report.warning:
                        print(
                            f"[WARNING] {spec.id} strength={strength} FPR={target:g}: "
                            f"{report.warning}"
                        )
            if compute_quality:
                _emit("psnr", _mean(psnrs))
                _emit("ssim", _mean(ssims))
                _emit("lpips", _mean(lpips_vals))
                if fid is not None:
                    _emit("fid", fid.compute())
                    _emit("fid_backend", fid.backend)

            mean_acc = _mean(bit_accs)
            if bit_accs:
                extra = ""
                if compute_detection and det_ok and pos_stats:
                    r01 = compute_detection_report(pos_stats, clean_stats, k, 1e-3)
                    r1 = compute_detection_report(pos_stats, clean_stats, k, 1e-2)
                    extra = (
                        f" TPR@0.1%FPR={r01.tpr_analytic:.4f} "
                        f"TPR@1%FPR={r1.tpr_analytic:.4f}"
                    )
                print(
                    f"{spec.id} strength={strength}: bit-accuracy = {mean_acc:.4f} "
                    f"pert_l2={_mean(pert_l2):.4f} pert_linf={_mean(pert_linf):.4f}{extra}"
                )
            else:
                print(f"{spec.id} strength={strength}: bit-accuracy = n/a")

    return rows


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    attacks: Sequence[str] = ("clean", "jpeg", "diffpure", "rinse_2", "unseen_blur"),
    msg_len: int = 64,
    regen_proxy: Optional[nn.Module] = None,
    config: Optional[Mapping[str, Any]] = None,
    text_conditioner: Optional[TextConditioner] = None,
) -> Dict[str, List[float]]:
    """Attack sweep over the validation loader; returns per-attack bit-accuracy lists."""
    rows = evaluate_protocol(
        encoder,
        decoder,
        val_loader,
        device=device,
        attacks=list(attacks),
        msg_len=msg_len,
        regen_proxy=regen_proxy,
        config=config,
        text_conditioner=text_conditioner,
        sweep_strengths=False,
        compute_detection=False,
        compute_fid=False,
        compute_quality=False,
    )
    out: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("metric_name") == "bit_accuracy":
            out.setdefault(str(row["attack"]), []).append(float(row["metric_value"]))
    return out


@torch.no_grad()
def evaluate_single_image(
    encoder: nn.Module,
    decoder: nn.Module,
    image: torch.Tensor,
    device: str = "cuda",
    attack: str = "clean",
    msg_len: int = 64,
    msg: Optional[torch.Tensor] = None,
    regen_proxy: Optional[nn.Module] = None,
    config: Optional[Mapping[str, Any]] = None,
    text_conditioner: Optional[TextConditioner] = None,
) -> Dict[str, Any]:
    """Embed + optional attack + decode for a single image tensor in [-1, 1]."""
    from src.eval.registry import apply_spec as _apply
    from src.eval.registry import get_attack as _get

    config = config or {}
    regen_cfg = config.get("regen", {})
    encoder.eval()
    decoder.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0)
    x = image.to(device)
    if msg is None:
        m = torch.randint(0, 2, (1, msg_len), device=device).float()
    else:
        m = msg.to(device).float()
        if m.ndim == 1:
            m = m.unsqueeze(0)

    x_w = encoder(x, m)
    text_embeds = make_text_embeds(
        x.size(0),
        device,
        seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
        dim=int(regen_cfg.get("text_embed_dim", 768)),
        conditioner=text_conditioner,
        regen_proxy=regen_proxy,
    )

    x_test = x_w
    if attack and attack != "clean":
        spec = _get(attack)
        if _needs_grad(attack):
            with torch.enable_grad():
                x_test = _apply(
                    spec, x_w, message=m, decoder=decoder,
                    regen_proxy=regen_proxy, text_embeds=text_embeds, config=config,
                )
        else:
            x_test = _apply(
                spec, x_w, message=m, decoder=decoder,
                regen_proxy=regen_proxy, text_embeds=text_embeds, config=config,
            )

    logits = decoder(x_test)
    bits = (torch.sigmoid(logits) > 0.5).float()
    bit_acc = (bits == m).float().mean().item()

    def _to01(t: torch.Tensor) -> torch.Tensor:
        return ((t.detach().cpu().clamp(-1, 1) + 1) / 2).squeeze(0)

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    x01 = _to01(x).permute(1, 2, 0).numpy()
    xw01 = _to01(x_w).permute(1, 2, 0).numpy()
    psnr = float(peak_signal_noise_ratio(x01, xw01, data_range=1.0))
    ssim = float(structural_similarity(x01, xw01, channel_axis=2, data_range=1.0))
    norms = perturbation_norms(x_w, x_test)

    return {
        "bit_accuracy": bit_acc,
        "bits": bits.detach().cpu(),
        "message": m.detach().cpu(),
        "psnr_cover_wm": psnr,
        "ssim_cover_wm": ssim,
        "pert_l2": float(sum(norms["pert_l2"]) / max(len(norms["pert_l2"]), 1)),
        "pert_linf": float(sum(norms["pert_linf"]) / max(len(norms["pert_linf"]), 1)),
        "x_w": x_w.detach().cpu(),
        "x_test": x_test.detach().cpu(),
    }
