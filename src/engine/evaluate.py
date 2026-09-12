"""Evaluation: attack sweep including UNSEEN attacks held out from training."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.attacks.registry import (
    ATTACK_REGISTRY,
    apply_spec,
    default_strength,
    get_attack,
)
from src.attacks.regeneration import TextConditioner, make_text_embeds
from src.utils.metrics import (
    FIDMeter,
    LPIPSMeter,
    bit_accuracy_per_image,
    long_row,
    psnr_batch,
    ssim_batch,
    tpr_at_fpr,
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
    """Apply a named attack for evaluation (registry-backed)."""
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
    return attack_id in ("guided_regen", "adv")


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
) -> List[Dict[str, Any]]:
    """Full evaluation protocol → long-format rows (bit-acc, TPR, PSNR/SSIM/LPIPS/FID)."""
    config = config or {}
    regen_cfg = config.get("regen", {})
    encoder.eval()
    decoder.eval()

    if attacks is None:
        attacks = [s.id for s in ATTACK_REGISTRY]
    if quality_only:
        attacks = ["clean"]

    lpips_meter = LPIPSMeter().to(device) if compute_quality else None
    rows: List[Dict[str, Any]] = []

    for attack_id in attacks:
        try:
            spec = get_attack("unseen_blur" if attack_id == "unseen" else attack_id)
        except KeyError:
            print(f"[eval] skipping unknown attack '{attack_id}'")
            continue
        strengths: Sequence[float]
        if sweep_strengths:
            strengths = spec.strengths
        else:
            strengths = (default_strength(spec),)

        for strength in strengths:
            fid = FIDMeter(device=device) if compute_fid and compute_quality else None
            bit_accs: List[float] = []
            scores_pos: List[float] = []
            scores_neg: List[float] = []
            psnrs: List[float] = []
            ssims: List[float] = []
            lpips_vals: List[float] = []

            for batch in val_loader:
                x = batch[0].to(device)
                m = torch.randint(0, 2, (x.size(0), msg_len), device=device).float()
                with torch.no_grad():
                    x_w = encoder(x, m)
                text_embeds = make_text_embeds(
                    x.size(0),
                    device,
                    seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
                    dim=int(regen_cfg.get("text_embed_dim", 768)),
                    conditioner=text_conditioner,
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
                    scores_pos.extend(acc.detach().cpu().tolist())
                    if compute_detection:
                        logits_neg = decoder(x)
                        scores_neg.extend(
                            bit_accuracy_per_image(logits_neg, m).detach().cpu().tolist()
                        )
                    if compute_quality:
                        # Attacked image vs original cover (paired quality).
                        psnrs.extend(psnr_batch(x, x_test))
                        ssims.extend(ssim_batch(x, x_test))
                        if lpips_meter is not None:
                            lpips_vals.extend(lpips_meter(x, x_test))
                        if fid is not None:
                            fid.update(x, x_test)

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
            )

            def _emit(name: str, value: Any) -> None:
                rows.append(long_row(metric_name=name, metric_value=value, **common))

            _emit("bit_accuracy", _mean(bit_accs))
            if compute_detection:
                _emit("tpr_at_0.1pct_fpr", tpr_at_fpr(scores_pos, scores_neg, 0.001))
                _emit("tpr_at_1pct_fpr", tpr_at_fpr(scores_pos, scores_neg, 0.01))
            if compute_quality:
                _emit("psnr", _mean(psnrs))
                _emit("ssim", _mean(ssims))
                _emit("lpips", _mean(lpips_vals))
                if fid is not None:
                    _emit("fid", fid.compute())
                    _emit("fid_backend", fid.backend)

            mean_acc = _mean(bit_accs)
            if bit_accs:
                print(f"{spec.id} strength={strength}: bit-accuracy = {mean_acc:.4f}")
            else:
                print(f"{spec.id} strength={strength}: bit-accuracy = n/a")

    return rows


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    attacks: Sequence[str] = ("clean", "jpeg", "regen", "guided_regen", "unseen_blur"),
    msg_len: int = 64,
    regen_proxy: Optional[nn.Module] = None,
    config: Optional[Mapping[str, Any]] = None,
    text_conditioner: Optional[TextConditioner] = None,
) -> Dict[str, List[float]]:
    """Attack sweep over the validation loader; returns per-attack bit-accuracy lists.

    Backward-compatible wrapper around :func:`evaluate_protocol`.
    """
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
    results: Dict[str, List[float]] = {a: [] for a in attacks}
    for row in rows:
        if row.get("metric_name") == "bit_accuracy":
            try:
                results[str(row["attack"])].append(float(row["metric_value"]))
            except (TypeError, ValueError, KeyError):
                pass
    for a in attacks:
        if results[a]:
            mean_acc = sum(results[a]) / len(results[a])
            print(f"{a}: bit-accuracy = {mean_acc:.4f}")
        else:
            print(f"{a}: bit-accuracy = n/a")
    return results


def evaluate_single_image(
    encoder: nn.Module,
    decoder: nn.Module,
    image: torch.Tensor,
    device: str = "cuda",
    msg: Optional[torch.Tensor] = None,
    msg_len: int = 64,
    attack: Optional[str] = None,
    regen_proxy: Optional[nn.Module] = None,
    config: Optional[Mapping[str, Any]] = None,
    text_conditioner: Optional[TextConditioner] = None,
) -> Dict[str, Any]:
    """Embed / optionally attack / decode a single image; report bits + PSNR/SSIM."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

    with torch.no_grad():
        x_w = encoder(x, m)

    text_embeds = make_text_embeds(
        x.size(0),
        device,
        seq_len=int(regen_cfg.get("text_embed_seq_len", 77)),
        dim=int(regen_cfg.get("text_embed_dim", 768)),
        conditioner=text_conditioner,
    )

    x_test = x_w
    if attack and attack != "clean":
        if _needs_grad(attack):
            with torch.enable_grad():
                x_test = apply_attack(
                    attack, x_w, m, decoder, regen_proxy, text_embeds, config
                )
        else:
            with torch.no_grad():
                x_test = apply_attack(
                    attack, x_w, m, decoder, regen_proxy, text_embeds, config
                )

    with torch.no_grad():
        logits = decoder(x_test)
        bits = (torch.sigmoid(logits) > 0.5).float()
        bit_acc = (bits == m).float().mean().item()

    def _to01(t: torch.Tensor) -> torch.Tensor:
        return ((t.detach().cpu().clamp(-1, 1) + 1) / 2).squeeze(0)

    x01 = _to01(x).permute(1, 2, 0).numpy()
    xw01 = _to01(x_w).permute(1, 2, 0).numpy()
    psnr = float(peak_signal_noise_ratio(x01, xw01, data_range=1.0))
    ssim = float(structural_similarity(x01, xw01, channel_axis=2, data_range=1.0))

    return {
        "message": m.detach().cpu(),
        "recovered": bits.detach().cpu(),
        "bit_accuracy": bit_acc,
        "psnr": psnr,
        "ssim": ssim,
        "x": x.detach().cpu(),
        "x_w": x_w.detach().cpu(),
        "x_test": x_test.detach().cpu(),
    }
