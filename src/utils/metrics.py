"""Image-quality, detection, cost, and long-format metric helpers.

All identifiers are English. Figure/table *display* strings live in ``viz.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Canonical long-format schema used by metrics.csv and results.csv.
LONG_FIELDS: List[str] = [
    "epoch",
    "seed",
    "attack",
    "strength",
    "epsilon",
    "delta",
    "knowledge_level",
    "seen_flag",
    "metric_name",
    "metric_value",
]

NOT_EVALUATED = "not_evaluated"
PROTOCOL_VERSION = "1.0"

# Student's t critical values (two-sided 95%) indexed by degrees of freedom.
# df >= 30 uses 1.96. Source: standard t-table.
_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 1.960,
}


def t_crit_95(n: int) -> float:
    """Two-sided 95% t critical value for sample size ``n`` (df = n-1)."""
    if n <= 1:
        return float("inf")
    df = n - 1
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    keys = sorted(_T_CRIT_95)
    for k in keys:
        if df <= k:
            return _T_CRIT_95[k]
    return 1.960


def mean_ci95(values: Sequence[float]) -> Tuple[float, float, int]:
    """Return ``(mean, halfwidth, n)`` for a 95% CI. NaNs are dropped."""
    arr = np.asarray([v for v in values if v is not None and _is_finite_number(v)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if n == 1:
        return mean, float("nan"), 1
    std = float(arr.std(ddof=1))
    half = t_crit_95(n) * std / math.sqrt(n)
    return mean, float(half), n


def _is_finite_number(v: Any) -> bool:
    if isinstance(v, str):
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def parse_metric_value(v: Any) -> Any:
    """Parse a CSV cell: numeric, ``not_evaluated``, or NaN."""
    if v is None:
        return NOT_EVALUATED
    if isinstance(v, str):
        s = v.strip()
        if s in ("", "nan", "NaN", "None", NOT_EVALUATED, "не вимірювалося"):
            return NOT_EVALUATED
        try:
            return float(s)
        except ValueError:
            return NOT_EVALUATED
    try:
        f = float(v)
    except (TypeError, ValueError):
        return NOT_EVALUATED
    if not np.isfinite(f):
        return NOT_EVALUATED
    return f


def to_01(t: torch.Tensor) -> torch.Tensor:
    """Map a tensor in [-1, 1] or [0, 1] to [0, 1]."""
    x = t.detach().float()
    if x.min() < -1e-4:
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)


def psnr_batch(cover: torch.Tensor, other: torch.Tensor) -> List[float]:
    """Per-image PSNR (dB) for a batch in [-1, 1] or [0, 1]."""
    a = to_01(cover)
    b = to_01(other)
    mse = (a - b).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).detach().cpu().tolist()


def ssim_batch(cover: torch.Tensor, other: torch.Tensor) -> List[float]:
    """Per-image SSIM; uses scikit-image when available, else a differentiable approx."""
    a = to_01(cover).cpu()
    b = to_01(other).cpu()
    out: List[float] = []
    try:
        from skimage.metrics import structural_similarity

        for i in range(a.size(0)):
            x = a[i].permute(1, 2, 0).numpy()
            y = b[i].permute(1, 2, 0).numpy()
            out.append(
                float(structural_similarity(x, y, channel_axis=2, data_range=1.0))
            )
        return out
    except Exception:
        return _ssim_torch(to_01(cover), to_01(other))


def _ssim_torch(a: torch.Tensor, b: torch.Tensor) -> List[float]:
    """Channel-mean SSIM with a Gaussian 11-tap window (fallback)."""
    c1, c2 = 0.01**2, 0.03**2
    k = 11
    coords = torch.arange(k, dtype=a.dtype, device=a.device) - k // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(a.size(1), 1, k, k).contiguous()
    pad = k // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=a.size(1))
    mu_b = F.conv2d(b, window, padding=pad, groups=a.size(1))
    mu_a2, mu_b2, mu_ab = mu_a.pow(2), mu_b.pow(2), mu_a * mu_b
    sig_a = F.conv2d(a * a, window, padding=pad, groups=a.size(1)) - mu_a2
    sig_b = F.conv2d(b * b, window, padding=pad, groups=b.size(1)) - mu_b2
    sig_ab = F.conv2d(a * b, window, padding=pad, groups=a.size(1)) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sig_a + sig_b + c2)
    )
    return ssim_map.flatten(1).mean(dim=1).detach().cpu().tolist()


class LPIPSMeter:
    """Reusable LPIPS (AlexNet) meter; falls back to MSE-as-distance if weights missing."""

    def __init__(self) -> None:
        self._fn = None
        self.backend = "mse_fallback"
        try:
            import lpips

            self._fn = lpips.LPIPS(net="alex")
            for p in self._fn.parameters():
                p.requires_grad_(False)
            self._fn.eval()
            self.backend = "lpips_alex"
        except Exception:
            self._fn = None

    def to(self, device: torch.device | str) -> "LPIPSMeter":
        if self._fn is not None:
            self._fn = self._fn.to(device)
        return self

    @torch.no_grad()
    def __call__(self, cover: torch.Tensor, other: torch.Tensor) -> List[float]:
        a = cover.float()
        b = other.float()
        # lpips expects [-1, 1]
        if to_01(a).min() >= 0 and a.min() >= 0:
            a = a * 2 - 1
            b = b * 2 - 1
        if self._fn is not None:
            device = a.device
            self._fn = self._fn.to(device)
            d = self._fn(a, b)
            return d.reshape(d.size(0), -1).mean(dim=1).detach().cpu().tolist()
        return (a - b).pow(2).flatten(1).mean(dim=1).detach().cpu().tolist()


def _inception_features(x01: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Inception-v3 pool features; ``x01`` is BCHW in [0, 1]."""
    x = F.interpolate(x01, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x - torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225], device=x.device
    )[:, None, None]
    feat = model(x)
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    return feat.reshape(feat.size(0), -1).float()


def _spatial_features(x01: torch.Tensor) -> torch.Tensor:
    """Offline FID stand-in: 8x8x3 mean + std per cell (48-d after flatten of stats)."""
    pooled = F.adaptive_avg_pool2d(x01, 8)
    pooled_sq = F.adaptive_avg_pool2d(x01 * x01, 8)
    var = (pooled_sq - pooled.pow(2)).clamp_min(0).sqrt()
    return torch.cat([pooled.flatten(1), var.flatten(1)], dim=1).float()


class FIDMeter:
    """Accumulate features for Frechet Inception Distance (or a spatial proxy)."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.backend = "spatial_proxy"
        self._model: Optional[torch.nn.Module] = None
        self._real: List[torch.Tensor] = []
        self._fake: List[torch.Tensor] = []
        try:
            import os

            from torchvision.models import Inception_V3_Weights, inception_v3

            # Do not download weights during smoke tests / offline runs.
            hub = Path(torch.hub.get_dir()) / "checkpoints"
            cached = any(hub.glob("inception_v3_google*")) if hub.exists() else False
            if not cached and os.environ.get("WRDM_INCEPTION", "0") != "1":
                raise FileNotFoundError("Inception-v3 weights not cached")
            weights = Inception_V3_Weights.IMAGENET1K_V1
            model = inception_v3(weights=weights, aux_logits=True)
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(device)
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
            self.backend = "inception_v3"
        except Exception:
            self._model = None
            self.backend = "spatial_proxy"

    def reset(self) -> None:
        self._real.clear()
        self._fake.clear()

    @torch.no_grad()
    def update(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        r = to_01(real).to(self.device)
        f = to_01(fake).to(self.device)
        if self._model is not None:
            self._real.append(_inception_features(r, self._model).cpu())
            self._fake.append(_inception_features(f, self._model).cpu())
        else:
            self._real.append(_spatial_features(r).cpu())
            self._fake.append(_spatial_features(f).cpu())

    def compute(self) -> float:
        if not self._real or not self._fake:
            return float("nan")
        real = torch.cat(self._real, dim=0).numpy().astype(np.float64)
        fake = torch.cat(self._fake, dim=0).numpy().astype(np.float64)
        return float(_frechet_distance(real, fake))


def _frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Frechet distance between two Gaussians fitted to rows of ``x`` and ``y``."""
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float(np.linalg.norm(mu_x - mu_y))
    if sigma_x.ndim == 0:
        sigma_x = np.atleast_2d(sigma_x)
        sigma_y = np.atleast_2d(sigma_y)
    diff = mu_x - mu_y
    covmean = _sqrtm_psd(sigma_x @ sigma_y)
    tr = np.trace(sigma_x) + np.trace(sigma_y) - 2.0 * np.trace(covmean)
    val = float(diff.dot(diff) + tr)
    return max(val, 0.0)


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigen-decomposition."""
    mat = np.nan_to_num(0.5 * (mat + mat.T), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        vals, vecs = np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(mat + 1e-6 * np.eye(mat.shape[0]))
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def bit_accuracy_per_image(logits: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """Per-row bit accuracy in [0, 1]."""
    bits = (torch.sigmoid(logits) > 0.5).float()
    return (bits == message).float().mean(dim=1)


def tpr_at_fpr(scores_pos: Sequence[float], scores_neg: Sequence[float], fpr: float) -> float:
    """TPR at a target FPR using higher-is-more-watermarked scores."""
    pos = np.asarray(list(scores_pos), dtype=np.float64)
    neg = np.asarray(list(scores_neg), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    fpr = float(np.clip(fpr, 0.0, 1.0))
    thresh = float(np.quantile(neg, 1.0 - fpr))
    return float((pos >= thresh).mean())


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def peak_gpu_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def canonical_config(cfg: Mapping[str, Any], ignore: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Deep-copy a config with run-specific keys removed (for hashing / F2 diffs)."""
    skip = set(
        ignore
        or (
            "run_name",
            "seed",
            "resume",
            "device",
            "runs_root",
            "export_algorithm",
            "num_workers",
        )
    )

    def _strip(obj: Any, prefix: str = "") -> Any:
        if isinstance(obj, Mapping):
            out = {}
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if k in skip or key in skip:
                    continue
                out[k] = _strip(v, key)
            return out
        if isinstance(obj, (list, tuple)):
            return [_strip(v, prefix) for v in obj]
        return obj

    return _strip(dict(cfg))


def config_hash(cfg: Mapping[str, Any], ignore: Optional[Iterable[str]] = None) -> str:
    canon = canonical_config(cfg, ignore=ignore)
    blob = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit_hash(repo_root: Optional[Path] = None) -> str:
    try:
        kw: Dict[str, Any] = {
            "args": ["git", "rev-parse", "HEAD"],
            "capture_output": True,
            "text": True,
            "check": False,
        }
        if repo_root is not None:
            kw["cwd"] = str(repo_root)
        proc = subprocess.run(**kw)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_yaml_file(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def long_row(
    *,
    epoch: Any = "",
    seed: Any = "",
    attack: str = "",
    strength: Any = "",
    epsilon: Any = "",
    delta: Any = "",
    knowledge_level: str = "",
    seen_flag: Any = "",
    metric_name: str,
    metric_value: Any,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "seed": seed,
        "attack": attack,
        "strength": strength,
        "epsilon": epsilon,
        "delta": delta,
        "knowledge_level": knowledge_level,
        "seen_flag": seen_flag,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


@dataclass
class LatencyStats:
    embed_ms: float
    decode_ms: float
    n_images: int


@torch.no_grad()
def measure_embed_decode_latency(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    images: torch.Tensor,
    msg_len: int,
    device: str,
    n_images: int = 100,
    warmup: int = 10,
) -> LatencyStats:
    """Batch-size-1 embed/decode latency averaged over ``n_images`` (after warmup)."""
    encoder.eval()
    decoder.eval()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    pool = images.detach()
    n_pool = max(int(pool.size(0)), 1)

    def _one(i: int) -> Tuple[float, float]:
        x = pool[i % n_pool : i % n_pool + 1].to(device)
        m = torch.randint(0, 2, (1, msg_len), device=device).float()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") and torch.cuda.is_available() else None
        t1 = torch.cuda.Event(enable_timing=True) if t0 is not None else None
        t2 = torch.cuda.Event(enable_timing=True) if t0 is not None else None
        if t0 is not None:
            t0.record()
            x_w = encoder(x, m)
            t1.record()
            _ = decoder(x_w)
            t2.record()
            torch.cuda.synchronize()
            return float(t0.elapsed_time(t1)), float(t1.elapsed_time(t2))
        import time

        a = time.perf_counter()
        x_w = encoder(x, m)
        b = time.perf_counter()
        _ = decoder(x_w)
        c = time.perf_counter()
        return (b - a) * 1000.0, (c - b) * 1000.0

    for i in range(max(warmup, 0)):
        _one(i)
    embed, decode = [], []
    for i in range(max(n_images, 1)):
        e, d = _one(i + warmup)
        embed.append(e)
        decode.append(d)
    return LatencyStats(
        embed_ms=float(np.mean(embed)),
        decode_ms=float(np.mean(decode)),
        n_images=len(embed),
    )
