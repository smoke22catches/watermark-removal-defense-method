#!/usr/bin/env python3
"""Paper figure/table generators (Ukrainian display text; English identifiers).

CLI::

    python src/utils/viz.py --figure all --runs <run_dir> ... --out <dir>
    python scripts/viz.py   --figure d4 --runs ... --out ... --x-axis aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.registry import (
    ATTACK_REGISTRY,
    CLASS_LABEL_UK,
    SCHEME_INFO,
    SCHEME_ORDER,
    get_attack,
)
from src.attacks.sampler import curriculum_probs
from src.data.datasets import load_manifest
from src.utils.metrics import (
    NOT_EVALUATED,
    canonical_config,
    load_json,
    load_yaml_file,
    mean_ci95,
    parse_metric_value,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Ukrainian display strings (glossary, verbatim)
# ---------------------------------------------------------------------------
UK = {
    "bit_accuracy": "Точність декодування, біт",
    "decoding_accuracy": "Точність декодування",
    "quality_degradation": "Деградація якості",
    "attack_strength": "Сила атаки",
    "epoch": "Епоха",
    "loss": "Функція втрат",
    "total_loss": "Сумарні втрати",
    "decode_loss": "Втрати декодування",
    "perceptual_loss": "Перцептивні втрати",
    "scheme": "Схема",
    "attack": "Атака",
    "attack_class": "Клас атаки",
    "dist_geometric": "Дисторсійні (геометричні)",
    "dist_photometric": "Дисторсійні (фотометричні)",
    "dist_degradation": "Дисторсійні (деградаційні)",
    "regen": "Регенераційні",
    "adv": "Змагальні",
    "sem": "Семантично-керовані",
    "clean": "Без атаки",
    "seen": "Відомі під час навчання",
    "unseen": "Невідомі (відкладені)",
    "knowledge": "Рівень обізнаності",
    "eps": "Бюджет збурення",
    "delta": "Бюджет якості",
    "dataset": "Набір даних",
    "payload": "Довжина повідомлення, біт",
    "time_epoch": "Час навчання на епоху, с",
    "peak_gpu": "Пік пам'яті GPU, ГБ",
    "embed_ms": "Затримка вбудовування, мс",
    "decode_ms": "Затримка декодування, мс",
    "params_m": "Параметри, млн",
    "grad_yes": "Градієнт проходить",
    "grad_no": "Градієнт не проходить (заморожено)",
    "not_eval": "не вимірювалося",
    "ours": "Запропонований метод",
    "mean_ci": "Середнє ± 95% ДІ",
    "seeds": "Запусків (seed)",
}

_ATTACK_LABELS: Dict[str, str] = {
    "clean": UK["clean"],
    "jpeg": "JPEG",
    "regen": "регенерація",
    "guided_regen": "керована регенерація",
    "unseen_blur": "невідоме розмиття",
    "unseen": "невідома атака",
}
for _spec in ATTACK_REGISTRY:
    _ATTACK_LABELS[_spec.id] = _spec.label_uk

_LOSS_LABELS: Dict[str, str] = {
    "loss": UK["total_loss"],
    "loss_decode": UK["decode_loss"],
    "loss_perc": UK["perceptual_loss"],
}

# Colorblind-safe Okabe–Ito palette + distinct markers/linestyles (not color alone).
SCHEME_STYLE: Dict[str, Dict[str, Any]] = {
    "ours": {"color": "#0072B2", "marker": "o", "ls": "-", "label": UK["ours"]},
    "stegastamp": {"color": "#E69F00", "marker": "s", "ls": "--", "label": "StegaStamp"},
    "trustmark": {"color": "#009E73", "marker": "D", "ls": "-.", "label": "TrustMark"},
    "vine": {"color": "#CC79A7", "marker": "^", "ls": ":", "label": "VINE"},
    "stable_signature": {
        "color": "#D55E00",
        "marker": "v",
        "ls": (0, (3, 1, 1, 1)),
        "label": "Stable Signature",
    },
}

BRANCH_STYLE = {
    "ddim_proxy": {"color": "#0072B2", "marker": "o", "label": "DDIM-проксі"},
    "blur_surrogate": {"color": "#E69F00", "marker": "s", "label": "Сурогат-розмиття"},
    "none": {"color": "#999999", "marker": "x", "label": "Без регенераційної гілки"},
}

CLASS_ORDER = (
    "clean",
    "dist_geometric",
    "dist_photometric",
    "dist_degradation",
    "regen",
    "adv",
    "sem",
)
D1_COLUMNS = (
    "clean",
    "dist_geometric",
    "dist_photometric",
    "dist_degradation",
    "regen",
    "adv",
    "sem",
)
MIN_SEEDS = 3
FIGURE_IDS = ("d4", "c3", "g1", "f2", "f3", "b1", "d1", "e1", "c1", "c2", "h1")
F2_BRANCHES = ("ddim_proxy", "blur_surrogate", "none")
F2_IGNORE = {
    "run_name",
    "seed",
    "resume",
    "device",
    "runs_root",
    "export_algorithm",
    "num_workers",
    "regen_branch",
    "blur_sigma",
}
# Proxy-steps sweep may use a reduced epoch budget vs the main runs.
F3_IGNORE = {
    "run_name",
    "seed",
    "resume",
    "device",
    "runs_root",
    "export_algorithm",
    "num_workers",
    "epochs",
    "regen.n_steps",
}


def configure_style() -> None:
    """Deterministic matplotlib style with a Cyrillic-capable font."""
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["axes.titlesize"] = 9
    matplotlib.rcParams["axes.labelsize"] = 8
    matplotlib.rcParams["legend.fontsize"] = 7
    matplotlib.rcParams["xtick.labelsize"] = 7
    matplotlib.rcParams["ytick.labelsize"] = 7
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["savefig.bbox"] = "tight"


def _display_label(name: str, mapping: Mapping[str, str] | None = None) -> str:
    if mapping and name in mapping:
        return mapping[name]
    return name.replace("_", " ")


def _to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [-1, 1] (or [0, 1]) to HWC uint8 RGB."""
    x = t.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def plot_training_curves(
    metrics_csv: Path | str,
    out_dir: Path | str,
    loss_keys: Sequence[str] = ("loss", "loss_decode", "loss_perc"),
    bitacc_prefix: str = "bit_acc_",
) -> List[Path]:
    """Plot loss curves and per-attack bit-accuracy vs epoch from metrics.csv."""
    configure_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    saved: List[Path] = []

    if "metric_name" in df.columns:
        loss_df = df[df["metric_name"].isin(list(loss_keys))]
        if not loss_df.empty and "epoch" in loss_df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            for k in loss_keys:
                sub = loss_df[loss_df["metric_name"] == k]
                if sub.empty:
                    continue
                ax.plot(
                    sub["epoch"],
                    pd.to_numeric(sub["metric_value"], errors="coerce"),
                    label=_display_label(k, _LOSS_LABELS),
                )
            ax.set_xlabel(UK["epoch"])
            ax.set_ylabel(UK["loss"])
            ax.set_title("Втрати під час навчання")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = out_dir / "losses.png"
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            saved.append(path)

        bit = df[df["metric_name"] == "bit_accuracy"]
        if not bit.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for attack, sub in bit.groupby("attack"):
                if str(attack) == "train":
                    continue
                ax.plot(
                    sub["epoch"],
                    pd.to_numeric(sub["metric_value"], errors="coerce"),
                    label=_display_label(str(attack), _ATTACK_LABELS),
                )
            ax.set_xlabel(UK["epoch"])
            ax.set_ylabel(UK["bit_accuracy"])
            ax.set_title("Точність бітів відносно епохи")
            ax.set_ylim(0.0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = out_dir / "bit_accuracy.png"
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            saved.append(path)
        return saved

    if "epoch" not in df.columns:
        return saved

    present = [k for k in loss_keys if k in df.columns]
    if present:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in present:
            ax.plot(df["epoch"], df[k], label=_display_label(k, _LOSS_LABELS))
        ax.set_xlabel(UK["epoch"])
        ax.set_ylabel(UK["loss"])
        ax.set_title("Втрати під час навчання")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / "losses.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    # Bit-accuracy per attack.
    # Validation runs only every val_every epochs, so non-eval rows are NaN.
    # Matplotlib breaks lines at NaN and draws no markers by default, so those
    # isolated validation points are invisible unless we drop NaNs first.
    bit_cols = [c for c in df.columns if c.startswith(bitacc_prefix)]
    if bit_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in bit_cols:
            attack = c.replace(bitacc_prefix, "")
            sub = df[["epoch", c]].dropna()
            if sub.empty:
                continue
            ax.plot(
                sub["epoch"],
                sub[c],
                label=_display_label(attack, _ATTACK_LABELS),
                marker="o",
                markersize=3,
            )
        ax.set_xlabel(UK["epoch"])
        ax.set_ylabel(UK["bit_accuracy"])
        ax.set_title("Точність бітів відносно епохи")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / "bit_accuracy.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_eval_bit_accuracy(
    results: Mapping[str, float],
    out_path: Path | str,
    title: str = "Точність бітів за атакою",
) -> Path:
    """Bar chart of mean bit-accuracy per attack name."""
    configure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    labels = [_display_label(n, _ATTACK_LABELS) for n in names]
    vals = [results[k] for k in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color="#0072B2")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(UK["bit_accuracy"])
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_qualitative_grid(
    original: torch.Tensor,
    watermarked: torch.Tensor,
    attacked: Mapping[str, torch.Tensor],
    out_path: Path | str,
    max_attacks: int = 4,
) -> Path:
    """Save a small grid: original | watermarked | attacked... | residual(wm-orig)."""
    configure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    residual = (watermarked - original).abs()
    residual = residual / residual.amax().clamp_min(1e-8)

    cols: List[tuple[str, torch.Tensor]] = [
        ("оригінал", original),
        ("з водяним знаком", watermarked),
    ]
    for name, t in list(attacked.items())[:max_attacks]:
        cols.append((_display_label(name, _ATTACK_LABELS), t))
    cols.append(("|вз−ориг|", residual))

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, (title, tensor) in zip(axes, cols):
        ax.imshow(_to_uint8_image(tensor))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Run loading / aggregation
# ---------------------------------------------------------------------------


class RunBundle:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.config = load_yaml_file(self.path / "config.yaml")
        self.meta = load_json(self.path / "run_meta.json")
        if (self.path / "results_meta.json").exists():
            rm = load_json(self.path / "results_meta.json")
            merged = dict(rm)
            merged.update({k: v for k, v in self.meta.items() if k not in merged or merged[k] in (None, "")})
            self.meta = {**self.meta, **rm}
        self.cost = load_json(self.path / "cost.json")
        self.metrics = _read_csv(self.path / "metrics.csv")
        self.results = _read_csv(self.path / "results.csv")
        if "scheme" not in self.meta:
            self.meta["scheme"] = self.config.get("scheme", "ours")
        if "seed" not in self.meta:
            self.meta["seed"] = self.config.get("seed", "")
        if "regen_branch" not in self.meta:
            self.meta["regen_branch"] = self.config.get("regen_branch", "ddim_proxy")

    @property
    def scheme(self) -> str:
        return str(self.meta.get("scheme", "ours"))

    @property
    def seed(self) -> Any:
        return self.meta.get("seed", self.config.get("seed", ""))

    @property
    def dataset(self) -> str:
        return str(self.meta.get("dataset", self.config.get("dataset", "")))


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_runs(paths: Sequence[Path | str]) -> List[RunBundle]:
    runs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[viz] warning: missing run dir {path}")
            continue
        runs.append(RunBundle(path))
    return runs


def _eval_frame(runs: Sequence[RunBundle]) -> pd.DataFrame:
    """Concatenate evaluation rows, one run per (scheme, seed, regen_branch).

    Prefer ``results.csv`` (eval protocol) over training ``metrics.csv`` so the
    two are never averaged together for the same identity.
    """
    chosen: Dict[Tuple[Any, ...], RunBundle] = {}
    for r in runs:
        key = (r.scheme, str(r.seed), str(r.meta.get("regen_branch", r.config.get("regen_branch", ""))))
        has_res = r.results is not None and not r.results.empty
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = r
        else:
            prev_res = prev.results is not None and not prev.results.empty
            if has_res and not prev_res:
                chosen[key] = r
            elif has_res and prev_res:
                n_new = int(r.results["attack"].nunique()) if "attack" in r.results.columns else 0
                n_old = int(prev.results["attack"].nunique()) if "attack" in prev.results.columns else 0
                if n_new > n_old:
                    chosen[key] = r
    frames: List[pd.DataFrame] = []
    for r in chosen.values():
        src = r.results if (r.results is not None and not r.results.empty) else r.metrics
        if src is None or src.empty:
            continue
        df = src.copy()
        if "metric_name" not in df.columns:
            continue
        df["scheme"] = r.scheme
        df["seed"] = r.seed
        df["dataset"] = r.dataset
        df["regen_branch"] = r.meta.get("regen_branch", r.config.get("regen_branch", ""))
        df["run"] = str(r.path)
        if "attack" in df.columns:
            df = df[df["attack"].astype(str) != "train"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["metric_value_num"] = out["metric_value"].map(parse_metric_value)
    return out


def _numeric_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda v: isinstance(v, (int, float)) and np.isfinite(v))


def _scheme_label(scheme: str) -> str:
    if scheme in SCHEME_STYLE:
        return SCHEME_STYLE[scheme]["label"]
    info = SCHEME_INFO.get(scheme, {})
    return str(info.get("label_uk", scheme))


def _n_seeds(df: pd.DataFrame, scheme: Optional[str] = None) -> int:
    if df.empty or "seed" not in df.columns:
        return 0
    sub = df if scheme is None else df[df["scheme"] == scheme]
    return int(sub["seed"].nunique())


def _warn_seeds(n: int, fig_id: str, ax: Any = None, fig: Any = None) -> str:
    if n >= MIN_SEEDS:
        return ""
    msg = f"WARNING [{fig_id}]: aggregated over n={n} seeds (< {MIN_SEEDS})."
    print(msg)
    note = f"УВАГА: n={n} < 3"
    target = fig if fig is not None else (ax.figure if ax is not None else None)
    if target is not None:
        target.text(
            0.99,
            0.99,
            note,
            transform=target.transFigure,
            color="crimson",
            ha="right",
            va="top",
            fontsize=7,
            fontweight="bold",
        )
    return note


def _save_fig(fig: plt.Figure, out_dir: Path, fig_id: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{fig_id}.pdf"
    png = out_dir / f"{fig_id}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _write_caption(out_dir: Path, fig_id: str, text: str) -> Path:
    path = out_dir / f"{fig_id}_caption.tex"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _write_table(
    out_dir: Path,
    fig_id: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    colspec: Optional[str] = None,
    title: Optional[str] = None,
    also_figure: bool = True,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{fig_id}.csv"
    tex_path = out_dir / f"{fig_id}.tex"
    pd.DataFrame(list(rows), columns=list(headers)).to_csv(csv_path, index=False, encoding="utf-8")
    spec = colspec or ("l" + "c" * (len(headers) - 1))
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\input{" + f"{fig_id}_caption.tex" + "}",
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts: Dict[str, str] = {"csv": str(csv_path), "tex": str(tex_path)}
    if also_figure:
        artifacts.update(
            _render_table_figure(headers, rows, out_dir, fig_id, title or fig_id.upper())
        )
    return artifacts


def _render_table_figure(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    out_dir: Path,
    fig_id: str,
    title: str,
) -> Dict[str, str]:
    """Raster/vector rendering of a booktabs table so Cyrillic can be inspected."""
    configure_style()
    n_rows = max(len(rows), 1)
    n_cols = max(len(headers), 1)
    wrap_w = 28 if n_cols <= 3 else max(10, int(96 / n_cols))
    fig_w = min(14.0, max(7.0, 1.45 * n_cols))
    fig_h = min(18.0, max(2.8, 0.52 * (n_rows + 4)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=10)
    cell = [[_wrap_cell(c, wrap_w) for c in row] for row in rows] or [[UK["not_eval"]] * n_cols]
    col_lab = [_wrap_cell(h, wrap_w) for h in headers]
    table = ax.table(
        cellText=cell,
        colLabels=col_lab,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5 if n_cols < 8 else 5.8)
    table.scale(1.0, 1.55)
    try:
        table.auto_set_column_width(col=list(range(n_cols)))
    except Exception:
        pass
    for (r, _c), cell_obj in table.get_celld().items():
        if r == 0:
            cell_obj.set_facecolor("#dceaf7")
            cell_obj.set_text_props(fontweight="bold")
        cell_obj.set_edgecolor("#888")
    fig.tight_layout()
    return _save_fig(fig, out_dir, fig_id)


def _strip_tex(s: Any) -> str:
    t = str(s)
    t = t.replace(r"${\pm}$", "±").replace(r"$\pm$", "±")
    t = t.replace(r"$\lambda_{\mathrm{perc}}$", "λ_perc")
    t = t.replace(r"\lambda_{\mathrm{perc}}", "λ_perc")
    t = t.replace(r"PGD $(\varepsilon, \alpha, T)$", "PGD (ε, α, T)")
    t = t.replace(r"$\varepsilon$", "ε").replace(r"$\delta$", "δ")
    t = t.replace(r"\textbf{", "").replace(r"\_", "_")
    t = t.replace(r"\mathrm{", "")
    t = t.replace("{", "").replace("}", "").replace("$", "")
    return t


def _wrap_cell(s: Any, width: int) -> str:
    t = _strip_tex(s)
    if width < 4 or len(t) <= width:
        return t
    return "\n".join(textwrap.wrap(t, width=width, break_long_words=True) or [t])


def _fmt_ci(mean: float, half: float, n: int) -> str:
    if n <= 0 or not np.isfinite(mean):
        return UK["not_eval"]
    if n == 1 or not np.isfinite(half):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{half:.3f}"


def _fmt_ci_tex(mean: float, half: float, n: int, bold: bool = False) -> str:
    s = _fmt_ci(mean, half, n)
    if s == UK["not_eval"]:
        return s
    inner = s.replace("±", r"${\pm}$")
    # 0.123±0.01 → 0.123${\pm}$0.01
    if "±" in s:
        a, b = s.split("±", 1)
        inner = f"{a}${{\\pm}}${b}"
    return f"\\textbf{{{inner}}}" if bold else inner


def _pivot_metric(
    df: pd.DataFrame,
    metric: str,
    keys: Sequence[str],
) -> pd.DataFrame:
    sub = df[df["metric_name"] == metric]
    sub = sub[_numeric_mask(sub["metric_value_num"])]
    return sub


def _attack_class(attack_id: str) -> str:
    try:
        return get_attack("unseen_blur" if attack_id == "unseen" else str(attack_id)).cls
    except KeyError:
        if attack_id in CLASS_LABEL_UK:
            return str(attack_id)
        return "regen"


def _is_number(v: Any) -> bool:
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _normalized_rank(attack_id: str, strength: Any) -> float:
    """Map a registry strength to [0, 1] (weak → strong) for D4 aggregation."""
    try:
        spec = get_attack("unseen_blur" if attack_id == "unseen" else str(attack_id))
        sts = [float(s) for s in spec.strengths]
        if len(sts) <= 1 or not _is_number(strength):
            return 0.5 if sts else 0.0
        sv = float(strength)
        i = min(range(len(sts)), key=lambda j: abs(sts[j] - sv))
        return i / (len(sts) - 1)
    except Exception:
        return 0.5


def _quality_degradation(row_acc: pd.Series, refs: Mapping[str, float], x_axis: str) -> float:
    psnr = float(row_acc.get("psnr", np.nan))
    ssim = float(row_acc.get("ssim", np.nan))
    lpips = float(row_acc.get("lpips", np.nan))
    if x_axis == "psnr":
        return psnr
    if x_axis == "lpips":
        return lpips
    psnr_ref = refs.get("psnr", 40.0)
    ssim_ref = refs.get("ssim", 1.0)
    lpips_ref = refs.get("lpips", 0.0)
    d_psnr = np.clip((psnr_ref - psnr) / max(abs(psnr_ref), 1.0), 0, 1) if np.isfinite(psnr) else np.nan
    d_ssim = np.clip(ssim_ref - ssim, 0, 1) if np.isfinite(ssim) else np.nan
    d_lpips = np.clip(lpips - lpips_ref, 0, 1) if np.isfinite(lpips) else np.nan
    vals = [v for v in (d_psnr, d_ssim, d_lpips) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _wide_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long metrics to one row per (scheme, seed, attack, strength)."""
    if df.empty:
        return df
    keep = df[df["metric_name"].isin(["bit_accuracy", "psnr", "ssim", "lpips", "fid",
                                      "tpr_at_0.1pct_fpr", "tpr_at_1pct_fpr"])]
    if keep.empty:
        return keep
    idx = ["scheme", "seed", "attack", "strength", "epsilon", "delta",
           "knowledge_level", "seen_flag", "dataset", "regen_branch"]
    idx = [c for c in idx if c in keep.columns]
    keep = keep.copy()
    keep["metric_value_num"] = keep["metric_value"].map(parse_metric_value)
    keep = keep[_numeric_mask(keep["metric_value_num"])]
    if keep.empty:
        return keep
    wide = keep.pivot_table(
        index=idx,
        columns="metric_name",
        values="metric_value_num",
        aggfunc="mean",
    ).reset_index()
    return wide


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_d4(
    runs: Sequence[RunBundle],
    out_dir: Path,
    x_axis: str = "aggregate",
) -> Dict[str, Any]:
    """Robustness ↔ quality curves, one panel per attack class."""
    configure_style()
    df = _eval_frame(runs)
    wide = _wide_quality(df)
    classes = [c for c in CLASS_ORDER if c != "clean"]
    n_seeds = _n_seeds(df)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), sharey=True)
    axes = axes.ravel()
    warn = ""
    datasets = sorted({r.dataset for r in runs if r.dataset})

    warn = _warn_seeds(n_seeds, "d4", fig=fig)
    for i, cls in enumerate(classes):
        ax = axes[i]
        ax.set_title(CLASS_LABEL_UK.get(cls, cls), fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        if x_axis == "psnr":
            ax.set_xlabel("PSNR, дБ")
        elif x_axis == "lpips":
            ax.set_xlabel("LPIPS")
        else:
            ax.set_xlabel(UK["quality_degradation"])
        if i % 3 == 0:
            ax.set_ylabel(UK["decoding_accuracy"])

        cls_attacks = {s.id for s in ATTACK_REGISTRY if s.cls == cls}
        panel = wide[wide["attack"].astype(str).isin(cls_attacks)] if not wide.empty else wide
        if panel.empty or "bit_accuracy" not in panel.columns:
            ax.text(0.5, 0.5, UK["not_eval"], ha="center", va="center", transform=ax.transAxes)
            continue

        for scheme in SCHEME_ORDER:
            sub = panel[panel["scheme"] == scheme]
            if sub.empty:
                continue
            st = SCHEME_STYLE[scheme]
            refs = {}
            if not wide.empty:
                clean = wide[(wide["scheme"] == scheme) & (wide["attack"] == "clean")]
                for m in ("psnr", "ssim", "lpips"):
                    if m in clean.columns and clean[m].notna().any():
                        refs[m] = float(clean[m].mean())
            ranked = sub.copy()
            ranked["rank"] = [
                _normalized_rank(str(a), s)
                for a, s in zip(ranked["attack"], ranked["strength"] if "strength" in ranked.columns else [""] * len(ranked))
            ]
            ranked["rank_key"] = (ranked["rank"] * 4).round() / 4.0
            ranked["qdeg"] = ranked.apply(lambda row: _quality_degradation(row, refs, x_axis), axis=1)
            points = []
            for rank, g in ranked.groupby("rank_key"):
                seed_y, seed_x = [], []
                for _, sg in g.groupby("seed"):
                    seed_y.append(float(sg["bit_accuracy"].mean()))
                    seed_x.append(float(sg["qdeg"].mean()))
                y_m, y_h, n = mean_ci95(seed_y)
                x_m, _, _ = mean_ci95(seed_x)
                points.append((x_m, y_m, y_h, n, rank))
            points = [p for p in points if np.isfinite(p[0]) and np.isfinite(p[1])]
            points.sort(key=lambda p: (p[4], p[0]))
            if not points:
                continue
            xs = np.array([p[0] for p in points], dtype=float)
            ys = np.array([p[1] for p in points], dtype=float)
            yerr = np.array([p[2] if np.isfinite(p[2]) else 0.0 for p in points], dtype=float)
            order = np.argsort(xs, kind="mergesort")
            ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
                    label=_scheme_label(scheme), markersize=4)
            ax.fill_between(
                xs[order],
                ys[order] - yerr[order],
                ys[order] + yerr[order],
                color=st["color"],
                alpha=0.15,
                linewidth=0,
            )
            for x, y, _, _, rank in points:
                ax.annotate(
                    f"{rank:g}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=5,
                    color=st["color"],
                )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Стійкість ↔ якість", fontsize=10)
    fig.tight_layout()
    artifacts = _save_fig(fig, out_dir, "d4")
    cap = (
        f"\\caption{{Стійкість як функція деградації якості (ось X: "
        f"{UK['quality_degradation'] if x_axis == 'aggregate' else x_axis}; "
        f"ось Y: {UK['decoding_accuracy']}). "
        f"Набір даних: {', '.join(datasets) or UK['not_eval']}. "
        f"Метрика: {UK['bit_accuracy']}. {UK['mean_ci']}, $n={n_seeds}$ {UK['seeds']}. "
        f"Кожна точка підписана силою атаки. "
        f"Бюджети $(\\varepsilon,\\delta)$ і рівень обізнаності подано в табл.~C3. "
        f"{'Увага: менше ніж 3 запусків (seed).' if n_seeds < MIN_SEEDS else ''}}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "d4", cap))
    artifacts["warning"] = warn
    return artifacts


# Transforms whose ``strength`` is a genuine configured L∞ ball radius.
_C3_CONFIGURED_LINF_IDS = frozenset({"adv", "guided_regen"})


def _c3_strength_key(strength: Any) -> float:
    """Stable key so CSV round-trips match registry floats (e.g. 4/255)."""
    return round(float(strength), 10)


def _fmt_ci_meas(mean: float, half: float, n: int) -> str:
    """CI formatter for measured budgets (more digits than bit-accuracy)."""
    if n <= 0 or not np.isfinite(mean):
        return UK["not_eval"]
    if n == 1 or not np.isfinite(half):
        return f"{mean:.4g}"
    return f"{mean:.4g}±{half:.4g}"


def _c3_collect_measurements(
    df: pd.DataFrame,
    scheme: str,
) -> Dict[Tuple[str, float], Dict[str, List[float]]]:
    """Map (attack, strength) → metric → list of per-seed values.

    Uses the metric names written by ``evaluate_protocol``: ``pert_l2``,
    ``pert_linf``, ``pert_lpips``.
    """
    out: Dict[Tuple[str, float], Dict[str, List[float]]] = {}
    if df.empty or "metric_name" not in df.columns:
        return out
    panel = df
    if "scheme" in panel.columns:
        panel = panel[panel["scheme"].astype(str) == str(scheme)]
    if panel.empty:
        return out
    keep = panel[panel["metric_name"].isin(["pert_l2", "pert_linf", "pert_lpips"])].copy()
    if keep.empty:
        return out
    keep["metric_value_num"] = keep["metric_value"].map(parse_metric_value)
    keep = keep[_numeric_mask(keep["metric_value_num"])]
    if keep.empty:
        return out
    # One value per (seed, attack, strength, metric) — already a per-run mean over samples.
    for (attack, strength, metric), g in keep.groupby(
        ["attack", "strength", "metric_name"], sort=False
    ):
        try:
            s_key = _c3_strength_key(strength)
        except (TypeError, ValueError):
            continue
        key = (str(attack), s_key)
        bucket = out.setdefault(key, {"pert_l2": [], "pert_linf": [], "pert_lpips": []})
        if metric not in bucket:
            continue
        # Prefer one value per seed (mean if duplicates).
        if "seed" in g.columns:
            for _, sg in g.groupby("seed"):
                bucket[metric].append(float(sg["metric_value_num"].mean()))
        else:
            bucket[metric].extend(float(v) for v in g["metric_value_num"].tolist())
    return out


def _c3_sample_and_seed_counts(
    runs: Sequence[RunBundle], scheme: str, meas: Mapping[Tuple[str, float], Any]
) -> Tuple[int, int]:
    """Return ``(n_samples_estimate, n_seeds)`` for the caption."""
    seeds: set = set()
    samples_per_run: List[int] = []
    for r in runs:
        if str(r.scheme) != str(scheme):
            continue
        if r.results is None or r.results.empty:
            continue
        has_pert = False
        if "metric_name" in r.results.columns:
            has_pert = bool(
                r.results["metric_name"].isin(["pert_l2", "pert_linf", "pert_lpips"]).any()
            )
        if not has_pert:
            continue
        seeds.add(str(r.seed))
        counts = (r.meta.get("splits") or {}).get("counts") or {}
        n = counts.get("test")
        if n is None:
            n = r.config.get("num_samples")
        if n is not None:
            try:
                samples_per_run.append(int(n))
            except (TypeError, ValueError):
                pass
    n_seeds = len(seeds)
    if not n_seeds and meas:
        # Fall back: count distinct seeds implied by measurement lists.
        n_seeds = max((len(v.get("pert_l2") or []) for v in meas.values()), default=0)
    if samples_per_run:
        # Typical test-set size × seeds (same split reused across seeds).
        n_per = int(round(float(np.median(samples_per_run))))
        n_samples = n_per * max(n_seeds, 1)
    else:
        n_samples = 0
    return n_samples, n_seeds


def generate_c3(
    runs: Sequence[RunBundle],
    out_dir: Path,
    *,
    c3_scheme: str = "ours",
    **_: Any,
) -> Dict[str, Any]:
    """Protocol table with *measured* perturbation / LPIPS budgets from results.csv."""
    configure_style()
    scheme = str(c3_scheme or "ours")
    df = _eval_frame(runs) if runs else pd.DataFrame()
    meas = _c3_collect_measurements(df, scheme)
    n_samples, n_seeds = _c3_sample_and_seed_counts(runs or [], scheme, meas)

    headers = [
        UK["attack"],
        UK["attack_class"],
        UK["knowledge"],
        "Вид",
        UK["attack_strength"],
        r"$\|T(x_w)-x_w\|_2$",
        r"$\|T(x_w)-x_w\|_\infty$",
        r"LPIPS$(x_w,T(x_w))$",
        "Заданий бюджет",
    ]
    csv_headers = [
        "attack",
        "class",
        "knowledge_level",
        "seen_in_training",
        "strength",
        "pert_l2",
        "pert_linf",
        "pert_lpips",
        "configured_linf_budget",
        "scheme",
    ]
    rows: List[List[str]] = []
    csv_rows: List[List[Any]] = []
    k_uk = {"black": "чорний", "grey": "сірий", "white": "білий"}

    for spec in ATTACK_REGISTRY:
        seen = UK["seen"] if spec.seen_in_training else UK["unseen"]
        k_label = k_uk.get(spec.knowledge_level, spec.knowledge_level)
        strengths = list(spec.strengths) if spec.strengths else (0.0,)
        for strength in strengths:
            s_key = _c3_strength_key(strength)
            bucket = meas.get((spec.id, s_key), {})
            m2, h2, n2 = mean_ci95(bucket.get("pert_l2") or [])
            mi, hi, ni = mean_ci95(bucket.get("pert_linf") or [])
            ml, hl, nl = mean_ci95(bucket.get("pert_lpips") or [])
            cell_l2 = _fmt_ci_meas(m2, h2, n2)
            cell_inf = _fmt_ci_meas(mi, hi, ni)
            cell_lp = _fmt_ci_meas(ml, hl, nl)
            if spec.id in _C3_CONFIGURED_LINF_IDS:
                # Strength *is* the configured L∞ radius for PGD / guided regen.
                configured = f"$\\varepsilon_\\infty={s_key:g}$"
                configured_csv = s_key
            else:
                configured = "—"
                configured_csv = ""
            rows.append(
                [
                    spec.label_uk,
                    CLASS_LABEL_UK.get(spec.cls, spec.cls),
                    k_label,
                    seen,
                    f"{s_key:g}",
                    cell_l2.replace("±", r"${\pm}$") if "±" in cell_l2 else cell_l2,
                    cell_inf.replace("±", r"${\pm}$") if "±" in cell_inf else cell_inf,
                    cell_lp.replace("±", r"${\pm}$") if "±" in cell_lp else cell_lp,
                    configured,
                ]
            )
            csv_rows.append(
                [
                    spec.id,
                    spec.cls,
                    spec.knowledge_level,
                    spec.seen_in_training,
                    s_key,
                    cell_l2,
                    cell_inf,
                    cell_lp,
                    configured_csv,
                    scheme,
                ]
            )

    artifacts = _write_table(
        out_dir,
        "c3",
        headers,
        rows,
        colspec="llllc cccc",
        title="Єдиний протокол оцінювання (виміряні бюджети)",
    )
    pd.DataFrame(csv_rows, columns=csv_headers).to_csv(
        out_dir / "c3.csv", index=False, encoding="utf-8"
    )
    datasets = sorted({r.dataset for r in runs if r.dataset}) if runs else []
    scheme_label = _scheme_label(scheme)
    if n_samples > 0 and n_seeds > 0:
        meas_note = (
            f"Виміри для схеми «{scheme_label}» ($\\texttt{{{scheme}}}$): "
            f"{n_samples} зразків, $n={n_seeds}$ {UK['seeds']} "
            f"({UK['mean_ci']} за seed)."
        )
    elif n_seeds > 0:
        meas_note = (
            f"Виміри для схеми «{scheme_label}» ($\\texttt{{{scheme}}}$): "
            f"$n={n_seeds}$ {UK['seeds']} ({UK['mean_ci']})."
        )
    else:
        meas_note = (
            f"Виміряні стовпці: «{UK['not_eval']}» "
            f"(немає evaluation-запусків для схеми \\texttt{{{scheme}}})."
        )
    cap = (
        f"\\caption{{Єдиний протокол оцінювання. Рядок — пара (трансформація, сила). "
        f"Стовпці: клас, рівень обізнаності $k$, вид (відома/невідома), сила, "
        f"виміряні $\\|T(x_w)-x_w\\|_2$, $\\|T(x_w)-x_w\\|_\\infty$ та "
        f"LPIPS$(x_w,T(x_w))$ (імена метрик як у \\texttt{{results.csv}}: "
        f"\\texttt{{pert\\_l2}}, \\texttt{{pert\\_linf}}, \\texttt{{pert\\_lpips}}). "
        f"«Заданий бюджет» — сконфігурований $L_\\infty$-радіус лише для "
        f"градієнтних / керованих атак (\\texttt{{adv}}, \\texttt{{guided\\_regen}}). "
        f"{meas_note} "
        f"Набір даних: {', '.join(datasets) or 'протокол (реєстр)'}.}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "c3", cap))
    artifacts["scheme"] = scheme
    artifacts["n_seeds"] = n_seeds
    artifacts["n_samples"] = n_samples
    return artifacts


def generate_g1(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Held-out (unseen) attack results + seen→unseen drop.

    Regeneration class must expose both a non-empty seen group (``regen_proxy``,
    ``guided_regen``) and a non-empty unseen group (DiffPure / rinse / codecs).
    """
    configure_style()
    df = _eval_frame(runs)
    wide = _wide_quality(df)
    n_seeds = _n_seeds(df)
    from src.eval.registry import regeneration_seen_unseen

    regen_groups = regeneration_seen_unseen()
    regen_seen_ids = list(regen_groups["seen"])
    regen_unseen_ids = list(regen_groups["unseen"])
    if not regen_seen_ids or not regen_unseen_ids:
        raise RuntimeError(
            "G1 requires non-empty seen and unseen groups in the regeneration class; "
            f"got seen={regen_seen_ids}, unseen={regen_unseen_ids}"
        )
    unseen_ids = [s.id for s in ATTACK_REGISTRY if not s.seen_in_training and not s.not_evaluated]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax, ax2 = axes
    _warn_seeds(n_seeds, "g1", fig=fig)

    present_unseen = []
    for aid in unseen_ids:
        has = False
        if not wide.empty and "bit_accuracy" in wide.columns:
            has = bool((wide["attack"] == aid).any()) and _numeric_mask(
                wide.loc[wide["attack"] == aid, "bit_accuracy"]
            ).any()
        present_unseen.append(aid if has else None)
    plot_ids = [a for a in unseen_ids if a in {x for x in present_unseen if x}]
    if not plot_ids:
        plot_ids = list(unseen_ids)

    x = np.arange(len(plot_ids))
    width = 0.16
    table_headers = [UK["attack"]] + [_scheme_label(s) for s in SCHEME_ORDER]
    table_rows: List[List[str]] = []

    for j, scheme in enumerate(SCHEME_ORDER):
        st = SCHEME_STYLE[scheme]
        means, errs = [], []
        for aid in plot_ids:
            vals = []
            if not wide.empty and "bit_accuracy" in wide.columns:
                sub = wide[(wide["scheme"] == scheme) & (wide["attack"] == aid)]
                for seed, g in sub.groupby("seed"):
                    vals.append(float(g["bit_accuracy"].mean()))
            m, h, n = mean_ci95(vals)
            means.append(m if n else np.nan)
            errs.append(h if n and np.isfinite(h) else 0.0)
        if not any(np.isfinite(m) for m in means):
            continue
        offset = (j - len(SCHEME_ORDER) / 2) * width + width / 2
        ax.bar(
            x + offset,
            [0 if not np.isfinite(m) else m for m in means],
            width,
            yerr=errs,
            color=st["color"],
            hatch="" if scheme == "ours" else "..",
            label=_scheme_label(scheme),
            error_kw={"elinewidth": 0.8, "capsize": 2},
        )

    # table rows per attack
    for aid in unseen_ids:
        row = [_ATTACK_LABELS.get(aid, aid)]
        for scheme in SCHEME_ORDER:
            vals = []
            if not wide.empty and "bit_accuracy" in wide.columns:
                sub = wide[(wide["scheme"] == scheme) & (wide["attack"] == aid)]
                for _, g in sub.groupby("seed"):
                    vals.append(float(g["bit_accuracy"].mean()))
            m, h, n = mean_ci95(vals)
            row.append(_fmt_ci(m, h, n))
        table_rows.append(row)

    ax.set_xticks(x)
    ax.set_xticklabels([_ATTACK_LABELS.get(a, a) for a in plot_ids], rotation=35, ha="right")
    ax.set_ylabel(UK["bit_accuracy"])
    ax.set_ylim(0, 1.05)
    ax.set_title(UK["unseen"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, ncol=1, loc="upper right")

    # seen → unseen drop (all classes) + regeneration-class drop
    drop_means, drop_errs, drop_labels = [], [], []
    drop_row = ["seen→unseen"]
    regen_drop_row = ["regen seen→unseen"]
    for scheme in SCHEME_ORDER:
        seen_vals, unseen_vals = [], []
        regen_seen_vals, regen_unseen_vals = [], []
        if not wide.empty and "bit_accuracy" in wide.columns:
            for seed, g in wide[wide["scheme"] == scheme].groupby("seed"):
                g = g.copy()
                g["seen_bool"] = g["seen_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
                g = g[g["attack"] != "clean"]
                sv = g[g["seen_bool"]]["bit_accuracy"]
                uv = g[~g["seen_bool"]]["bit_accuracy"]
                if len(sv) and len(uv):
                    seen_vals.append(float(sv.mean()))
                    unseen_vals.append(float(uv.mean()))
                rs = g[g["attack"].isin(regen_seen_ids)]["bit_accuracy"]
                ru = g[g["attack"].isin(regen_unseen_ids)]["bit_accuracy"]
                if len(rs) and len(ru):
                    regen_seen_vals.append(float(rs.mean()))
                    regen_unseen_vals.append(float(ru.mean()))
        diffs = [s - u for s, u in zip(seen_vals, unseen_vals)]
        m, h, n = mean_ci95(diffs)
        drop_means.append(0.0 if not np.isfinite(m) else m)
        drop_errs.append(0.0 if not (n and np.isfinite(h)) else h)
        drop_labels.append(_scheme_label(scheme))
        drop_row.append(_fmt_ci(m, h, n))
        rdiffs = [s - u for s, u in zip(regen_seen_vals, regen_unseen_vals)]
        rm, rh, rn = mean_ci95(rdiffs)
        regen_drop_row.append(_fmt_ci(rm, rh, rn))
    table_rows.append(drop_row)
    table_rows.append(regen_drop_row)

    colors = [SCHEME_STYLE[s]["color"] for s in SCHEME_ORDER]
    ax2.bar(np.arange(len(SCHEME_ORDER)), drop_means, yerr=drop_errs,
            color=colors, error_kw={"elinewidth": 0.8, "capsize": 2})
    ax2.set_xticks(np.arange(len(SCHEME_ORDER)))
    ax2.set_xticklabels(drop_labels, rotation=30, ha="right")
    ax2.set_title("Спад: відомі → невідомі")
    ax2.set_ylabel(UK["decoding_accuracy"])
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    artifacts = _save_fig(fig, out_dir, "g1")
    artifacts.update(_write_table(out_dir, "g1", table_headers, table_rows, also_figure=False))
    artifacts["regen_seen"] = regen_seen_ids
    artifacts["regen_unseen"] = regen_unseen_ids
    datasets = sorted({r.dataset for r in runs if r.dataset})
    cap = (
        f"\\caption{{Результати на невідомих (відкладених) атаках. "
        f"Метрика: {UK['bit_accuracy']}. {UK['mean_ci']}, $n={n_seeds}$ {UK['seeds']}. "
        f"Набір даних: {', '.join(datasets) or UK['not_eval']}. "
        f"Клас регенерації: відомі={{{', '.join(regen_seen_ids)}}}, "
        f"невідомі={{{', '.join(regen_unseen_ids)}}}. "
        f"Права панель: спад точності seen$\\to$unseen (менший спад — краще узагальнення). "
        f"Бюджети та $k$ — за протоколом C3. "
        f"{'Увага: менше ніж 3 запусків (seed).' if n_seeds < MIN_SEEDS else ''}}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "g1", cap))
    return artifacts


def _config_diff(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    ignore: Optional[Iterable[str]] = None,
) -> List[str]:
    skip = set(ignore) if ignore is not None else set(F2_IGNORE)
    ca, cb = canonical_config(a, ignore=skip), canonical_config(b, ignore=skip)

    def _walk(x, y, prefix=""):
        diffs = []
        if type(x) is not type(y) and not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return [f"{prefix}: {x!r} vs {y!r}"]
        if isinstance(x, dict):
            keys = set(x) | set(y)
            for k in sorted(keys):
                diffs.extend(_walk(x.get(k), y.get(k), f"{prefix}.{k}" if prefix else k))
            return diffs
        if x != y:
            return [f"{prefix}: {x!r} vs {y!r}"]
        return []

    return _walk(ca, cb)


def _config_fingerprint(cfg: Mapping[str, Any], ignore: Iterable[str]) -> str:
    return json.dumps(canonical_config(cfg, ignore=ignore), sort_keys=True, default=str)


def _assert_config_identity(
    run_groups: Sequence[Sequence[RunBundle]],
    *,
    ignore: Iterable[str],
    fig_id: str,
    allowed_msg: str,
) -> None:
    """Abort if any two runs differ outside ``ignore`` keys."""
    refs: List[RunBundle] = []
    for group in run_groups:
        if group:
            refs.append(group[0])
        for r in group[1:]:
            diffs = _config_diff(group[0].config, r.config, ignore=ignore)
            if diffs:
                raise SystemExit(
                    f"{fig_id} abort: intra-group configs differ beyond {allowed_msg} "
                    f"({r.path}):\n  " + "\n  ".join(diffs)
                )
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            diffs = _config_diff(refs[i].config, refs[j].config, ignore=ignore)
            if diffs:
                raise SystemExit(
                    f"{fig_id} abort: configs differ beyond {allowed_msg}:\n  "
                    + "\n  ".join(diffs)
                )


def _select_coherent_branch_runs(
    by_branch: Dict[str, List[RunBundle]],
    *,
    ignore: Iterable[str],
) -> Dict[str, List[RunBundle]]:
    """Keep the largest config-fingerprint cluster so mixed sweeps do not break F2."""
    from collections import Counter

    all_runs = [r for rs in by_branch.values() for r in rs]
    if not all_runs:
        return by_branch
    counts = Counter(_config_fingerprint(r.config, ignore) for r in all_runs)
    best_fp = counts.most_common(1)[0][0]
    return {
        b: [r for r in rs if _config_fingerprint(r.config, ignore) == best_fp]
        for b, rs in by_branch.items()
    }


def _f2_seed_means(
    wide: pd.DataFrame,
    branches: Sequence[str],
    *,
    seen: Optional[bool] = None,
    regen_unseen_only: bool = False,
) -> Dict[str, Dict[Any, float]]:
    """Per-branch map seed → mean bit_accuracy under the given filter."""
    out: Dict[str, Dict[Any, float]] = {b: {} for b in branches}
    if wide.empty or "bit_accuracy" not in wide.columns:
        return out
    panel = wide[wide["attack"].astype(str) != "clean"].copy()
    if panel.empty:
        return out
    if "seen_flag" in panel.columns:
        flag = panel["seen_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        flag = pd.Series([False] * len(panel), index=panel.index)
    if regen_unseen_only:
        regen_ids = {s.id for s in ATTACK_REGISTRY if s.cls == "regen" and not s.seen_in_training}
        panel = panel[(~flag) & panel["attack"].astype(str).isin(regen_ids)]
    elif seen is True:
        panel = panel[flag]
    elif seen is False:
        panel = panel[~flag]
    for branch in branches:
        sub = panel[panel["regen_branch"].astype(str) == branch]
        for seed, g in sub.groupby("seed"):
            out[branch][seed] = float(g["bit_accuracy"].mean())
    return out


def _paired_diff_ci(
    a: Mapping[Any, float], b: Mapping[Any, float]
) -> Tuple[float, float, int]:
    common = sorted(set(a) & set(b), key=lambda s: str(s))
    diffs = [float(a[s]) - float(b[s]) for s in common]
    return mean_ci95(diffs)


def generate_f2(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Ablation over regen_branch ∈ {ddim_proxy, blur_surrogate, none}."""
    configure_style()
    train_runs = [r for r in runs if r.cost]
    pool = train_runs or list(runs)
    by_branch: Dict[str, List[RunBundle]] = {b: [] for b in F2_BRANCHES}
    for r in pool:
        b = str(r.meta.get("regen_branch", r.config.get("regen_branch", "")))
        if b in by_branch:
            by_branch[b].append(r)

    # Prefer a coherent config cluster (so proxy-step sweeps do not pollute F2),
    # but if that leaves <2 branches, assert on the full set to surface real mismatches.
    filtered = _select_coherent_branch_runs(by_branch, ignore=F2_IGNORE)
    present = [b for b in F2_BRANCHES if filtered[b]]
    if len(present) >= 2:
        by_branch = filtered
    else:
        present = [b for b in F2_BRANCHES if by_branch[b]]
        if len(present) < 2:
            raise SystemExit(
                "F2 abort: need runs for at least two of "
                f"{list(F2_BRANCHES)}; found {present or 'none'}."
            )
    _assert_config_identity(
        [by_branch[b] for b in present],
        ignore=F2_IGNORE,
        fig_id="F2",
        allowed_msg="regen_branch / seed / run_name",
    )

    df = _eval_frame(runs)
    wide = _wide_quality(df)
    n_seeds = max(
        (_n_seeds(df[df["regen_branch"] == b]) if not df.empty else 0) for b in present
    )
    groups = (
        ("seen", UK["seen"], True, False),
        ("unseen", UK["unseen"], False, False),
        ("unseen_regen", "Невідомі регенераційні", None, True),
    )
    group_vals: Dict[str, Dict[str, Dict[Any, float]]] = {}
    for gid, _title, seen_flag, regen_only in groups:
        group_vals[gid] = _f2_seed_means(
            wide, present, seen=seen_flag, regen_unseen_only=regen_only
        )

    # Figure: grouped bars (groups × variants) with CI error bars.
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _warn_seeds(n_seeds, "f2", fig=fig)
    x = np.arange(len(groups))
    width = 0.8 / max(len(present), 1)
    for j, branch in enumerate(present):
        means, errs = [], []
        for gid, *_ in groups:
            m, h, n = mean_ci95(list(group_vals[gid][branch].values()))
            means.append(m if n else np.nan)
            errs.append(h if n and np.isfinite(h) else 0.0)
        offset = (j - (len(present) - 1) / 2) * width
        st = BRANCH_STYLE[branch]
        ax.bar(
            x + offset,
            [0.0 if not np.isfinite(m) else m for m in means],
            width,
            yerr=errs,
            color=st["color"],
            label=st["label"],
            error_kw={"elinewidth": 0.8, "capsize": 2},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t, _, _ in groups], rotation=12, ha="right")
    ax.set_ylabel(UK["decoding_accuracy"])
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()

    # Table: three variant columns + two paired-difference columns.
    headers = (
        ["Група"]
        + [BRANCH_STYLE[b]["label"] for b in F2_BRANCHES]
        + ["Δ (DDIM − сурогат)", "Δ (DDIM − без гілки)"]
    )
    rows: List[List[str]] = []
    for gid, title, _, _ in groups:
        row = [title]
        for branch in F2_BRANCHES:
            if branch in present:
                m, h, n = mean_ci95(list(group_vals[gid][branch].values()))
                row.append(_fmt_ci(m, h, n))
            else:
                row.append(UK["not_eval"])
        ddim = group_vals[gid].get("ddim_proxy", {})
        blur = group_vals[gid].get("blur_surrogate", {})
        none = group_vals[gid].get("none", {})
        if "ddim_proxy" in present and "blur_surrogate" in present:
            md, hd, nd = _paired_diff_ci(ddim, blur)
            row.append(_fmt_ci(md, hd, nd))
        else:
            row.append(UK["not_eval"])
        if "ddim_proxy" in present and "none" in present:
            md, hd, nd = _paired_diff_ci(ddim, none)
            row.append(_fmt_ci(md, hd, nd))
        else:
            row.append(UK["not_eval"])
        rows.append(row)

    artifacts = _save_fig(fig, out_dir, "f2")
    artifacts.update(_write_table(out_dir, "f2", headers, rows, also_figure=False))
    artifacts["branches"] = list(present)
    datasets = sorted({r.dataset for r in runs if r.dataset})
    branch_labels = ", ".join(BRANCH_STYLE[b]["label"] for b in present)
    cap = (
        f"\\caption{{Абляція навчальної гілки регенерації ({branch_labels}). "
        f"Рядки: відомі трансформації, невідомі трансформації та невідомі регенераційні. "
        f"Метрика: {UK['decoding_accuracy']}. {UK['mean_ci']}, "
        f"$n={n_seeds}$ {UK['seeds']}. Набір даних: {', '.join(datasets) or UK['not_eval']}. "
        f"$\\Delta$ — попарні різниці за seed (DDIM $-$ сурогат / DDIM $-$ без гілки). "
        f"Конфігурації ідентичні з точністю до \\texttt{{regen\\_branch}}, seed і run\\_name. "
        f"{'Увага: менше ніж 3 запусків (seed).' if n_seeds < MIN_SEEDS else ''}}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "f2", cap))
    return artifacts


def _n_steps_of(run: RunBundle) -> int:
    regen = run.config.get("regen") or {}
    if isinstance(regen, Mapping) and "n_steps" in regen:
        return int(regen["n_steps"])
    return int(run.config.get("n_steps", run.meta.get("n_steps", 4) or 4))


def _epochs_of(run: RunBundle) -> int:
    return int(run.config.get("epochs", run.meta.get("epochs", 0) or 0))


def _cost_seconds_per_epoch(run: RunBundle) -> Optional[float]:
    if not run.cost:
        return None
    if "seconds_per_epoch_mean" in run.cost:
        try:
            return float(run.cost["seconds_per_epoch_mean"])
        except (TypeError, ValueError):
            pass
    v = run.cost.get("seconds_per_epoch")
    if isinstance(v, list) and v:
        return float(np.mean(v))
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _cost_peak_gpu_gb(run: RunBundle) -> Optional[float]:
    if not run.cost:
        return None
    if "peak_gpu_memory_gb_max" in run.cost:
        try:
            return float(run.cost["peak_gpu_memory_gb_max"])
        except (TypeError, ValueError):
            pass
    v = run.cost.get("peak_gpu_memory_bytes")
    if isinstance(v, list) and v:
        return float(max(v) / (1024**3))
    try:
        return float(v) / (1024**3) if v is not None else None
    except (TypeError, ValueError):
        return None


def generate_f3(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Proxy-steps sweep: unseen-regen accuracy and training cost vs ``regen.n_steps``."""
    configure_style()
    train_runs = [r for r in runs if r.cost] or list(runs)
    # Prefer the largest F3-coherent cluster that still spans ≥2 distinct S values
    # (so main ablation runs with a single S do not block the sweep).
    from collections import defaultdict

    fp_groups: Dict[str, List[RunBundle]] = defaultdict(list)
    for r in train_runs:
        fp_groups[_config_fingerprint(r.config, F3_IGNORE)].append(r)

    def _score(rs: List[RunBundle]) -> Tuple[int, int]:
        return (len({_n_steps_of(r) for r in rs}), len(rs))

    if not fp_groups:
        raise SystemExit("F3 abort: no runs available.")
    chosen = max(fp_groups.values(), key=_score)
    by_s: Dict[int, List[RunBundle]] = {}
    for r in chosen:
        by_s.setdefault(_n_steps_of(r), []).append(r)
    if len(by_s) < 2:
        raise SystemExit(
            "F3 abort: need training runs for at least two distinct regen.n_steps values "
            f"(with configs identical aside from n_steps / epochs / seed / run_name); "
            f"found {sorted(by_s) or 'none'}."
        )
    _assert_config_identity(
        [by_s[s] for s in sorted(by_s)],
        ignore=F3_IGNORE,
        fig_id="F3",
        allowed_msg="n_steps / epochs / seed / run_name",
    )

    steps = sorted(by_s)
    epochs_used = sorted({_epochs_of(r) for rs in by_s.values() for r in rs if _epochs_of(r) > 0})
    epochs_str = (
        str(epochs_used[0])
        if len(epochs_used) == 1
        else (", ".join(str(e) for e in epochs_used) if epochs_used else UK["not_eval"])
    )

    df = _eval_frame(runs)
    wide = _wide_quality(df)
    # Attach n_steps from matching training identity (scheme, seed) when missing.
    if not wide.empty and "n_steps" not in wide.columns:
        wide = wide.copy()
        wide["n_steps"] = np.nan
        for r in runs:
            s = _n_steps_of(r)
            mask = (wide["seed"].astype(str) == str(r.seed)) & (
                wide["scheme"].astype(str) == str(r.scheme)
            )
            if "regen_branch" in wide.columns:
                rb = str(r.meta.get("regen_branch", r.config.get("regen_branch", "")))
                mask = mask & (wide["regen_branch"].astype(str) == rb)
            wide.loc[mask, "n_steps"] = s

    regen_unseen_ids = {
        s.id for s in ATTACK_REGISTRY if s.cls == "regen" and not s.seen_in_training
    }
    acc_by_s: Dict[int, List[float]] = {s: [] for s in steps}
    if not wide.empty and "bit_accuracy" in wide.columns:
        flag = wide["seen_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
        panel = wide[(~flag) & wide["attack"].astype(str).isin(regen_unseen_ids)]
        for s in steps:
            sub = panel[panel["n_steps"] == s] if "n_steps" in panel.columns else panel.iloc[0:0]
            if sub.empty:
                # Fall back: evaluate rows from runs with this S.
                run_keys = {(str(r.scheme), str(r.seed)) for r in by_s[s]}
                sub = panel[
                    panel.apply(
                        lambda row: (str(row["scheme"]), str(row["seed"])) in run_keys,
                        axis=1,
                    )
                ] if not panel.empty else panel
            for seed, g in sub.groupby("seed"):
                acc_by_s[s].append(float(g["bit_accuracy"].mean()))

    sec_by_s: Dict[int, List[float]] = {s: [] for s in steps}
    mem_by_s: Dict[int, List[float]] = {s: [] for s in steps}
    for s in steps:
        for r in by_s[s]:
            sec = _cost_seconds_per_epoch(r)
            mem = _cost_peak_gpu_gb(r)
            if sec is not None:
                sec_by_s[s].append(sec)
            if mem is not None:
                mem_by_s[s].append(mem)

    seeds_per_s = [len(set(str(r.seed) for r in by_s[s])) for s in steps]
    single_seed = max(seeds_per_s) <= 1
    n_seeds = max(seeds_per_s) if seeds_per_s else 0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    _warn_seeds(n_seeds, "f3", fig=fig)
    if single_seed:
        fig.text(
            0.01,
            0.99,
            "1 запуск на значення S",
            transform=fig.transFigure,
            color="crimson",
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
        )

    # (a) accuracy vs S
    means, errs = [], []
    for s in steps:
        m, h, n = mean_ci95(acc_by_s[s])
        means.append(m if n else np.nan)
        errs.append(h if n and np.isfinite(h) else 0.0)
    ax0.errorbar(
        steps,
        [0.0 if not np.isfinite(m) else m for m in means],
        yerr=errs,
        color="#0072B2",
        marker="o",
        capsize=3,
        linewidth=1.2,
    )
    ax0.set_xlabel("Кількість кроків проксі $S$")
    ax0.set_ylabel(UK["decoding_accuracy"])
    ax0.set_xticks(steps)
    ax0.set_ylim(0, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Невідомі регенераційні")

    # (b) cost vs S (twin y-axes)
    sec_m, sec_e, mem_m, mem_e = [], [], [], []
    for s in steps:
        m, h, n = mean_ci95(sec_by_s[s])
        sec_m.append(m if n else np.nan)
        sec_e.append(h if n and np.isfinite(h) else 0.0)
        m2, h2, n2 = mean_ci95(mem_by_s[s])
        mem_m.append(m2 if n2 else np.nan)
        mem_e.append(h2 if n2 and np.isfinite(h2) else 0.0)
    ax1.errorbar(
        steps,
        [0.0 if not np.isfinite(m) else m for m in sec_m],
        yerr=sec_e,
        color="#0072B2",
        marker="o",
        capsize=3,
        label=UK["time_epoch"],
    )
    ax1b = ax1.twinx()
    ax1b.errorbar(
        steps,
        [0.0 if not np.isfinite(m) else m for m in mem_m],
        yerr=mem_e,
        color="#E69F00",
        marker="s",
        capsize=3,
        label=UK["peak_gpu"],
    )
    ax1.set_xlabel("Кількість кроків проксі $S$")
    ax1.set_ylabel(UK["time_epoch"], color="#0072B2")
    ax1b.set_ylabel(UK["peak_gpu"], color="#E69F00")
    ax1.set_xticks(steps)
    ax1.grid(True, alpha=0.3)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=6, loc="upper left")
    ax1.set_title("Вартість навчання")

    fig.tight_layout()
    artifacts = _save_fig(fig, out_dir, "f3")

    headers = [
        "$S$",
        UK["decoding_accuracy"],
        UK["time_epoch"],
        UK["peak_gpu"],
        UK["seeds"],
    ]
    rows: List[List[str]] = []
    for s in steps:
        ma, ha, na = mean_ci95(acc_by_s[s])
        ms, hs, ns = mean_ci95(sec_by_s[s])
        mm, hm, nm = mean_ci95(mem_by_s[s])
        rows.append(
            [
                str(s),
                _fmt_ci(ma, ha, na),
                _fmt_ci(ms, hs, ns),
                _fmt_ci(mm, hm, nm),
                str(len(set(str(r.seed) for r in by_s[s]))),
            ]
        )
    artifacts.update(_write_table(out_dir, "f3", headers, rows, also_figure=False))
    artifacts["n_steps"] = list(steps)
    artifacts["epochs"] = epochs_used
    datasets = sorted({r.dataset for r in runs if r.dataset})
    single_note = " 1 запуск на значення $S$." if single_seed else ""
    cap = (
        f"\\caption{{Залежність від кількості кроків проксі $S$ "
        f"(\\texttt{{regen.n\\_steps}}). "
        f"(а) {UK['decoding_accuracy']} на невідомих регенераційних трансформаціях; "
        f"(б) {UK['time_epoch']} та {UK['peak_gpu']} (з \\texttt{{cost.json}}). "
        f"Навчання sweep: {epochs_str} епох. "
        f"{UK['mean_ci']}, $n={n_seeds}$ {UK['seeds']}. "
        f"Набір даних: {', '.join(datasets) or UK['not_eval']}. "
        f"Конфігурації ідентичні з точністю до $S$, epochs, seed і run\\_name."
        f"{single_note} "
        f"{'Увага: менше ніж 3 запусків (seed).' if n_seeds < MIN_SEEDS and not single_seed else ''}}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "f3", cap))
    return artifacts


def generate_b1(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Architecture / gradient-flow schematic from the resolved config."""
    configure_style()
    cfg = runs[0].config if runs else {}
    p_dist, p_regen, p_adv = curriculum_probs(10**9, cfg)
    p_dist, p_regen, p_adv = curriculum_probs(10**9, cfg)
    branch = str(cfg.get("regen_branch", "ddim_proxy"))
    lam = float(cfg.get("lambda_perc", 1.0))
    n_steps = int(cfg.get("regen", {}).get("n_steps", 4))
    if branch == "ddim_proxy":
        regen_name = f"DDIM\n({n_steps} кроків)"
    elif branch == "blur_surrogate":
        regen_name = "Gaussian\nblur"
    else:
        regen_name = "вимкнено"
        p_regen = 0.0
        p_dist = 1.0 - p_adv

    dot_path = out_dir / "b1.dot"
    out_dir.mkdir(parents=True, exist_ok=True)
    dot = f"""digraph B1 {{
  graph [rankdir=LR, fontname="DejaVu Sans"];
  node [fontname="DejaVu Sans", shape=box, style=rounded];
  E [label="E"];
  xw [label="x_w", shape=ellipse];
  dist [label="A_dist\\np={p_dist:.2f}"];
  regen [label="A_regen {regen_name}\\np={p_regen:.2f}"];
  adv [label="A_adv PGD\\np={p_adv:.2f}"];
  D [label="D"];
  Ldec [label="L_dec"];
  Lperc [label="L_perc λ={lam:g}"];
  vae [label="VAE/UNet", style="dashed,rounded"];
  E -> xw [label="{UK['grad_yes']}"];
  xw -> dist; xw -> regen; xw -> adv;
  dist -> D; regen -> D; adv -> D;
  D -> Ldec;
  xw -> Lperc;
  vae -> regen [style=dashed, label="{UK['grad_no']}"];
}}
"""
    dot_path.write_text(dot, encoding="utf-8")
    rendered = False
    try:
        import graphviz  # type: ignore

        g = graphviz.Source(dot)
        g.render(str(out_dir / "b1_graphviz"), format="pdf", cleanup=True)
        g.render(str(out_dir / "b1_graphviz"), format="png", cleanup=True)
        rendered = True
    except Exception:
        rendered = False

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Архітектура та потік градієнта")

    def box(x, y, w, h, text, fc="#dceaf7", ls="solid", lw=1.4):
        p = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=fc, edgecolor="#222", linestyle=ls, linewidth=lw,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)

    def arrow(x1, y1, x2, y2, dashed=False):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color="#222",
                lw=1.2,
                linestyle=(0, (4, 2)) if dashed else "solid",
            ),
        )

    box(0.3, 2.4, 1.4, 1.1, "E")
    box(2.2, 2.4, 1.4, 1.1, r"$x_w$", fc="#f7f3dc")
    box(4.6, 4.3, 2.2, 1.1, f"dist\n$p$={p_dist:.2f}", fc="#e8f5e9")
    box(4.6, 2.4, 2.2, 1.1, f"regen\n{regen_name}\n$p$={p_regen:.2f}", fc="#fff3e0")
    box(4.6, 0.5, 2.2, 1.1, f"adv PGD\n$p$={p_adv:.2f}", fc="#fce4ec")
    box(7.6, 2.4, 1.4, 1.1, "D")
    box(9.6, 3.3, 2.0, 1.0, r"$\mathcal{L}_{\mathrm{dec}}$", fc="#eceff1")
    box(9.6, 1.5, 2.0, 1.0, rf"$\mathcal{{L}}_{{\mathrm{{perc}}}}$ $\lambda$={lam:g}", fc="#eceff1")
    box(4.6, 3.55, 2.2, 0.6, "VAE/UNet", fc="#eeeeee", ls="dashed") if False else None
    box(2.2, 4.5, 1.8, 0.9, "VAE/UNet", fc="#eeeeee", ls="dashed")

    arrow(1.7, 2.95, 2.2, 2.95)
    arrow(3.6, 3.2, 4.6, 4.7)
    arrow(3.6, 2.95, 4.6, 2.95)
    arrow(3.6, 2.6, 4.6, 1.1)
    arrow(6.8, 4.8, 8.3, 3.3)
    arrow(6.8, 2.95, 7.6, 2.95)
    arrow(6.8, 1.1, 8.3, 2.6)
    arrow(9.0, 3.1, 9.6, 3.7)
    arrow(9.0, 2.6, 9.6, 2.1)
    arrow(2.9, 3.5, 3.1, 4.5, dashed=True)
    arrow(4.0, 4.9, 4.6, 3.0, dashed=True)

    ax.plot([0.4, 1.2], [0.35, 0.35], color="#222", lw=1.4)
    ax.text(1.35, 0.35, UK["grad_yes"], va="center", fontsize=7)
    ax.plot([5.4, 6.2], [0.35, 0.35], color="#222", lw=1.4, ls="--")
    ax.text(6.35, 0.35, UK["grad_no"], va="center", fontsize=7)

    artifacts = _save_fig(fig, out_dir, "b1")
    artifacts["dot"] = str(dot_path)
    n = _n_seeds(_eval_frame(runs)) if runs else 0
    cap = (
        f"\\caption{{Схема архітектури та потоку градієнта, згенерована з розв'язаної конфігурації. "
        f"Суцільні стрілки: {UK['grad_yes']}; пунктир: {UK['grad_no']} "
        f"(переднавчені VAE/UNet). Імовірності гілок атаки: "
        f"$p_{{\\mathrm{{dist}}}}={p_dist:.2f}$, $p_{{\\mathrm{{regen}}}}={p_regen:.2f}$, "
        f"$p_{{\\mathrm{{adv}}}}={p_adv:.2f}$; regen-branch={branch}. "
        f"$n={n}$ {UK['seeds']}.}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "b1", cap))
    artifacts["graphviz"] = rendered
    return artifacts


def generate_d1(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Main comparison table: schemes × attack classes."""
    configure_style()
    df = _eval_frame(runs)
    wide = _wide_quality(df)
    n_seeds = _n_seeds(df)
    headers = [UK["scheme"]] + [CLASS_LABEL_UK.get(c, c) for c in D1_COLUMNS]
    body: List[List[str]] = []
    stats: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for scheme in SCHEME_ORDER:
        stats[scheme] = {}
        for cls in D1_COLUMNS:
            ids = [s.id for s in ATTACK_REGISTRY if s.cls == cls]
            vals = []
            if not wide.empty and "bit_accuracy" in wide.columns:
                sub = wide[(wide["scheme"] == scheme) & (wide["attack"].isin(ids))]
                for _, g in sub.groupby("seed"):
                    vals.append(float(g["bit_accuracy"].mean()))
            stats[scheme][cls] = mean_ci95(vals)

    best = {}
    for cls in D1_COLUMNS:
        finite = [(s, stats[s][cls][0]) for s in SCHEME_ORDER if np.isfinite(stats[s][cls][0])]
        best[cls] = max(finite, key=lambda t: t[1])[0] if finite else None

    for scheme in SCHEME_ORDER:
        row = [_scheme_label(scheme)]
        for cls in D1_COLUMNS:
            m, h, n = stats[scheme][cls]
            row.append(_fmt_ci_tex(m, h, n, bold=(best[cls] == scheme and n > 0)))
        body.append(row)

    artifacts = _write_table(out_dir, "d1", headers, body, title="Порівняння схем за класами атак")
    datasets = sorted({r.dataset for r in runs if r.dataset})
    cap = (
        f"\\caption{{Порівняння схем за класами атак (геометричні дисторсії — окрема колонка). "
        f"Метрика: {UK['bit_accuracy']}. {UK['mean_ci']}, $n={n_seeds}$ {UK['seeds']}. "
        f"Набір даних: {', '.join(datasets) or UK['not_eval']}. "
        f"Найкраще в колонці виділено жирним. Порожні виміри: «{UK['not_eval']}». "
        f"Бюджети — протокол C3 (типова сила). "
        f"{'Увага: менше ніж 3 запусків (seed).' if n_seeds < MIN_SEEDS else ''}}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "d1", cap))
    if n_seeds < MIN_SEEDS:
        print(f"WARNING [d1]: n={n_seeds} seeds (< {MIN_SEEDS}).")
    return artifacts


def generate_e1(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Watermarked-image quality (no attack)."""
    configure_style()
    df = _eval_frame(runs)
    wide = _wide_quality(df)
    n_seeds = _n_seeds(df)
    headers = [
        UK["scheme"],
        UK["dataset"],
        UK["payload"],
        "PSNR",
        "SSIM",
        "LPIPS",
        "FID",
    ]
    rows: List[List[str]] = []
    payload_map = {}
    for r in runs:
        info = SCHEME_INFO.get(r.scheme, {})
        if r.scheme == "ours":
            payload_map[r.scheme] = int(
                r.config.get("msg_len") or r.meta.get("payload_bits") or 0
            )
        else:
            payload_map[r.scheme] = int(
                info.get("payload_bits")
                or r.meta.get("payload_bits")
                or r.config.get("msg_len")
                or 0
            )
    schemes = list(SCHEME_ORDER)
    datasets = sorted({r.dataset for r in runs if r.dataset}) or [""]
    for scheme in schemes:
        for ds in datasets:
            def _col(metric: str) -> str:
                vals = []
                if not wide.empty and metric in wide.columns:
                    sub = wide[(wide["scheme"] == scheme) & (wide["attack"] == "clean")]
                    if ds and "dataset" in sub.columns:
                        sub = sub[sub["dataset"] == ds]
                    for _, g in sub.groupby("seed"):
                        vals.append(float(g[metric].mean()))
                m, h, n = mean_ci95(vals)
                return _fmt_ci(m, h, n)

            bits = payload_map.get(scheme)
            if not bits:
                bits = SCHEME_INFO.get(scheme, {}).get("payload_bits") or UK["not_eval"]
            rows.append(
                [
                    _scheme_label(scheme),
                    ds or UK["not_eval"],
                    str(bits),
                    _col("psnr"),
                    _col("ssim"),
                    _col("lpips"),
                    _col("fid"),
                ]
            )
    artifacts = _write_table(out_dir, "e1", headers, rows, title="Якість зображення з водяним знаком")
    cap = (
        f"\\caption{{Якість зображення з водяним знаком відносно обкладинки (без атаки). "
        f"{UK['mean_ci']}, $n={n_seeds}$ {UK['seeds']}. "
        f"Набір даних: {', '.join(datasets) or UK['not_eval']}. "
        f"Колонка «{UK['payload']}» потрібна, бо якість і ємність є компромісом. "
        f"{'Увага: менше ніж 3 запусків (seed).' if n_seeds < MIN_SEEDS else ''}}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "e1", cap))
    if n_seeds < MIN_SEEDS:
        print(f"WARNING [e1]: n={n_seeds} seeds (< {MIN_SEEDS}).")
    return artifacts


def generate_c1(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Datasets table from the manifest (+ run split)."""
    configure_style()
    data_root = Path("./data")
    if runs:
        data_root = Path(str(runs[0].config.get("data_root", "./data")))
    manifest = load_manifest(data_root)
    datasets = manifest.get("datasets") or {}
    headers = [
        UK["dataset"],
        "Спліт",
        "К-сть зображень",
        "Роздільність",
        "Роль",
        "Джерело",
    ]
    rows: List[List[str]] = []
    role_map = {
        "coco": "train",
        "div2k": "test",
        "imagenette": "val",
        "imagenet": "test",
        "clic": "test",
        "synthetic": "train/val (smoke)",
    }
    source_map = {
        "coco": "MS-COCO",
        "div2k": "DIV2K",
        "imagenette": "Imagenette",
        "imagenet": "ImageNet (ручне)",
        "clic": "CLIC",
        "synthetic": "процедурний (torch.rand)",
    }
    if not datasets and runs:
        cfg = runs[0].config
        name = str(cfg.get("dataset", "synthetic"))
        n = cfg.get("num_samples") or ""
        val_f = float(cfg.get("val_fraction", 0.1))
        res = int(cfg.get("image_size", 128))
        rows.append([name, "train", str(n), str(res), "train", source_map.get(name, name)])
        rows.append([name, "val", f"val_fraction={val_f}", str(res), "val", source_map.get(name, name)])
    else:
        for name, entry in datasets.items():
            split = str(entry.get("split") or "")
            n = entry.get("num_images", entry.get("n", ""))
            res = ""
            if runs:
                res = str(runs[0].config.get("image_size", ""))
            rows.append(
                [
                    name,
                    split,
                    str(n),
                    str(res) if res else UK["not_eval"],
                    role_map.get(name, ""),
                    source_map.get(name, str(entry.get("path", ""))),
                ]
            )
        # Make the train vs eval split visible from the run config even on one source.
        if runs:
            cfg = runs[0].config
            name = str(cfg.get("dataset", ""))
            val_f = float(cfg.get("val_fraction", 0.1))
            rows.append(
                [
                    name or "run",
                    f"train/val split, val_fraction={val_f}",
                    str(cfg.get("num_samples") or ""),
                    str(cfg.get("image_size", "")),
                    "train ≠ eval split",
                    "конфігурація запуску",
                ]
            )
    if not rows:
        rows.append([UK["not_eval"]] * 6)
    artifacts = _write_table(out_dir, "c1", headers, rows, title="Набори даних")
    cap = (
        f"\\caption{{Набори даних. Навчання і оцінювання використовують різні спліти "
        f"(колонка «Роль»). Джерело — маніфест \\texttt{{data/manifest.json}} та конфіг запуску.}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "c1", cap))
    return artifacts


def generate_c2(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Training hyperparameters from resolved config + run_meta."""
    configure_style()
    cfg = runs[0].config if runs else {}
    meta = runs[0].meta if runs else {}
    seeds = sorted({str(r.seed) for r in runs})
    cur = cfg.get("curriculum", {})
    pgd = cfg.get("pgd", {})
    regen = cfg.get("regen", {})
    hw = meta.get("hardware") or ("cuda" if torch.cuda.is_available() else "cpu")
    items = [
        ("Оптимізатор", "Adam"),
        ("Швидкість навчання", str(cfg.get("lr", ""))),
        ("Розмір батча", str(cfg.get("batch_size", ""))),
        (UK["epoch"], str(cfg.get("epochs", ""))),
        (UK["payload"], str(cfg.get("msg_len", meta.get("payload_bits", "")))),
        ("Роздільність", str(cfg.get("image_size", meta.get("train_resolution", "")))),
        (r"$\lambda_{\mathrm{perc}}$", str(cfg.get("lambda_perc", ""))),
        ("Кроки DDIM-проксі", str(regen.get("n_steps", ""))),
        (r"PGD $(\varepsilon, \alpha, T)$",
         f"({pgd.get('eps', '')}, {pgd.get('alpha', '')}, {pgd.get('steps', '')})"),
        ("Навчальний план",
         f"p0={cur.get('p0', 0.1)}, kappa={cur.get('kappa', 0.01)}, "
         f"p_max={cur.get('p_max', 0.5)}, p_adv={cur.get('p_adv', 0.2)}"),
        ("Обладнання", str(hw)),
        (UK["seeds"], ", ".join(seeds) or UK["not_eval"]),
        ("Гілка регенерації", str(cfg.get("regen_branch", meta.get("regen_branch", "")))),
        ("Хеш конфігурації", str(meta.get("config_hash", ""))),
    ]
    headers = ["Гіперпараметр", "Значення"]
    rows = [[k, v] for k, v in items]
    artifacts = _write_table(out_dir, "c2", headers, rows, colspec="ll", title="Гіперпараметри навчання")
    cap = (
        f"\\caption{{Гіперпараметри навчання (розв'язана конфігурація та \\texttt{{run\\_meta.json}}). "
        f"Набір даних: {meta.get('dataset', cfg.get('dataset', ''))}. "
        f"$n={len(seeds)}$ {UK['seeds']}.}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "c2", cap))
    return artifacts


def generate_h1(runs: Sequence[RunBundle], out_dir: Path, **_: Any) -> Dict[str, Any]:
    """Computational cost table from cost.json."""
    configure_style()
    headers = [
        UK["scheme"],
        UK["time_epoch"],
        UK["peak_gpu"],
        "Повний час, с",
        UK["params_m"],
        UK["embed_ms"],
        UK["decode_ms"],
    ]
    rows: List[List[str]] = []
    by_scheme: Dict[str, List[RunBundle]] = {s: [] for s in SCHEME_ORDER}
    for r in runs:
        by_scheme.setdefault(r.scheme, []).append(r)

    for scheme in SCHEME_ORDER:
        group = [r for r in by_scheme.get(scheme, []) if r.cost]
        def col(key: str, scale: float = 1.0) -> str:
            vals = []
            for r in group:
                v = r.cost.get(key)
                if isinstance(v, list) and v:
                    v = float(np.mean(v))
                try:
                    vals.append(float(v) * scale)
                except (TypeError, ValueError):
                    pass
            m, h, n = mean_ci95(vals)
            return _fmt_ci(m, h, n)

        params_vals = []
        for r in group:
            p = r.cost.get("params_total")
            if p is not None:
                params_vals.append(float(p) / 1e6)
        pm, ph, pn = mean_ci95(params_vals)
        rows.append(
            [
                _scheme_label(scheme),
                col("seconds_per_epoch_mean") if any("seconds_per_epoch_mean" in r.cost for r in group)
                else col("seconds_per_epoch"),
                col("peak_gpu_memory_gb_max") if any("peak_gpu_memory_gb_max" in r.cost for r in group)
                else col("peak_gpu_memory_bytes", 1.0 / (1024**3)),
                col("total_training_wall_clock_s"),
                _fmt_ci(pm, ph, pn),
                col("embed_latency_ms"),
                col("decode_latency_ms"),
            ]
        )
    artifacts = _write_table(out_dir, "h1", headers, rows, title="Обчислювальна вартість")
    n_seeds = len({r.seed for r in runs if r.cost})
    cap = (
        f"\\caption{{Обчислювальна вартість (з \\texttt{{cost.json}}). "
        f"{UK['mean_ci']}, $n={n_seeds}$ {UK['seeds']}. "
        f"Значення «{UK['not_eval']}» означають відсутність виміру, а не нульову вартість. "
        f"Запропонований метод може бути дорожчим за базові схеми — це відображено чесно. "
        f"{'Увага: менше ніж 3 запусків (seed).' if n_seeds < MIN_SEEDS else ''}}}"
    )
    artifacts["caption"] = str(_write_caption(out_dir, "h1", cap))
    if n_seeds < MIN_SEEDS:
        print(f"WARNING [h1]: n={n_seeds} seeds (< {MIN_SEEDS}).")
    return artifacts


GENERATORS = {
    "d4": generate_d4,
    "c3": generate_c3,
    "g1": generate_g1,
    "f2": generate_f2,
    "f3": generate_f3,
    "b1": generate_b1,
    "d1": generate_d1,
    "e1": generate_e1,
    "c1": generate_c1,
    "c2": generate_c2,
    "h1": generate_h1,
}


def generate_all(
    figure: str,
    run_dirs: Sequence[Path | str],
    out_dir: Path | str,
    x_axis: str = "aggregate",
    c3_scheme: str = "ours",
) -> Dict[str, Any]:
    configure_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(run_dirs)
    ids = list(FIGURE_IDS) if figure == "all" else [figure.lower()]
    manifest: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "source_runs": [str(Path(p).resolve()) for p in run_dirs],
        "figures": {},
        "n_runs": len(runs),
        "c3_scheme": c3_scheme,
    }
    for fid in ids:
        if fid not in GENERATORS:
            raise SystemExit(f"Unknown figure id: {fid}. Choose from {', '.join(FIGURE_IDS)}, all")
        print(f"[viz] generating {fid} ...")
        try:
            art = GENERATORS[fid](
                runs, out_dir, x_axis=x_axis, c3_scheme=c3_scheme
            )
        except SystemExit as e:
            # Keep ``--figure all`` robust: record the abort and continue.
            if figure == "all":
                print(f"[viz] FAILED {fid}: {e}")
                art = {"error": str(e)}
            else:
                raise
        except Exception as e:
            print(f"[viz] FAILED {fid}: {e}")
            art = {"error": str(e)}
        manifest["figures"][fid] = art
    man_path = out_dir / "manifest.json"
    if figure != "all" and man_path.exists():
        try:
            prev = json.loads(man_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and isinstance(prev.get("figures"), dict):
                prev["figures"].update(manifest["figures"])
                prev["generated_at"] = manifest["generated_at"]
                prev["source_runs"] = manifest["source_runs"]
                manifest = prev
        except Exception:
            pass
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print(f"[viz] wrote {man_path}")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper figures/tables from run directories")
    p.add_argument(
        "--figure",
        type=str,
        required=True,
        help="Figure id (d4,c3,g1,f2,f3,b1,d1,e1,c1,c2,h1) or 'all'",
    )
    p.add_argument("--runs", nargs="+", required=True, help="Run directories")
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument(
        "--x-axis",
        type=str,
        default="aggregate",
        choices=["aggregate", "psnr", "lpips"],
        help="D4 x-axis: aggregate quality degradation, raw PSNR, or raw LPIPS",
    )
    p.add_argument(
        "--c3-scheme",
        type=str,
        default="ours",
        help="Scheme id whose measured pert_l2 / pert_linf / pert_lpips fill C3 "
        "(default: ours)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    generate_all(
        args.figure,
        args.runs,
        args.out,
        x_axis=args.x_axis,
        c3_scheme=args.c3_scheme,
    )


if __name__ == "__main__":
    main()
