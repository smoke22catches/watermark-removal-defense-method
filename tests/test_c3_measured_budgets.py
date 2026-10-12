"""Acceptance tests for C3 measured perturbation / LPIPS budgets."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics import mean_ci95
from src.utils.viz import UK, _fmt_ci_meas, generate_all, generate_c3, load_runs


def _write_run(
    root: Path,
    *,
    name: str,
    seed: int,
    scheme: str = "ours",
    num_samples: int = 8,
    pert_rows: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg: Dict[str, Any] = {
        "seed": seed,
        "run_name": name,
        "dataset": "synthetic",
        "scheme": scheme,
        "msg_len": 16,
        "image_size": 64,
        "batch_size": 4,
        "num_samples": num_samples,
    }
    (run_dir / "config.yaml").write_text(
        __import__("yaml").safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    meta = {
        "scheme": scheme,
        "seed": seed,
        "dataset": "synthetic",
        "splits": {"counts": {"train": 32, "val": 8, "test": num_samples}},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "results_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "cost.json").write_text("{}", encoding="utf-8")

    fields = [
        "epoch",
        "global_step",
        "seed",
        "attack",
        "strength",
        "epsilon",
        "delta",
        "knowledge_level",
        "seen_flag",
        "source",
        "metric_name",
        "metric_value",
    ]
    rows: List[Dict[str, Any]] = []
    for pr in pert_rows or []:
        for metric, value in (
            ("pert_l2", pr["pert_l2"]),
            ("pert_linf", pr["pert_linf"]),
            ("pert_lpips", pr["pert_lpips"]),
            ("bit_accuracy", pr.get("bit_accuracy", 0.9)),
        ):
            rows.append(
                {
                    "epoch": "",
                    "global_step": "",
                    "seed": str(seed),
                    "attack": pr["attack"],
                    "strength": str(pr["strength"]),
                    # Static spec fields — must NOT leak into measured C3 columns.
                    "epsilon": str(pr.get("epsilon", 99.0)),
                    "delta": str(pr.get("delta", 88.0)),
                    "knowledge_level": pr.get("knowledge_level", "black"),
                    "seen_flag": pr.get("seen_flag", "true"),
                    "source": "",
                    "metric_name": metric,
                    "metric_value": str(value),
                }
            )
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return run_dir


def test_c3_reproduces_known_means_and_cis() -> None:
    """Synthetic results.csv → C3 mean±95% CI matches mean_ci95 on seed values."""
    # Two seeds, jpeg@70: L2 = {0.10, 0.30} → mean 0.20
    # adv@4/255: L∞ = {0.01, 0.03}
    jpeg_l2 = [0.10, 0.30]
    jpeg_linf = [0.02, 0.06]
    jpeg_lpips = [0.04, 0.08]
    adv_strength = 4 / 255
    adv_l2 = [0.5, 0.7]
    adv_linf = [0.01, 0.03]
    adv_lpips = [0.1, 0.2]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        dirs = []
        for i, seed in enumerate((1, 2)):
            dirs.append(
                _write_run(
                    root,
                    name=f"ours_s{seed}",
                    seed=seed,
                    scheme="ours",
                    num_samples=8,
                    pert_rows=[
                        {
                            "attack": "jpeg",
                            "strength": 70.0,
                            "pert_l2": jpeg_l2[i],
                            "pert_linf": jpeg_linf[i],
                            "pert_lpips": jpeg_lpips[i],
                            "epsilon": 0.70,  # static — must not appear as measured
                            "delta": 0.20,
                        },
                        {
                            "attack": "adv",
                            "strength": adv_strength,
                            "pert_l2": adv_l2[i],
                            "pert_linf": adv_linf[i],
                            "pert_lpips": adv_lpips[i],
                            "epsilon": 8 / 255,
                            "delta": 0.10,
                        },
                    ],
                )
            )
        # Distractor scheme — must be ignored when c3_scheme=ours
        dirs.append(
            _write_run(
                root,
                name="baseline_s1",
                seed=1,
                scheme="stegastamp",
                pert_rows=[
                    {
                        "attack": "jpeg",
                        "strength": 70.0,
                        "pert_l2": 9.99,
                        "pert_linf": 9.99,
                        "pert_lpips": 9.99,
                    }
                ],
            )
        )
        runs = load_runs(dirs)
        art = generate_c3(runs, out, c3_scheme="ours")
        assert (out / "c3.csv").exists()
        assert (out / "c3.tex").exists()
        assert (out / "c3_caption.tex").exists()

        csv_df = __import__("pandas").read_csv(out / "c3.csv")
        jpeg = csv_df[(csv_df["attack"] == "jpeg") & (csv_df["strength"] == 70.0)].iloc[0]
        adv = csv_df[
            (csv_df["attack"] == "adv")
            & (abs(csv_df["strength"].astype(float) - adv_strength) < 1e-9)
        ].iloc[0]

        exp_l2 = _fmt_ci_meas(*mean_ci95(jpeg_l2))
        exp_linf = _fmt_ci_meas(*mean_ci95(jpeg_linf))
        exp_lp = _fmt_ci_meas(*mean_ci95(jpeg_lpips))
        assert jpeg["pert_l2"] == exp_l2
        assert jpeg["pert_linf"] == exp_linf
        assert jpeg["pert_lpips"] == exp_lp
        # Static epsilon/delta must not appear in measured cells.
        assert "0.70" not in str(jpeg["pert_l2"])
        assert "99" not in str(jpeg["pert_l2"])

        assert adv["pert_l2"] == _fmt_ci_meas(*mean_ci95(adv_l2))
        assert adv["pert_linf"] == _fmt_ci_meas(*mean_ci95(adv_linf))
        assert adv["pert_lpips"] == _fmt_ci_meas(*mean_ci95(adv_lpips))
        # Configured L∞ budget column for gradient-based attack.
        assert math.isclose(
            float(adv["configured_linf_budget"]), adv_strength, rel_tol=0, abs_tol=1e-9
        )
        # jpeg has no configured budget.
        assert str(jpeg["configured_linf_budget"]) in ("", "nan")

        cap = (out / "c3_caption.tex").read_text(encoding="utf-8")
        assert "ours" in cap
        assert "n=2" in cap
        assert "16" in cap or str(art["n_samples"]) in cap  # 8 samples × 2 seeds
        assert art["scheme"] == "ours"
        assert art["n_seeds"] == 2
        assert art["n_samples"] == 16


def test_c3_without_runs_shows_not_measured() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        art = generate_c3([], out, c3_scheme="ours")
        assert (out / "c3.csv").exists()
        csv_df = __import__("pandas").read_csv(out / "c3.csv")
        assert len(csv_df) > 0
        for col in ("pert_l2", "pert_linf", "pert_lpips"):
            assert (csv_df[col] == UK["not_eval"]).all(), col
        # No static epsilon/delta columns at all.
        assert "epsilon" not in csv_df.columns
        assert "delta" not in csv_df.columns
        tex = (out / "c3.tex").read_text(encoding="utf-8")
        assert UK["not_eval"] in tex
        # Static registry epsilon values must not appear as measured cells.
        # (e.g. jpeg epsilon=0.70 was previously dumped into ε column)
        assert "0.70" not in tex or "Заданий бюджет" in tex
        cap = (out / "c3_caption.tex").read_text(encoding="utf-8")
        assert UK["not_eval"] in cap or "немає evaluation" in cap
        assert art["n_seeds"] == 0


def test_c3_scheme_cli_option() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        run = _write_run(
            root,
            name="ss_s1",
            seed=7,
            scheme="stegastamp",
            num_samples=4,
            pert_rows=[
                {
                    "attack": "noise",
                    "strength": 0.05,
                    "pert_l2": 0.42,
                    "pert_linf": 0.11,
                    "pert_lpips": 0.03,
                }
            ],
        )
        generate_all("c3", [run], out, c3_scheme="stegastamp")
        csv_df = __import__("pandas").read_csv(out / "c3.csv")
        noise = csv_df[
            (csv_df["attack"] == "noise") & (csv_df["strength"] == 0.05)
        ].iloc[0]
        assert noise["pert_l2"] == "0.42"
        assert noise["scheme"] == "stegastamp"
        cap = (out / "c3_caption.tex").read_text(encoding="utf-8")
        assert "stegastamp" in cap


if __name__ == "__main__":
    test_c3_reproduces_known_means_and_cis()
    test_c3_without_runs_shows_not_measured()
    test_c3_scheme_cli_option()
    print("ok")
