"""Acceptance tests for F2 (regen-branch ablation) and F3 (proxy-steps sweep)."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.viz import (
    BRANCH_STYLE,
    FIGURE_IDS,
    GENERATORS,
    F2_BRANCHES,
    generate_all,
    generate_f2,
    generate_f3,
    load_runs,
)


def _write_run(
    root: Path,
    *,
    name: str,
    seed: int,
    regen_branch: str = "ddim_proxy",
    n_steps: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    bit_acc_seen: float = 0.9,
    bit_acc_unseen: float = 0.7,
    bit_acc_unseen_regen: float = 0.65,
    seconds_per_epoch: float = 10.0,
    peak_gpu_gb: float = 4.0,
    extra_cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg: Dict[str, Any] = {
        "seed": seed,
        "run_name": name,
        "regen_branch": regen_branch,
        "epochs": epochs,
        "lr": lr,
        "dataset": "synthetic",
        "scheme": "ours",
        "msg_len": 16,
        "image_size": 64,
        "batch_size": 4,
        "regen": {"n_steps": n_steps, "t_start": 0.3, "use_placeholder": True},
        "curriculum": {"p0": 0.1, "kappa": 0.0, "p_max": 0.1, "p_adv": 0.2},
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    (run_dir / "config.yaml").write_text(
        __import__("yaml").safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    meta = {
        "scheme": "ours",
        "seed": seed,
        "dataset": "synthetic",
        "regen_branch": regen_branch,
        "epochs": epochs,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "results_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "cost.json").write_text(
        json.dumps(
            {
                "seconds_per_epoch_mean": seconds_per_epoch + 0.1 * seed,
                "seconds_per_epoch": [seconds_per_epoch] * 3,
                "peak_gpu_memory_gb_max": peak_gpu_gb + 0.05 * seed,
                "peak_gpu_memory_bytes": int((peak_gpu_gb + 0.05 * seed) * 1024**3),
                "total_training_wall_clock_s": epochs * seconds_per_epoch,
            }
        ),
        encoding="utf-8",
    )
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
    # Seen: jpeg; unseen non-regen: unseen_blur; unseen regen: diffpure
    for attack, seen, acc in (
        ("jpeg", "true", bit_acc_seen + 0.01 * seed),
        ("unseen_blur", "false", bit_acc_unseen + 0.01 * seed),
        ("diffpure", "false", bit_acc_unseen_regen + 0.01 * seed),
    ):
        rows.append(
            {
                "epoch": "",
                "global_step": "",
                "seed": str(seed),
                "attack": attack,
                "strength": "0",
                "epsilon": "0",
                "delta": "0",
                "knowledge_level": "black",
                "seen_flag": seen,
                "source": "",
                "metric_name": "bit_accuracy",
                "metric_value": str(acc),
            }
        )
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return run_dir


def test_f2_three_variants_table_and_diffs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dirs = []
        # Make ddim clearly better so Δ columns are nonzero.
        for seed in (0, 1, 2):
            dirs.append(
                _write_run(
                    root,
                    name=f"ddim_s{seed}",
                    seed=seed,
                    regen_branch="ddim_proxy",
                    bit_acc_seen=0.92,
                    bit_acc_unseen=0.80,
                    bit_acc_unseen_regen=0.75,
                )
            )
            dirs.append(
                _write_run(
                    root,
                    name=f"blur_s{seed}",
                    seed=seed,
                    regen_branch="blur_surrogate",
                    bit_acc_seen=0.85,
                    bit_acc_unseen=0.70,
                    bit_acc_unseen_regen=0.60,
                )
            )
            dirs.append(
                _write_run(
                    root,
                    name=f"none_s{seed}",
                    seed=seed,
                    regen_branch="none",
                    bit_acc_seen=0.80,
                    bit_acc_unseen=0.55,
                    bit_acc_unseen_regen=0.40,
                )
            )
        out = root / "figs"
        arts = generate_f2(load_runs(dirs), out)
        assert set(arts["branches"]) == set(F2_BRANCHES)
        csv_text = (out / "f2.csv").read_text(encoding="utf-8")
        header = csv_text.splitlines()[0]
        for label in (
            BRANCH_STYLE["ddim_proxy"]["label"],
            BRANCH_STYLE["blur_surrogate"]["label"],
            BRANCH_STYLE["none"]["label"],
        ):
            assert label in header
        assert "Δ (DDIM − сурогат)" in header
        assert "Δ (DDIM − без гілки)" in header
        assert len(csv_text.splitlines()) == 4  # header + 3 groups
        # Difference cells should not be the not_eval marker for all rows.
        body = "\n".join(csv_text.splitlines()[1:])
        assert "не вимірювалося" not in body or "±" in body
        cap = (out / "f2_caption.tex").read_text(encoding="utf-8")
        assert "DDIM" in cap
        assert (out / "f2.pdf").is_file() and (out / "f2.png").is_file()
        print("[ok] F2 three-variant table with two difference columns")


def test_f2_aborts_on_config_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = _write_run(root, name="a", seed=0, regen_branch="ddim_proxy", lr=1e-3)
        b = _write_run(
            root, name="b", seed=0, regen_branch="blur_surrogate", lr=2e-3  # mismatch
        )
        try:
            generate_f2(load_runs([a, b]), root / "out")
            raised = False
        except SystemExit as e:
            raised = True
            msg = str(e)
            assert "F2 abort" in msg
            assert "lr" in msg
        assert raised, "expected SystemExit on config mismatch"
        print("[ok] F2 aborts when configs differ beyond regen_branch/seed/run_name")


def test_f3_proxy_steps_panels_and_caption() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dirs = []
        for s in (2, 4, 6, 8):
            dirs.append(
                _write_run(
                    root,
                    name=f"s{s}",
                    seed=0,
                    n_steps=s,
                    epochs=20,  # reduced budget vs a typical main run
                    seconds_per_epoch=5.0 + s,
                    peak_gpu_gb=3.0 + 0.2 * s,
                    bit_acc_unseen_regen=0.5 + 0.03 * s,
                )
            )
        out = root / "figs"
        arts = generate_f3(load_runs(dirs), out)
        assert arts["n_steps"] == [2, 4, 6, 8]
        assert 20 in arts["epochs"]
        cap = (out / "f3_caption.tex").read_text(encoding="utf-8")
        assert "20" in cap  # epoch count
        assert "кроків проксі" in cap or "n_steps" in cap or "$S$" in cap
        assert "1 запуск на значення" in cap
        assert (out / "f3.pdf").is_file() and (out / "f3.png").is_file()
        csv_text = (out / "f3.csv").read_text(encoding="utf-8")
        assert "Точність декодування" in csv_text
        assert "Час навчання на епоху" in csv_text
        assert "Пік пам'яті GPU" in csv_text
        print("[ok] F3 panels + caption with epoch count")


def test_viz_all_includes_f2_f3() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dirs = []
        # Main ablation cluster (n_steps=4, epochs=100) — used by F2.
        for seed in (0, 1):
            for branch in F2_BRANCHES:
                dirs.append(
                    _write_run(
                        root,
                        name=f"{branch}_s{seed}",
                        seed=seed,
                        regen_branch=branch,
                        n_steps=4,
                        epochs=100,
                    )
                )
        # Reduced-budget proxy-steps sweep — used by F3.
        for s in (2, 4, 6, 8):
            dirs.append(
                _write_run(
                    root,
                    name=f"sweep_s{s}",
                    seed=0,
                    n_steps=s,
                    epochs=20,
                    regen_branch="ddim_proxy",
                    seconds_per_epoch=5.0 + s,
                    peak_gpu_gb=3.0 + 0.1 * s,
                )
            )
        out = root / "all_figs"
        assert "f2" in FIGURE_IDS and "f3" in FIGURE_IDS
        assert "f2" in GENERATORS and "f3" in GENERATORS
        man = generate_all("all", dirs, out)
        assert "f2" in man["figures"] and "f3" in man["figures"]
        assert "error" not in man["figures"]["f2"], man["figures"]["f2"]
        assert "error" not in man["figures"]["f3"], man["figures"]["f3"]
        assert (out / "f2.pdf").is_file() and (out / "f3.pdf").is_file()
        print("[ok] viz --figure all manifest includes f2 and f3 outputs")


def test_branch_uk_labels() -> None:
    assert BRANCH_STYLE["ddim_proxy"]["label"] == "DDIM-проксі"
    assert BRANCH_STYLE["blur_surrogate"]["label"] == "Сурогат-розмиття"
    assert BRANCH_STYLE["none"]["label"] == "Без регенераційної гілки"
    print("[ok] Ukrainian branch labels")


if __name__ == "__main__":
    test_branch_uk_labels()
    test_f2_three_variants_table_and_diffs()
    test_f2_aborts_on_config_mismatch()
    test_f3_proxy_steps_panels_and_caption()
    test_viz_all_includes_f2_f3()
    print("=== all F2/F3 acceptance tests passed ===")
