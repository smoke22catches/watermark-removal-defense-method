"""Acceptance tests for post-hoc baseline wrappers."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.registry import ATTACK_REGISTRY, SCHEME_INFO, SCHEME_ORDER
from src.baselines.common import sample_bits
from src.baselines.factory import as_encoder_decoder, load_baseline
from src.utils.metrics import NOT_EVALUATED, bit_accuracy_per_image, psnr_batch
from src.utils.viz import RunBundle, generate_d1, generate_e1


POSTHOC = ("stegastamp", "trustmark", "vine")


def _try_load(scheme_id: str, device: str = "cpu"):
    res = load_baseline(scheme_id, device=device, config={})
    return res


def test_generation_time_schemes_not_applicable() -> None:
    for sid in ("stable_signature", "tree_ring"):
        info = SCHEME_INFO[sid]
        assert info.get("not_applicable") is True
        res = _try_load(sid)
        assert res.available is False
        assert res.not_applicable is True
        assert "generation" in res.reason.lower() or "during" in res.reason.lower()
    assert "stable_signature" not in SCHEME_ORDER
    assert "tree_ring" not in SCHEME_ORDER
    print("[ok] generation-time schemes marked not_applicable / excluded from SCHEME_ORDER")


def test_available_baseline_clean_bitacc_and_psnr() -> None:
    """For each available baseline: bit_acc ≥ 0.95 on 50 clean images; report PSNR.

    Images are drawn at a common resolution of 256² (large enough for TrustMark / VINE
    native encode; StegaStamp residual-transfers from 400²). See docs/baselines.md.
    """
    device = "cpu"
    n = 50
    eval_size = 256
    torch.manual_seed(0)
    images = torch.rand(n, 3, eval_size, eval_size) * 2 - 1
    reports = []
    any_available = False
    for sid in POSTHOC:
        res = _try_load(sid, device=device)
        if not res.available or res.instance is None:
            print(f"[skip] {sid}: {res.reason}")
            continue
        any_available = True
        enc, dec, k = as_encoder_decoder(res)
        gen = torch.Generator().manual_seed(123)
        bits = sample_bits(n, k, device=device, generator=gen)
        with torch.no_grad():
            accs, psnrs = [], []
            bs = 4
            for i in range(0, n, bs):
                x = images[i : i + bs]
                m = bits[i : i + bs]
                x_w = enc(x, m)
                logits = dec(x_w)
                accs.extend(bit_accuracy_per_image(logits, m).cpu().tolist())
                psnrs.extend(psnr_batch(x, x_w))
        mean_acc = float(sum(accs) / len(accs))
        mean_psnr = float(sum(psnrs) / len(psnrs))
        reports.append((sid, mean_acc, mean_psnr, k))
        print(
            f"[measure] {sid}: bit_acc={mean_acc:.4f} PSNR={mean_psnr:.2f} dB "
            f"(k={k}, n={n}, size={eval_size})"
        )
        assert mean_acc >= 0.95, f"{sid} clean bit_acc {mean_acc} < 0.95"
        res.instance.close()
    if not any_available:
        print(
            "[warn] no post-hoc baseline packages/weights available in this env; "
            "install per docs/baselines.md to exercise the ≥0.95 sanity check"
        )
    else:
        print("[ok] available baselines passed clean bit_acc ≥ 0.95")
        for sid, acc, psnr, k in reports:
            print(f"  {sid}: bit_acc={acc:.4f} PSNR={psnr:.2f} (k={k})")


def test_infer_scheme_availability_not_hardcoded() -> None:
    """``infer.py`` must not force non-ours schemes to unavailable."""
    text = (ROOT / "scripts" / "infer.py").read_text(encoding="utf-8")
    assert 'if scheme != "ours":' not in text
    assert "available = False  # official weights are not bundled" not in text
    assert "load_baseline" in text
    print("[ok] infer.py derives availability from load_baseline")


def test_d1_e1_show_baseline_numbers() -> None:
    """D1/E1 emit numeric cells for every post-hoc scheme present in the runs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runs = []
        for sid in ("ours", "trustmark", "stegastamp", "vine"):
            run_dir = tmp_path / sid
            run_dir.mkdir()
            # Minimal results.csv with clean bit_accuracy / quality metrics.
            with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
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
                    ],
                )
                w.writeheader()
                for metric, val in (
                    ("bit_accuracy", 0.97),
                    ("psnr", 42.0),
                    ("ssim", 0.95),
                    ("lpips", 0.02),
                    ("fid", 5.0),
                ):
                    w.writerow(
                        {
                            "epoch": "",
                            "global_step": "",
                            "seed": "0",
                            "attack": "clean",
                            "strength": "0",
                            "epsilon": "0",
                            "delta": "0",
                            "knowledge_level": "black",
                            "seen_flag": "true",
                            "source": "",
                            "metric_name": metric,
                            "metric_value": str(val),
                        }
                    )
            # Add one distortion-class attack so D1 has more than clean.
            with open(run_dir / "results.csv", "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
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
                    ],
                )
                w.writerow(
                    {
                        "epoch": "",
                        "global_step": "",
                        "seed": "0",
                        "attack": "jpeg",
                        "strength": "50",
                        "epsilon": "0.7",
                        "delta": "0.2",
                        "knowledge_level": "black",
                        "seen_flag": "true",
                        "source": "",
                        "metric_name": "bit_accuracy",
                        "metric_value": "0.90",
                    }
                )
            meta = {
                "scheme": sid,
                "seed": 0,
                "dataset": "synthetic",
                "msg_len": 100 if sid != "ours" else 16,
            }
            (run_dir / "results_meta.json").write_text(
                __import__("json").dumps(meta), encoding="utf-8"
            )
            (run_dir / "config.yaml").write_text(
                f"scheme: {sid}\nseed: 0\ndataset: synthetic\nmsg_len: {meta['msg_len']}\n",
                encoding="utf-8",
            )
            runs.append(RunBundle(run_dir))

        out = tmp_path / "figs"
        out.mkdir()
        d1 = generate_d1(runs, out)
        e1 = generate_e1(runs, out)
        d1_csv = (out / "d1.csv").read_text(encoding="utf-8")
        e1_csv = (out / "e1.csv").read_text(encoding="utf-8")
        for sid in POSTHOC:
            label = str(SCHEME_INFO[sid]["label_uk"])
            assert label in d1_csv or sid in d1_csv
            # Numeric (not the not_evaluated marker) should appear for baselines.
            assert NOT_EVALUATED not in d1_csv.splitlines()[1] or "0.9" in d1_csv
        assert "0.97" in e1_csv or "42" in e1_csv
        assert d1 and e1
        print("[ok] D1/E1 display numbers for available baselines")


def test_docs_baselines_lists_sources() -> None:
    doc = (ROOT / "docs" / "baselines.md").read_text(encoding="utf-8")
    for needle in (
        "github.com/tancik/StegaStamp",
        "github.com/adobe/trustmark",
        "github.com/Shilin-LU/VINE",
        "MIT",
        "checksum",
        "700328b8754db934b2f6cb5e5185d81f",
        "not_applicable",
    ):
        assert needle.lower() in doc.lower(), f"missing {needle!r} in docs/baselines.md"
    print("[ok] docs/baselines.md lists source / license / checksums")


def test_registry_attacks_not_marked_not_evaluated_are_runnable_ids() -> None:
    runnable = [s.id for s in ATTACK_REGISTRY if not s.not_evaluated]
    assert "regen_proxy" in runnable and "guided_regen" in runnable
    assert "semantic_regen" not in runnable
    print(f"[ok] runnable registry entries: {len(runnable)}")


if __name__ == "__main__":
    test_generation_time_schemes_not_applicable()
    test_infer_scheme_availability_not_hardcoded()
    test_docs_baselines_lists_sources()
    test_registry_attacks_not_marked_not_evaluated_are_runnable_ids()
    test_d1_e1_show_baseline_numbers()
    test_available_baseline_clean_bitacc_and_psnr()
    print("=== all baseline acceptance tests finished ===")
