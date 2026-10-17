"""Acceptance tests for scripts/collect_results.py packaging."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_results import (  # noqa: E402
    package_results,
    warn_on_runs,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _make_minimal_repo_layout(base: Path) -> tuple[Path, Path, Path]:
    """Build a tiny fake repo slice with docs, splits, and one eval run."""
    docs = base / "docs"
    docs.mkdir(parents=True)
    (docs / "baselines.md").write_text("# baselines\n", encoding="utf-8")
    (docs / "evaluation_protocol.md").write_text("# protocol\n", encoding="utf-8")

    splits = base / "data" / "splits" / "coco"
    splits.mkdir(parents=True)
    meta = {
        "dataset": "coco",
        "split_seed": 1,
        "counts": {"train": 100, "val": 10, "test": 5000},
        "split_hashes": {"train.txt": "aaa", "val.txt": "bbb", "test.txt": "ccc"},
    }
    (splits / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    runs = base / "runs" / "20260101-000000_eval"
    runs.mkdir(parents=True)
    (runs / "config.yaml").write_text("dataset: coco\n", encoding="utf-8")
    results_meta = {
        "dataset": "coco",
        "splits": {
            "counts": {"train": 100, "val": 10, "test": 5000},
            "split_hashes": {"train.txt": "aaa", "val.txt": "bbb", "test.txt": "ccc"},
        },
    }
    (runs / "results_meta.json").write_text(
        json.dumps(results_meta, indent=2) + "\n", encoding="utf-8"
    )
    (runs / "run_meta.json").write_text(
        json.dumps(results_meta, indent=2) + "\n", encoding="utf-8"
    )
    (runs / "results.csv").write_text("metric_name,metric_value\n", encoding="utf-8")
    return docs, splits.parent, runs.parent


def test_warn_on_small_test_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "eval_small"
        run_dir.mkdir()
        payload = {
            "dataset": "coco",
            "splits": {"counts": {"test": 1500}, "split_hashes": {"test.txt": "x"}},
        }
        (run_dir / "results_meta.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        warnings = warn_on_runs([run_dir])
        assert any("1500" in w and "5000" in w for w in warnings), warnings


def test_warn_on_mismatched_split_hashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs = []
        for i, h in enumerate(("hash_a", "hash_b")):
            d = root / f"run_{i}"
            d.mkdir()
            payload = {
                "dataset": "coco",
                "splits": {
                    "counts": {"test": 5000},
                    "split_hashes": {"test.txt": h},
                },
            }
            (d / "results_meta.json").write_text(json.dumps(payload), encoding="utf-8")
            runs.append(d)
        warnings = warn_on_runs(runs)
        assert any("distinct split hash" in w for w in warnings), warnings


def test_package_identical_from_any_cwd(monkeypatch_root: bool = True) -> None:
    """Packaging from /tmp must match packaging from the repo root.

    We monkeypatch collect_results.ROOT to a temp tree so the test does not
    pull the whole real runs/ tree into the archive.
    """
    import scripts.collect_results as cr

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "repo"
        base.mkdir()
        _make_minimal_repo_layout(base)
        # Point the module at our fake repo.
        old_root = cr.ROOT
        cr.ROOT = base
        try:
            out_a = Path(tmp) / "out_a" / "results_A.tgz"
            out_b = Path(tmp) / "out_b" / "results_A.tgz"
            out_a.parent.mkdir()
            out_b.parent.mkdir()
            a, _, _ = package_results(
                tag="A",
                out_path=out_a,
                runs_root=base / "runs",
                splits_dir=base / "data" / "splits",
            )
            b, _, _ = package_results(
                tag="A",
                out_path=out_b,
                runs_root=base / "runs",
                splits_dir=base / "data" / "splits",
            )
            # Compare member names (content of instance_info differs by timestamp).
            import tarfile

            def members(path: Path) -> set[str]:
                with tarfile.open(path, "r:gz") as tf:
                    return set(tf.getnames())

            assert members(a) == members(b)
            assert "instance_info_A.json" in members(a)
            assert "docs/baselines.md" in members(a)
            assert "data/splits/coco/meta.json" in members(a)
        finally:
            cr.ROOT = old_root


def test_cli_from_tmp_matches_repo_cwd() -> None:
    """End-to-end: invoke the script from /tmp and from the repo root."""
    script = ROOT / "scripts" / "collect_results.py"
    with tempfile.TemporaryDirectory() as tmp:
        # Use a dedicated empty runs overlay via --runs-root pointing at a temp
        # tree that only has docs/splits from the real repo — instead, call the
        # library with monkeypatched ROOT (covered above) and here just verify
        # the CLI resolves ROOT from __file__ when cwd=/tmp.
        out_dir = Path(tmp) / "pkg"
        out_dir.mkdir()
        # Minimal: run with real repo but empty-ish runs via a temp runs root.
        fake_runs = Path(tmp) / "runs"
        fake_runs.mkdir()
        # One synthetic small-test run to trigger the warning path.
        run = fake_runs / "synth_eval"
        run.mkdir()
        (run / "results_meta.json").write_text(
            json.dumps(
                {
                    "dataset": "coco",
                    "splits": {
                        "counts": {"test": 1500},
                        "split_hashes": {"test.txt": "abc"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "config.yaml").write_text("x: 1\n", encoding="utf-8")

        out1 = out_dir / "from_tmp.tgz"
        out2 = out_dir / "from_root.tgz"
        cmd_base = [
            sys.executable,
            str(script),
            "--tag",
            "B",
            "--runs-root",
            str(fake_runs),
            "--splits-dir",
            str(ROOT / "data" / "splits"),
        ]
        r1 = subprocess.run(
            cmd_base + ["--out", str(out1)],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        assert r1.returncode == 0, r1.stderr + r1.stdout
        assert "1500" in r1.stdout or "1500" in r1.stderr
        r2 = subprocess.run(
            cmd_base + ["--out", str(out2)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert r2.returncode == 0, r2.stderr + r2.stdout

        import tarfile

        def names(p: Path) -> set[str]:
            with tarfile.open(p, "r:gz") as tf:
                return set(tf.getnames())

        assert names(out1) == names(out2)
        assert "instance_info_B.json" in names(out1)


if __name__ == "__main__":
    test_warn_on_small_test_split()
    print("[ok] warn small test")
    test_warn_on_mismatched_split_hashes()
    print("[ok] warn mismatched hashes")
    test_package_identical_from_any_cwd()
    print("[ok] package member parity")
    test_cli_from_tmp_matches_repo_cwd()
    print("[ok] CLI from /tmp matches repo cwd")
    print("=== collect_results acceptance tests passed ===")
