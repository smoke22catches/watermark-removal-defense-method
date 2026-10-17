#!/usr/bin/env python3
"""Package experiment artifacts into a portable results_<tag>.tgz archive.

Resolves the repository root from this script's location so it works from any
working directory (e.g. ``python /path/to/scripts/collect_results.py`` from /tmp).
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ARTIFACTS = (
    "config.yaml",
    "run_meta.json",
    "results_meta.json",
    "cost.json",
    "results.csv",
    "metrics.csv",
    "log.txt",
)

MIN_TEST_IMAGES = 5000


def _git_info(repo: Path) -> Dict[str, Any]:
    commit = "unknown"
    dirty = False
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip() or "unknown"
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty_proc.returncode == 0:
            dirty = bool(dirty_proc.stdout.strip())
    except Exception:
        pass
    return {"git_commit": commit, "git_dirty": dirty}


def _gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "gpu_name": None,
        "driver_version": None,
        "gpu_memory_total_mb": None,
        "torch_version": None,
        "cuda_version": None,
    }
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["cuda_version"] = str(getattr(torch.version, "cuda", None))
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_total_mb"] = int(props.total_memory // (1024 * 1024))
    except Exception:
        pass
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            # Use first GPU line.
            line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                if not info["gpu_name"]:
                    info["gpu_name"] = parts[0]
                info["driver_version"] = parts[1]
                try:
                    info["gpu_memory_total_mb"] = int(float(parts[2]))
                except ValueError:
                    pass
    except Exception:
        pass
    return info


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _split_hashes_from_meta(splits_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    splits_dir = splits_dir if splits_dir.is_absolute() else (ROOT / splits_dir)
    splits_dir = splits_dir.resolve()
    if not splits_dir.is_dir():
        return out
    for meta_path in sorted(splits_dir.glob("*/meta.json")):
        meta = _load_json(meta_path)
        if not meta:
            continue
        dataset = meta.get("dataset") or meta_path.parent.name
        try:
            rel = meta_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = meta_path.as_posix()
        out[str(dataset)] = {
            "split_hashes": meta.get("split_hashes") or {},
            "counts": meta.get("counts") or {},
            "split_seed": meta.get("split_seed"),
            "path": rel,
        }
    return out


def build_instance_info(tag: str, splits_dir: Path) -> Dict[str, Any]:
    info = {
        "tag": tag,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(ROOT),
    }
    info.update(_git_info(ROOT))
    info.update(_gpu_info())
    info["split_hashes"] = _split_hashes_from_meta(splits_dir)
    return info


def iter_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.is_dir():
        return []
    dirs: List[Path] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        # A run directory has at least one of the tracked artifacts / checkpoints.
        if any((child / name).exists() for name in RUN_ARTIFACTS) or (
            child / "checkpoints"
        ).is_dir():
            dirs.append(child)
    return dirs


def _canonical_split_hashes(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    splits = payload.get("splits") or {}
    hashes = splits.get("split_hashes")
    if not hashes:
        return None
    return json.dumps(hashes, sort_keys=True)


def _test_count(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not payload:
        return None
    splits = payload.get("splits") or {}
    counts = splits.get("counts") or {}
    test = counts.get("test")
    if test is None:
        return None
    try:
        return int(test)
    except (TypeError, ValueError):
        return None


def warn_on_runs(run_dirs: Iterable[Path]) -> List[str]:
    """Return warning strings for undersized test splits / mismatched hashes."""
    warnings: List[str] = []
    hash_keys: Set[str] = set()
    small_tests: List[Tuple[str, int]] = []

    for run_dir in run_dirs:
        meta = _load_json(run_dir / "results_meta.json")
        if meta is None:
            meta = _load_json(run_dir / "run_meta.json")
        n_test = _test_count(meta)
        if n_test is not None and n_test < MIN_TEST_IMAGES:
            # Synthetic runs are intentionally small; skip those.
            dataset = str((meta or {}).get("dataset") or "").lower()
            if dataset != "synthetic":
                small_tests.append((run_dir.name, n_test))
        key = _canonical_split_hashes(meta)
        if key is not None:
            hash_keys.add(key)

    for name, n in small_tests:
        warnings.append(
            f"evaluation run {name!r} used a test split of {n} < {MIN_TEST_IMAGES} images"
        )
    if len(hash_keys) > 1:
        warnings.append(
            f"runs were produced with {len(hash_keys)} distinct split hash sets "
            f"(results may not be comparable)"
        )
    return warnings


def _add_if_exists(tar: tarfile.TarFile, path: Path, arcname: str) -> bool:
    if not path.exists():
        return False
    tar.add(path, arcname=arcname)
    return True


def resolve_archive_path(out_path: Path, tag: str) -> Path:
    """``--out`` is either a ``.tgz`` file path or a directory to write into."""
    if str(out_path).endswith(".tgz") or str(out_path).endswith(".tar.gz"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path / f"results_{tag}.tgz"


def package_results(
    *,
    tag: str,
    out_path: Path,
    runs_root: Path,
    splits_dir: Path,
    with_checkpoints: bool = False,
) -> Tuple[Path, Optional[Path], List[str]]:
    tag = tag.upper()
    archive_path = resolve_archive_path(out_path, tag)

    instance_info = build_instance_info(tag, splits_dir)
    info_name = f"instance_info_{tag}.json"
    info_path = archive_path.parent / info_name
    info_path.write_text(
        json.dumps(instance_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_dirs = iter_run_dirs(runs_root)
    warnings = warn_on_runs(run_dirs)
    for w in warnings:
        print(f"[warn] {w}")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(info_path, arcname=info_name)
        for doc in ("docs/baselines.md", "docs/evaluation_protocol.md"):
            _add_if_exists(tar, ROOT / doc, doc)
        if splits_dir.is_dir():
            for meta_path in sorted(splits_dir.glob("*/meta.json")):
                rel = meta_path.relative_to(ROOT).as_posix()
                tar.add(meta_path, arcname=rel)
        for run_dir in run_dirs:
            try:
                rel_run = run_dir.relative_to(ROOT).as_posix()
            except ValueError:
                rel_run = f"runs/{run_dir.name}"
            for art in RUN_ARTIFACTS:
                _add_if_exists(tar, run_dir / art, f"{rel_run}/{art}")

    ckpt_path: Optional[Path] = None
    if with_checkpoints:
        ckpt_path = archive_path.with_name(f"checkpoints_{tag}.tgz")
        with tarfile.open(ckpt_path, "w:gz") as tar:
            for run_dir in run_dirs:
                best = run_dir / "checkpoints" / "best.pt"
                if not best.is_file():
                    continue
                try:
                    rel_run = run_dir.relative_to(ROOT).as_posix()
                except ValueError:
                    rel_run = f"runs/{run_dir.name}"
                tar.add(best, arcname=f"{rel_run}/checkpoints/best.pt")

    return archive_path, ckpt_path, warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package run artifacts into results_<tag>.tgz")
    p.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Experiment tag (A, B, or C)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .tgz path (or directory to write results_<tag>.tgz into)",
    )
    p.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help="Runs directory (default: <repo>/runs)",
    )
    p.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Splits directory (default: <repo>/data/splits)",
    )
    p.add_argument(
        "--with-checkpoints",
        action="store_true",
        help="Also write checkpoints_<tag>.tgz with training best.pt files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = str(args.tag).strip().upper()
    if tag not in {"A", "B", "C"}:
        print(f"[warn] unusual tag {tag!r}; expected A, B, or C — continuing")

    runs_root = Path(args.runs_root) if args.runs_root else ROOT / "runs"
    if not runs_root.is_absolute():
        # Relative --runs-root is resolved against the repo root, not cwd.
        runs_root = (ROOT / runs_root).resolve()
    else:
        runs_root = runs_root.resolve()

    splits_dir = Path(args.splits_dir) if args.splits_dir else ROOT / "data" / "splits"
    if not splits_dir.is_absolute():
        splits_dir = (ROOT / splits_dir).resolve()
    else:
        splits_dir = splits_dir.resolve()

    out = Path(args.out)
    # Relative --out stays relative to the caller's cwd (user-chosen destination).
    if not out.is_absolute():
        out = Path.cwd() / out

    archive, ckpt, warnings = package_results(
        tag=tag,
        out_path=out,
        runs_root=runs_root,
        splits_dir=splits_dir,
        with_checkpoints=bool(args.with_checkpoints),
    )
    print(f"Wrote {archive}")
    info_sidecar = archive.parent / f"instance_info_{tag}.json"
    if info_sidecar.is_file():
        print(f"Wrote {info_sidecar}")
    if ckpt is not None:
        print(f"Wrote {ckpt}")
    if warnings:
        print(f"({len(warnings)} warning(s) before packaging)")


if __name__ == "__main__":
    main()
