#!/usr/bin/env python3
"""Standalone dataset downloader for watermark robustness experiments.

Downloads (or documents manual steps for) common cover-image sources:
  - MS-COCO val2017 / val2014
  - DIV2K (HR validation / train)
  - Imagenette (ImageNet-style subset; full ImageNet needs manual credentials)
  - CLIC professional / mobile (optional)

Writes ``data/manifest.json`` with paths and sample counts.

``--fraction`` / ``--num-samples`` apply only to training splits by default
(COCO train2017, DIV2K train, Imagenette train). Evaluation / test source
folders are kept in full unless ``--cap-eval-splits`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Official image counts used to detect previously capped / incomplete folders.
OFFICIAL_IMAGE_COUNTS: Dict[Tuple[str, str], int] = {
    ("coco", "train2017"): 118_287,
    ("coco", "val2017"): 5_000,
    ("coco", "train2014"): 82_783,
    ("coco", "val2014"): 40_504,
    ("div2k", "train"): 800,
    ("div2k", "valid"): 100,
    ("imagenette", "train"): 9_469,
    ("imagenette", "val"): 3_925,
}

# Splits that feed training (eligible for --fraction / --num-samples by default).
TRAINING_SPLITS: Dict[str, frozenset] = {
    "coco": frozenset({"train2017", "train2014"}),
    "div2k": frozenset({"train"}),
    "imagenette": frozenset({"train"}),
    "imagenet": frozenset({"train"}),
}


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _archive_looks_valid(path: Path) -> bool:
    """Return True if ``path`` opens as a complete zip/tar (truncated downloads fail).

    Incomplete HTTP downloads usually lack the end-of-central-directory record, so
    ``ZipFile`` raises ``BadZipFile`` — that is the check that matters, not size.
    """
    if not path.is_file() or path.stat().st_size < 22:  # empty zip lower bound
        return False
    name = path.name.lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(path, "r") as zf:
                return bool(zf.namelist())
        if (
            name.endswith((".tgz", ".tar", ".tar.gz", ".tar.bz2", ".tar.xz"))
            or ".tar" in name
        ):
            with tarfile.open(path, "r:*") as tf:
                return bool(tf.getmembers())
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError):
        return False
    return True


def _download(url: str, dest: Path, expected_sha256: Optional[str] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha256:
        got = _sha256(dest)
        if got == expected_sha256:
            print(f"  [ok] cached {dest.name} (sha256 match)")
            return dest
        print(f"  [warn] checksum mismatch for {dest.name}; re-downloading")
        dest.unlink()
    elif dest.exists():
        if _archive_looks_valid(dest):
            print(f"  [ok] cached {dest.name}")
            return dest
        print(
            f"  [warn] corrupt/incomplete cache {dest.name} "
            f"({dest.stat().st_size} bytes); re-downloading"
        )
        dest.unlink()

    print(f"  downloading {url}")
    print(f"         -> {dest}")

    def _reporthook(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = 100.0 * done / total
        sys.stdout.write(f"\r  progress: {pct:5.1f}% ({done // (1 << 20)} / {total // (1 << 20)} MiB)")
        sys.stdout.flush()

    try:
        urlretrieve(url, dest, reporthook=_reporthook)
        print()
    except Exception as e:
        print(f"\n  [error] download failed: {e}")
        if dest.exists():
            dest.unlink()
        raise

    if not _archive_looks_valid(dest):
        size = dest.stat().st_size if dest.exists() else 0
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded archive {dest.name} is corrupt/incomplete ({size} bytes)"
        )

    if expected_sha256:
        got = _sha256(dest)
        if got != expected_sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {dest.name}: expected {expected_sha256}, got {got}"
            )
        print(f"  [ok] sha256 verified")
    return dest


def _extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {out_dir}")
    try:
        if archive.suffix == ".zip" or archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(out_dir)
        elif (
            archive.suffix in {".tgz", ".gz", ".bz2", ".xz"}
            or ".tar" in archive.name
            or archive.name.endswith(".tgz")
        ):
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(out_dir)
        else:
            raise ValueError(f"Unknown archive type: {archive}")
    except Exception:
        # Leave a broken archive on disk → next run would "cache" it forever.
        print(f"  [warn] extract failed; removing corrupt archive {archive}")
        archive.unlink(missing_ok=True)
        raise


def _cap_images(
    root: Path,
    num_samples: Optional[int] = None,
    fraction: Optional[float] = None,
) -> int:
    """Optionally keep a subset of images (sorted) for quick tests.

    ``fraction`` is in ``[0, 1]`` and keeps that share of the dataset.
    When both ``fraction`` and ``num_samples`` are set, the stricter (smaller) cap wins.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    n = len(files)
    limit = n
    if fraction is not None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        limit = min(limit, int(n * fraction))
    if num_samples is not None:
        limit = min(limit, num_samples)
    if limit >= n:
        return n
    keep = set(files[:limit])
    removed = 0
    for p in files:
        if p not in keep:
            p.unlink()
            removed += 1
    label = f"{limit} images"
    if fraction is not None:
        label += f" ({fraction:.0%} of {n})"
    print(f"  capped to {label} (removed {removed})")
    return limit


def _target_cap(
    official: Optional[int],
    num_samples: Optional[int],
    fraction: Optional[float],
) -> Optional[int]:
    """Images needed after applying fraction/num_samples to an official-sized folder."""
    if official is None:
        return None
    limit = official
    if fraction is not None:
        limit = min(limit, int(official * fraction))
    if num_samples is not None:
        limit = min(limit, num_samples)
    return limit


def _prompt_restore(message: str, *, assume_yes: bool) -> bool:
    print(message)
    if assume_yes or not sys.stdin.isatty():
        print("  -> restoring (non-interactive / --yes)")
        return True
    try:
        answer = input("  Restore now? [Y/n] ").strip().lower()
    except EOFError:
        print("  -> restoring (no stdin)")
        return True
    return answer in ("", "y", "yes")


def _clear_image_dir(root: Path) -> None:
    if not root.exists():
        return
    shutil.rmtree(root)


def _ensure_complete_or_restore(
    *,
    dataset: str,
    split: str,
    out: Path,
    archive: Path,
    extract_parent: Path,
    url: str,
    official: Optional[int],
    will_cap: bool,
    num_samples: Optional[int],
    fraction: Optional[float],
    assume_yes: bool,
) -> Tuple[bool, str]:
    """Return (ready_to_use_without_redownload, note).

    If the folder is incomplete relative to what this run needs, offer restore
    (re-extract from archive if valid, otherwise re-download).
    """
    n = _count_images(out)
    if n <= 0:
        return False, "missing"

    needed = (
        _target_cap(official, num_samples, fraction)
        if will_cap
        else official
    )
    if official is None or needed is None or n >= needed:
        if official is not None and n < official and will_cap:
            return True, f"cached (capped {n}/{official})"
        if official is not None and n < official:
            # Fewer than official but somehow needed is None — treat as incomplete.
            pass
        else:
            return True, "cached"

    if n < official:
        action = (
            f"{dataset} {split}: folder has {n} images < official {official}"
            + (
                f" (need {needed} for this run's cap)"
                if will_cap and needed is not None
                else " (evaluation/full download expects the complete split)"
            )
            + f".\n  Offer to restore from {'archive ' + archive.name if archive.exists() else 're-download'}."
        )
        if not _prompt_restore(action, assume_yes=assume_yes):
            print(f"  [warn] keeping incomplete {out} ({n} images)")
            return True, f"incomplete_kept ({n}/{official})"

        _clear_image_dir(out)
        try:
            if archive.exists() and _archive_looks_valid(archive):
                print(f"  re-extracting {archive.name}")
                _extract(archive, extract_parent)
            else:
                if archive.exists():
                    archive.unlink(missing_ok=True)
                _download(url, archive)
                _extract(archive, extract_parent)
        except Exception as e:
            print(f"  [error] restore failed: {e}")
            return False, "restore_failed"
        restored_n = _count_images(out)
        print(f"  restored {out} ({restored_n} images)")
        return True, "restored"

    return True, "cached"


def is_training_split(dataset: str, split: str) -> bool:
    """True if this split feeds training and is capped by --fraction by default."""
    return split in TRAINING_SPLITS.get(dataset.lower(), frozenset())


def resolve_cap_args(
    dataset: str,
    split: str,
    *,
    num_samples: Optional[int],
    fraction: Optional[float],
    cap_eval_splits: bool,
) -> Tuple[Optional[int], Optional[float], bool]:
    """Return (num_samples, fraction, will_cap) for this dataset/split."""
    train = is_training_split(dataset, split)
    will_cap = (fraction is not None or num_samples is not None) and (
        train or cap_eval_splits
    )
    if will_cap:
        return num_samples, fraction, True
    return None, None, False


# ---------------------------------------------------------------------------
# Dataset handlers
# ---------------------------------------------------------------------------

def download_coco(
    data_root: Path,
    split: str = "val2017",
    num_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    *,
    will_cap: bool = False,
    assume_yes: bool = False,
) -> Dict[str, Any]:
    """MS-COCO images (val2017 by default; val2014 also supported)."""
    split = split if split in ("val2017", "val2014", "train2017", "train2014") else "val2017"
    out = data_root / "coco" / split
    url = f"http://images.cocodataset.org/zips/{split}.zip"
    archive = data_root / "coco" / f"{split}.zip"
    official = OFFICIAL_IMAGE_COUNTS.get(("coco", split))
    cap_n = num_samples if will_cap else None
    cap_f = fraction if will_cap else None

    if out.exists() and _count_images(out) > 0:
        ready, note = _ensure_complete_or_restore(
            dataset="coco",
            split=split,
            out=out,
            archive=archive,
            extract_parent=data_root / "coco",
            url=url,
            official=official,
            will_cap=will_cap,
            num_samples=cap_n,
            fraction=cap_f,
            assume_yes=assume_yes,
        )
        if ready:
            n = _cap_images(out, cap_n, cap_f) if will_cap else _count_images(out)
            print(f"COCO {split}: already present at {out} ({n} images) [{note}]")
            return {
                "name": "coco",
                "path": str(out),
                "split": split,
                "num_images": n,
                "status": "cached",
                "capped": bool(will_cap and (cap_n is not None or cap_f is not None)),
                "kept_full": not will_cap,
                "official_count": official,
            }

    try:
        _download(url, archive)
        _extract(archive, data_root / "coco")
    except Exception as e:
        print(f"[coco] download failed ({e}).")
        print("  Manual: download from https://cocodataset.org/#download")
        print(f"  Place images under {out}")
        return {
            "name": "coco",
            "path": str(out),
            "split": split,
            "num_images": 0,
            "status": "manual_required",
            "capped": False,
            "kept_full": True,
            "official_count": official,
        }

    n = _cap_images(out, cap_n, cap_f) if will_cap else _count_images(out)
    print(f"COCO {split}: {out} ({n} images)")
    return {
        "name": "coco",
        "path": str(out),
        "split": split,
        "num_images": n,
        "status": "ok",
        "capped": bool(will_cap),
        "kept_full": not will_cap,
        "official_count": official,
    }


def download_div2k(
    data_root: Path,
    split: str = "valid",
    num_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    *,
    will_cap: bool = False,
    assume_yes: bool = False,
) -> Dict[str, Any]:
    """DIV2K high-resolution images (train or valid HR)."""
    split = "valid" if split in ("val", "valid", "validation") else "train"
    name = "DIV2K_valid_HR" if split == "valid" else "DIV2K_train_HR"
    out = data_root / "div2k" / name
    url = f"http://data.vision.ee.ethz.ch/cvl/DIV2K/{name}.zip"
    archive = data_root / "div2k" / f"{name}.zip"
    official = OFFICIAL_IMAGE_COUNTS.get(("div2k", split))
    cap_n = num_samples if will_cap else None
    cap_f = fraction if will_cap else None

    if out.exists() and _count_images(out) > 0:
        ready, note = _ensure_complete_or_restore(
            dataset="div2k",
            split=split,
            out=out,
            archive=archive,
            extract_parent=data_root / "div2k",
            url=url,
            official=official,
            will_cap=will_cap,
            num_samples=cap_n,
            fraction=cap_f,
            assume_yes=assume_yes,
        )
        if ready:
            n = _cap_images(out, cap_n, cap_f) if will_cap else _count_images(out)
            print(f"DIV2K {name}: already present at {out} ({n} images) [{note}]")
            return {
                "name": "div2k",
                "path": str(out),
                "split": split,
                "num_images": n,
                "status": "cached",
                "capped": bool(will_cap),
                "kept_full": not will_cap,
                "official_count": official,
            }

    try:
        _download(url, archive)
        _extract(archive, data_root / "div2k")
    except Exception as e:
        print(f"[div2k] download failed ({e}).")
        print("  Manual: http://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print(f"  Extract so images land in {out}")
        return {
            "name": "div2k",
            "path": str(out),
            "split": split,
            "num_images": 0,
            "status": "manual_required",
            "capped": False,
            "kept_full": True,
            "official_count": official,
        }

    n = _cap_images(out, cap_n, cap_f) if will_cap else _count_images(out)
    print(f"DIV2K {name}: {out} ({n} images)")
    return {
        "name": "div2k",
        "path": str(out),
        "split": split,
        "num_images": n,
        "status": "ok",
        "capped": bool(will_cap),
        "kept_full": not will_cap,
        "official_count": official,
    }


def download_imagenette(
    data_root: Path,
    split: str = "val",
    num_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    *,
    will_cap: bool = False,
    assume_yes: bool = False,
) -> Dict[str, Any]:
    """Imagenette (fast.ai) — ImageNet-style subset practical to auto-download."""
    out_root = data_root / "imagenette"
    extracted = out_root / "imagenette2-320"
    val_dir = extracted / "val"
    train_dir = extracted / "train"
    split_key = "val" if split in ("val", "valid") else "train"
    target = val_dir if split_key == "val" else train_dir
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    archive = out_root / "imagenette2-320.tgz"
    official = OFFICIAL_IMAGE_COUNTS.get(("imagenette", split_key))
    cap_n = num_samples if will_cap else None
    cap_f = fraction if will_cap else None

    if target.exists() and _count_images(target) > 0:
        ready, note = _ensure_complete_or_restore(
            dataset="imagenette",
            split=split_key,
            out=target,
            archive=archive,
            extract_parent=out_root,
            url=url,
            official=official,
            will_cap=will_cap,
            num_samples=cap_n,
            fraction=cap_f,
            assume_yes=assume_yes,
        )
        if ready:
            n = _cap_images(target, cap_n, cap_f) if will_cap else _count_images(target)
            print(f"Imagenette: already present at {target} ({n} images) [{note}]")
            return {
                "name": "imagenette",
                "path": str(target),
                "split": split_key,
                "num_images": n,
                "status": "cached",
                "capped": bool(will_cap),
                "kept_full": not will_cap,
                "official_count": official,
            }

    try:
        _download(url, archive, expected_sha256=None)
        _extract(archive, out_root)
    except Exception as e:
        print(f"[imagenette] download failed ({e}).")
        print("  Manual: https://github.com/fastai/imagenette")
        print(f"  Extract under {out_root}")
        return {
            "name": "imagenette",
            "path": str(val_dir),
            "split": split_key,
            "num_images": 0,
            "status": "manual_required",
            "capped": False,
            "kept_full": True,
            "official_count": official,
        }

    target = val_dir if split_key == "val" else train_dir
    n = _cap_images(target, cap_n, cap_f) if will_cap else _count_images(target)
    print(f"Imagenette: {target} ({n} images)")
    return {
        "name": "imagenette",
        "path": str(target),
        "split": split_key,
        "num_images": n,
        "status": "ok",
        "capped": bool(will_cap),
        "kept_full": not will_cap,
        "official_count": official,
    }


def download_imagenet(
    data_root: Path,
    split: str = "val",
    num_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    *,
    will_cap: bool = False,
    assume_yes: bool = False,
) -> Dict[str, Any]:
    """Full ImageNet requires registration — print instructions, do not hard-fail."""
    _ = assume_yes
    split_key = "val" if split in ("val", "valid") else "train"
    out = data_root / "imagenet" / split_key
    n = _count_images(out)
    cap_n = num_samples if will_cap else None
    cap_f = fraction if will_cap else None
    if n > 0:
        n = _cap_images(out, cap_n, cap_f) if will_cap else n
        print(f"ImageNet: found local copy at {out} ({n} images)")
        return {
            "name": "imagenet",
            "path": str(out),
            "split": split_key,
            "num_images": n,
            "status": "cached",
            "capped": bool(will_cap),
            "kept_full": not will_cap,
            "official_count": None,
        }

    print("[imagenet] Full ImageNet cannot be auto-downloaded (requires ILSVRC credentials).")
    print("  Manual steps:")
    print("    1. Register at https://image-net.org/ and accept the terms.")
    print("    2. Download ILSVRC2012 train/val tarballs.")
    print(f"    3. Extract so images live under {out}")
    print("  Alternatively use --dataset imagenette for an ImageNet-style subset.")
    return {
        "name": "imagenet",
        "path": str(out),
        "split": split_key,
        "num_images": 0,
        "status": "manual_required",
        "capped": False,
        "kept_full": True,
        "official_count": None,
    }


def download_clic(
    data_root: Path,
    split: str = "professional",
    num_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    *,
    will_cap: bool = False,
    assume_yes: bool = False,
) -> Dict[str, Any]:
    """CLIC professional/mobile test sets (learned compression + watermarking)."""
    _ = assume_yes
    split = split if split in ("professional", "mobile") else "professional"
    out = data_root / "clic" / split
    # CLIC is an evaluation-only source; never treat as a training split.
    cap_n = num_samples if will_cap else None
    cap_f = fraction if will_cap else None
    if out.exists() and _count_images(out) > 0:
        n = _cap_images(out, cap_n, cap_f) if will_cap else _count_images(out)
        print(f"CLIC {split}: already present at {out} ({n} images)")
        return {
            "name": "clic",
            "path": str(out),
            "split": split,
            "num_images": n,
            "status": "cached",
            "capped": bool(will_cap),
            "kept_full": not will_cap,
            "official_count": None,
        }

    urls = {
        "professional": [
            "https://data.vision.ee.ethz.ch/cvl/clic/professional_test_2020.zip",
            "https://storage.googleapis.com/clic2021_public/professional_valid_2021.zip",
        ],
        "mobile": [
            "https://data.vision.ee.ethz.ch/cvl/clic/mobile_test_2020.zip",
        ],
    }
    archive = data_root / "clic" / f"{split}.zip"
    ok = False
    last_err: Optional[Exception] = None
    for url in urls.get(split, []):
        try:
            if archive.exists():
                archive.unlink()
            _download(url, archive)
            _extract(archive, out)
            ok = True
            break
        except Exception as e:
            last_err = e
            print(f"  [warn] failed {url}: {e}")

    if not ok:
        print(f"[clic] auto-download failed ({last_err}).")
        print("  Manual: https://clic.compression.cc/  (Challenge on Learned Image Compression)")
        print(f"  Place images under {out}")
        return {
            "name": "clic",
            "path": str(out),
            "split": split,
            "num_images": 0,
            "status": "manual_required",
            "capped": False,
            "kept_full": True,
            "official_count": None,
        }

    n = _cap_images(out, cap_n, cap_f) if will_cap else _count_images(out)
    print(f"CLIC {split}: {out} ({n} images)")
    return {
        "name": "clic",
        "path": str(out),
        "split": split,
        "num_images": n,
        "status": "ok",
        "capped": bool(will_cap),
        "kept_full": not will_cap,
        "official_count": None,
    }


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "coco": download_coco,
    "div2k": download_div2k,
    "imagenette": download_imagenette,
    "imagenet": download_imagenet,
    "clic": download_clic,
}

DEFAULT_SET = ["coco", "div2k", "imagenette", "clic"]


def write_manifest(data_root: Path, entries: List[Dict[str, Any]]) -> Path:
    path = data_root / "manifest.json"
    existing: Dict[str, Any] = {"datasets": {}}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    datasets = existing.setdefault("datasets", {})
    for e in entries:
        datasets[e["name"]] = e
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"\nWrote manifest: {path}")
    return path


def _parse_fraction(value: str) -> float:
    f = float(value)
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError(f"fraction must be in [0, 1], got {f}")
    return f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download watermarking benchmark datasets")
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"One of {{{','.join(list(HANDLERS)+['all'])}}} (default: all)",
    )
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--split", type=str, default=None, help="Dataset-specific split")
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional absolute image cap (training splits only, unless --cap-eval-splits)",
    )
    p.add_argument(
        "--fraction",
        type=_parse_fraction,
        default=None,
        help=(
            "Keep only this fraction of training splits after extract "
            "(0-1; evaluation splits stay full unless --cap-eval-splits)"
        ),
    )
    p.add_argument(
        "--cap-eval-splits",
        action="store_true",
        help=(
            "Also apply --fraction / --num-samples to validation/test source folders "
            "(COCO val2017, DIV2K valid, Imagenette val). Default: leave eval full."
        ),
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive: restore incomplete folders without prompting",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    names = DEFAULT_SET if args.dataset.lower() == "all" else [args.dataset.lower()]
    # Always document ImageNet manual path when downloading "all"
    if args.dataset.lower() == "all":
        names = DEFAULT_SET + ["imagenet"]

    entries: List[Dict[str, Any]] = []
    cap_summary: List[Tuple[str, str, str]] = []  # dataset, split, "capped"|"full"

    for name in names:
        if name not in HANDLERS:
            print(f"[skip] unknown dataset '{name}'")
            continue
        print(f"\n=== {name} ===")
        # Default: download BOTH train and eval folders so fixed splits never
        # resolve train and test to the same directory.
        jobs: List[Dict[str, Any]] = []
        if args.split is not None:
            jobs.append({"split": args.split})
        elif name == "coco":
            jobs = [{"split": "train2017"}, {"split": "val2017"}]
        elif name == "div2k":
            jobs = [{"split": "train"}, {"split": "valid"}]
        elif name == "imagenette":
            jobs = [{"split": "train"}, {"split": "val"}]
        elif name == "clic":
            jobs = [{"split": "professional"}]
        else:
            jobs = [{}]

        role_paths: Dict[str, str] = {}
        role_counts: Dict[str, int] = {}
        role_status: Dict[str, str] = {}
        last_entry: Dict[str, Any] = {
            "name": name,
            "path": str(data_root / name),
            "num_images": 0,
            "status": "error",
        }
        for job in jobs:
            split_hint = str(job.get("split") or ("train" if name == "imagenet" else ""))
            # CLIC / unspecified ImageNet val: treat as eval (no default cap).
            if name == "clic":
                will_cap = bool(
                    args.cap_eval_splits
                    and (args.fraction is not None or args.num_samples is not None)
                )
                cap_n = args.num_samples if will_cap else None
                cap_f = args.fraction if will_cap else None
            elif not split_hint:
                will_cap = False
                cap_n, cap_f = None, None
            else:
                cap_n, cap_f, will_cap = resolve_cap_args(
                    name,
                    split_hint,
                    num_samples=args.num_samples,
                    fraction=args.fraction,
                    cap_eval_splits=bool(args.cap_eval_splits),
                )
            kwargs: Dict[str, Any] = {
                "data_root": data_root,
                "num_samples": cap_n,
                "fraction": cap_f,
                "will_cap": will_cap,
                "assume_yes": bool(args.yes),
                **job,
            }
            try:
                entry = HANDLERS[name](**kwargs)
                last_entry = entry
                split_name = str(entry.get("split") or job.get("split") or "")
                if split_name:
                    role_paths[split_name] = str(entry.get("path", ""))
                    role_counts[split_name] = int(entry.get("num_images") or 0)
                    role_status[split_name] = str(entry.get("status") or "error")
                    if entry.get("capped"):
                        cap_summary.append((name, split_name, "capped"))
                    else:
                        cap_summary.append((name, split_name, "full"))
                print(
                    f"  role={split_name or 'default'}: {entry.get('status')} "
                    f"n={entry.get('num_images')} "
                    f"{'CAPPED' if entry.get('capped') else 'FULL'}"
                )
            except Exception as e:
                print(f"[{name}] error (continuing): {e}")
                split_name = str(job.get("split") or "")
                last_entry = {
                    "name": name,
                    "path": str(data_root / name),
                    "split": job.get("split"),
                    "num_images": 0,
                    "status": "error",
                    "error": str(e),
                }
                if split_name:
                    role_paths[split_name] = str(data_root / name / split_name)
                    role_counts[split_name] = 0
                    role_status[split_name] = "error"
                    cap_summary.append((name, split_name, "error"))
        # Prefer train path as the primary manifest path; keep all roles.
        if "train2017" in role_paths:
            last_entry["path"] = role_paths["train2017"]
            last_entry["path_val"] = role_paths.get("val2017", "")
            last_entry["num_images"] = role_counts.get("train2017", 0)
        elif "train" in role_paths:
            last_entry["path"] = role_paths["train"]
            last_entry["path_val"] = role_paths.get("val") or role_paths.get("valid", "")
            last_entry["num_images"] = role_counts.get("train", 0)
        last_entry["paths"] = role_paths
        last_entry["role_counts"] = role_counts
        # Require every downloaded role to have images; do not mark ok if train failed.
        if role_status and any(
            role_counts.get(r, 0) <= 0 or role_status.get(r) not in ("ok", "cached")
            for r in role_status
        ):
            last_entry["status"] = "incomplete"
        entries.append(last_entry)

    write_manifest(data_root, entries)
    print("\nSummary:")
    for e in entries:
        print(f"  {e['name']}: {e.get('status')}  path={e.get('path')}  n={e.get('num_images')}")

    print("\nCap policy (this run):")
    if not cap_summary:
        print("  (no splits processed)")
    else:
        for ds, sp, kind in cap_summary:
            print(f"  {ds}/{sp}: {kind}")
    if args.fraction is not None or args.num_samples is not None:
        if args.cap_eval_splits:
            print(
                "\nNote: --cap-eval-splits is set; evaluation folders may be smaller "
                "than the paper default test_size=5000."
            )
        else:
            print(
                "\nNote: --fraction / --num-samples applied to training splits only; "
                "evaluation folders (e.g. COCO val2017) were kept in full. "
                "Pass --cap-eval-splits to override."
            )
    print(
        "\nNext: python scripts/make_splits.py --dataset all "
        "(writes data/splits/*/{{train,val,test}}.txt)"
    )


if __name__ == "__main__":
    main()
