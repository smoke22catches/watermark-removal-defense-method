#!/usr/bin/env python3
"""Standalone dataset downloader for watermark robustness experiments.

Downloads (or documents manual steps for) common cover-image sources:
  - MS-COCO val2017 / val2014
  - DIV2K (HR validation / train)
  - Imagenette (ImageNet-style subset; full ImageNet needs manual credentials)
  - CLIC professional / mobile (optional)

Writes ``data/manifest.json`` with paths and sample counts.
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
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlretrieve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
        print(f"  [ok] cached {dest.name}")
        return dest

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


def _cap_images(root: Path, num_samples: Optional[int]) -> int:
    """Optionally keep only the first ``num_samples`` images (sorted) for quick tests."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if num_samples is None or num_samples >= len(files):
        return len(files)
    keep = set(files[:num_samples])
    removed = 0
    for p in files:
        if p not in keep:
            p.unlink()
            removed += 1
    print(f"  capped to {num_samples} images (removed {removed})")
    return num_samples


# ---------------------------------------------------------------------------
# Dataset handlers
# ---------------------------------------------------------------------------

def download_coco(
    data_root: Path,
    split: str = "val2017",
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """MS-COCO images (val2017 by default; val2014 also supported)."""
    split = split if split in ("val2017", "val2014", "train2017", "train2014") else "val2017"
    out = data_root / "coco" / split
    if out.exists() and _count_images(out) > 0:
        n = _cap_images(out, num_samples)
        print(f"COCO {split}: already present at {out} ({n} images)")
        return {"name": "coco", "path": str(out), "split": split, "num_images": n, "status": "cached"}

    url = f"http://images.cocodataset.org/zips/{split}.zip"
    # Official archives are large; checksums optional (not always published for all mirrors).
    archive = data_root / "coco" / f"{split}.zip"
    try:
        _download(url, archive)
        _extract(archive, data_root / "coco")
    except Exception as e:
        print(f"[coco] download failed ({e}).")
        print("  Manual: download from https://cocodataset.org/#download")
        print(f"  Place images under {out}")
        return {"name": "coco", "path": str(out), "split": split, "num_images": 0, "status": "manual_required"}

    n = _cap_images(out, num_samples)
    print(f"COCO {split}: {out} ({n} images)")
    return {"name": "coco", "path": str(out), "split": split, "num_images": n, "status": "ok"}


def download_div2k(
    data_root: Path,
    split: str = "valid",
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """DIV2K high-resolution images (train or valid HR)."""
    split = "valid" if split in ("val", "valid", "validation") else "train"
    name = "DIV2K_valid_HR" if split == "valid" else "DIV2K_train_HR"
    out = data_root / "div2k" / name
    if out.exists() and _count_images(out) > 0:
        n = _cap_images(out, num_samples)
        print(f"DIV2K {name}: already present at {out} ({n} images)")
        return {"name": "div2k", "path": str(out), "split": split, "num_images": n, "status": "cached"}

    # Official DIV2K mirrors (data.vision.ee.ethz.ch)
    url = f"http://data.vision.ee.ethz.ch/cvl/DIV2K/{name}.zip"
    archive = data_root / "div2k" / f"{name}.zip"
    try:
        _download(url, archive)
        _extract(archive, data_root / "div2k")
    except Exception as e:
        print(f"[div2k] download failed ({e}).")
        print("  Manual: http://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print(f"  Extract so images land in {out}")
        return {"name": "div2k", "path": str(out), "split": split, "num_images": 0, "status": "manual_required"}

    n = _cap_images(out, num_samples)
    print(f"DIV2K {name}: {out} ({n} images)")
    return {"name": "div2k", "path": str(out), "split": split, "num_images": n, "status": "ok"}


def download_imagenette(
    data_root: Path,
    split: str = "val",
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Imagenette (fast.ai) — ImageNet-style subset practical to auto-download."""
    out_root = data_root / "imagenette"
    extracted = out_root / "imagenette2-320"
    val_dir = extracted / "val"
    if val_dir.exists() and _count_images(val_dir) > 0:
        target = val_dir if split in ("val", "valid") else extracted / "train"
        n = _cap_images(target, num_samples)
        print(f"Imagenette: already present at {target} ({n} images)")
        return {
            "name": "imagenette",
            "path": str(target),
            "split": split,
            "num_images": n,
            "status": "cached",
        }

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    # TODO: pin an official sha256 when publishing paper release artifacts.
    expected = None
    archive = out_root / "imagenette2-320.tgz"
    try:
        _download(url, archive, expected_sha256=expected)
        _extract(archive, out_root)
    except Exception as e:
        print(f"[imagenette] download failed ({e}).")
        print("  Manual: https://github.com/fastai/imagenette")
        print(f"  Extract under {out_root}")
        return {
            "name": "imagenette",
            "path": str(val_dir),
            "split": split,
            "num_images": 0,
            "status": "manual_required",
        }

    target = val_dir if split in ("val", "valid") else extracted / "train"
    n = _cap_images(target, num_samples)
    print(f"Imagenette: {target} ({n} images)")
    return {
        "name": "imagenette",
        "path": str(target),
        "split": split,
        "num_images": n,
        "status": "ok",
    }


def download_imagenet(
    data_root: Path,
    split: str = "val",
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Full ImageNet requires registration — print instructions, do not hard-fail."""
    out = data_root / "imagenet" / ("val" if split in ("val", "valid") else "train")
    n = _count_images(out)
    if n > 0:
        n = _cap_images(out, num_samples)
        print(f"ImageNet: found local copy at {out} ({n} images)")
        return {"name": "imagenet", "path": str(out), "split": split, "num_images": n, "status": "cached"}

    print("[imagenet] Full ImageNet cannot be auto-downloaded (requires ILSVRC credentials).")
    print("  Manual steps:")
    print("    1. Register at https://image-net.org/ and accept the terms.")
    print("    2. Download ILSVRC2012 train/val tarballs.")
    print(f"    3. Extract so images live under {out}")
    print("  Alternatively use --dataset imagenette for an ImageNet-style subset.")
    return {
        "name": "imagenet",
        "path": str(out),
        "split": split,
        "num_images": 0,
        "status": "manual_required",
    }


def download_clic(
    data_root: Path,
    split: str = "professional",
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """CLIC professional/mobile test sets (learned compression + watermarking)."""
    split = split if split in ("professional", "mobile") else "professional"
    out = data_root / "clic" / split
    if out.exists() and _count_images(out) > 0:
        n = _cap_images(out, num_samples)
        print(f"CLIC {split}: already present at {out} ({n} images)")
        return {"name": "clic", "path": str(out), "split": split, "num_images": n, "status": "cached"}

    # CLIC challenge mirrors change over years; try a known public archive, else instruct.
    # 2021 test professional (zip of PNGs) — URL may require fallback.
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
        }

    n = _cap_images(out, num_samples)
    print(f"CLIC {split}: {out} ({n} images)")
    return {"name": "clic", "path": str(out), "split": split, "num_images": n, "status": "ok"}


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
    p.add_argument("--num-samples", type=int, default=None, help="Optional image cap")
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
    for name in names:
        if name not in HANDLERS:
            print(f"[skip] unknown dataset '{name}'")
            continue
        print(f"\n=== {name} ===")
        kwargs: Dict[str, Any] = {"data_root": data_root, "num_samples": args.num_samples}
        if args.split is not None:
            kwargs["split"] = args.split
        elif name == "coco":
            kwargs["split"] = "val2017"
        elif name == "div2k":
            kwargs["split"] = "valid"
        elif name == "clic":
            kwargs["split"] = "professional"
        try:
            entries.append(HANDLERS[name](**kwargs))
        except Exception as e:
            print(f"[{name}] error (continuing): {e}")
            entries.append(
                {
                    "name": name,
                    "path": str(data_root / name),
                    "split": args.split,
                    "num_images": 0,
                    "status": "error",
                    "error": str(e),
                }
            )

    write_manifest(data_root, entries)
    print("\nSummary:")
    for e in entries:
        print(f"  {e['name']}: {e.get('status')}  path={e.get('path')}  n={e.get('num_images')}")


if __name__ == "__main__":
    main()
