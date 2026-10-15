"""Fixed, seed-independent train/val/test image lists.

Splits are defined by text files under ``data/splits/<dataset>/`` and are
generated once with a dedicated ``split_seed`` (never the training seed).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_SPLIT_SEED = 20241010
DEFAULT_SPLITS_DIR = "data/splits"

# Recommended sizes (overridable via config / CLI).
COCO_VAL_SIZE = 1000
COCO_TEST_SIZE = 5000
DIV2K_VAL_FROM_TRAIN = 50
IMAGENETTE_VAL_FROM_TRAIN = 500


@dataclass
class SplitSpec:
    """Resolved split paths + metadata for one dataset."""

    dataset: str
    splits_dir: Path
    train_list: Path
    val_list: Path
    test_list: Path
    meta_path: Path
    train_paths: List[str] = field(default_factory=list)
    val_paths: List[str] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    file_hashes: Dict[str, str] = field(default_factory=dict)  # list filename -> sha256
    split_seed: int = DEFAULT_SPLIT_SEED

    def as_meta(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "splits_dir": str(self.splits_dir).replace("\\", "/"),
            "split_seed": int(self.split_seed),
            "split_files": {
                "train": self.train_list.name,
                "val": self.val_list.name,
                "test": self.test_list.name,
            },
            "split_hashes": dict(self.file_hashes),
            "counts": {
                "train": len(self.train_paths),
                "val": len(self.val_paths),
                "test": len(self.test_paths),
            },
        }


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_list_file(path: Path, lines: Sequence[str]) -> str:
    """Write newline-terminated sorted-stable list; return content sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve caller order (already determined by split_seed shuffle).
    body = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(body, encoding="utf-8", newline="\n")
    return text_sha256(body)


def read_list_file(path: Path) -> List[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def list_file_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    return text_sha256(path.read_text(encoding="utf-8"))


def split_dir_for(dataset: str, splits_root: str | Path = DEFAULT_SPLITS_DIR) -> Path:
    return Path(splits_root) / dataset.lower()


def resolve_role_root(
    data_root: str | Path,
    dataset: str,
    role: str,
) -> Optional[Path]:
    """Resolve the image folder for ``role`` in {train, val, test}.

    Train and test never share the same default folder for multi-folder datasets.
    """
    data_root = Path(data_root)
    name = dataset.lower()
    role = role.lower()

    if name == "synthetic":
        return None

    if name == "coco":
        if role == "train":
            # Never fall back to val2017 — train/test must stay in different folders.
            candidates = [
                data_root / "coco" / "train2017",
                data_root / "coco" / "train2014",
            ]
        else:  # val / test both drawn from val2017 via disjoint lists
            candidates = [
                data_root / "coco" / "val2017",
                data_root / "coco" / "val2014",
            ]
    elif name == "div2k":
        if role == "train" or role == "val":
            # val images are a held-out slice of train_HR (lists enforce this)
            candidates = [data_root / "div2k" / "DIV2K_train_HR"]
        else:
            candidates = [data_root / "div2k" / "DIV2K_valid_HR"]
    elif name in ("imagenette", "imagenet"):
        base = data_root / name
        if name == "imagenette":
            extracted = base / "imagenette2-320"
            if role == "test":
                candidates = [extracted / "val", base / "val"]
            else:
                candidates = [extracted / "train", base / "train"]
        else:
            if role == "test":
                candidates = [base / "val"]
            else:
                candidates = [base / "train"]
    elif name == "clic":
        candidates = [
            data_root / "clic" / "professional",
            data_root / "clic" / "mobile",
        ]
    else:
        candidates = [data_root / name]

    for c in candidates:
        if c.exists() and collect_images(c):
            return c
    return None


def _shuffle(paths: List[str], split_seed: int) -> List[str]:
    rng = random.Random(int(split_seed))
    out = list(paths)
    rng.shuffle(out)
    return out


def generate_coco_splits(
    data_root: Path,
    splits_root: Path,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_max: Optional[int] = None,
    val_size: int = COCO_VAL_SIZE,
    test_size: int = COCO_TEST_SIZE,
) -> SplitSpec:
    """Build COCO train/val/test lists.

    Official ``val2017`` has exactly 5,000 images. Defaults request test=5,000, so
    both val and test cannot come from ``val2017``. Policy:

    * If ``val2017`` is large enough for ``val_size + test_size``, draw both from it
      (disjoint).
    * Otherwise (real COCO): use ``val2017`` for **test** only, and hold out
      ``val_size`` images from ``train2017`` for validation (same pattern as DIV2K /
      Imagenette). Train and test remain in different folders.
    """
    train_root = resolve_role_root(data_root, "coco", "train")
    val_root = resolve_role_root(data_root, "coco", "val")
    if train_root is None or val_root is None:
        raise FileNotFoundError(
            "COCO train2017 and val2017 are required to generate splits. "
            "Run: python scripts/download_data.py --dataset coco"
        )
    train_pool = [rel_posix(p, data_root) for p in collect_images(train_root)]
    val_pool = [rel_posix(p, data_root) for p in collect_images(val_root)]
    train_pool = _shuffle(train_pool, split_seed)
    if train_max is not None:
        train_pool = train_pool[: int(train_max)]
    val_pool = _shuffle(val_pool, split_seed + 1)

    val_from_train = False
    if len(val_pool) >= val_size + test_size:
        val_imgs = val_pool[:val_size]
        test_imgs = val_pool[val_size : val_size + test_size]
        train_imgs = train_pool
        val_test_root = rel_posix(val_root, data_root)
    else:
        if len(val_pool) < test_size:
            raise RuntimeError(
                f"COCO val2017 has {len(val_pool)} images; need at least "
                f"test_size={test_size}. "
                "Run: python scripts/download_data.py --dataset coco"
            )
        if len(train_pool) <= val_size:
            raise RuntimeError(
                f"COCO train2017 has {len(train_pool)} images after train_max; "
                f"need > val_size={val_size} to hold out a validation split"
            )
        # Real COCO: full val2017 → test; carve val from train2017.
        test_imgs = val_pool[:test_size]
        val_imgs = train_pool[:val_size]
        train_imgs = train_pool[val_size:]
        val_from_train = True
        val_test_root = rel_posix(val_root, data_root)
        print(
            f"[splits] coco: val2017 has {len(val_pool)} < val+test="
            f"{val_size + test_size}; using val2017 for test ({len(test_imgs)}) "
            f"and holding out val={len(val_imgs)} from train2017"
        )

    return _write_dataset_splits(
        "coco",
        data_root,
        splits_root,
        train_imgs,
        val_imgs,
        test_imgs,
        split_seed=split_seed,
        extra_meta={
            "train_root": rel_posix(train_root, data_root),
            "val_test_root": val_test_root,
            "val_root": (
                rel_posix(train_root, data_root)
                if val_from_train
                else rel_posix(val_root, data_root)
            ),
            "val_from_train": val_from_train,
            "val_size": val_size,
            "test_size": test_size,
            "train_max": train_max,
        },
    )


def generate_div2k_splits(
    data_root: Path,
    splits_root: Path,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_size: int = DIV2K_VAL_FROM_TRAIN,
) -> SplitSpec:
    train_root = resolve_role_root(data_root, "div2k", "train")
    test_root = resolve_role_root(data_root, "div2k", "test")
    if train_root is None or test_root is None:
        raise FileNotFoundError(
            "DIV2K train_HR and valid_HR are required. "
            "Run: python scripts/download_data.py --dataset div2k"
        )
    if train_root.resolve() == test_root.resolve():
        raise RuntimeError("DIV2K train and test resolved to the same folder")
    train_pool = _shuffle(
        [rel_posix(p, data_root) for p in collect_images(train_root)], split_seed
    )
    if len(train_pool) <= val_size:
        raise RuntimeError(f"DIV2K train has {len(train_pool)} images; need > val_size={val_size}")
    val_imgs = train_pool[:val_size]
    train_imgs = train_pool[val_size:]
    test_imgs = _shuffle(
        [rel_posix(p, data_root) for p in collect_images(test_root)], split_seed + 2
    )
    return _write_dataset_splits(
        "div2k",
        data_root,
        splits_root,
        train_imgs,
        val_imgs,
        test_imgs,
        split_seed=split_seed,
        extra_meta={
            "train_root": rel_posix(train_root, data_root),
            "test_root": rel_posix(test_root, data_root),
            "val_size": val_size,
        },
    )


def generate_imagenette_splits(
    data_root: Path,
    splits_root: Path,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_size: int = IMAGENETTE_VAL_FROM_TRAIN,
) -> SplitSpec:
    train_root = resolve_role_root(data_root, "imagenette", "train")
    test_root = resolve_role_root(data_root, "imagenette", "test")
    if train_root is None or test_root is None:
        raise FileNotFoundError(
            "Imagenette train/ and val/ are required. "
            "Run: python scripts/download_data.py --dataset imagenette"
        )
    if train_root.resolve() == test_root.resolve():
        raise RuntimeError("Imagenette train and test resolved to the same folder")
    train_pool = _shuffle(
        [rel_posix(p, data_root) for p in collect_images(train_root)], split_seed
    )
    n_val = min(val_size, max(1, len(train_pool) // 10))
    val_imgs = train_pool[:n_val]
    train_imgs = train_pool[n_val:]
    test_imgs = _shuffle(
        [rel_posix(p, data_root) for p in collect_images(test_root)], split_seed + 3
    )
    return _write_dataset_splits(
        "imagenette",
        data_root,
        splits_root,
        train_imgs,
        val_imgs,
        test_imgs,
        split_seed=split_seed,
        extra_meta={
            "train_root": rel_posix(train_root, data_root),
            "test_root": rel_posix(test_root, data_root),
            "val_size": n_val,
        },
    )


def generate_synthetic_splits(
    data_root: Path,
    splits_root: Path,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    num_samples: int = 64,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> SplitSpec:
    """Index-based synthetic splits (no disk images). Entries like ``synthetic/00042``."""
    n = int(num_samples)
    ids = [f"synthetic/{i:05d}" for i in range(n)]
    ids = _shuffle(ids, split_seed)
    n_test = max(1, int(n * test_fraction)) if n > 2 else 1
    n_val = max(1, int(n * val_fraction)) if n > 2 else 0
    n_train = max(1, n - n_test - n_val)
    # Adjust if rounding overflowed.
    while n_train + n_val + n_test > n:
        if n_val > 0:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    test_imgs = ids[:n_test]
    val_imgs = ids[n_test : n_test + n_val]
    train_imgs = ids[n_test + n_val : n_test + n_val + n_train]
    return _write_dataset_splits(
        "synthetic",
        data_root,
        splits_root,
        train_imgs,
        val_imgs,
        test_imgs,
        split_seed=split_seed,
        extra_meta={"num_samples": n, "val_fraction": val_fraction, "test_fraction": test_fraction},
    )


def _write_dataset_splits(
    dataset: str,
    data_root: Path,
    splits_root: Path,
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
    *,
    split_seed: int,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> SplitSpec:
    out = split_dir_for(dataset, splits_root)
    out.mkdir(parents=True, exist_ok=True)
    train_list = out / "train.txt"
    val_list = out / "val.txt"
    test_list = out / "test.txt"
    h_train = write_list_file(train_list, train)
    h_val = write_list_file(val_list, val)
    h_test = write_list_file(test_list, test)
    # Disjointness on basenames and full relative paths.
    _assert_disjoint(train, val, test, dataset)
    meta = {
        "dataset": dataset,
        "split_seed": int(split_seed),
        # Portable hint only — absolute machine paths must not be committed.
        "data_root": Path(data_root).as_posix(),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "split_hashes": {
            "train.txt": h_train,
            "val.txt": h_val,
            "test.txt": h_test,
        },
        "extra": dict(extra_meta or {}),
    }
    meta_path = out / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SplitSpec(
        dataset=dataset,
        splits_dir=out,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        meta_path=meta_path,
        train_paths=list(train),
        val_paths=list(val),
        test_paths=list(test),
        file_hashes={"train.txt": h_train, "val.txt": h_val, "test.txt": h_test},
        split_seed=int(split_seed),
    )


def _assert_disjoint(
    train: Sequence[str], val: Sequence[str], test: Sequence[str], dataset: str
) -> None:
    sets = {
        "train": set(train),
        "val": set(val),
        "test": set(test),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise RuntimeError(
                f"{dataset} split overlap between {a} and {b}: "
                f"{len(overlap)} entries e.g. {next(iter(overlap))!r}"
            )
    # Basename collision across train/test (leakage even if folders differ).
    def basenames(paths: Iterable[str]) -> Set[str]:
        return {Path(p).name.lower() for p in paths}

    bn_train, bn_test = basenames(train), basenames(test)
    bn_overlap = bn_train & bn_test
    if bn_overlap and dataset != "synthetic":
        raise RuntimeError(
            f"{dataset} train/test share {len(bn_overlap)} basenames "
            f"(e.g. {next(iter(bn_overlap))!r})"
        )


def load_split_spec(
    dataset: str,
    *,
    data_root: str | Path = "./data",
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> SplitSpec:
    dataset = dataset.lower()
    out = split_dir_for(dataset, splits_dir)
    train_list = out / "train.txt"
    val_list = out / "val.txt"
    test_list = out / "test.txt"
    meta_path = out / "meta.json"
    if not train_list.is_file() or not test_list.is_file():
        raise FileNotFoundError(
            f"Missing split lists under {out}. "
            f"Run: python scripts/make_splits.py --dataset {dataset}"
        )
    train = read_list_file(train_list)
    val = read_list_file(val_list) if val_list.is_file() else []
    test = read_list_file(test_list)
    hashes = {
        "train.txt": list_file_hash(train_list),
        "val.txt": list_file_hash(val_list) if val_list.is_file() else "",
        "test.txt": list_file_hash(test_list),
    }
    split_seed = DEFAULT_SPLIT_SEED
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        split_seed = int(meta.get("split_seed", DEFAULT_SPLIT_SEED))
        # Prefer hashes recorded at generation time if present.
        stored = meta.get("split_hashes") or {}
        for k, v in stored.items():
            if k in hashes and v:
                hashes[k] = str(v)
    _assert_disjoint(train, val, test, dataset)
    return SplitSpec(
        dataset=dataset,
        splits_dir=out,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        meta_path=meta_path,
        train_paths=train,
        val_paths=val,
        test_paths=test,
        file_hashes=hashes,
        split_seed=split_seed,
    )


def resolve_split_paths(
    rel_paths: Sequence[str],
    data_root: str | Path,
) -> List[Path]:
    """Map relative split entries to absolute paths (skip synthetic:// indices)."""
    root = Path(data_root)
    out: List[Path] = []
    for rel in rel_paths:
        if rel.startswith("synthetic/"):
            continue
        p = Path(rel)
        abs_p = p if p.is_absolute() else (root / p)
        out.append(abs_p)
    return out


def content_hashes_for_paths(
    paths: Sequence[Path],
    *,
    max_files: Optional[int] = None,
) -> Dict[str, str]:
    """Map basename -> content sha256 for overlap checks."""
    out: Dict[str, str] = {}
    for i, p in enumerate(paths):
        if max_files is not None and i >= max_files:
            break
        if not p.is_file():
            continue
        out[p.name.lower()] = file_sha256(p)
    return out


def assert_train_test_disjoint(
    train_rel: Sequence[str],
    test_rel: Sequence[str],
    *,
    data_root: str | Path,
    check_content: bool = True,
) -> None:
    """Abort if train/test lists overlap by relative path, basename, or content hash."""
    train_set, test_set = set(train_rel), set(test_rel)
    path_overlap = train_set & test_set
    if path_overlap:
        raise RuntimeError(
            f"Train/test split overlap ({len(path_overlap)} paths), "
            f"e.g. {next(iter(path_overlap))!r}"
        )
    bn_train = {Path(p).name.lower() for p in train_rel}
    bn_test = {Path(p).name.lower() for p in test_rel}
    bn_overlap = bn_train & bn_test
    if bn_overlap and not all(p.startswith("synthetic/") for p in train_rel):
        raise RuntimeError(
            f"Train/test basename overlap ({len(bn_overlap)}), "
            f"e.g. {next(iter(bn_overlap))!r}"
        )
    if not check_content:
        return
    # Content-hash check for real image files that exist on disk.
    train_paths = [p for p in resolve_split_paths(train_rel, data_root) if p.is_file()]
    test_paths = [p for p in resolve_split_paths(test_rel, data_root) if p.is_file()]
    if not train_paths or not test_paths:
        return
    th = set(content_hashes_for_paths(train_paths).values())
    sh = content_hashes_for_paths(test_paths)
    for name, h in sh.items():
        if h in th:
            raise RuntimeError(
                f"Train/test content-hash overlap for file basename {name!r} (sha256={h[:12]}…)"
            )


def verify_infer_no_train_overlap(
    test_rel: Sequence[str],
    train_rel_from_meta: Sequence[str],
    *,
    data_root: str | Path,
) -> None:
    """Inference safety check against training list recorded in run metadata."""
    assert_train_test_disjoint(
        train_rel_from_meta,
        test_rel,
        data_root=data_root,
        check_content=True,
    )


def generate_all_available(
    data_root: str | Path,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    datasets: Optional[Sequence[str]] = None,
    coco_train_max: Optional[int] = None,
    coco_val_size: int = COCO_VAL_SIZE,
    coco_test_size: int = COCO_TEST_SIZE,
    synthetic_n: int = 64,
) -> Dict[str, SplitSpec]:
    """Generate splits for each available dataset (skips missing sources)."""
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    wanted = [d.lower() for d in (datasets or ("coco", "div2k", "imagenette", "synthetic"))]
    out: Dict[str, SplitSpec] = {}
    for name in wanted:
        try:
            if name == "coco":
                out[name] = generate_coco_splits(
                    data_root,
                    splits_dir,
                    split_seed=split_seed,
                    train_max=coco_train_max,
                    val_size=coco_val_size,
                    test_size=coco_test_size,
                )
            elif name == "div2k":
                out[name] = generate_div2k_splits(
                    data_root, splits_dir, split_seed=split_seed
                )
            elif name == "imagenette":
                out[name] = generate_imagenette_splits(
                    data_root, splits_dir, split_seed=split_seed
                )
            elif name == "synthetic":
                out[name] = generate_synthetic_splits(
                    data_root, splits_dir, split_seed=split_seed, num_samples=synthetic_n
                )
            else:
                print(f"[splits] skip unknown dataset {name}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[splits] skip {name}: {e}")
    return out
