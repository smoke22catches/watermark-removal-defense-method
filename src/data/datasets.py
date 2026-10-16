"""Image datasets and dataloader helpers for watermark training/eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLITS_DIR,
    SplitSpec,
    generate_synthetic_splits,
    load_split_spec,
    resolve_role_root,
    resolve_split_paths,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _default_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
        ]
    )


def _collect_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


class ImageFolderDataset(Dataset):
    """Flat / nested folder of RGB images; yields ``(image_tensor,)``."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 128,
        transform: Optional[transforms.Compose] = None,
        num_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.paths = _collect_images(self.root)
        if num_samples is not None:
            self.paths = self.paths[: int(num_samples)]
        self.transform = transform or _default_transform(image_size)
        self.rel_paths: List[str] = [p.name for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return (tensor,)


class ImageListDataset(Dataset):
    """Dataset from an explicit list of image paths (fixed split)."""

    def __init__(
        self,
        paths: Sequence[Path],
        image_size: int = 128,
        transform: Optional[transforms.Compose] = None,
        rel_paths: Optional[Sequence[str]] = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        missing = [p for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} split images missing on disk "
                f"(e.g. {missing[0]}). Download data or regenerate splits."
            )
        self.rel_paths = list(rel_paths) if rel_paths is not None else [p.name for p in self.paths]
        self.transform = transform or _default_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return (self.transform(img),)


class SyntheticImageDataset(Dataset):
    """Random RGB tensors in [-1, 1] for offline smoke tests without real data."""

    def __init__(
        self,
        num_samples: int = 32,
        image_size: int = 128,
        seed: int = 0,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.image_size = image_size
        # Full pool is always generated from a fixed seed so split indices are stable.
        g = torch.Generator().manual_seed(int(seed))
        pool_n = int(num_samples) if indices is None else max(int(num_samples), max(indices) + 1)
        self._pool = torch.rand(pool_n, 3, image_size, image_size, generator=g) * 2 - 1
        if indices is None:
            self._indices = list(range(int(num_samples)))
        else:
            self._indices = [int(i) for i in indices]
        self.num_samples = len(self._indices)
        self.rel_paths = [f"synthetic/{i:05d}" for i in self._indices]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        return (self._pool[self._indices[idx]],)


def load_manifest(data_root: str | Path) -> Dict[str, Any]:
    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.exists():
        return {"datasets": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_path(
    data_root: str | Path,
    name: str,
    role: str = "train",
) -> Optional[Path]:
    """Resolve a dataset folder for ``role`` in {train, val, test}.

    Train and test never resolve to the same folder by default for COCO / DIV2K /
    Imagenette. Prefer :func:`src.data.splits.resolve_role_root`.
    """
    return resolve_role_root(data_root, name, role)


def list_available_datasets(data_root: str | Path) -> List[str]:
    names = ["synthetic"]
    for name in ("coco", "div2k", "imagenette", "imagenet", "clic"):
        if resolve_dataset_path(data_root, name, role="train") is not None:
            names.append(name)
    return names


def _synthetic_indices_from_rel(rel_paths: Sequence[str]) -> List[int]:
    out: List[int] = []
    for rel in rel_paths:
        # synthetic/00042
        stem = Path(rel).name
        out.append(int(stem))
    return out


def ensure_split_spec(
    dataset: str,
    *,
    data_root: str | Path = "./data",
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    split_seed: int = DEFAULT_SPLIT_SEED,
    num_samples: Optional[int] = None,
) -> SplitSpec:
    """Load split lists; auto-create synthetic splits if missing."""
    dataset = dataset.lower()
    try:
        return load_split_spec(dataset, data_root=data_root, splits_dir=splits_dir)
    except FileNotFoundError:
        if dataset == "synthetic":
            n = int(num_samples) if num_samples is not None else 64
            return generate_synthetic_splits(
                Path(data_root),
                Path(splits_dir),
                split_seed=int(split_seed),
                num_samples=n,
            )
        raise


def build_split_dataset(
    dataset: str,
    role: str,
    *,
    data_root: str | Path = "./data",
    image_size: int = 128,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    split_seed: int = DEFAULT_SPLIT_SEED,
    num_samples: Optional[int] = None,
    split_spec: Optional[SplitSpec] = None,
) -> Tuple[Dataset, SplitSpec]:
    """Build a dataset for one role from fixed split lists."""
    dataset = dataset.lower()
    role = role.lower()
    spec = split_spec or ensure_split_spec(
        dataset,
        data_root=data_root,
        splits_dir=splits_dir,
        split_seed=split_seed,
        num_samples=num_samples,
    )
    rel = {"train": spec.train_paths, "val": spec.val_paths, "test": spec.test_paths}[role]
    if num_samples is not None:
        rel = list(rel)[: int(num_samples)]

    if dataset == "synthetic":
        # Pool size from meta or max index + 1.
        max_idx = max(_synthetic_indices_from_rel(spec.train_paths + spec.val_paths + spec.test_paths), default=0)
        pool_n = max(max_idx + 1, int(num_samples or 64))
        ds: Dataset = SyntheticImageDataset(
            num_samples=pool_n,
            image_size=image_size,
            seed=0,  # fixed pool; split indices select rows
            indices=_synthetic_indices_from_rel(rel),
        )
        return ds, spec

    abs_paths = resolve_split_paths(rel, data_root)
    ds = ImageListDataset(abs_paths, image_size=image_size, rel_paths=rel)
    return ds, spec


def build_dataset(
    name: str,
    data_root: str | Path,
    image_size: int = 128,
    num_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """Build a dataset by name. Falls back to synthetic if files/splits are missing."""
    name = name.lower()
    try:
        ds, _ = build_split_dataset(
            name,
            split,
            data_root=data_root,
            image_size=image_size,
            num_samples=num_samples,
        )
        return ds
    except FileNotFoundError:
        print(
            f"[data] Dataset '{name}' split '{split}' not available under {data_root}; "
            f"using synthetic. Run scripts/download_data.py and scripts/make_splits.py."
        )
        n = num_samples if num_samples is not None else 64
        return SyntheticImageDataset(num_samples=n, image_size=image_size, seed=0)


def epoch_seed(seed: int, epoch: int) -> int:
    """Deterministic seed for epoch ``e`` derived from the run seed."""
    return int(seed) + int(epoch) * 1_000_003


def dataloader_kwargs(
    num_workers: int = 0,
    *,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared DataLoader performance knobs for train / val / test.

    ``pin_memory`` defaults to True when ``device`` looks like CUDA.
    ``persistent_workers`` / ``prefetch_factor`` only apply when ``num_workers > 0``.
    """
    nw = max(0, int(num_workers))
    if pin_memory is None:
        pin_memory = bool(device is not None and str(device).startswith("cuda"))
    kw: Dict[str, Any] = {
        "num_workers": nw,
        "pin_memory": bool(pin_memory),
        "drop_last": False,
    }
    if nw > 0:
        if persistent_workers is None:
            persistent_workers = True
        kw["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is None:
            prefetch_factor = 2
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
    return kw


def make_epoch_train_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    device: Optional[str] = None,
) -> DataLoader:
    """Train loader with shuffle order fixed by ``(seed, epoch)``.

    Prefer :func:`make_reusable_train_loader` + :func:`reseed_train_loader` inside
    the training loop so worker processes stay warm across epochs.
    """
    g = torch.Generator()
    g.manual_seed(epoch_seed(seed, epoch))
    sampler = torch.utils.data.RandomSampler(dataset, generator=g)
    kw = dataloader_kwargs(
        num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        device=device,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, **kw)
    loader._shuffle_generator = g  # type: ignore[attr-defined]
    return loader


def make_reusable_train_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    device: Optional[str] = None,
) -> DataLoader:
    """One long-lived train loader; reseed with :func:`reseed_train_loader` each epoch."""
    return make_epoch_train_loader(
        dataset,
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        device=device,
    )


def reseed_train_loader(loader: DataLoader, seed: int, epoch: int) -> None:
    """Reset the train shuffle generator for deterministic epoch ``e`` order."""
    g = getattr(loader, "_shuffle_generator", None)
    if g is None:
        raise AttributeError(
            "DataLoader has no _shuffle_generator; build it with make_epoch_train_loader "
            "or make_reusable_train_loader."
        )
    g.manual_seed(epoch_seed(seed, epoch))


def skip_batches(loader: DataLoader, n_skip: int) -> Iterator[Any]:
    """Yield batches from ``loader`` after skipping the first ``n_skip``."""
    it = iter(loader)
    for _ in range(max(0, int(n_skip))):
        try:
            next(it)
        except StopIteration:
            return
    yield from it


def build_dataloaders(
    dataset_name: str,
    data_root: str | Path,
    image_size: int = 128,
    batch_size: int = 4,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    split_seed: int = DEFAULT_SPLIT_SEED,
    include_test: bool = False,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader] | Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create loaders from fixed split lists (independent of ``seed``).

    ``seed`` only affects the epoch shuffle order of the train loader, not membership.
    ``val_fraction`` is ignored when split files exist (kept for CLI compatibility).

    When ``include_test=True``, also returns ``(train, val, test, split_info)``.
    """
    _ = val_fraction  # membership comes from split files
    dataset_name = dataset_name.lower()
    try:
        spec = ensure_split_spec(
            dataset_name,
            data_root=data_root,
            splits_dir=splits_dir,
            split_seed=split_seed,
            num_samples=num_samples,
        )
    except FileNotFoundError:
        print(
            f"[data] No fixed splits for '{dataset_name}'; falling back to synthetic splits."
        )
        spec = generate_synthetic_splits(
            Path(data_root),
            Path(splits_dir),
            split_seed=int(split_seed),
            num_samples=int(num_samples) if num_samples is not None else 64,
        )
        dataset_name = "synthetic"

    train_ds, spec = build_split_dataset(
        dataset_name,
        "train",
        data_root=data_root,
        image_size=image_size,
        splits_dir=splits_dir,
        split_seed=split_seed,
        num_samples=num_samples,
        split_spec=spec,
    )
    val_ds, spec = build_split_dataset(
        dataset_name,
        "val",
        data_root=data_root,
        image_size=image_size,
        splits_dir=splits_dir,
        split_seed=split_seed,
        num_samples=None,
        split_spec=spec,
    )
    if len(val_ds) == 0:
        # Degenerate tiny synthetic: reuse a one-sample view of train for model selection.
        val_ds = train_ds

    dl_kw = dataloader_kwargs(
        num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        device=device,
    )
    train_loader = make_reusable_train_loader(
        train_ds,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        device=device,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **dl_kw,
    )
    info = spec.as_meta()
    info["train_list"] = list(spec.train_paths)
    info["val_list"] = list(spec.val_paths)
    info["test_list"] = list(spec.test_paths)

    if not include_test:
        # Attach for callers that still unpack two loaders.
        train_loader.split_info = info  # type: ignore[attr-defined]
        val_loader.split_info = info  # type: ignore[attr-defined]
        return train_loader, val_loader

    test_ds, spec = build_split_dataset(
        dataset_name,
        "test",
        data_root=data_root,
        image_size=image_size,
        splits_dir=splits_dir,
        split_seed=split_seed,
        num_samples=None,
        split_spec=spec,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **dl_kw,
    )
    if len(test_ds) < 5000 and dataset_name != "synthetic":
        print(
            f"[WARNING] test split has {len(test_ds)} < 5000 images; "
            "FID and TPR@0.1%FPR cannot be estimated reliably."
        )
    return train_loader, val_loader, test_loader, info


def build_test_loader(
    dataset_name: str,
    data_root: str | Path,
    *,
    image_size: int = 128,
    batch_size: int = 4,
    num_workers: int = 0,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    split_seed: int = DEFAULT_SPLIT_SEED,
    num_samples: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """Inference-only loader over the fixed test split."""
    dataset_name = dataset_name.lower()
    spec = ensure_split_spec(
        dataset_name,
        data_root=data_root,
        splits_dir=splits_dir,
        split_seed=split_seed,
        num_samples=num_samples,
    )
    test_ds, spec = build_split_dataset(
        dataset_name,
        "test",
        data_root=data_root,
        image_size=image_size,
        splits_dir=splits_dir,
        split_seed=split_seed,
        num_samples=num_samples,
        split_spec=spec,
    )
    if len(test_ds) < 5000 and dataset_name != "synthetic":
        print(
            f"[WARNING] clean/test set has {len(test_ds)} < 5000 images; "
            "0.1% FPR and FID cannot be estimated reliably."
        )
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs(
            num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            device=device,
        ),
    )
    info = spec.as_meta()
    info["train_list"] = list(spec.train_paths)
    info["test_list"] = list(spec.test_paths)
    return loader, info
