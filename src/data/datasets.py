"""Image datasets and dataloader helpers for watermark training/eval."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


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
    """Flat / nested folder of RGB images; yields ``(image_tensor,)`` to match start.py."""

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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return (tensor,)


class SyntheticImageDataset(Dataset):
    """Random RGB tensors in [-1, 1] for offline smoke tests without real data."""

    def __init__(self, num_samples: int = 32, image_size: int = 128, seed: int = 0) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        g = torch.Generator().manual_seed(seed)
        self._data = torch.rand(num_samples, 3, image_size, image_size, generator=g) * 2 - 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        return (self._data[idx],)


def load_manifest(data_root: str | Path) -> Dict[str, Any]:
    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.exists():
        return {"datasets": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_path(data_root: str | Path, name: str) -> Optional[Path]:
    """Resolve a dataset name via manifest or conventional folder layout."""
    data_root = Path(data_root)
    name = name.lower()
    if name == "synthetic":
        return None

    manifest = load_manifest(data_root)
    entry = manifest.get("datasets", {}).get(name)
    if entry and entry.get("path"):
        p = Path(entry["path"])
        if p.exists():
            return p

    # Conventional layouts written by scripts/download_data.py
    candidates = {
        "coco": [
            data_root / "coco" / "val2017",
            data_root / "coco" / "val2014",
            data_root / "coco",
        ],
        "div2k": [
            data_root / "div2k" / "DIV2K_valid_HR",
            data_root / "div2k" / "DIV2K_train_HR",
            data_root / "div2k",
        ],
        "imagenette": [
            data_root / "imagenette" / "imagenette2-320" / "val",
            data_root / "imagenette" / "val",
            data_root / "imagenette",
        ],
        "imagenet": [
            data_root / "imagenet" / "val",
            data_root / "imagenet",
        ],
        "clic": [
            data_root / "clic" / "professional",
            data_root / "clic" / "mobile",
            data_root / "clic",
        ],
    }
    for c in candidates.get(name, [data_root / name]):
        if c.exists() and _collect_images(c):
            return c
    return None


def list_available_datasets(data_root: str | Path) -> List[str]:
    names = ["synthetic"]
    for name in ("coco", "div2k", "imagenette", "imagenet", "clic"):
        if resolve_dataset_path(data_root, name) is not None:
            names.append(name)
    return names


def build_dataset(
    name: str,
    data_root: str | Path,
    image_size: int = 128,
    num_samples: Optional[int] = None,
    split: str = "train",
) -> Dataset:
    """Build a dataset by name. Falls back to synthetic if files are missing."""
    name = name.lower()
    if name == "synthetic":
        n = num_samples if num_samples is not None else 64
        return SyntheticImageDataset(num_samples=n, image_size=image_size)

    path = resolve_dataset_path(data_root, name)
    if path is None:
        print(
            f"[data] Dataset '{name}' not found under {data_root}; "
            f"using synthetic images. Run scripts/download_data.py --dataset {name}"
        )
        n = num_samples if num_samples is not None else 64
        return SyntheticImageDataset(num_samples=n, image_size=image_size)

    return ImageFolderDataset(path, image_size=image_size, num_samples=num_samples)


def build_dataloaders(
    dataset_name: str,
    data_root: str | Path,
    image_size: int = 128,
    batch_size: int = 4,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders yielding ``(image,)`` batches."""
    full = build_dataset(
        dataset_name, data_root, image_size=image_size, num_samples=num_samples
    )
    n = len(full)
    if n == 0:
        raise RuntimeError(f"Empty dataset for '{dataset_name}'")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction)) if n > 1 else 0
    if n_val == 0:
        train_ds, val_ds = full, full
    else:
        val_idx = indices[:n_val]
        train_idx = indices[n_val:] or indices[:1]
        train_ds = Subset(full, train_idx)
        val_ds = Subset(full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
