"""Acceptance: --fraction must not silently shrink evaluation folders."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_data import (  # noqa: E402
    OFFICIAL_IMAGE_COUNTS,
    _cap_images,
    is_training_split,
    resolve_cap_args,
)


def test_fraction_caps_train_only_by_default() -> None:
    n, f, will = resolve_cap_args(
        "coco", "train2017", num_samples=None, fraction=0.3, cap_eval_splits=False
    )
    assert will and f == 0.3 and n is None

    n, f, will = resolve_cap_args(
        "coco", "val2017", num_samples=None, fraction=0.3, cap_eval_splits=False
    )
    assert not will and f is None and n is None

    n, f, will = resolve_cap_args(
        "coco", "val2017", num_samples=None, fraction=0.3, cap_eval_splits=True
    )
    assert will and f == 0.3


def test_training_split_helpers() -> None:
    assert is_training_split("coco", "train2017")
    assert not is_training_split("coco", "val2017")
    assert is_training_split("div2k", "train")
    assert not is_training_split("div2k", "valid")
    assert is_training_split("imagenette", "train")
    assert not is_training_split("imagenette", "val")


def test_cap_images_fraction_on_train_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(100):
            (root / f"img_{i:03d}.jpg").write_bytes(b"x" + bytes([i]))
        n = _cap_images(root, fraction=0.3)
        assert n == 30
        assert sum(1 for p in root.glob("*.jpg")) == 30


def test_official_val2017_count() -> None:
    assert OFFICIAL_IMAGE_COUNTS[("coco", "val2017")] == 5000


def test_fraction_leaves_eval_folder_uncapped() -> None:
    """--fraction 0.3 caps train2017 but leaves val2017 at its full local count."""
    import scripts.download_data as dd

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        train = data / "coco" / "train2017"
        val = data / "coco" / "val2017"
        train.mkdir(parents=True)
        val.mkdir(parents=True)
        for i in range(100):
            (train / f"t{i:05d}.jpg").write_bytes(b"t" + bytes([i]))
        for i in range(50):
            (val / f"v{i:05d}.jpg").write_bytes(b"v" + bytes([i]))

        old = dict(dd.OFFICIAL_IMAGE_COUNTS)
        dd.OFFICIAL_IMAGE_COUNTS[("coco", "train2017")] = 100
        dd.OFFICIAL_IMAGE_COUNTS[("coco", "val2017")] = 50
        try:
            cn, cf, will = resolve_cap_args(
                "coco",
                "train2017",
                num_samples=None,
                fraction=0.3,
                cap_eval_splits=False,
            )
            e_train = dd.download_coco(
                data,
                split="train2017",
                num_samples=cn,
                fraction=cf,
                will_cap=will,
                assume_yes=True,
            )
            cn, cf, will = resolve_cap_args(
                "coco",
                "val2017",
                num_samples=None,
                fraction=0.3,
                cap_eval_splits=False,
            )
            e_val = dd.download_coco(
                data,
                split="val2017",
                num_samples=cn,
                fraction=cf,
                will_cap=will,
                assume_yes=True,
            )
            assert e_train["num_images"] == 30 and e_train["capped"]
            assert e_val["num_images"] == 50 and e_val["kept_full"] and not e_val["capped"]
        finally:
            dd.OFFICIAL_IMAGE_COUNTS.clear()
            dd.OFFICIAL_IMAGE_COUNTS.update(old)


if __name__ == "__main__":
    test_fraction_caps_train_only_by_default()
    print("[ok] fraction caps train only")
    test_training_split_helpers()
    print("[ok] training split helpers")
    test_cap_images_fraction_on_train_folder()
    print("[ok] cap images")
    test_official_val2017_count()
    print("[ok] official val2017 count")
    test_fraction_leaves_eval_folder_uncapped()
    print("[ok] fraction leaves eval uncapped")
    print("=== download cap policy tests passed ===")
