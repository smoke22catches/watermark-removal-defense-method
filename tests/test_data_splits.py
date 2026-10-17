"""Acceptance tests for fixed, seed-independent train/val/test splits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataloaders, build_test_loader
from src.data.splits import (
    COCO_TEST_SIZE,
    COCO_VAL_SIZE,
    DEFAULT_SPLIT_SEED,
    InsufficientTestSplitError,
    assert_train_test_disjoint,
    generate_coco_splits,
    generate_synthetic_splits,
    list_file_hash,
    verify_infer_no_train_overlap,
    write_list_file,
)


def _write_unique_image(path: Path, tag: bytes) -> None:
    """Minimal unique file with an image extension (valid for split indexing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"WMSPLIT\n" + tag)


def _make_coco_tree(
    data_root: Path,
    *,
    n_train: int = 40,
    n_val: int = COCO_VAL_SIZE + COCO_TEST_SIZE,
) -> None:
    train_dir = data_root / "coco" / "train2017"
    val_dir = data_root / "coco" / "val2017"
    for i in range(n_train):
        _write_unique_image(train_dir / f"train_{i:05d}.jpg", b"train" + i.to_bytes(4, "big"))
    for i in range(n_val):
        _write_unique_image(val_dir / f"val_{i:05d}.jpg", b"val" + i.to_bytes(4, "big"))


def test_generate_twice_byte_identical() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_a = root / "splits_a"
        splits_b = root / "splits_b"
        generate_synthetic_splits(
            data_root, splits_a, split_seed=DEFAULT_SPLIT_SEED, num_samples=64
        )
        generate_synthetic_splits(
            data_root, splits_b, split_seed=DEFAULT_SPLIT_SEED, num_samples=64
        )
        for name in ("train.txt", "val.txt", "test.txt"):
            a = (splits_a / "synthetic" / name).read_bytes()
            b = (splits_b / "synthetic" / name).read_bytes()
            assert a == b, f"{name} not byte-identical across regenerations"

        _make_coco_tree(data_root, n_train=30, n_val=COCO_VAL_SIZE + COCO_TEST_SIZE)
        generate_coco_splits(data_root, splits_a, split_seed=DEFAULT_SPLIT_SEED)
        generate_coco_splits(data_root, splits_b, split_seed=DEFAULT_SPLIT_SEED)
        for name in ("train.txt", "val.txt", "test.txt", "meta.json"):
            a = (splits_a / "coco" / name).read_bytes()
            b = (splits_b / "coco" / name).read_bytes()
            assert a == b, f"coco/{name} not byte-identical"


def test_train_seeds_record_identical_split_hashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_dir = root / "splits"
        generate_synthetic_splits(
            data_root, splits_dir, split_seed=DEFAULT_SPLIT_SEED, num_samples=48
        )
        hashes = []
        for seed in (0, 1, 2):
            train_loader, _ = build_dataloaders(
                dataset_name="synthetic",
                data_root=data_root,
                image_size=32,
                batch_size=4,
                num_workers=0,
                num_samples=48,
                seed=seed,
                splits_dir=splits_dir,
                split_seed=DEFAULT_SPLIT_SEED,
            )
            info = getattr(train_loader, "split_info", {})
            hashes.append(dict(info.get("split_hashes") or {}))
        assert hashes[0] == hashes[1] == hashes[2]
        assert hashes[0].get("test.txt")


def test_train_test_disjoint_names_and_content() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_dir = root / "splits"
        _make_coco_tree(data_root, n_train=40, n_val=COCO_VAL_SIZE + COCO_TEST_SIZE)
        spec = generate_coco_splits(
            data_root, splits_dir, split_seed=DEFAULT_SPLIT_SEED
        )
        assert_train_test_disjoint(
            spec.train_paths,
            spec.test_paths,
            data_root=data_root,
            check_content=True,
        )
        # Basename / path unit checks
        assert set(spec.train_paths).isdisjoint(spec.test_paths)
        assert set(spec.val_paths).isdisjoint(spec.test_paths)
        train_bn = {Path(p).name.lower() for p in spec.train_paths}
        test_bn = {Path(p).name.lower() for p in spec.test_paths}
        assert train_bn.isdisjoint(test_bn)


def test_infer_same_test_list_and_overlap_abort() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_dir = root / "splits"
        generate_synthetic_splits(
            data_root, splits_dir, split_seed=DEFAULT_SPLIT_SEED, num_samples=40
        )
        loader_a, info_a = build_test_loader(
            "synthetic",
            data_root,
            image_size=32,
            batch_size=4,
            splits_dir=splits_dir,
            split_seed=DEFAULT_SPLIT_SEED,
        )
        loader_b, info_b = build_test_loader(
            "synthetic",
            data_root,
            image_size=32,
            batch_size=4,
            splits_dir=splits_dir,
            split_seed=DEFAULT_SPLIT_SEED,
            num_samples=None,
        )
        assert info_a["test_list"] == info_b["test_list"]
        assert info_a["split_hashes"] == info_b["split_hashes"]
        assert len(loader_a) == len(loader_b)

        train_list = list(info_a["train_list"])
        test_list = list(info_a["test_list"])
        # Different "training seed" must not matter — overlap check uses lists.
        verify_infer_no_train_overlap(
            test_list, train_list, data_root=data_root
        )

        bad_test = list(test_list)
        if train_list:
            bad_test[0] = train_list[0]
        else:
            raise AssertionError("expected non-empty train_list")
        try:
            verify_infer_no_train_overlap(bad_test, train_list, data_root=data_root)
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "overlapping train/test must abort inference"


def test_coco_test_split_size_5000() -> None:
    assert COCO_TEST_SIZE == 5000
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_dir = root / "splits"
        # Real COCO layout: val2017 has exactly 5000 images (not 6000).
        _make_coco_tree(
            data_root,
            n_train=COCO_VAL_SIZE + 20,
            n_val=COCO_TEST_SIZE,
        )
        spec = generate_coco_splits(
            data_root, splits_dir, split_seed=DEFAULT_SPLIT_SEED
        )
        assert len(spec.test_paths) == 5000
        assert len(spec.val_paths) == COCO_VAL_SIZE
        assert all(p.startswith("coco/val2017/") for p in spec.test_paths)
        assert all(p.startswith("coco/train2017/") for p in spec.val_paths)
        assert all(p.startswith("coco/train2017/") for p in spec.train_paths)
        assert (splits_dir / "coco" / "test.txt").read_text(encoding="utf-8").count(
            "\n"
        ) == 5000
        # Hash recorded in meta matches live file
        meta = json.loads((splits_dir / "coco" / "meta.json").read_text(encoding="utf-8"))
        live = list_file_hash(splits_dir / "coco" / "test.txt")
        assert meta["split_hashes"]["test.txt"] == live
        assert meta["extra"]["val_from_train"] is True


def test_coco_small_val_refuses_without_allow_small_test() -> None:
    """A capped val2017 must fail make_splits unless --allow-small-test is set."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_dir = root / "splits"
        capped_val = 1500
        capped_train = 2000
        _make_coco_tree(data_root, n_train=capped_train, n_val=capped_val)
        try:
            generate_coco_splits(data_root, splits_dir, split_seed=DEFAULT_SPLIT_SEED)
            raised = False
        except InsufficientTestSplitError as e:
            raised = True
            msg = str(e)
            assert "1500" in msg
            assert str(COCO_TEST_SIZE) in msg
            assert "capped" in msg.lower() or "allow-small-test" in msg.lower()
        assert raised, "small val2017 must refuse without allow_small_test"

        spec = generate_coco_splits(
            data_root,
            splits_dir,
            split_seed=DEFAULT_SPLIT_SEED,
            allow_small_test=True,
        )
        assert len(spec.test_paths) == capped_val
        assert len(spec.val_paths) == COCO_VAL_SIZE
        assert len(spec.train_paths) == capped_train - COCO_VAL_SIZE
        meta = json.loads((splits_dir / "coco" / "meta.json").read_text(encoding="utf-8"))
        assert meta["extra"]["test_size"] == capped_val
        assert meta["extra"]["requested_test_size"] == COCO_TEST_SIZE
        assert meta["extra"]["allow_small_test"] is True
        assert meta["extra"]["val_from_train"] is True


def test_missing_coco_splits_do_not_fallback_to_synthetic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_dir = root / "splits"
        try:
            build_dataloaders(
                dataset_name="coco",
                data_root=data_root,
                image_size=32,
                batch_size=4,
                num_workers=0,
                seed=0,
                splits_dir=splits_dir,
                split_seed=DEFAULT_SPLIT_SEED,
            )
            raised = False
        except FileNotFoundError as e:
            raised = True
            assert "synthetic" in str(e).lower() or "make_splits" in str(e)
        assert raised, "missing coco splits must not silently become synthetic"


def test_coco_both_from_val_when_pool_large_enough() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        splits_dir = root / "splits"
        _make_coco_tree(
            data_root,
            n_train=30,
            n_val=COCO_VAL_SIZE + COCO_TEST_SIZE,
        )
        spec = generate_coco_splits(
            data_root, splits_dir, split_seed=DEFAULT_SPLIT_SEED
        )
        assert len(spec.test_paths) == COCO_TEST_SIZE
        assert len(spec.val_paths) == COCO_VAL_SIZE
        assert all(p.startswith("coco/val2017/") for p in spec.val_paths)
        assert all(p.startswith("coco/val2017/") for p in spec.test_paths)
        meta = json.loads((splits_dir / "coco" / "meta.json").read_text(encoding="utf-8"))
        assert meta["extra"]["val_from_train"] is False


def test_write_list_file_deterministic_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.txt"
        lines = ["b/x.jpg", "a/y.jpg"]
        h1 = write_list_file(p, lines)
        h2 = write_list_file(p, lines)
        assert h1 == h2 == list_file_hash(p)


if __name__ == "__main__":
    test_generate_twice_byte_identical()
    print("[ok] generate twice byte-identical")
    test_train_seeds_record_identical_split_hashes()
    print("[ok] train seeds identical split hashes")
    test_train_test_disjoint_names_and_content()
    print("[ok] train/test disjoint")
    test_infer_same_test_list_and_overlap_abort()
    print("[ok] infer same test list + overlap abort")
    test_coco_test_split_size_5000()
    print("[ok] coco test size 5000 (real val2017 layout)")
    test_coco_small_val_refuses_without_allow_small_test()
    print("[ok] coco small val refuses without --allow-small-test")
    test_missing_coco_splits_do_not_fallback_to_synthetic()
    print("[ok] missing coco splits do not fall back to synthetic")
    test_coco_both_from_val_when_pool_large_enough()
    print("[ok] coco both from val when pool large enough")
    test_write_list_file_deterministic_hash()
    print("[ok] list file hash")
    print("=== all data-split acceptance tests passed ===")
