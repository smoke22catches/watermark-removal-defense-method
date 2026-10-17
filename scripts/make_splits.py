#!/usr/bin/env python3
"""Generate fixed train/val/test image lists under data/splits/.

Uses a dedicated ``split_seed`` that is independent of the training seed.
Re-running with the same arguments yields byte-identical list files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLITS_DIR,
    COCO_TEST_SIZE,
    COCO_VAL_SIZE,
    InsufficientTestSplitError,
    generate_all_available,
    generate_coco_splits,
    generate_div2k_splits,
    generate_imagenette_splits,
    generate_synthetic_splits,
    list_file_hash,
    load_split_spec,
    resolve_role_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fixed dataset splits")
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="coco | div2k | imagenette | synthetic | all",
    )
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--splits-dir", type=str, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--coco-train-max", type=int, default=None)
    p.add_argument("--coco-val-size", type=int, default=COCO_VAL_SIZE)
    p.add_argument("--coco-test-size", type=int, default=COCO_TEST_SIZE)
    p.add_argument("--synthetic-n", type=int, default=64)
    p.add_argument(
        "--allow-small-test",
        action="store_true",
        help=(
            "Allow a COCO test split smaller than --coco-test-size (e.g. after a "
            "capped val2017 download). Records allow_small_test=true in meta.json."
        ),
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash existing split files and print hashes (no rewrite)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    name = args.dataset.lower()

    if args.verify:
        names = (
            ["coco", "div2k", "imagenette", "synthetic"]
            if name == "all"
            else [name]
        )
        for n in names:
            try:
                spec = load_split_spec(n, data_root=data_root, splits_dir=splits_dir)
            except FileNotFoundError as e:
                print(f"[verify] {n}: missing ({e})")
                continue
            print(f"[verify] {n}: seed={spec.split_seed} counts={spec.as_meta()['counts']}")
            for fn, h in spec.file_hashes.items():
                live = list_file_hash(spec.splits_dir / fn) if (spec.splits_dir / fn).exists() else ""
                ok = "ok" if live == h else "MISMATCH"
                print(f"  {fn}: {h[:16]}… [{ok}]")
        return

    try:
        if name == "all":
            specs = generate_all_available(
                data_root,
                splits_dir,
                split_seed=int(args.split_seed),
                coco_train_max=args.coco_train_max,
                coco_val_size=int(args.coco_val_size),
                coco_test_size=int(args.coco_test_size),
                synthetic_n=int(args.synthetic_n),
                allow_small_test=bool(args.allow_small_test),
            )
            # Surface skips so a fraction-capped COCO failure is not mistaken for success.
            expected = {"coco", "div2k", "imagenette", "synthetic"}
            missing = sorted(expected - set(specs))
            if missing:
                print(
                    f"\n[warn] skipped datasets (no usable images / split error): {missing}"
                )
                coco_train = resolve_role_root(data_root, "coco", "train")
                coco_val = resolve_role_root(data_root, "coco", "val")
                if "coco" in missing and (coco_train is not None or coco_val is not None):
                    if coco_train is None:
                        raise SystemExit(
                            "COCO val2017 is present but train2017 is missing or empty. "
                            "A prior download left a corrupt train2017.zip — re-run:\n"
                            "  python scripts/download_data.py --dataset coco --data-root ./data\n"
                            "then re-run make_splits."
                        )
                    if coco_val is None:
                        raise SystemExit(
                            "COCO train2017 is present but val2017 is missing or empty. "
                            "Re-run: python scripts/download_data.py --dataset coco --data-root ./data"
                        )
                    raise SystemExit(
                        "COCO images are present but splits were not generated. "
                        "Check train2017/val2017 under data/coco/, then re-run make_splits."
                    )
        elif name == "coco":
            specs = {
                "coco": generate_coco_splits(
                    data_root,
                    splits_dir,
                    split_seed=int(args.split_seed),
                    train_max=args.coco_train_max,
                    val_size=int(args.coco_val_size),
                    test_size=int(args.coco_test_size),
                    allow_small_test=bool(args.allow_small_test),
                )
            }
        elif name == "div2k":
            specs = {
                "div2k": generate_div2k_splits(
                    data_root, splits_dir, split_seed=int(args.split_seed)
                )
            }
        elif name == "imagenette":
            specs = {
                "imagenette": generate_imagenette_splits(
                    data_root, splits_dir, split_seed=int(args.split_seed)
                )
            }
        elif name == "synthetic":
            specs = {
                "synthetic": generate_synthetic_splits(
                    data_root,
                    splits_dir,
                    split_seed=int(args.split_seed),
                    num_samples=int(args.synthetic_n),
                )
            }
        else:
            raise SystemExit(f"Unknown dataset: {name}")
    except InsufficientTestSplitError as e:
        raise SystemExit(str(e)) from e

    print("\nFinal train/val/test counts:")
    for n, spec in specs.items():
        meta = spec.as_meta()
        c = meta["counts"]
        print(
            f"  {n}: train={c['train']} val={c['val']} test={c['test']}  "
            f"hashes={meta['split_hashes']}"
        )


if __name__ == "__main__":
    main()
