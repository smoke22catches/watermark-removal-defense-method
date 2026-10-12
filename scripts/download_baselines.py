#!/usr/bin/env python3
"""Download / verify official baseline watermark weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download official baseline weights")
    p.add_argument(
        "--scheme",
        type=str,
        default="all",
        choices=["all", "stegastamp", "trustmark", "vine"],
    )
    p.add_argument("--weights-root", type=str, default="weights")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        ["stegastamp", "trustmark", "vine"]
        if args.scheme == "all"
        else [args.scheme]
    )
    for name in names:
        print(f"\n=== {name} ===")
        if name == "stegastamp":
            from src.baselines.common import file_sha256
            from src.baselines.stegastamp_wm import (
                STEGA_WEIGHTS_URL,
                ensure_stegastamp_weights,
            )

            root = Path(args.weights_root) / "stegastamp"
            try:
                model_dir = ensure_stegastamp_weights(root)
                archive = root / "saved_models.tar.xz"
                digest = file_sha256(archive) if archive.is_file() else ""
                print(f"model_dir={model_dir}")
                print(f"archive_url={STEGA_WEIGHTS_URL}")
                print(f"archive_sha256={digest}")
                print("Record this sha256 in docs/baselines.md / results_meta.json.")
            except Exception as e:
                print(f"[stegastamp] failed: {e}")
            continue

        from src.baselines.factory import load_baseline

        res = load_baseline(name, device=args.device, config={})
        if not res.available or res.instance is None:
            print(f"[{name}] not available: {res.reason}")
            continue
        prov = res.instance.provenance().as_dict()
        print(f"loaded ok: {prov}")
        res.instance.close()


if __name__ == "__main__":
    main()
