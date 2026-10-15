#!/usr/bin/env python3
"""Download / verify official baseline watermark weights.

Packages must be installed first (TrustMark / VINE / TensorFlow). Prefer:

  python scripts/install_baselines.py

which installs the official repos without clobbering this project's torch pin,
then fetches weights. Pass ``--install`` here to run that step automatically.
"""

from __future__ import annotations

import argparse
import subprocess
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
    p.add_argument(
        "--install",
        action="store_true",
        help="Run scripts/install_baselines.py for the selected scheme first",
    )
    return p.parse_args()


def _maybe_install(scheme: str, weights_root: str) -> None:
    script = ROOT / "scripts" / "install_baselines.py"
    cmd = [
        sys.executable,
        str(script),
        "--scheme",
        scheme,
        "--weights-root",
        weights_root,
    ]
    print(f"[download_baselines] ensuring packages via {script.name} …")
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    names = (
        ["stegastamp", "trustmark", "vine"]
        if args.scheme == "all"
        else [args.scheme]
    )
    if args.install:
        _maybe_install(args.scheme, args.weights_root)

    for name in names:
        print(f"\n=== {name} ===")
        if name == "stegastamp":
            from src.baselines.common import file_sha256
            from src.baselines.stegastamp_wm import (
                STEGA_HF_REPO,
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
                print(f"hf_mirror={STEGA_HF_REPO}")
                print(f"archive_sha256={digest or '(n/a — HF mirror folder download)'}")
                print("Record this sha256 in docs/baselines.md / results_meta.json.")
            except Exception as e:
                print(f"[stegastamp] failed: {e}")
                print(
                    "Hint: python scripts/install_baselines.py --scheme stegastamp"
                )
            continue

        from src.baselines.factory import load_baseline

        res = load_baseline(name, device=args.device, config={})
        if not res.available or res.instance is None:
            print(f"[{name}] not available: {res.reason}")
            print(f"Hint: python scripts/install_baselines.py --scheme {name}")
            continue
        prov = res.instance.provenance().as_dict()
        print(f"loaded ok: {prov}")
        res.instance.close()


if __name__ == "__main__":
    main()
