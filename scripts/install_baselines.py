#!/usr/bin/env python3
"""Install official baseline packages and fetch weights for comparative eval.

StegaStamp / TrustMark / VINE are optional dependencies (see environment.yml).
Naive ``pip install git+...VINE`` pulls torch==2.0.1 and would break this project's
CUDA 12.4 stack — this script installs VINE with ``--no-deps`` into
``third_party/VINE`` and only adds the missing runtime packages (e.g. peft).

Usage:
  python scripts/install_baselines.py                 # all three
  python scripts/install_baselines.py --scheme vine
  python scripts/install_baselines.py --scheme trustmark --no-weights
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THIRD_PARTY = ROOT / "third_party"
VINE_REPO = "https://github.com/Shilin-LU/VINE.git"
VINE_DIR = THIRD_PARTY / "VINE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Install StegaStamp / TrustMark / VINE packages + weights"
    )
    p.add_argument(
        "--scheme",
        type=str,
        default="all",
        choices=["all", "stegastamp", "trustmark", "vine"],
    )
    p.add_argument("--weights-root", type=str, default="weights")
    p.add_argument(
        "--no-weights",
        action="store_true",
        help="Only install Python packages; skip weight downloads",
    )
    p.add_argument(
        "--pip",
        type=str,
        default=sys.executable,
        help="Python interpreter used for pip installs (default: current)",
    )
    return p.parse_args()


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def _pip(python: str, *args: str) -> None:
    _run([python, "-m", "pip", "install", *args])


def _module_ok(python: str, code: str) -> bool:
    r = subprocess.run(
        [python, "-c", code],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def install_stegastamp(python: str, weights_root: Path, *, download_weights: bool) -> None:
    print("\n=== install stegastamp ===")
    if not _module_ok(python, "import tensorflow"):
        # CPU wheel is enough for SavedModel encode/decode; GPU TF is optional.
        _pip(python, "tensorflow>=2.12")
    else:
        print("[stegastamp] tensorflow already importable")

    if not download_weights:
        print("[stegastamp] skipping weights (--no-weights)")
        return

    from src.baselines.stegastamp_wm import ensure_stegastamp_weights

    model_dir = ensure_stegastamp_weights(weights_root / "stegastamp")
    print(f"[stegastamp] model_dir={model_dir}")


def install_trustmark(python: str, *, download_weights: bool) -> None:
    print("\n=== install trustmark ===")
    if not _module_ok(python, "from trustmark import TrustMark"):
        try:
            _pip(python, "trustmark")
        except subprocess.CalledProcessError:
            print("[trustmark] PyPI install failed; trying Adobe GitHub python/ subdir")
            _pip(
                python,
                "git+https://github.com/adobe/trustmark.git#subdirectory=python",
            )
    else:
        print("[trustmark] package already importable")

    if not download_weights:
        print("[trustmark] skipping weight warm-up (--no-weights)")
        return

    # First TrustMark() call downloads encoder/decoder ckpts from Adobe's host.
    print("[trustmark] warming up official Q weights (Adobe CDN)…")
    code = (
        "from trustmark import TrustMark; "
        "TrustMark(verbose=True, model_type='Q', use_ECC=False, "
        "secret_len=100, device='cpu', loadRemover=False, loadBBoxDetector=False); "
        "print('trustmark weights ok')"
    )
    _run([python, "-c", code])


def _ensure_vine_checkout() -> Path:
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if (VINE_DIR / "setup.py").is_file():
        print(f"[vine] using existing checkout at {VINE_DIR}")
        return VINE_DIR
    if VINE_DIR.exists():
        raise RuntimeError(
            f"{VINE_DIR} exists but has no setup.py — remove it and re-run"
        )
    _run(["git", "clone", "--depth", "1", VINE_REPO, str(VINE_DIR)])
    return VINE_DIR


def install_vine(python: str, *, download_weights: bool) -> None:
    print("\n=== install vine ===")
    # Prefer the official package path (vine.src.*) after editable --no-deps install.
    if not _module_ok(python, "from vine.src.vine_turbo import VINE_Turbo"):
        vine_dir = _ensure_vine_checkout()
        # Critical: do NOT let VINE's setup.py pin torch==2.0.1 / xformers / etc.
        _pip(python, "--no-deps", "-e", str(vine_dir))
    else:
        print("[vine] vine.src.vine_turbo already importable")

    # Module-level import in vine_turbo.py requires peft even for no-LoRA inference.
    if not _module_ok(python, "import peft"):
        _pip(python, "peft")
    else:
        print("[vine] peft already importable")

    if not _module_ok(
        python,
        "from vine.src.vine_turbo import VINE_Turbo; "
        "from vine.src.stega_encoder_decoder import CustomConvNeXt",
    ):
        raise RuntimeError(
            "VINE installed but imports failed. Try:\n"
            f"  {python} -c \"from vine.src.vine_turbo import VINE_Turbo\""
        )
    print("[vine] package import ok")

    if not download_weights:
        print("[vine] skipping HF weight warm-up (--no-weights)")
        return

    print(
        "[vine] warming up Hugging Face weights "
        "(Shilin-LU/VINE-R-Enc + VINE-R-Dec; several GB)…"
    )
    code = (
        "from vine.src.vine_turbo import VINE_Turbo; "
        "from vine.src.stega_encoder_decoder import CustomConvNeXt; "
        "enc = VINE_Turbo.from_pretrained('Shilin-LU/VINE-R-Enc', device='cpu'); "
        "dec = CustomConvNeXt.from_pretrained('Shilin-LU/VINE-R-Dec'); "
        "print('vine weights ok', type(enc).__name__, type(dec).__name__)"
    )
    _run([python, "-c", code])


def main() -> int:
    args = parse_args()
    names = (
        ["stegastamp", "trustmark", "vine"]
        if args.scheme == "all"
        else [args.scheme]
    )
    weights_root = Path(args.weights_root)
    if not weights_root.is_absolute():
        weights_root = ROOT / weights_root

    failures: list[str] = []
    for name in names:
        try:
            if name == "stegastamp":
                install_stegastamp(
                    args.pip, weights_root, download_weights=not args.no_weights
                )
            elif name == "trustmark":
                install_trustmark(args.pip, download_weights=not args.no_weights)
            elif name == "vine":
                install_vine(args.pip, download_weights=not args.no_weights)
        except Exception as e:
            failures.append(f"{name}: {e}")
            print(f"[{name}] FAILED: {e}", file=sys.stderr)

    if failures:
        print("\nSome baseline installs failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        "\nDone. Verify with:\n"
        "  python scripts/download_baselines.py\n"
        "  # or: python tests/test_baselines_sanity.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
