#!/usr/bin/env python3
"""Paper figure/table CLI: ``python scripts/viz.py --figure all --runs ... --out ...``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.viz import main

if __name__ == "__main__":
    main()
