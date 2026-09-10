"""CLI helpers for loading YAML configs and applying overrides."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for k, v in overrides.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), MutableMapping):
            out[k] = deep_update(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_nested(cfg: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c`` on a nested dict."""
    parts = dotted_key.split(".")
    cur: MutableMapping[str, Any] = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]  # type: ignore[assignment]
    cur[parts[-1]] = value


def _parse_literal(raw: str) -> Any:
    """Parse CLI string into bool/int/float/None/list/str."""
    low = raw.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none"):
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_literal(x.strip()) for x in inner.split(",")]
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_set_overrides(cfg: MutableMapping[str, Any], items: Optional[Sequence[str]]) -> None:
    """Apply ``--set key=value`` overrides (supports dotted keys)."""
    if not items:
        return
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        set_nested(cfg, key.strip(), _parse_literal(raw.strip()))


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--msg-len", type=int, default=None)
    parser.add_argument("--encoder-strength", type=float, default=None, dest="encoder_strength")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lambda-perc", type=float, default=None)
    parser.add_argument("--regen-steps", type=int, default=None, dest="regen_steps")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Dotted override, e.g. --set regen.use_placeholder=true",
    )


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Load default YAML and apply known CLI flags + --set overrides."""
    cfg = load_yaml(args.config)
    mapping = {
        "seed": "seed",
        "device": "device",
        "run_name": "run_name",
        "data_root": "data_root",
        "dataset": "dataset",
        "image_size": "image_size",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "num_samples": "num_samples",
        "msg_len": "msg_len",
        "encoder_strength": "encoder_strength",
        "lr": "lr",
        "epochs": "epochs",
        "lambda_perc": "lambda_perc",
        "resume": "resume",
    }
    for attr, key in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            cfg[key] = val
    if getattr(args, "regen_steps", None) is not None:
        cfg.setdefault("regen", {})
        cfg["regen"]["n_steps"] = args.regen_steps
    apply_set_overrides(cfg, getattr(args, "set_overrides", None))
    return cfg
