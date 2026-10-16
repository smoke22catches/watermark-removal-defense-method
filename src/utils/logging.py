"""Run-directory creation, metric logging, and config snapshots."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

import yaml

from src.utils.metrics import LONG_FIELDS, dump_json


def create_run_dir(
    runs_root: str | Path,
    run_name: str = "default",
    seed: Optional[int] = None,
) -> Path:
    """Create ``runs/<YYYYmmdd-HHMMSS>_<run-name>_s<seed>/`` with standard subfolders."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if seed is not None:
        run_name = f"{run_name}_s{int(seed)}"
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name) or "run"
    run_dir = runs_root / f"{stamp}_{safe_name}"
    # Never overwrite: if collision (same second), append a counter.
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{stamp}_{safe_name}_{suffix}"
        suffix += 1
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "plots").mkdir(parents=True)
    return run_dir


def save_config(config: Mapping[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False, default_flow_style=False)


class _Tee:
    """Write to multiple text streams (console + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    """Dedup key: (global_step, metric_name, attack, epoch)."""
    return (
        str(row.get("global_step", "")),
        str(row.get("metric_name", "")),
        str(row.get("attack", "")),
        str(row.get("epoch", "")),
    )


class RunLogger:
    """Per-run logger: config snapshot, long-format CSV metrics, tee'd console log."""

    def __init__(
        self,
        run_dir: Path,
        config: Optional[Mapping[str, Any]] = None,
        metrics_filename: str = "metrics.csv",
        long_format: bool = True,
        append_metrics: bool = True,
        metrics_flush_every: int = 50,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / metrics_filename
        self.log_path = self.run_dir / "log.txt"
        self.long_format = long_format
        self.metrics_flush_every = max(1, int(metrics_flush_every))
        self._fieldnames: Optional[List[str]] = list(LONG_FIELDS) if long_format else None
        self._seen_keys: set = set()
        self._pending_rows: List[Mapping[str, Any]] = []
        if append_metrics and self.metrics_path.exists():
            self._load_existing_keys()
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._log_file)  # type: ignore[assignment]
        if config is not None and not (self.run_dir / "config.yaml").exists():
            save_config(config, self.run_dir / "config.yaml")
        elif config is not None and not append_metrics:
            save_config(config, self.run_dir / "config.yaml")

    def _load_existing_keys(self) -> None:
        try:
            with open(self.metrics_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    self._fieldnames = list(reader.fieldnames)
                for row in reader:
                    self._seen_keys.add(_row_key(row))
        except OSError:
            pass

    def log(self, message: str) -> None:
        print(message)

    def log_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        path = self.run_dir / filename
        dump_json(path, payload)
        return path

    def log_metrics(self, row: Dict[str, Any]) -> None:
        """Append a metrics row. Long-format rows must include ``metric_name``."""
        self.log_metric_rows([row])

    def log_metric_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        flush: bool = False,
    ) -> None:
        if not rows:
            return
        for row in rows:
            key = _row_key(row)
            # Dedup when global_step is present; always allow blank-step rows once keyed.
            if key in self._seen_keys and str(row.get("global_step", "")) != "":
                continue
            self._seen_keys.add(key)
            self._pending_rows.append(row)
        if flush or len(self._pending_rows) >= self.metrics_flush_every:
            self.flush_metrics()

    def flush_metrics(self) -> None:
        """Write any buffered metric rows to disk."""
        if not self._pending_rows:
            return
        new_rows = self._pending_rows
        self._pending_rows = []
        write_header = not self.metrics_path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(new_rows[0].keys())
        for row in new_rows:
            for k in row:
                if k not in self._fieldnames:
                    self._fieldnames.append(k)
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in new_rows:
                writer.writerow(dict(row))

    def close(self) -> None:
        self.flush_metrics()
        sys.stdout = self._stdout
        self._log_file.close()
