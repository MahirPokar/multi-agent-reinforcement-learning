"""Small file-output helpers for experiment runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


def make_run_dir(output_dir: str | None, experiment_name: str) -> Path:
    if output_dir:
        run_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"{timestamp}-{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class CsvMetricLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        return self

    def write(self, row: dict) -> None:
        if self.writer is None:
            raise RuntimeError("Metric logger is not open.")
        self.writer.writerow(row)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.file is not None:
            self.file.close()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
