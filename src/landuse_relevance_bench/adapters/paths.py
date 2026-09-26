"""Environment-overridable paths used as the CLI's default inputs and outputs."""

import os
from collections.abc import Mapping
from pathlib import Path

DATA_DIR_ENV = "LRB_DATA_DIR"
RESULTS_DIR_ENV = "LRB_RESULTS_DIR"


def default_paths(environ: Mapping[str, str] | None = None) -> tuple[Path, Path, Path]:
    """Return benchmark CSV, prompt template, and results directory defaults."""
    environment = os.environ if environ is None else environ
    data_dir = Path(environment.get(DATA_DIR_ENV, "data"))
    results_dir = Path(environment.get(RESULTS_DIR_ENV, "results"))
    return data_dir / "benchmark.csv", data_dir / "prompt.txt", results_dir
