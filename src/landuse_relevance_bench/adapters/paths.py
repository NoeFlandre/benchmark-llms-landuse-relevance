"""Environment-overridable paths used as the CLI's default inputs and outputs."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV = "LRB_DATA_DIR"
RESULTS_DIR_ENV = "LRB_RESULTS_DIR"


@dataclass(frozen=True)
class DefaultPaths:
    """The CLI's default data root, prompt templates and results directory."""

    data_root: Path
    prompt: Path
    scorer_prompt: Path
    results: Path


def default_paths(environ: Mapping[str, str] | None = None) -> DefaultPaths:
    """Resolve defaults under ``LRB_DATA_DIR`` and ``LRB_RESULTS_DIR`` (repo-relative otherwise)."""
    environment = os.environ if environ is None else environ
    data_dir = Path(environment.get(DATA_DIR_ENV, "data"))
    return DefaultPaths(
        data_root=data_dir / "translations",
        prompt=data_dir / "prompt.txt",
        scorer_prompt=data_dir / "prompt_reranker.txt",
        results=Path(environment.get(RESULTS_DIR_ENV, "results")),
    )
