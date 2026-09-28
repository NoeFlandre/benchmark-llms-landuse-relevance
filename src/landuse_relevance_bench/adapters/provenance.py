"""Code and model provenance helpers shared by the CLI and pipeline."""

import logging
import os
import subprocess

SOURCE_COMMIT_ENV = "LRB_SOURCE_COMMIT"


def source_commit() -> str:
    """Read the pinned source commit or resolve the current Git checkout."""
    pinned = os.environ.get(SOURCE_COMMIT_ENV, "").strip()
    if pinned:
        return pinned
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        completed = None
    commit = completed.stdout.strip() if completed else ""
    if not commit:
        logging.getLogger("landuse_relevance_bench").warning(
            "could not determine the source commit; set %s to record it", SOURCE_COMMIT_ENV
        )
    return commit
