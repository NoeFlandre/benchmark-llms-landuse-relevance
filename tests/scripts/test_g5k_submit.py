from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SUBMIT = REPOSITORY / "scripts" / "g5k_submit.sh"


def _run_submit(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "LRB_JOB_TYPES": "exotic",
        "LRB_QUEUE": "default",
        "LRB_GPU_FILTER": "gpu_model = 'A100-SXM4-40GB'",
    }
    return subprocess.run(
        ["bash", str(SUBMIT), *arguments],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def test_submit_array_dry_run_assigns_array_size_to_shard_count() -> None:
    result = _run_submit("0:15", "--array", "4", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "--array 4" in result.stdout
    assert "walltime=0:15" in result.stdout
    assert "LRB_SHARD_COUNT=4" in result.stdout
    assert "LRB_SHARD_INDEX=" not in result.stdout


@pytest.mark.parametrize("count", ("0", "-2", "four"))
def test_submit_rejects_invalid_array_size(count: str) -> None:
    result = _run_submit("--array", count, "--dry-run")

    assert result.returncode == 2
    assert "positive integer" in result.stderr
