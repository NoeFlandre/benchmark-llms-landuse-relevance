from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import g5k_status_multisite as status


def test_multisite_status_reports_complete_running_queued_and_unclaimed_pairs(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "sites.json"
    config.write_text(
        json.dumps({"sites": [{"name": "nancy", "frontend": "nancy", "weight": 3}]}),
        encoding="utf-8",
    )
    jobs = {
        "100": {
            "id": 100,
            "state": "Running",
            "command": "LRB_RESULTS=/remote/results LRB_SHARD_INDEX=0 "
            "LRB_SHARD_COUNT=3 /remote/scripts/g5k_node_run.sh",
        },
        "101": {
            "id": 101,
            "state": "Waiting",
            "command": "LRB_RESULTS=/remote/results LRB_SHARD_INDEX=1 "
            "LRB_SHARD_COUNT=3 /remote/scripts/g5k_node_run.sh",
        },
        "102": {
            "id": 102,
            "state": "Running",
            "command": "LRB_RESULTS=/other/project LRB_SHARD_INDEX=1 "
            "LRB_SHARD_COUNT=3 /remote/scripts/g5k_node_run.sh",
        },
    }

    def fake_run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        remote_command = command[-1]
        if remote_command == "oarstat -u -J":
            stdout = json.dumps(jobs)
        elif "--shard-index 0" in remote_command:
            stdout = "model-a\tde\tcomplete\nmodel-b\tde\tpending\n"
        elif "--shard-index 1" in remote_command:
            stdout = "model-a\ten\tpending\nmodel-b\ten\tpending\n"
        elif "--shard-index 2" in remote_command:
            stdout = "model-a\tfr\tpending\nmodel-b\tfr\tpending\n"
        else:
            raise AssertionError(remote_command)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(status.subprocess, "run", fake_run)
    rows = status.multisite_status(
        config,
        remote_root="/remote/lrb",
        remote_results="/remote/results",
        models=("model-a", "model-b"),
        languages=("de", "en", "fr"),
    )

    assert {(row.model_id, row.language, row.state) for row in rows} == {
        ("model-a", "de", "complete"),
        ("model-a", "en", "queued"),
        ("model-a", "fr", "unclaimed"),
        ("model-b", "de", "running"),
        ("model-b", "en", "queued"),
        ("model-b", "fr", "unclaimed"),
    }
    assert {row.job_id for row in rows if row.state == "running"} == {"100"}


def test_multisite_status_refuses_incomplete_remote_status_coverage(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "sites.json"
    config.write_text(
        json.dumps({"sites": [{"name": "nancy", "frontend": "nancy", "weight": 1}]}),
        encoding="utf-8",
    )

    def fake_run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        remote_command = command[-1]
        stdout = "{}" if remote_command == "oarstat -u -J" else "model-a\ten\tcomplete\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(status.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="coverage"):
        status.multisite_status(
            config,
            remote_root="/remote/lrb",
            remote_results="/remote/results",
            models=("model-a", "model-b"),
            languages=("en",),
        )
