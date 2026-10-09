from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import g5k_status_multisite as status


@pytest.fixture
def fake_ssh(monkeypatch, tmp_path: Path) -> Path:
    """Provide an ssh client on PATH so the status code can resolve one.

    The status module resolves ``ssh`` once at import time, so the module attribute is
    repointed as well. Every remote call is still answered by the test's ``fake_run``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]))
    monkeypatch.setattr(status, "SSH_EXECUTABLE", str(ssh))
    return ssh


def test_multisite_status_reports_complete_running_queued_and_unclaimed_pairs(
    monkeypatch, tmp_path: Path, fake_ssh: Path
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
    monkeypatch, tmp_path: Path, fake_ssh: Path
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


def _write_sites(tmp_path: Path, *sites: dict[str, object]) -> Path:
    config = tmp_path / "sites.json"
    config.write_text(json.dumps({"sites": list(sites)}), encoding="utf-8")
    return config


def _remote(
    *,
    jobs: object,
    status_lines: dict[int, str],
) -> object:
    """A stand-in for ssh: answers the scheduler query and each shard's status command."""

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        remote_command = command[-1]
        if remote_command == "oarstat -u -J":
            stdout = json.dumps(jobs)
        else:
            shard = int(remote_command.split("--shard-index ")[1].split()[0])
            stdout = status_lines[shard]
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake_run


def test_remote_command_failure_names_the_frontend_and_its_stderr(
    monkeypatch, tmp_path: Path, fake_ssh: Path
) -> None:
    config = _write_sites(tmp_path, {"name": "nancy", "frontend": "nancy", "weight": 1})

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 255, "", "Permission denied\n")

    monkeypatch.setattr(status.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=r"ssh nancy failed \(255\): Permission denied"):
        status.multisite_status(
            config,
            remote_root="/remote/lrb",
            remote_results="/remote/results",
            models=("model-a",),
            languages=("en",),
        )


def test_scheduler_jobs_are_classified_by_their_most_urgent_state(
    monkeypatch, tmp_path: Path, fake_ssh: Path
) -> None:
    config = _write_sites(tmp_path, {"name": "nancy", "frontend": "nancy", "weight": 2})

    def job(job_id: int, state: str, shard: int, *, results: str = "/remote/results") -> dict:
        return {
            "id": job_id,
            "state": state,
            "command": f"LRB_RESULTS={results} LRB_SHARD_INDEX={shard} "
            "LRB_SHARD_COUNT=2 /remote/scripts/g5k_node_run.sh",
        }

    jobs = [
        "not a job record",
        {"id": 300, "state": "Running", "command": "LRB_SHARD_INDEX=0"},
        {"id": 301, "state": "Running", "command": "LRB_RESULTS=/remote/results LRB_SHARD_INDEX=x"},
        job(302, "Terminated", 0),
        job(303, "Waiting", 0),
        job(304, "Error", 0),
        job(305, "Waiting", 1),
        job(306, "Running", 1),
        job(307, "Error", 1, results="/other/results"),
    ]
    monkeypatch.setattr(
        status.subprocess,
        "run",
        _remote(
            jobs=jobs,
            status_lines={
                0: "model-a\ten\tpending\nmodel-b\ten\tpending\n",
                1: "model-a\tfr\tpending\nmodel-b\tfr\tpending\n",
            },
        ),
    )

    rows = status.multisite_status(
        config,
        remote_root="/remote/lrb",
        remote_results="/remote/results",
        models=("model-a", "model-b"),
        languages=("en", "fr"),
    )

    by_pair = {(row.model_id, row.language): (row.state, row.job_id) for row in rows}
    assert by_pair[("model-a", "en")] == ("failed", "304")
    assert by_pair[("model-b", "en")] == ("failed", "304")
    assert by_pair[("model-a", "fr")] == ("running", "306")
    assert by_pair[("model-b", "fr")] == ("running", "306")


@pytest.mark.parametrize(
    ("status_text", "message"),
    [
        ("model-a\ten\n", "unexpected lrb status row"),
        ("model-a\ten\tcomplete\textra\n", "unexpected lrb status row"),
        ("model-a\ten\tbogus\n", "unexpected lrb status state"),
        ("model-a\ten\tcomplete\nmodel-a\ten\tpending\n", "does not match its plan"),
    ],
)
def test_malformed_remote_status_rows_are_refused(
    monkeypatch, tmp_path: Path, fake_ssh: Path, status_text: str, message: str
) -> None:
    config = _write_sites(tmp_path, {"name": "nancy", "frontend": "nancy", "weight": 1})

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = "{}" if command[-1] == "oarstat -u -J" else status_text
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(status.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match=message):
        status.multisite_status(
            config,
            remote_root="/remote/lrb",
            remote_results="/remote/results",
            models=("model-a",),
            languages=("en",),
        )


def _status_argv(config: Path, *extra: str) -> list[str]:
    return [
        "g5k_status_multisite.py",
        "--sites-config",
        str(config),
        "--remote-root",
        "/remote/lrb/",
        *extra,
    ]


def test_status_command_prints_every_pair_and_a_state_summary(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_sites(tmp_path, {"name": "nancy", "frontend": "nancy", "weight": 1})
    calls: list[dict[str, object]] = []

    def fake_multisite_status(sites_config: Path, **kwargs: object) -> list[status.PairStatus]:
        calls.append({"sites_config": sites_config, **kwargs})
        return [
            status.PairStatus("nancy", "model-a", "en", "complete", "11"),
            status.PairStatus("nancy", "model-a", "fr", "running", "12"),
            status.PairStatus("nancy", "model-b", "en", "unclaimed"),
        ]

    monkeypatch.setattr(status, "multisite_status", fake_multisite_status)
    monkeypatch.setattr(sys, "argv", _status_argv(config))

    status.main()
    output = capsys.readouterr().out

    assert calls == [
        {
            "sites_config": config,
            "remote_root": "/remote/lrb/",
            "remote_results": "/remote/lrb/results-live",
        }
    ]
    assert output.splitlines()[0] == "site\tmodel\tlanguage\tstate\tjob_id"
    assert "nancy\tmodel-a\ten\tcomplete\t11" in output.splitlines()
    assert "nancy\tmodel-b\ten\tunclaimed\t-" in output.splitlines()
    assert output.splitlines()[-1] == "complete=1, running=1, queued=0, failed=0, unclaimed=1"


def test_status_command_keeps_an_absolute_results_directory(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_sites(tmp_path, {"name": "nancy", "frontend": "nancy", "weight": 1})
    seen: dict[str, object] = {}

    def fake_multisite_status(_sites_config: Path, **kwargs: object) -> list[status.PairStatus]:
        seen.update(kwargs)
        return []

    monkeypatch.setattr(status, "multisite_status", fake_multisite_status)
    monkeypatch.setattr(sys, "argv", _status_argv(config, "--results-dir", "/scratch/results"))

    status.main()

    assert seen["remote_results"] == "/scratch/results"
    assert capsys.readouterr().out.splitlines()[-1] == (
        "complete=0, running=0, queued=0, failed=0, unclaimed=0"
    )


@pytest.mark.parametrize(
    "error",
    [RuntimeError("ssh nancy failed (255): no route"), ValueError("bad coverage")],
)
def test_status_command_reports_refusals_as_usage_errors(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    config = _write_sites(tmp_path, {"name": "nancy", "frontend": "nancy", "weight": 1})

    def refusing(_sites_config: Path, **_kwargs: object) -> list[status.PairStatus]:
        raise error

    monkeypatch.setattr(status, "multisite_status", refusing)
    monkeypatch.setattr(sys, "argv", _status_argv(config))

    with pytest.raises(SystemExit) as exit_info:
        status.main()

    assert exit_info.value.code == 2
    assert str(error) in capsys.readouterr().err
