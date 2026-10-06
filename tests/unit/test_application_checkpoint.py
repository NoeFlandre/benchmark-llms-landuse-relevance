from pathlib import Path

import pytest

from landuse_relevance_bench import application
from landuse_relevance_bench.adapters.results_store import run_filename


def test_checkpoint_state_pending_when_absent(tmp_path: Path) -> None:
    assert application._checkpoint_state(tmp_path / "x.json", "m", "en") == "pending"


def test_checkpoint_state_tolerates_oserror_and_valueerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / run_filename("m", "en")
    path.write_text("{}")

    def boom(_: Path) -> None:
        raise OSError("denied")

    monkeypatch.setattr(application, "read_run", boom)
    assert application._checkpoint_state(path, "m", "en") == "invalid"
    assert application.completed_pairs([("m", "en")], tmp_path) == set()

    def bad(_: Path) -> None:
        raise ValueError("bad")

    monkeypatch.setattr(application, "read_run", bad)
    assert application._checkpoint_state(path, "m", "en") == "invalid"


def test_checkpoint_state_complete_and_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "r.json"
    path.write_text("{}")
    meta = type("M", (), {"name": "m", "language": "en"})()
    monkeypatch.setattr(application, "read_run", lambda _: type("R", (), {"metadata": meta})())
    assert application._checkpoint_state(path, "m", "en") == "complete"
    assert application._checkpoint_state(path, "m", "fr") == "invalid"
