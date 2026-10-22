from pathlib import Path

import pytest

from landuse_relevance_bench import application


def _stored(tmp_path: Path) -> Path:
    path = tmp_path / "r.json"
    path.write_text("{}")
    return path


def test_checkpoint_state_pending_when_absent(tmp_path: Path) -> None:
    assert application._checkpoint_state(tmp_path / "x.json", "m", "en") == "pending"


def test_checkpoint_state_tolerates_oserror_and_valueerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _stored(tmp_path)

    def denied(_: Path) -> None:
        raise OSError("denied")

    monkeypatch.setattr(application, "read_run", denied)
    assert application._checkpoint_state(path, "m", "en") == "invalid"
    assert application.completed_pairs([("m", "en")], tmp_path) == set()

    def bad(_: Path) -> None:
        raise ValueError("bad")

    monkeypatch.setattr(application, "read_run", bad)
    assert application._checkpoint_state(path, "m", "en") == "invalid"


def test_checkpoint_state_complete_and_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _stored(tmp_path)
    meta = type(
        "M",
        (),
        {"name": "m", "language": "en", "experiment_identity": {"schema_version": 2}},
    )()
    monkeypatch.setattr(application, "read_run", lambda _: type("R", (), {"metadata": meta})())
    assert application._checkpoint_state(path, "m", "en") == "complete"
    assert application._checkpoint_state(path, "m", "fr") == "invalid"


def test_checkpoint_state_marks_legacy_metadata_as_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _stored(tmp_path)
    metadata_by_version = [
        type(
            "M",
            (),
            {"name": "m", "language": "en", "experiment_identity": None},
        )(),
        type(
            "M",
            (),
            {"name": "m", "language": "en", "experiment_identity": {}},
        )(),
        type(
            "M",
            (),
            {"name": "m", "language": "en", "experiment_identity": {"schema_version": 1}},
        )(),
        type(
            "M",
            (),
            {"name": "m", "language": "en", "experiment_identity": {"schema_version": True}},
        )(),
    ]

    for meta in metadata_by_version:
        monkeypatch.setattr(
            application,
            "read_run",
            lambda _, metadata=meta: type("R", (), {"metadata": metadata})(),
        )
        assert application._checkpoint_state(path, "m", "en") == "legacy"


def test_identity_mismatches_rejects_extra_null_setting() -> None:
    assert application._identity_mismatches(
        {"schema_version": 2, "removed_setting": None}, {"schema_version": 2}
    ) == ["removed_setting"]
