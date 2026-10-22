import logging
import sys
from types import SimpleNamespace

import pytest

from landuse_relevance_bench.adapters import revision

COMMIT = "a" * 40


def test_an_immutable_candidate_wins_without_asking_the_hub(monkeypatch) -> None:
    monkeypatch.setattr(revision, "hub_commit", pytest.fail)
    assert revision.resolve_revision("m/x", None, "", COMMIT) == COMMIT


def test_a_mutable_candidate_is_resolved_to_its_commit(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def hub_commit(model_id: str, requested: str | None = None) -> str:
        calls.append((model_id, requested))
        return COMMIT

    monkeypatch.setattr(revision, "hub_commit", hub_commit)
    assert revision.resolve_revision("m/x", "main") == COMMIT
    assert calls == [("m/x", "main")]


def test_an_unpinned_revision_resolves_the_current_commit(monkeypatch) -> None:
    monkeypatch.setattr(revision, "hub_commit", lambda model_id, requested=None: COMMIT)
    assert revision.resolve_revision("m/x", None, "") == COMMIT


def test_an_unresolvable_revision_is_empty_and_warned(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert revision.resolve_revision("m/x", None) == ""
    assert "immutable weights revision" in caplog.text


class FakeApi:
    def __init__(self, sha: str | None = None, error: Exception | None = None) -> None:
        self._sha, self._error = sha, error

    def __call__(self) -> "FakeApi":
        return self

    def model_info(self, model_id: str, *, revision: str | None = None) -> SimpleNamespace:
        assert model_id == "m/x"
        if self._error is not None:
            raise self._error
        return SimpleNamespace(sha=self._sha, requested_revision=revision)


@pytest.fixture
def real_hub_commit(monkeypatch):
    monkeypatch.undo()
    return revision.hub_commit


def test_hub_commit_reads_the_model_info_sha(monkeypatch, real_hub_commit) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi(COMMIT)))
    assert real_hub_commit("m/x") == COMMIT


def test_hub_commit_resolves_the_requested_ref(monkeypatch, real_hub_commit) -> None:
    calls: list[str | None] = []

    class Api(FakeApi):
        def model_info(self, model_id: str, *, revision: str | None = None) -> SimpleNamespace:
            calls.append(revision)
            return super().model_info(model_id, revision=revision)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api(COMMIT)))
    assert real_hub_commit("m/x", "release-tag") == COMMIT
    assert calls == ["release-tag"]


def test_hub_commit_is_empty_when_the_hub_is_unreachable(monkeypatch, real_hub_commit) -> None:
    api = FakeApi(error=OSError("offline"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=api))
    assert real_hub_commit("m/x") == ""
