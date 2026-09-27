import logging
import sys
from types import SimpleNamespace

import pytest

from landuse_relevance_bench.adapters import revision


def test_a_known_candidate_wins_without_asking_the_hub(monkeypatch) -> None:
    monkeypatch.setattr(revision, "hub_commit", pytest.fail)
    assert revision.resolve_revision("m/x", None, "", "abc") == "abc"


def test_the_hub_commit_fills_in_an_unknown_revision(monkeypatch) -> None:
    monkeypatch.setattr(revision, "hub_commit", lambda model_id: f"sha-of-{model_id}")
    assert revision.resolve_revision("m/x", None, "") == "sha-of-m/x"


def test_an_unresolvable_revision_is_empty_and_warned(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert revision.resolve_revision("m/x", None) == ""
    assert "pass --revision" in caplog.text


class FakeApi:
    def __init__(self, sha: str | None = None, error: Exception | None = None) -> None:
        self._sha, self._error = sha, error

    def __call__(self) -> "FakeApi":
        return self

    def model_info(self, model_id: str) -> SimpleNamespace:
        assert model_id == "m/x"
        if self._error is not None:
            raise self._error
        return SimpleNamespace(sha=self._sha)


@pytest.fixture
def real_hub_commit(monkeypatch):
    monkeypatch.undo()
    return revision.hub_commit


def test_hub_commit_reads_the_model_info_sha(monkeypatch, real_hub_commit) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi("c0ffee")))
    assert real_hub_commit("m/x") == "c0ffee"


def test_hub_commit_is_empty_when_the_hub_is_unreachable(monkeypatch, real_hub_commit) -> None:
    api = FakeApi(error=OSError("offline"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=api))
    assert real_hub_commit("m/x") == ""
