import pytest

from landuse_relevance_bench.adapters import revision


@pytest.fixture(autouse=True)
def _offline_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never reach the Hub; revision lookups there resolve to nothing."""
    monkeypatch.setattr(revision, "hub_commit", lambda _model_id: "")
