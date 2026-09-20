from pathlib import Path
import re

from landuse_relevance_bench.adapters.hf_publish import read_published_runs
from landuse_relevance_bench.adapters.results_store import read_runs
from landuse_relevance_bench.adapters.translations import load_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_DOCS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "index.md",
    PROJECT_ROOT / "docs" / "benchmark.md",
    PROJECT_ROOT / "docs" / "results.md",
    PROJECT_ROOT / "docs" / "grid5000.md",
    PROJECT_ROOT / "docs" / "architecture.md",
    PROJECT_ROOT / "docs" / "debt.md",
    *sorted((PROJECT_ROOT / "docs" / "adr").glob("*.md")),
    PROJECT_ROOT / "results" / "README.md",
)


def test_active_data_is_the_multilingual_manifest_only() -> None:
    assert not (PROJECT_ROOT / "data" / "benchmark.csv").exists()
    manifest = load_manifest(PROJECT_ROOT / "data" / "translations")
    assert len(manifest.languages) == 85
    assert manifest.row_count == 300


def test_active_result_readers_ignore_every_archive_subtree(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True)
    (archive / "old.json").write_text("not active JSON", encoding="utf-8")

    assert read_runs(tmp_path) == []
    assert read_published_runs(tmp_path) == ()


def test_active_documentation_has_no_historical_benchmark_references() -> None:
    forbidden = (
        "data/benchmark.csv",
        "154 items",
        "154 prompts",
        "max_new_tokens=8",
        "sha256(sentence)",
    )
    for path in ACTIVE_DOCS:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert not any(needle.lower() in lowered for needle in forbidden), path
        assert re.search(r"\bv1\b", lowered) is None, path
