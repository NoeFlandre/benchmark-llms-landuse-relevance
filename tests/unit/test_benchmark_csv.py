from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.benchmark_csv import (
    BenchmarkFileError,
    load_benchmark,
)
from landuse_relevance_bench.domain.labels import Label


def test_loads_every_row_in_file_order(benchmark_path: Path) -> None:
    items = load_benchmark(benchmark_path)
    assert [i.sentence for i in items] == [
        "Dense mangrove forest lines the lagoon.",
        "The council was dissolved in 1974.",
    ]
    assert [i.label for i in items] == [Label.YES, Label.NO]
    assert [i.region for i in items] == ["fiji", "antarctica"]


def test_reports_the_offending_line_for_a_bad_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "sentence,label,source_item_id,language\n"
        "Trees.,yes,source-1,en\n"
        "Rocks.,perhaps,source-2,en\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkFileError, match="line 3"):
        load_benchmark(path)


def test_rejects_a_file_without_the_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("text,gold\nTrees.,yes\n", encoding="utf-8")
    with pytest.raises(BenchmarkFileError, match="column"):
        load_benchmark(path)


def test_rejects_a_benchmark_with_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("sentence,label,source_item_id,language\n", encoding="utf-8")
    with pytest.raises(BenchmarkFileError, match="no rows"):
        load_benchmark(path)


def test_rejects_duplicate_sentences_so_item_ids_stay_unique(tmp_path: Path) -> None:
    path = tmp_path / "dupes.csv"
    path.write_text(
        "sentence,label,source_item_id,language\nTrees.,yes,source-1,en\nTrees.,no,source-1,en\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkFileError, match="duplicate"):
        load_benchmark(path)


def test_rejects_the_archived_single_file_benchmark(legacy_benchmark_path: Path) -> None:
    with pytest.raises(BenchmarkFileError, match="column"):
        load_benchmark(legacy_benchmark_path)
