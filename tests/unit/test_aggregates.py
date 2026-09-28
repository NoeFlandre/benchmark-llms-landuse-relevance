from pathlib import Path

from landuse_relevance_bench.adapters.results_store import (
    aggregate_rows,
    write_aggregates_csv,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult


def _result(language: str, outcomes: tuple[tuple[Label, Label | None], ...]) -> RunResult:
    metadata = RunMetadata(
        model_id="model/a",
        language=language,
        model_revision="rev",
        prompt_sha256="p" * 64,
        benchmark_sha256="b" * 64,
        max_new_tokens=8,
        batch_size=16,
        seed=0,
        decoding="greedy",
        dtype="bfloat16",
        started_at="2026-09-13T10:00:00Z",
        duration_seconds=1.0,
    )
    predictions = tuple(
        Prediction(
            item_id=f"{index:016x}",
            expected=expected,
            predicted=predicted,
            raw_output="yes" if predicted is Label.YES else "no",
        )
        for index, (expected, predicted) in enumerate(outcomes)
    )
    return RunResult(metadata=metadata, predictions=predictions, metrics=evaluate(outcomes))


def test_aggregate_rows_are_macro_metrics_with_f1_spread() -> None:
    perfect = _result("en", ((Label.YES, Label.YES), (Label.NO, Label.NO)))
    weak = _result("fr", ((Label.YES, Label.NO), (Label.NO, Label.NO)))

    (row,) = aggregate_rows([weak, perfect])

    assert row["model_id"] == "model/a"
    assert row["language_count"] == 2
    assert row["accuracy_macro"] == 0.75
    assert row["f1_macro"] == 0.5
    assert row["f1_min"] == 0.0
    assert row["f1_max"] == 1.0
    assert row["f1_std"] == 0.5


def test_aggregate_csv_has_a_stable_header(tmp_path: Path) -> None:
    result = _result("en", ((Label.YES, Label.YES),))

    path = write_aggregates_csv([result], tmp_path / "aggregates.csv")

    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("model_id,")
