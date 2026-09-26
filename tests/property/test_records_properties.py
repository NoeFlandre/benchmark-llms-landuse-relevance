"""Record round trips preserve Unicode names and nullable model revisions."""

from pathlib import Path

from hypothesis import given
from strategies import run_results

from landuse_relevance_bench.adapters.results_store import read_run, write_run
from landuse_relevance_bench.domain.records import RunMetadata, RunResult


@given(result=run_results())
def test_run_metadata_and_result_round_trip_in_memory(result: RunResult) -> None:
    assert RunMetadata.from_dict(result.metadata.to_dict()) == result.metadata
    assert RunResult.from_dict(result.to_dict()) == result
    assert result.metadata.model_id.startswith("模型/")
    assert result.metadata.model_revision is None


@given(result=run_results())
def test_a_run_round_trips_through_the_results_store(tmp_path: Path, result: RunResult) -> None:
    path = write_run(result, tmp_path)
    assert read_run(path) == result
