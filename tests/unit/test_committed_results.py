from pathlib import Path

from landuse_relevance_bench.adapters.hf_publish import dataset_card
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import read_runs, write_leaderboard_csv

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def test_committed_archive_is_not_treated_as_active_runs() -> None:
    assert read_runs(RESULTS) == []


def test_historical_result_groups_remain_documented_and_valid() -> None:
    all_results = read_runs(RESULTS, recursive=True)
    assert len(all_results) >= len(read_runs(RESULTS))
    assert all(result.metrics.n_items == len(result.predictions) for result in all_results)
    docs = (RESULTS / "archive" / "historical-single-language" / "PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    assert "more-models-20260913" in docs
    assert "qwen3-20260913" in docs
    assert "recent-models-20260913" in docs


def test_committed_leaderboard_and_card_are_generated_from_saved_runs(tmp_path: Path) -> None:
    all_results = read_runs(RESULTS, recursive=True)
    generated_board = write_leaderboard_csv(all_results, tmp_path / "leaderboard.csv")
    first_bytes = generated_board.read_bytes()
    write_leaderboard_csv(all_results, tmp_path / "leaderboard-repeat.csv")
    assert (tmp_path / "leaderboard-repeat.csv").read_bytes() == first_bytes
    assert len(first_bytes.splitlines()) == len(all_results) + 1
    generated_card = dataset_card(
        all_results,
        benchmark_name="benchmark.csv",
        prompt_text=load_prompt(ROOT / "data" / "prompt.txt"),
    )
    assert all(result.metadata.model_id in generated_card for result in all_results)
