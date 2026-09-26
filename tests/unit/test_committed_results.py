from pathlib import Path

from landuse_relevance_bench.adapters.hf_publish import dataset_card
from landuse_relevance_bench.adapters.results_store import read_runs, write_leaderboard_csv
from landuse_relevance_bench.domain.roster import model_ids

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def test_committed_current_runs_are_in_the_roster() -> None:
    current = read_runs(RESULTS)
    assert current
    assert {result.metadata.name for result in current} <= set(model_ids())


def test_historical_result_groups_remain_documented_and_valid() -> None:
    all_results = read_runs(RESULTS, recursive=True)
    assert len(all_results) >= len(read_runs(RESULTS))
    assert all(result.metrics.n_items == len(result.predictions) for result in all_results)
    docs = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
    assert "more-models-20260913" in docs
    assert "qwen3-20260913" in docs
    assert "recent-models-20260913" in docs


def test_committed_leaderboard_and_card_are_generated_from_saved_runs(tmp_path: Path) -> None:
    all_results = read_runs(RESULTS, recursive=True)
    generated_board = write_leaderboard_csv(all_results, tmp_path / "leaderboard.csv")
    assert generated_board.read_bytes() == (RESULTS / "leaderboard.csv").read_bytes()
    generated_card = dataset_card(all_results, benchmark_name="benchmark.csv")
    assert generated_card == (RESULTS / "README.md").read_text(encoding="utf-8")
