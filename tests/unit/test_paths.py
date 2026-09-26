from pathlib import Path

from landuse_relevance_bench.adapters.paths import default_paths


def test_default_paths_can_be_relocated_with_environment_settings() -> None:
    benchmark, prompt, results = default_paths(
        {"LRB_DATA_DIR": "/data/bench", "LRB_RESULTS_DIR": "/data/runs"}
    )
    assert benchmark == Path("/data/bench/benchmark.csv")
    assert prompt == Path("/data/bench/prompt.txt")
    assert results == Path("/data/runs")


def test_default_paths_keep_repo_relative_defaults() -> None:
    assert default_paths({}) == (
        Path("data/benchmark.csv"),
        Path("data/prompt.txt"),
        Path("results"),
    )
