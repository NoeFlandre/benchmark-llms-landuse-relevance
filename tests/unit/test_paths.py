from pathlib import Path

from landuse_relevance_bench.adapters.paths import DefaultPaths, default_paths


def test_default_paths_can_be_relocated_with_environment_settings() -> None:
    paths = default_paths({"LRB_DATA_DIR": "/data/bench", "LRB_RESULTS_DIR": "/data/runs"})
    assert paths == DefaultPaths(
        data_root=Path("/data/bench/translations"),
        prompt=Path("/data/bench/prompt.txt"),
        scorer_prompt=Path("/data/bench/prompt_reranker.txt"),
        results=Path("/data/runs"),
    )


def test_default_paths_keep_repo_relative_defaults() -> None:
    assert default_paths({}) == DefaultPaths(
        data_root=Path("data/translations"),
        prompt=Path("data/prompt.txt"),
        scorer_prompt=Path("data/prompt_reranker.txt"),
        results=Path("results"),
    )
