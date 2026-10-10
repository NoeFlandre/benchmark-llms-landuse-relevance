"""RunRequest.for_model is the one factory the CLI uses for every model-language pair."""

from pathlib import Path

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import model_ids
from landuse_relevance_bench.domain.scorers import scorer_ids

COMMON = {
    "language": "en",
    "benchmark_path": Path("en.csv"),
    "prompt_path": Path("prompt.txt"),
    "output_dir": Path("out"),
    "close_generator": False,
}


def test_a_scoring_id_gets_the_plain_request_built_from_its_options() -> None:
    model = scorer_ids()[0]
    options = {"revision": "rev", "batch_size": 8, "seed": 2, "dtype": "float32", **COMMON}

    built = RunRequest.for_model(model, **options)

    assert built == RunRequest(model, **options)


def test_a_scoring_id_without_a_batch_size_uses_the_default_batch_size() -> None:
    built = RunRequest.for_model(scorer_ids()[0], batch_size=None, **COMMON)

    assert built.batch_size == DEFAULT_BATCH_SIZE


def test_a_roster_id_gets_the_same_request_as_for_run() -> None:
    model = model_ids()[0]
    options = {
        "revision": None,
        "batch_size": None,
        "max_new_tokens": 32,
        "seed": 4,
        "dtype": "float16",
        **COMMON,
    }

    assert RunRequest.for_model(model, **options) == RunRequest.for_run(model, **options)
