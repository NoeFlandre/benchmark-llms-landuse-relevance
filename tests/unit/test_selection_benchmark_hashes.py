"""Benchmark identity is checked within a language, never across languages."""

from factories import make_result
from landuse_relevance_bench.domain.selection import collection_comparability_errors

SCORING = {
    "inference": "scoring",
    "decision_rule": "argmax over the native yes/no scores",
    "max_new_tokens": 0,
    "decoding": "none",
}


def test_generative_runs_in_different_languages_may_use_different_benchmark_files() -> None:
    english = make_result(model_id="g", language="en", benchmark_sha256="x" * 64)
    french = make_result(model_id="g", language="fr", benchmark_sha256="y" * 64)

    assert collection_comparability_errors([english, french]) == ()


def test_scoring_runs_in_different_languages_may_use_different_benchmark_files() -> None:
    english = make_result(model_id="s", language="en", benchmark_sha256="x" * 64, **SCORING)
    french = make_result(model_id="s", language="fr", benchmark_sha256="y" * 64, **SCORING)

    assert collection_comparability_errors([english, french]) == ()


def test_generative_runs_in_one_language_must_share_their_benchmark_file() -> None:
    first = make_result(model_id="g1", language="en", benchmark_sha256="x" * 64)
    second = make_result(model_id="g2", language="en", benchmark_sha256="y" * 64)

    assert collection_comparability_errors([first, second]) == (
        "runs are not comparable: benchmark_sha256 differs for language en",
    )


def test_a_scorer_must_keep_its_prompt_and_budget_across_languages() -> None:
    english = make_result(model_id="s", language="en", prompt_sha256="p" * 64, **SCORING)
    french = make_result(
        model_id="s", language="fr", prompt_sha256="q" * 64, benchmark_sha256="y" * 64, **SCORING
    )

    errors = collection_comparability_errors([english, french])

    assert any(error.startswith("s:") and "prompt_sha256" in error for error in errors)
