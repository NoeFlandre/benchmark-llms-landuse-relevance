"""The pipeline's log lines: model load, GPU memory, batch progress and cleanup failures."""

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from factories import ScriptedGenerator
from landuse_relevance_bench.adapters import pipeline
from landuse_relevance_bench.adapters.benchmark_csv import BenchmarkFileError
from landuse_relevance_bench.adapters.pipeline import (
    ProgressReporting,
    RunRequest,
    execute,
    execute_scoring,
)
from landuse_relevance_bench.domain.scorers import scorer_ids

LOGGER = "landuse_relevance_bench.adapters.pipeline"
MODEL = "LiquidAI/LFM2.5-350M"
MIB = 2**20


@pytest.fixture
def request_for(tmp_path: Path, benchmark_path: Path, prompt_path: Path):
    def build(**overrides: object) -> RunRequest:
        return replace(
            RunRequest(
                model_id=MODEL,
                language="en",
                benchmark_path=benchmark_path,
                prompt_path=prompt_path,
                output_dir=tmp_path / "results",
            ),
            **overrides,
        )

    return build


def _provider(generator, revision: str = "rev0"):
    return lambda _request: (generator, revision)


class _Closing(ScriptedGenerator):
    """A generator whose cleanup raises, so the cleanup log line is exercised."""

    def __init__(self, outputs, *, fail_generation: bool = False) -> None:
        super().__init__(outputs)
        self._fail_generation = fail_generation

    def generate(self, prompts):
        if self._fail_generation:
            raise RuntimeError("generation failed")
        return super().generate(prompts)

    def close(self) -> None:
        raise OSError("cleanup failed")


def test_execute_logs_the_model_load_with_its_prompt_count(
    caplog, request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        execute(request_for(), _provider(ScriptedGenerator(["yes", "no"])))

    assert f"{MODEL}: loading model (2 prompts)" in caplog.messages


def test_execute_reports_free_gpu_memory_in_mebibytes_before_and_after_the_run(
    caplog, request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: int(3.5 * MIB))

    with caplog.at_level(logging.INFO, logger=LOGGER):
        execute(request_for(), _provider(ScriptedGenerator(["yes", "no"])))

    assert f"{MODEL}: free GPU memory before model load: 3.5 MiB" in caplog.messages
    assert f"{MODEL}: free GPU memory after run cleanup: 3.5 MiB" in caplog.messages
    assert not [m for m in caplog.messages if "failed run cleanup" in m]


def test_execute_logs_each_generation_batch_with_its_position(
    caplog, request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        execute(request_for(batch_size=1), _provider(ScriptedGenerator(["yes", "no"])))

    assert f"{MODEL}: batch 1/2 (1/2 prompts)" in caplog.messages
    assert f"{MODEL}: batch 2/2 (2/2 prompts)" in caplog.messages


def test_execute_hands_the_generator_the_request_batch_size(request_for, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)
    generator = ScriptedGenerator(["yes", "no"])

    execute(request_for(batch_size=1), _provider(generator))

    assert generator.batch_sizes == [1, 1]


@pytest.mark.parametrize(
    ("total", "batch_size", "batches"),
    [(5, 2, 3), (2, 2, 1), (5, 1, 5), (0, 4, 1)],
)
def test_progress_counts_batches_by_rounding_the_quotient_up(
    caplog, total: int, batch_size: int, batches: int
) -> None:
    progress = ProgressReporting(ScriptedGenerator(["a"] * 6), "x", total, batch_size)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        progress.generate(["a"] * min(batch_size, total) if total else [])

    assert caplog.messages == [f"x: batch 1/{batches} ({min(batch_size, total)}/{total} prompts)"]


def test_progress_accumulates_prompts_and_batches_across_calls(caplog) -> None:
    progress = ProgressReporting(ScriptedGenerator(["a"] * 5), "x", 5, 2)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        progress.generate(["a", "a"])
        progress.generate(["a", "a"])
        progress.generate(["a"])

    assert caplog.messages == [
        "x: batch 1/3 (2/5 prompts)",
        "x: batch 2/3 (4/5 prompts)",
        "x: batch 3/3 (5/5 prompts)",
    ]


def test_progress_logs_the_model_name_it_was_given(caplog) -> None:
    progress = ProgressReporting(ScriptedGenerator(["a"]), "run@variant", 1, 1)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        progress.generate(["a"])

    assert caplog.messages == ["run@variant: batch 1/1 (1/1 prompts)"]


def test_a_failed_generator_cleanup_is_logged_as_a_prediction_error_phase(
    caplog, request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: int(3.5 * MIB))

    with (
        caplog.at_level(logging.INFO, logger=LOGGER),
        pytest.raises(RuntimeError, match="generation failed"),
    ):
        execute(request_for(), _provider(_Closing([], fail_generation=True)))

    assert f"{MODEL}: generator cleanup failed after prediction error" in caplog.messages
    assert f"{MODEL}: free GPU memory after failed run cleanup: 3.5 MiB" in caplog.messages
    assert f"{MODEL}: free GPU memory after run cleanup: 3.5 MiB" not in caplog.messages


def test_a_failed_cleanup_after_success_is_logged_as_the_success_phase(
    caplog, request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        execute(request_for(), _provider(_Closing(["yes", "no"])))

    assert f"{MODEL}: generator cleanup failed after prediction" in caplog.messages
    assert f"{MODEL}: generator cleanup failed after prediction error" not in caplog.messages


def test_a_failed_cleanup_is_logged_at_error_level_with_its_traceback(
    caplog, request_for, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "free_gpu_memory_bytes", lambda: None)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        execute(request_for(), _provider(_Closing(["yes", "no"])))

    (record,) = [r for r in caplog.records if "generator cleanup failed" in r.getMessage()]
    assert record.levelno == logging.ERROR
    exc_type, exc_value, _ = record.exc_info or (None, None, None)
    assert exc_type is OSError
    assert str(exc_value) == "cleanup failed"


def test_scoring_loads_the_benchmark_in_the_request_language(request_for) -> None:
    model = scorer_ids()[0]

    with pytest.raises(BenchmarkFileError, match="expected language 'fr'"):
        execute_scoring(
            request_for(model_id=model, language="fr"),
            lambda _request: pytest.fail("the scorer must not load before the data is checked"),
        )


def test_generation_loads_the_benchmark_in_the_request_language(request_for) -> None:
    with pytest.raises(BenchmarkFileError, match="expected language 'fr'"):
        execute(
            request_for(language="fr"),
            lambda _request: pytest.fail("the model must not load before the data is checked"),
        )
