import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from factories import ScriptedGenerator
from landuse_relevance_bench.adapters import pipeline
from landuse_relevance_bench.adapters.hashing import sha256_of_file
from landuse_relevance_bench.adapters.pipeline import RunRequest, execute
from landuse_relevance_bench.adapters.results_store import read_run, run_filename
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.roster import ROSTER


@pytest.fixture
def request_for(tmp_path: Path, benchmark_path: Path, prompt_path: Path):
    def build(**overrides: Any) -> RunRequest:
        return replace(
            RunRequest(
                model_id="LiquidAI/LFM2.5-350M",
                benchmark_path=benchmark_path,
                prompt_path=prompt_path,
                output_dir=tmp_path / "results",
            ),
            **overrides,
        )

    return build


class StubProvider:
    """A generator provider that keeps the generator reachable for assertions."""

    def __init__(self, outputs: Sequence[str], revision: str = "rev0") -> None:
        self.generator = ScriptedGenerator(outputs)
        self._revision = revision

    def __call__(self, _: RunRequest) -> tuple[ScriptedGenerator, str]:
        return self.generator, self._revision


def test_scores_the_whole_benchmark_and_returns_the_result(request_for) -> None:
    result = execute(request_for(), StubProvider(["yes", "no"]))
    assert result.metrics.n_items == 2
    assert result.metrics.accuracy == 1.0
    assert [p.predicted for p in result.predictions] == [Label.YES, Label.NO]


def test_writes_one_result_file_named_after_the_model(request_for, tmp_path: Path) -> None:
    result = execute(request_for(), StubProvider(["yes", "no"]))
    path = tmp_path / "results" / run_filename("LiquidAI/LFM2.5-350M")
    assert read_run(path) == result


def test_records_the_inputs_it_actually_used(request_for, benchmark_path: Path) -> None:
    result = execute(
        request_for(batch_size=1, max_new_tokens=4, seed=7), StubProvider(["yes", "no"])
    )
    metadata = result.metadata
    assert metadata.benchmark_sha256 == sha256_of_file(benchmark_path)
    assert metadata.model_revision == "rev0"
    assert metadata.batch_size == 1
    assert metadata.max_new_tokens == 4
    assert metadata.seed == 7
    assert metadata.decoding == "greedy"
    assert metadata.duration_seconds >= 0.0
    assert metadata.package_version


def test_feeds_the_file_prompt_to_the_generator(request_for) -> None:
    provider = StubProvider(["yes", "no"])
    execute(request_for(), provider)
    assert provider.generator.prompts[0].startswith("Classify.")
    assert provider.generator.prompts[0].endswith("Dense mangrove forest lines the lagoon.")


def test_unparsable_generations_survive_into_the_stored_result(request_for) -> None:
    result = execute(request_for(), StubProvider(["I cannot tell", "no"]))
    assert result.predictions[0].predicted is None
    assert result.predictions[0].raw_output == "I cannot tell"
    assert result.metrics.unparsed_rate == 0.5


def test_a_rostered_run_resolves_its_pins_and_records_its_draft(request_for) -> None:
    rostered = RunRequest.for_run(
        "LiquidAI/LFM2.5-2.6B+DSpark",
        benchmark_path=request_for().benchmark_path,
        prompt_path=request_for().prompt_path,
        output_dir=request_for().output_dir,
    )
    assert (rostered.batch_size, rostered.runtime) == (1, "sglang")
    metadata = execute(rostered, StubProvider(["yes", "no"], "654f")).metadata
    spec = next(spec for spec in ROSTER if spec.name == rostered.name)
    assert metadata.name == "LiquidAI/LFM2.5-2.6B+DSpark"
    assert metadata.draft_model_id == "LiquidAI/LFM2.5-2.6B-DSpark"
    assert metadata.draft_model_revision == spec.draft_revision
    assert metadata.speculative["speculative_algorithm"] == "DSPARK"
    assert (request_for().output_dir / "LiquidAI__LFM2.5-2.6B+DSpark.json").exists()


def test_explicit_settings_override_the_roster_and_unknown_models_run_plainly() -> None:
    def for_run(name: str, **overrides: Any) -> RunRequest:
        return RunRequest.for_run(
            name,
            benchmark_path=Path("b"),
            prompt_path=Path("p"),
            output_dir=Path("o"),
            **overrides,
        )

    overridden = for_run("LiquidAI/LFM2.5-VL-3B+DSpark", revision="main", batch_size=4)
    assert (overridden.revision, overridden.batch_size) == ("main", 4)
    plain = for_run("some/model")
    assert (plain.model_id, plain.runtime, plain.revision, plain.batch_size) == (
        "some/model",
        "transformers",
        None,
        16,
    )


def test_the_result_records_each_sentence_s_latency(request_for) -> None:
    result = execute(request_for(batch_size=1), StubProvider(["yes", "no"]))
    assert all(p.latency_seconds is not None for p in result.predictions)
    assert result.speed.latency_p50_seconds is not None


def test_execute_closes_a_generator_once_after_success(request_for) -> None:
    provider = StubProvider(["yes", "no"])
    calls: list[str] = []
    provider.generator.close = lambda: calls.append("closed")

    execute(request_for(), provider)

    assert calls == ["closed"]


def test_execute_closes_a_generator_once_when_prediction_fails(request_for) -> None:
    class FailingGenerator:
        def __init__(self) -> None:
            self.close_calls = 0

        def generate(self, _prompts: Sequence[str]) -> Sequence[str]:
            raise RuntimeError("generation failed")

        def close(self) -> None:
            self.close_calls += 1

    generator = FailingGenerator()

    def provider(_: RunRequest) -> tuple[FailingGenerator, str]:
        return generator, "rev0"

    with pytest.raises(RuntimeError, match="generation failed"):
        execute(request_for(), provider)

    assert generator.close_calls == 1


def test_gpu_memory_probe_handles_missing_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: None)
    assert pipeline._free_gpu_memory_bytes() is None


def test_gpu_memory_probe_reports_least_free_nvidia_memory_in_bytes(monkeypatch) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="500\n 123\n"),
    )
    assert pipeline._free_gpu_memory_bytes() == 123 * 2**20


def test_gpu_memory_probe_ignores_empty_nvidia_smi_output(monkeypatch) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="\n"),
    )
    assert pipeline._free_gpu_memory_bytes() is None


def test_execute_logs_gpu_memory_before_load_and_after_cleanup(
    request_for, monkeypatch, caplog
) -> None:
    caplog.set_level(logging.INFO, logger=pipeline.logger.name)
    readings = iter((8 * 2**20, 5 * 2**20))
    monkeypatch.setattr(pipeline, "_free_gpu_memory_bytes", lambda: next(readings))

    execute(request_for(), StubProvider(["yes", "no"]))

    assert "free GPU memory before model load: 8.0 MiB" in caplog.text
    assert "free GPU memory after run cleanup: 5.0 MiB" in caplog.text


def test_execute_preserves_unpinned_draft_revision_through_progress_wrapper(request_for) -> None:
    generator = ScriptedGenerator(["yes", "no"])
    generator.draft_revision = "resolved-draft-revision"

    result = execute(
        request_for(draft_model_id="some/draft"),
        lambda _: (generator, "target-revision"),
    )

    assert result.metadata.draft_model_revision == "resolved-draft-revision"


def test_continuous_batching_request_uses_a_distinct_run_name() -> None:
    request = RunRequest.for_run(
        "LiquidAI/LFM2.5-2.6B",
        benchmark_path=Path("benchmark.csv"),
        prompt_path=Path("prompt.txt"),
        output_dir=Path("results"),
        batch_size=154,
        continuous_batching=True,
    )

    assert request.model_id == "LiquidAI/LFM2.5-2.6B"
    assert request.name == "LiquidAI/LFM2.5-2.6B@continuous-b154"
    assert request.continuous_batching


def test_sglang_throughput_request_keeps_batch_one_baseline_separate() -> None:
    request = RunRequest.for_run(
        "LiquidAI/LFM2.5-2.6B@sglang",
        benchmark_path=Path("benchmark.csv"),
        prompt_path=Path("prompt.txt"),
        output_dir=Path("results"),
        batch_size=8,
        throughput_mode=True,
    )

    assert request.runtime == "sglang"
    assert request.batch_size == 8
    assert request.name == "LiquidAI/LFM2.5-2.6B@sglang-throughput-b8"
    assert request.throughput_mode


def test_throughput_mode_requires_batch_size_above_one() -> None:
    with pytest.raises(ValueError, match="throughput mode requires batch_size > 1"):
        RunRequest.for_run(
            "LiquidAI/LFM2.5-2.6B@sglang",
            benchmark_path=Path("benchmark.csv"),
            prompt_path=Path("prompt.txt"),
            output_dir=Path("results"),
            throughput_mode=True,
        )
