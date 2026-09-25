from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from factories import ScriptedGenerator
from landuse_relevance_bench.adapters.hashing import sha256_of_file
from landuse_relevance_bench.adapters.pipeline import RunRequest, execute
from landuse_relevance_bench.adapters.results_store import read_run, run_filename
from landuse_relevance_bench.domain.labels import Label


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
    assert metadata.name == "LiquidAI/LFM2.5-2.6B+DSpark"
    assert metadata.draft_model_id == "LiquidAI/LFM2.5-2.6B-DSpark"
    assert metadata.draft_model_revision == "458cedab07d0f7b2b05700c77e1aa463d43d6f04"
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
