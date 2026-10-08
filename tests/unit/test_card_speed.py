"""Characterization of the runtime-performance card section, pinned before a refactor.

Each expected figure is derived by hand in the comment beside it. The golden file is the one
rendered value: it was written once from the code and then checked number by number.
"""

from pathlib import Path

from factories import make_result
from landuse_relevance_bench.adapters.hf_publish import (
    EXPECTED_FULL_SWEEP_LANGUAGE_COUNT,
    SPEED_COLUMNS,
    _agreement_section,
    _reproducibility_note,
    _speed_rows,
    _speed_section,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import Prediction, RunResult

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "speed_section_synthetic.md"

BASELINE_RUN = "LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16"
DSPARK_RUN = "LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16"

REPRODUCIBILITY_NOTE = (
    "Reproducibility probes show GPU sensitivity: on 15 overlapping LFM2.5-2.6B "
    "SGLang-throughput languages, A40 versus RTX 6000 Ada changed 474/4,500 "
    "verdicts; LFM2.5-8B-A1B changed 31/300 verdicts across GPU types. VL-3B "
    "SGLang throughput changed 158/25,500 verdicts between RTX A6000 and RTX "
    "6000 Ada, while its same-GPU DSpark comparison changed 0/25,500. Throughput "
    "mode versus batch size 1 changed 2/300 VL-3B English verdicts. Transformers "
    "continuous batching fails on LFM2 with `Invalid group type: conv`. Compare runs "
    "only with the same runtime, mode and GPU model."
)


def _timed(  # noqa: PLR0913 - one keyword per telemetry field the run may record
    item_id: str,
    *,
    latency: float | None = None,
    tokens: int | None = None,
    verify: int | None = None,
    accepted: int | None = None,
    proposed: int | None = None,
) -> Prediction:
    return Prediction(
        item_id=item_id,
        expected=Label.YES,
        predicted=Label.YES,
        raw_output="yes",
        latency_seconds=latency,
        generated_tokens=tokens,
        verify_steps=verify,
        accepted_drafts=accepted,
        proposed_drafts=proposed,
    )


def _run(
    model_id: str,
    language: str,
    duration: float,
    predictions: tuple[Prediction, ...],
    **settings: object,
) -> RunResult:
    return make_result(
        model_id=model_id,
        predictions=predictions,
        language=language,
        duration_seconds=duration,
        **settings,
    )


def test_optional_figures_are_rounded_to_their_recorded_digits() -> None:
    english = _run(
        "a/model",
        "en",
        1.728,
        (
            _timed("en-1", latency=0.1, tokens=3, verify=2, accepted=3, proposed=5),
            _timed("en-2", latency=0.2, tokens=5, verify=3, accepted=2, proposed=4),
            _timed("en-3", latency=0.41111, tokens=4, verify=2, accepted=3, proposed=5),
            _timed("en-4", latency=0.9123, tokens=7, verify=4, accepted=2, proposed=4),
        ),
        runtime="sglang",
        device_name="NVIDIA A100",
    )
    french = _run(
        "a/model",
        "fr",
        1.728,
        (
            _timed("fr-1", latency=0.3, tokens=2, verify=1, accepted=2, proposed=4),
            _timed("fr-2", latency=0.5, tokens=6, verify=3, accepted=3, proposed=4),
            _timed("fr-3", latency=0.6, tokens=5, verify=2, accepted=2, proposed=4),
        ),
        runtime="sglang",
        device_name="NVIDIA A100",
    )

    (row,) = _speed_rows([english, french])

    assert list(row) == list(SPEED_COLUMNS)
    assert row == {
        "model_id": "a/model",
        "runtime": "sglang",
        "device": "NVIDIA A100",
        "generation_mode": "static-batched",
        "batch_size": 16,
        "language_count": 2,
        # wall = 1.728 + 1.728 = 3.456 s; 2 digits
        "cumulative_wall_seconds": 3.46,
        # 7 items / 3.456 s = 2.0254629... ; 3 digits
        "sentences_per_second": 2.025,
        # latencies sum to 3.02341 over 7 values: 3.02341 / 7 = 0.4319157... ; 4 digits
        "latency_mean_seconds": 0.4319,
        # sorted 0.1, 0.2, 0.3, 0.41111, 0.5, 0.6, 0.9123; rank 0.5 * 6 = 3 -> 0.41111 ; 4 digits
        "latency_p50_seconds": 0.4111,
        # rank 0.95 * 6 = 5.7: 0.6 + (0.9123 - 0.6) * 0.7 = 0.81861 ; 4 digits
        "latency_p95_seconds": 0.8186,
        # 3+5+4+7 (en) + 2+6+5 (fr) = 32 tokens
        "generated_tokens": 32,
        # 32 tokens / 3.456 s = 9.2592592... ; 2 digits
        "output_tokens_per_second": 9.26,
        # 32 tokens / (2+3+2+4 + 1+3+2 = 17) verify steps = 1.8823529... ; 3 digits
        "mean_accept_length": 1.882,
        # (3+2+3+2 + 2+3+2 = 17) accepted / (5+4+5+4 + 4+4+4 = 30) proposed = 0.5666... ; 4 digits
        "draft_accept_rate": 0.5667,
    }


def test_figures_a_run_did_not_record_are_none_not_zero_or_blank() -> None:
    bare = _run(
        "b/bare",
        "en",
        2.0,
        tuple(_timed(f"b-{index}") for index in range(4)),
        device_name="NVIDIA A100",
    )
    partial = _run(
        "c/partial",
        "en",
        2.0,
        (
            _timed("c-1", latency=0.5, tokens=4, verify=2, accepted=1, proposed=2),
            # One prediction without latency makes every latency figure incomplete.
            _timed("c-2", tokens=6, verify=3, accepted=2, proposed=3),
        ),
        device_name="NVIDIA A100",
    )
    zero_wall = _run(
        "d/zero",
        "en",
        0.0,
        (
            _timed("d-1", latency=0.5, tokens=4, verify=2, accepted=1, proposed=2),
            _timed("d-2", latency=0.5, tokens=4, verify=2, accepted=1, proposed=2),
        ),
        device_name="NVIDIA A100",
    )

    rows = {row["model_id"]: row for row in _speed_rows([bare, partial, zero_wall])}

    assert rows["b/bare"] == {
        "model_id": "b/bare",
        "runtime": "transformers",
        "device": "NVIDIA A100",
        "generation_mode": "static-batched",
        "batch_size": 16,
        "language_count": 1,
        # 4 items / 2.0 s = 2.0
        "cumulative_wall_seconds": 2.0,
        "sentences_per_second": 2.0,
        "latency_mean_seconds": None,
        "latency_p50_seconds": None,
        "latency_p95_seconds": None,
        "generated_tokens": None,
        "output_tokens_per_second": None,
        "mean_accept_length": None,
        "draft_accept_rate": None,
    }
    assert rows["c/partial"] == {
        "model_id": "c/partial",
        "runtime": "transformers",
        "device": "NVIDIA A100",
        "generation_mode": "static-batched",
        "batch_size": 16,
        "language_count": 1,
        # 2 items / 2.0 s = 1.0
        "cumulative_wall_seconds": 2.0,
        "sentences_per_second": 1.0,
        "latency_mean_seconds": None,
        "latency_p50_seconds": None,
        "latency_p95_seconds": None,
        # 4 + 6 = 10 tokens; 10 / 2.0 s = 5.0
        "generated_tokens": 10,
        "output_tokens_per_second": 5.0,
        # 10 tokens / (2 + 3) verify steps = 2.0
        "mean_accept_length": 2.0,
        # (1 + 2) accepted / (2 + 3) proposed = 0.6
        "draft_accept_rate": 0.6,
    }
    assert rows["d/zero"] == {
        "model_id": "d/zero",
        "runtime": "transformers",
        "device": "NVIDIA A100",
        "generation_mode": "static-batched",
        "batch_size": 16,
        "language_count": 1,
        # zero wall time: no rate can be formed
        "cumulative_wall_seconds": 0.0,
        "sentences_per_second": None,
        # latencies 0.5, 0.5: mean, p50 and p95 are all 0.5
        "latency_mean_seconds": 0.5,
        "latency_p50_seconds": 0.5,
        "latency_p95_seconds": 0.5,
        # 4 + 4 = 8 tokens; the rate is None because the wall time is 0
        "generated_tokens": 8,
        "output_tokens_per_second": None,
        # 8 tokens / (2 + 2) verify steps = 2.0
        "mean_accept_length": 2.0,
        # (1 + 1) accepted / (2 + 2) proposed = 0.5
        "draft_accept_rate": 0.5,
    }


def test_settings_that_differ_between_runs_are_described_as_varying() -> None:
    runtime_mix = (
        _run(
            "e/runtime",
            "en",
            1.0,
            (_timed("e-1"),),
            runtime="sglang",
            device_name="NVIDIA A100",
        ),
        _run(
            "e/runtime",
            "fr",
            1.0,
            (_timed("e-2"),),
            runtime="transformers",
            device_name="NVIDIA A100",
        ),
    )
    batch_mix = (
        _run("f/batch", "en", 1.0, (_timed("f-1"),), batch_size=8, device_name="NVIDIA A100"),
        _run("f/batch", "fr", 1.0, (_timed("f-2"),), batch_size=16, device_name="NVIDIA A100"),
    )

    rows = {row["model_id"]: row for row in _speed_rows([*runtime_mix, *batch_mix])}

    assert rows["e/runtime"]["runtime"] == "varies"
    # One batch size across both runs is reported as that integer.
    assert rows["e/runtime"]["batch_size"] == 16
    assert type(rows["e/runtime"]["batch_size"]) is int
    assert rows["f/batch"]["batch_size"] == "varies"
    assert rows["f/batch"]["runtime"] == "transformers"


def _golden_fixture() -> list[RunResult]:
    full = [
        _run(
            "g/full",
            "en",
            1.0,
            (
                _timed("g-en-1", latency=0.1, tokens=3, verify=2, accepted=2, proposed=4),
                _timed("g-en-2", latency=0.2, tokens=5, verify=3, accepted=2, proposed=4),
            ),
            runtime="sglang",
            device_name="NVIDIA A100",
        ),
        _run(
            "g/full",
            "fr",
            2.0,
            (_timed("g-fr-1", latency=0.3, tokens=4, verify=2, accepted=1, proposed=3),),
            runtime="sglang",
            device_name="NVIDIA A100",
        ),
    ]
    bare = [
        _run(
            "g/bare",
            language,
            1.0,
            (_timed(f"g-bare-{language}"),),
            runtime="transformers",
            device_name="NVIDIA A100",
        )
        for language in ("en", "fr")
    ]
    return [*full, *bare]


def test_the_speed_section_of_a_full_and_a_bare_model_matches_the_golden() -> None:
    assert _speed_section(_golden_fixture()) == GOLDEN.read_text(encoding="utf-8")


def _sweep(run_id: str, languages: int) -> list[RunResult]:
    return [
        make_result(model_id="LiquidAI/LFM2.5-VL-3B", language=f"l{index:03d}", run_id=run_id)
        for index in range(languages)
    ]


def test_reproducibility_note_needs_a_full_sweep_of_both_probe_runs() -> None:
    assert EXPECTED_FULL_SWEEP_LANGUAGE_COUNT == 85

    full = [*_sweep(BASELINE_RUN, 85), *_sweep(DSPARK_RUN, 85)]

    assert _reproducibility_note(full) == REPRODUCIBILITY_NOTE


def test_reproducibility_note_is_withheld_when_either_probe_run_is_short() -> None:
    assert _reproducibility_note([*_sweep(BASELINE_RUN, 85), *_sweep(DSPARK_RUN, 84)]) == ""
    assert _reproducibility_note(_sweep(BASELINE_RUN, 85)) == ""


def _speedup_row(baseline_tokens: int | None) -> str:
    """The DSpark row for two languages, each run lasting 1.0 s per language.

    The baseline has 2 tokens per language (4 over 2.0 s = 2.0 tokens/s); the draft has 3
    per language (6 over 2.0 s = 3.0 tokens/s).
    """
    results: list[RunResult] = []
    for language in ("en", "fr"):
        results.append(
            _run(
                "a/model",
                language,
                1.0,
                (_timed("x-1", tokens=baseline_tokens),),
                run_id="a/model@sglang",
                runtime="sglang",
            )
        )
        results.append(
            _run(
                "a/model",
                language,
                1.0,
                (_timed("x-1", tokens=3),),
                run_id="a/model+DSpark",
                runtime="sglang",
                draft_model_id="a/draft",
            )
        )
    (row,) = [
        line for line in _agreement_section(results).splitlines() if line.startswith("| a/model |")
    ]
    return row


def test_speedup_is_the_draft_rate_over_the_baseline_rate_and_blank_without_one() -> None:
    # 3.0 / 2.0 = 1.5 -> "1.50x"
    assert _speedup_row(2) == (
        "| a/model | a/model@sglang | a/model+DSpark | 2.0 | 3.0 | 1.50x | yes | 2 | 2 | 0 | 0 |"
    )
    # Zero baseline rate: the speedup cell is empty, not "inf" or "0.00x".
    assert _speedup_row(0) == (
        "| a/model | a/model@sglang | a/model+DSpark | 0.0 | 3.0 |  | yes | 2 | 2 | 0 | 0 |"
    )
    # Missing baseline rate: the speedup cell is empty.
    assert _speedup_row(None) == (
        "| a/model | a/model@sglang | a/model+DSpark |  | 3.0 |  | yes | 2 | 2 | 0 | 0 |"
    )
