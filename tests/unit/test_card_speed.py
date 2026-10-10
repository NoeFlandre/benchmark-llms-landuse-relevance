"""speed_rows pins each runtime-performance figure and its rounding, exactly."""

from factories import make_result
from landuse_relevance_bench.adapters.publishing.card_speed import speed_rows
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import Prediction


def _timed(item_id: str, **telemetry: float | int) -> Prediction:
    return Prediction(
        item_id=item_id,
        expected=Label.YES,
        predicted=Label.YES,
        raw_output="yes",
        **telemetry,
    )


def test_speed_rows_pins_every_figure_and_its_rounding() -> None:
    # Two runs of one model: four timed items over 3.0 + 4.0 = 7.0 seconds.
    alpha_en = make_result(
        model_id="org/alpha",
        language="en",
        duration_seconds=3.0,
        device_name="NVIDIA A100",
        predictions=(
            _timed(
                "a1",
                latency_seconds=0.1,
                generated_tokens=10,
                verify_steps=4,
                accepted_drafts=3,
                proposed_drafts=4,
            ),
            _timed(
                "a2",
                latency_seconds=0.25,
                generated_tokens=7,
                verify_steps=3,
                accepted_drafts=2,
                proposed_drafts=3,
            ),
        ),
    )
    alpha_fr = make_result(
        model_id="org/alpha",
        language="fr",
        duration_seconds=4.0,
        device_name="NVIDIA A100",
        predictions=(
            _timed(
                "a3",
                latency_seconds=0.4,
                generated_tokens=9,
                verify_steps=5,
                accepted_drafts=4,
                proposed_drafts=5,
            ),
            _timed(
                "a4",
                latency_seconds=0.3,
                generated_tokens=6,
                verify_steps=2,
                accepted_drafts=1,
                proposed_drafts=2,
            ),
        ),
    )
    # One untimed item: every telemetry figure is unrecorded and stays None.
    beta = make_result(
        model_id="org/beta",
        language="en",
        duration_seconds=2.0,
        runtime="sglang",
        batch_size=8,
        predictions=(
            Prediction(item_id="b1", expected=Label.YES, predicted=Label.YES, raw_output="yes"),
        ),
    )

    rows = speed_rows([beta, alpha_fr, alpha_en])

    assert rows == [
        {
            "model_id": "org/alpha",
            "runtime": "transformers",
            "device": "NVIDIA A100",
            "generation_mode": "static-batched",
            "batch_size": 16,
            "language_count": 2,
            "cumulative_wall_seconds": 7.0,
            # 4 / 7 = 0.5714... to 3 places
            "sentences_per_second": 0.571,
            # mean 0.2625; p50 interpolates 0.25 and 0.30 at 0.275; p95 gives 0.385
            "latency_mean_seconds": 0.2625,
            "latency_p50_seconds": 0.275,
            "latency_p95_seconds": 0.385,
            "generated_tokens": 32,
            # 32 / 7 = 4.571... to 2 places
            "output_tokens_per_second": 4.57,
            # pooled 32 / 14 = 2.2857... to 3 places
            "mean_accept_length": 2.286,
            # pooled 10 / 14 = 0.7142... to 4 places
            "draft_accept_rate": 0.7143,
        },
        {
            "model_id": "org/beta",
            "runtime": "sglang",
            "device": "",
            "generation_mode": "static-batched",
            "batch_size": 8,
            "language_count": 1,
            "cumulative_wall_seconds": 2.0,
            "sentences_per_second": 0.5,
            "latency_mean_seconds": None,
            "latency_p50_seconds": None,
            "latency_p95_seconds": None,
            "generated_tokens": None,
            "output_tokens_per_second": None,
            "mean_accept_length": None,
            "draft_accept_rate": None,
        },
    ]
