"""Shared Hypothesis strategies for the benchmark's serialised records."""

from hypothesis import strategies as st

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import Outcome, evaluate
from landuse_relevance_bench.domain.records import Prediction, RunResult

labels = st.sampled_from([Label.YES, Label.NO])
predicted_labels = st.one_of(labels, st.none())
outcomes = st.tuples(labels, predicted_labels)
suffixes = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters="/\\"),
    max_size=12,
)


def predictions_for(pairs: list[Outcome]) -> tuple[Prediction, ...]:
    return tuple(
        Prediction(
            item_id=f"item-{index}",
            expected=expected,
            predicted=predicted,
            raw_output="" if predicted is None else predicted.value,
        )
        for index, (expected, predicted) in enumerate(pairs)
    )


def result_for(name: str, pairs: list[Outcome]) -> RunResult:
    from factories import make_metadata

    predictions = predictions_for(pairs)
    return RunResult(
        metadata=make_metadata(model_id=f"模型/{name}", model_revision=None),
        predictions=predictions,
        metrics=evaluate([prediction.outcome for prediction in predictions]),
    )


@st.composite
def run_results(draw: st.DrawFn) -> RunResult:
    pairs = draw(st.lists(outcomes, min_size=1, max_size=12))
    suffix = draw(suffixes)
    return result_for(f"run-{suffix}", pairs)


@st.composite
def comparable_leaderboards(draw: st.DrawFn) -> list[RunResult]:
    n_runs = draw(st.integers(min_value=1, max_value=5))
    n_items = draw(st.integers(min_value=1, max_value=8))
    expected = draw(st.lists(labels, min_size=n_items, max_size=n_items))
    predictions = draw(
        st.lists(
            st.lists(predicted_labels, min_size=n_items, max_size=n_items),
            min_size=n_runs,
            max_size=n_runs,
        )
    )
    return [
        result_for(
            f"run-{index}",
            list(zip(expected, predicted, strict=True)),
        )
        for index, predicted in enumerate(predictions)
    ]
