from landuse_relevance_bench.domain.pareto import (
    ModelPoint,
    is_runtime_variant,
    parameters_of,
    pareto_frontier,
    plotted_points,
)


def _point(
    model_id: str,
    parameters: int | None,
    balanced_accuracy: float,
    *,
    language_count: int = 85,
    generative: bool = True,
) -> ModelPoint:
    return ModelPoint(
        model_id=model_id,
        parameters=parameters,
        generative=generative,
        language_count=language_count,
        n_items_total=300 * language_count,
        balanced_accuracy=balanced_accuracy,
        f1=0.0,
        mcc=0.0,
    )


def test_frontier_keeps_models_no_smaller_model_matches() -> None:
    points = [
        _point("big-good", 9_000_000_000, 0.80),
        _point("small-bad", 1_000_000_000, 0.55),
        _point("small-good", 2_000_000_000, 0.70),
        _point("mid-worse", 4_000_000_000, 0.70),
        _point("tiny-chance", 500_000_000, 0.50),
    ]

    frontier = pareto_frontier(points)

    assert [point.model_id for point in frontier] == [
        "tiny-chance",
        "small-bad",
        "small-good",
        "big-good",
    ]


def test_frontier_skips_points_without_a_size() -> None:
    points = [_point("unsized", None, 0.99), _point("sized", 1_000_000_000, 0.6)]

    assert [point.model_id for point in pareto_frontier(points)] == ["sized"]


def test_frontier_of_no_points_is_empty() -> None:
    assert pareto_frontier([]) == ()


def test_runtime_variants_are_recognised_by_their_suffix() -> None:
    assert is_runtime_variant("LiquidAI/LFM2.5-2.6B@sglang-throughput-b16")
    assert is_runtime_variant("LiquidAI/LFM2.5-2.6B+DSpark-throughput-b16")
    assert not is_runtime_variant("LiquidAI/LFM2.5-2.6B")
    assert not is_runtime_variant("LiquidAI/LFM2.5-2.6B@logprob")


def test_plotted_points_need_every_language_a_size_and_no_runtime_suffix() -> None:
    points = [
        _point("full", 1_000_000_000, 0.6),
        _point("partial", 1_000_000_000, 0.9, language_count=15),
        _point("variant+DSpark-throughput-b16", 1_000_000_000, 0.9),
        _point("unsized", None, 0.9),
    ]

    assert [point.model_id for point in plotted_points(points, 85)] == ["full"]


def test_roster_and_scorer_parameter_counts_are_found() -> None:
    assert parameters_of("LiquidAI/LFM2.5-350M") == 354_483_968
    assert parameters_of("Qwen/Qwen3-0.6B") == 600_000_000
    assert parameters_of("LiquidAI/d1-3B") is not None
    assert parameters_of("not/in-the-roster") is None
