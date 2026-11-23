"""The balanced-accuracy-against-size plot on the dataset card, and the points it draws."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.optional_import import require
from landuse_relevance_bench.adapters.results_store import aggregate_rows
from landuse_relevance_bench.domain.pareto import ModelPoint, parameters_of, pareto_frontier
from landuse_relevance_bench.domain.records import RunResult

# The figure's path inside the published folder; the card links to it.
FIGURE_PATH = "figures/pareto-frontier.png"

# Colors from the dataviz reference palette, light surface. Generative models take the
# first categorical slot; scoring models are context, so they take the neutral gray.
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_GENERATIVE = "#2a78d6"
_SCORING = "#898781"


def model_points(results: Sequence[RunResult]) -> list[ModelPoint]:
    """One point per run name, with its macro metrics and its roster parameter count."""
    generative_names = {result.metadata.name for result in results if result.metadata.is_generative}
    return [
        ModelPoint(
            model_id=row["model_id"],
            parameters=parameters_of(row["model_id"]),
            generative=row["model_id"] in generative_names,
            language_count=int(row["language_count"]),
            n_items_total=int(row["n_items_total"]),
            balanced_accuracy=float(row["balanced_accuracy_macro"]),
            f1=float(row["f1_macro"]),
            mcc=float(row["matthews_corrcoef_macro"]),
        )
        for row in aggregate_rows(results)
    ]


def _billions(point: ModelPoint) -> float:
    return (point.parameters or 0) / 1e9


# The name column sits in the empty upper-left of the plot, one name per frontier model.
_NAME_X = 0.11
_NAME_TOP = 0.85
_NAME_STEP = 0.0175


def write_pareto_plot(points: Sequence[ModelPoint], path: Path) -> Path:
    """Draw balanced accuracy against total parameters, joining the frontier with a line.

    ``points`` are the plotted checkpoints (see ``plotted_points``); each needs a size.
    Every frontier model is named in a column, with a leader line to its point.
    """
    require("matplotlib", "publish")
    figure_module = require("matplotlib.figure", "publish")
    backend_module = require("matplotlib.backends.backend_agg", "publish")
    sized = [point for point in points if point.parameters is not None]
    if not sized:
        raise ValueError("the Pareto plot needs at least one model with a known size")
    frontier = pareto_frontier(sized)
    figure = figure_module.Figure(figsize=(7.2, 5.0), dpi=200, facecolor=_SURFACE)
    axes = figure.add_subplot()
    axes.set_facecolor(_SURFACE)

    for generative, color, label in (
        (False, _SCORING, "Scoring"),
        (True, _GENERATIVE, "Generative"),
    ):
        group = [point for point in sized if point.generative is generative]
        axes.scatter(
            [_billions(point) for point in group],
            [point.balanced_accuracy for point in group],
            s=56,
            color=color,
            edgecolors=_SURFACE,
            linewidths=1.2,
            zorder=3,
            label=label,
        )

    axes.plot(
        [_billions(point) for point in frontier],
        [point.balanced_accuracy for point in frontier],
        color=_INK_SECONDARY,
        linewidth=1.4,
        zorder=2,
        label="Pareto frontier",
    )

    axes.set_xscale("log")
    axes.set_xlim(0.1, 40)
    axes.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1B", "0.3B", "1B", "3B", "10B", "30B"])
    axes.minorticks_off()
    axes.set_ylim(0.48, 0.88)
    axes.set_yticks([0.5, 0.6, 0.7, 0.8], ["0.5", "0.6", "0.7", "0.8"])
    axes.grid(axis="y", color=_GRIDLINE, linewidth=0.6, zorder=0)
    axes.axhline(0.5, color=_INK_MUTED, linewidth=0.8, zorder=1)
    axes.text(
        0.99,
        0.5,
        "chance",
        transform=axes.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color=_INK_MUTED,
    )

    axes.set_xlabel("Total parameters (log scale)", color=_INK_SECONDARY, fontsize=9)
    axes.set_ylabel("Balanced accuracy, mean over languages", color=_INK_SECONDARY, fontsize=9)
    axes.tick_params(colors=_INK_SECONDARY, labelsize=8.5, length=0)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_BASELINE)
    axes.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=_INK_SECONDARY)

    figure.tight_layout()
    _name_frontier(figure, axes, backend_module, frontier)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches="tight", pad_inches=0.15)
    return path


def _name_frontier(
    figure: Any, axes: Any, backend_module: Any, frontier: Sequence[ModelPoint]
) -> None:
    """Name each frontier model, most accurate first, with a leader line to its point.

    Each leader starts just after the end of its name, so no line crosses a name.
    """
    renderer = backend_module.FigureCanvasAgg(figure).get_renderer()
    to_data = axes.transData.inverted()
    ordered = sorted(frontier, key=lambda point: -point.balanced_accuracy)
    for row, point in enumerate(ordered):
        y = _NAME_TOP - row * _NAME_STEP
        name = axes.text(
            _NAME_X,
            y,
            point.model_id.split("/")[-1],
            ha="left",
            va="center",
            fontsize=7.5,
            color=_INK_PRIMARY,
            zorder=4,
        )
        name_end = to_data.transform((name.get_window_extent(renderer).x1, 0))[0]
        axes.plot(
            [name_end * 1.06, _billions(point)],
            [y, point.balanced_accuracy],
            color=_INK_MUTED,
            linewidth=0.6,
            zorder=1.5,
        )
