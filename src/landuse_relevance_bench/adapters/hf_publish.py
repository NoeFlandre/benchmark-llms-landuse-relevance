"""Publishing run results to a Hugging Face dataset repository."""

import html
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.adapters.results_store import (
    aggregate_rows,
    read_runs,
    scoring_summary_rows,
)
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import RunResult, outcomes_of
from landuse_relevance_bench.domain.scorers import scorer_for

CARD_COLUMNS = (
    "model_id",
    "language_count",
    "accuracy_macro",
    "balanced_accuracy_macro",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "matthews_corrcoef_macro",
)

_SCORER_CARD_CONFIG = {
    "Alibaba-NLP/gte-multilingual-reranker-base": (
        "sequence classifier",
        "prompt + sentence pair",
        "sigmoid relevance logit",
    ),
    "mixedbread-ai/mxbai-rerank-base-v2": (
        "causal-LM reranker",
        "official query/document turn",
        "sigmoid(1-logit - 0-logit - 4.5)",
    ),
    "convaiinnovations/laya-multilingual": (
        "typed decision model",
        "JSON state + 4 `noul` questions/call",
        "Laya `noul` yes probability",
    ),
    "Qwen/Qwen3-Reranker-0.6B": (
        "causal-LM reranker",
        "manual yes/no reranker turn",
        "yes/no next-token probability",
    ),
    "Qwen/Qwen3-Reranker-4B": (
        "causal-LM reranker",
        "manual yes/no reranker turn",
        "yes/no next-token probability",
    ),
}


class DatasetHub(Protocol):
    """The slice of :class:`huggingface_hub.HfApi` this project depends on."""

    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...


def read_published_runs(results_dir: Path) -> tuple[RunResult, ...]:
    """Read every result below ``results_dir`` in stable model-id order."""
    runs = read_runs(results_dir)
    return tuple(
        sorted(runs, key=lambda result: (result.metadata.model_id, result.metadata.language))
    )


def _uniform(results: Sequence[RunResult], name: str, value_of: Any) -> Any:
    """Return the single value every run agrees on, or refuse to describe the sweep."""
    values = {value_of(result) for result in results}
    if len(values) != 1:
        raise ValueError(f"runs disagree on {name}: {sorted(values)}")
    return values.pop()


def dataset_card(
    results: Sequence[RunResult],
    *,
    benchmark_name: str,
    prompt_text: str,
    scorer_prompt_text: str = "",
    viewer_file: str = "",
    plot_files: Sequence[str] = (),
) -> str:
    """Build a terse card whose scores are recomputed from every prediction."""
    if not results:
        raise ValueError("cannot build a card from an empty set of results")
    for result in results:
        derived = evaluate(outcomes_of(result.predictions))
        if derived != result.metrics:
            raise ValueError(f"{result.metadata.model_id} metrics do not match predictions")
    generative_pre = [r for r in results if r.metadata.is_generative]
    if not generative_pre:
        raise ValueError("cannot build a card without any generative run")
    prompt_sha256 = _uniform(generative_pre, "prompt digest", lambda r: r.metadata.prompt_sha256)
    if sha256_of_text(prompt_text) != prompt_sha256:
        raise ValueError("prompt text does not match the digest recorded in the runs")
    generative = generative_pre
    scoring = [r for r in results if not r.metadata.is_generative]
    scorer_prompt_sha256 = ""
    if scoring:
        scorer_prompt_sha256 = _uniform(
            scoring, "scoring prompt digest", lambda r: r.metadata.prompt_sha256
        )
        if sha256_of_text(scorer_prompt_text) != scorer_prompt_sha256:
            raise ValueError(
                "scoring prompt text does not match the digest recorded in the scoring runs"
            )
    decoding = _uniform(generative, "decoding", lambda r: r.metadata.decoding)
    dtype = _uniform(generative, "dtype", lambda r: r.metadata.dtype)
    seed = _uniform(generative, "seed", lambda r: r.metadata.seed)
    max_new_tokens = _uniform(generative, "token budget", lambda r: r.metadata.max_new_tokens)
    batch_sizes = sorted({result.metadata.batch_size for result in generative})
    batch_size_line = (
        f"batch size {batch_sizes[0]}"
        if len(batch_sizes) == 1
        else "batch size varying by model, recorded per run"
    )
    n_items = _uniform(results, "items per language", lambda r: r.metrics.n_items)
    languages = sorted({result.metadata.language for result in results})
    header = "| " + " | ".join(CARD_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(CARD_COLUMNS)) + "|"

    def table(subset: Sequence[RunResult]) -> str:
        return "\n".join(
            "| " + " | ".join(str(row[column]) for column in CARD_COLUMNS) + " |"
            for row in aggregate_rows(subset)
        )

    body = table(generative)
    scoring_section = (
        ""
        if not scoring
        else _scoring_section(
            scoring,
            prompt_text=scorer_prompt_text,
            prompt_sha256=scorer_prompt_sha256,
        )
    )
    plots_section = _plots_section(plot_files)
    return f"""---
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

`{benchmark_name}` · model-level macro averages over completed language checkpoints.

- Code: https://github.com/NoeFlandre/benchmark-llms-landuse-relevance

## Benchmark

Binary sentence classification of land-use, land-cover, and geographic-environment
signal observable from remote sensing.

| setting | value |
|---|---|
| languages | {len(languages)} |
| items per language | {n_items} |
| labels | `yes`, `no` |
| positive class | `yes` |
| prompt language | English |
| generation | {decoding} decoding; seed {seed}; `max_new_tokens={max_new_tokens}` |
| dtype | `{dtype}` |
| batch size | {batch_size_line} |
| result layout | one JSON per model/language |
| dataset viewer | `{viewer_file or "not included"}` |
| prompt sha256 | `{prompt_sha256}` |
| benchmark hashes | recorded per language run |

### Prompt

Used verbatim; `{{}}` is replaced by the target sentence.

```text
{prompt_text}```

## Aggregate scores

Macro averages by language. Full aggregate metrics: [`aggregates.csv`](aggregates.csv).

{header}
{divider}
{body}
{scoring_section}{plots_section}"""


def _plots_section(plot_files: Sequence[str]) -> str:
    """Render stable relative links to the plots placed beside the card."""
    if not plot_files:
        return ""
    alt_text = {
        "quality_metrics.svg": "Best scoring metrics",
        "performance.svg": "Inference performance",
    }
    links = "\n\n".join(
        f"![{alt_text.get(Path(file).name, Path(file).stem)}]({file})" for file in plot_files
    )
    return f"\n\n## Plots\n\n{links}"


def _scoring_section(
    scoring: Sequence[RunResult],
    *,
    prompt_text: str,
    prompt_sha256: str,
) -> str:
    """Render compact public-facing scoring details and tables."""
    summary = scoring_summary_rows(scoring)
    summary_header = "| " + " | ".join(label for _, label, _ in SCORING_SUMMARY_CARD_COLUMNS) + " |"
    summary_divider = "|" + "|".join(["---"] * len(SCORING_SUMMARY_CARD_COLUMNS)) + "|"
    summary_body = "\n".join(
        "| "
        + " | ".join(
            _card_summary_value(row, value_column, threshold_column)
            for value_column, _, threshold_column in SCORING_SUMMARY_CARD_COLUMNS
        )
        + " |"
        for row in summary
    )
    setup_header = (
        "| model | handling | relevance score / decision rule | sequence length (tokens) | "
        "dtype / batch / seed | revision |"
    )
    setup_divider = "|" + "|".join(["---"] * 6) + "|"
    setup_body = "\n".join(_scoring_setup_row(model_runs) for model_runs in _by_model(scoring))
    return f"""
## Scoring models

Each model returns a normalized `yes` relevance score in [0, 1]. Best values below use
thresholds selected on this benchmark, so read them as an upper bound. All tested
thresholds and their metrics are in [`threshold_sweep.csv`](threshold_sweep.csv).

### Scoring setup

{setup_header}
{setup_divider}
{setup_body}

### Scoring prompt

```text
{prompt_text}```

### Best thresholded scoring metrics

{summary_header}
{summary_divider}
{summary_body}

Each metric is shown at its own best threshold; ROC-AUC is threshold-independent.
Scoring prompt SHA-256: `{prompt_sha256}`.
"""


def _by_model(results: Sequence[RunResult]) -> tuple[tuple[RunResult, ...], ...]:
    """Group scoring rows by model in stable order for the compact setup table."""
    grouped: dict[str, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.metadata.model_id, []).append(result)
    return tuple(tuple(grouped[model_id]) for model_id in sorted(grouped))


def _scoring_setup_row(results: Sequence[RunResult]) -> str:
    """Render the reproducibility-relevant adapter settings for one model."""
    model_id = results[0].metadata.model_id
    family, input_handling, score = _SCORER_CARD_CONFIG.get(
        model_id, ("scoring adapter", "recorded prompt + sentence", "normalized yes score")
    )
    try:
        rule = scorer_for(model_id).decision_rule
    except KeyError:
        rule = results[0].metadata.decision_rule or "recorded per run"
    sequence = _setting_or_varying(results, "sequence length", lambda r: r.metadata.sequence_length)
    if sequence == "None":
        sequence = "model-defined"
    dtype = _setting_or_varying(results, "dtype", lambda r: r.metadata.dtype)
    batch = _setting_or_varying(results, "batch size", lambda r: r.metadata.batch_size)
    seed = _setting_or_varying(results, "seed", lambda r: r.metadata.seed)
    revision = _setting_or_varying(results, "revision", lambda r: r.metadata.model_revision)
    cells = (
        model_id,
        f"{family}: {input_handling}",
        f"{score}; {rule}",
        sequence,
        f"{dtype}; batch {batch}; seed {seed}",
        revision,
    )
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _card_summary_value(
    row: dict[str, Any], value_column: str, threshold_column: str | None
) -> str:
    """Format compact card values without changing the downloadable CSV schema."""
    value = row.get(value_column)
    if value is None:
        return "n/a"
    if threshold_column is not None:
        threshold = row[threshold_column]
        return f"{_format_card_float(float(value))} @ {_format_card_threshold(float(threshold))}"
    if value_column == "peak_vram_bytes_max":
        return f"{float(value) / 1024**3:.2f}"
    if value_column == "throughput_items_per_second_macro":
        return f"{float(value):.2f}"
    return str(value) if isinstance(value, str | int) else _format_card_float(float(value))


def _format_card_float(value: float) -> str:
    """Format a score to four useful decimals, omitting redundant zeroes."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_card_threshold(value: float) -> str:
    """Keep small score boundaries readable without losing significant digits."""
    return f"{value:.6g}"


def _setting_or_varying(results: Sequence[RunResult], name: str, value_of: Any) -> str:
    """Describe a setting without rejecting a mixed, explicitly recorded roster."""
    values = sorted({value_of(result) for result in results}, key=str)
    if len(values) == 1:
        return str(values[0])
    return f"varies by model ({name} is recorded per run)"


def write_scoring_plots(results: Sequence[RunResult], directory: Path) -> tuple[Path, ...]:
    """Write deterministic SVG summaries for scoring quality and performance."""
    summary = scoring_summary_rows(results)
    if not summary:
        return ()
    directory.mkdir(parents=True, exist_ok=True)
    quality = directory / "quality_metrics.svg"
    performance = directory / "performance.svg"
    quality.write_text(_quality_svg(summary), encoding="utf-8")
    performance.write_text(_performance_svg(summary), encoding="utf-8")
    return quality, performance


def _quality_svg(summary: Sequence[dict[str, Any]]) -> str:
    """Build a compact horizontal bar chart with fixed geometry and ordering."""
    rows = sorted(summary, key=lambda row: str(row["model_id"]))
    metrics = (
        ("MCC", "best_mcc", -1.0, 1.0),
        ("F1", "best_f1", 0.0, 1.0),
        ("Balanced accuracy", "best_balanced_accuracy", 0.0, 1.0),
        ("Precision", "best_precision", 0.0, 1.0),
        ("Recall", "best_recall", 0.0, 1.0),
        ("ROC-AUC", "roc_auc_macro", 0.0, 1.0),
    )
    colors = ("#2563eb", "#ea580c", "#059669", "#7c3aed", "#ca8a04")
    width, left, right, top, row_height, bottom = 960, 220, 28, 76, 62, 34
    height = top + row_height * len(metrics) + bottom
    plot_width = width - left - right
    fragments = [
        _svg_open(width, height, "Best thresholded scoring metrics"),
        _svg_text("Best thresholded scoring metrics", 24, 32, size=20, weight="bold"),
        _svg_text("Higher is better; MCC uses a -1 to 1 scale.", 24, 54, size=12),
    ]
    legend_x = left
    for index, row in enumerate(rows):
        color = colors[index % len(colors)]
        label = str(row["model_id"])
        fragments.append(f'<rect x="{legend_x}" y="38" width="12" height="12" fill="{color}"/>')
        fragments.append(_svg_text(label, legend_x + 18, 49, size=12))
        legend_x += 18 + max(90, len(label) * 7)
    for metric_index, (label, key, minimum, maximum) in enumerate(metrics):
        y = top + metric_index * row_height
        fragments.append(_svg_text(label, 24, y + 18, size=13, weight="bold"))
        for tick in _ticks(minimum, maximum):
            x = left + _scale(tick, minimum, maximum) * plot_width
            fragments.append(
                f'<line x1="{x:.2f}" y1="{y + 24}" x2="{x:.2f}" '
                f'y2="{y + row_height - 8}" stroke="#e5e7eb"/>'
            )
            fragments.append(
                _svg_text(_format_number(tick), x, y + row_height - 2, size=10, anchor="middle")
            )
        for model_index, row in enumerate(rows):
            value = _as_float(row.get(key))
            bar_y = y + 29 + model_index * 14
            if value is None:
                fragments.append(_svg_text("n/a", left, bar_y + 10, size=10))
                continue
            clamped = min(maximum, max(minimum, value))
            x_zero = left + _scale(0.0, minimum, maximum) * plot_width
            x_value = left + _scale(clamped, minimum, maximum) * plot_width
            fragments.append(
                f'<rect x="{min(x_zero, x_value):.2f}" y="{bar_y}" '
                f'width="{abs(x_value - x_zero):.2f}" height="10" '
                f'fill="{colors[model_index % len(colors)]}" rx="2"/>'
            )
            value_x = min(plot_width + left - 2, max(left + 2, x_value + 5))
            fragments.append(_svg_text(_format_number(value), value_x, bar_y + 9, size=10))
    fragments.append("</svg>")
    return "".join(fragments) + "\n"


def _performance_svg(summary: Sequence[dict[str, Any]]) -> str:
    """Build deterministic throughput and VRAM panels with independent scales."""
    rows = sorted(summary, key=lambda row: str(row["model_id"]))
    width, left, right, top = 960, 220, 28, 74
    panel_height, panel_gap, bottom = 112, 32, 30
    height = top + 2 * panel_height + panel_gap + bottom
    plot_width = width - left - right
    panels = (
        ("Throughput (items / s)", "throughput_items_per_second_macro", 1.0),
        ("Peak VRAM (GiB)", "peak_vram_bytes_max", 1024**3),
    )
    colors = ("#2563eb", "#ea580c", "#059669", "#7c3aed", "#ca8a04")
    fragments = [
        _svg_open(width, height, "Inference performance"),
        _svg_text("Inference performance", 24, 32, size=20, weight="bold"),
        _svg_text("Macro throughput and maximum allocated VRAM.", 24, 54, size=12),
    ]
    for panel_index, (label, key, divisor) in enumerate(panels):
        y = top + panel_index * (panel_height + panel_gap)
        numeric_values = [_as_float(row.get(key)) for row in rows]
        values = [value / divisor for value in numeric_values if value is not None]
        maximum = max(max(values, default=1.0), 1.0)
        fragments.append(_svg_text(label, 24, y + 16, size=13, weight="bold"))
        for tick in _ticks(0.0, maximum, count=3):
            x = left + _scale(tick, 0.0, maximum) * plot_width
            fragments.append(
                f'<line x1="{x:.2f}" y1="{y + 22}" x2="{x:.2f}" '
                f'y2="{y + panel_height - 8}" stroke="#e5e7eb"/>'
            )
            fragments.append(
                _svg_text(_format_number(tick), x, y + panel_height - 2, size=10, anchor="middle")
            )
        for row_index, row in enumerate(rows):
            raw_value = _as_float(row.get(key))
            value = None if raw_value is None else raw_value / divisor
            bar_y = y + 28 + row_index * 16
            if value is None:
                fragments.append(_svg_text(f"{row['model_id']}: n/a", left, bar_y + 10, size=10))
                continue
            bar_width = _scale(value, 0.0, maximum) * plot_width
            fragments.append(
                f'<rect x="{left}" y="{bar_y}" width="{bar_width:.2f}" height="11" '
                f'fill="{colors[row_index % len(colors)]}" rx="2"/>'
            )
            fragments.append(
                _svg_text(
                    f"{row['model_id']}  {_format_number(value)}",
                    min(width - right, left + bar_width + 6),
                    bar_y + 10,
                    size=10,
                )
            )
    fragments.append("</svg>")
    return "".join(fragments) + "\n"


def _svg_open(width: int, height: int, title: str) -> str:
    escaped = html.escape(title, quote=True)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">'
        f'<title id="title">{escaped}</title><rect width="100%" height="100%" fill="white"/>'
    )


def _svg_text(
    value: str,
    x: float,
    y: float,
    *,
    size: int,
    weight: str = "normal",
    anchor: str = "start",
) -> str:
    escaped = html.escape(str(value), quote=True)
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#111827">{escaped}</text>'
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _scale(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _ticks(minimum: float, maximum: float, *, count: int = 5) -> tuple[float, ...]:
    if count <= 1:
        return (minimum,)
    step = (maximum - minimum) / (count - 1)
    return tuple(minimum + step * index for index in range(count))


def _format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def publish_results(
    repo_id: str,
    results_dir: Path,
    results: Sequence[RunResult],
    *,
    api: DatasetHub | None = None,
    private: bool = False,
    commit_message: str = "Publish small-LLM land-use relevance benchmark results",
    benchmark_name: str = "v3-multilingual",
    prompt_text: str,
    scorer_prompt_text: str = "",
    data_root: Path | None = None,
) -> str:
    """Write the card next to the results, then push the whole folder to the Hub."""
    if not results:
        raise ValueError("refusing to publish an empty set of results")
    _reject_archive_paths(results_dir)
    hub = api if api is not None else _default_api()
    results_dir.mkdir(parents=True, exist_ok=True)
    viewer_file = ""
    if data_root is not None:
        write_viewer_dataset(data_root, results_dir / "data" / "train.csv")
        viewer_file = "data/train.csv"
    plot_paths = write_scoring_plots(results, results_dir / "plots")
    plot_files = tuple(path.relative_to(results_dir).as_posix() for path in plot_paths)
    (results_dir / "README.md").write_text(
        dataset_card(
            results,
            benchmark_name=benchmark_name,
            prompt_text=prompt_text,
            scorer_prompt_text=scorer_prompt_text,
            viewer_file=viewer_file,
            plot_files=plot_files,
        ),
        encoding="utf-8",
    )
    hub.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    hub.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(results_dir),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def _default_api() -> DatasetHub:
    from huggingface_hub import HfApi

    # HfApi satisfies DatasetHub in practice; its **kwargs signatures are wider than
    # the protocol can express.
    return cast(DatasetHub, HfApi())


def _reject_archive_paths(results_dir: Path) -> None:
    paths = results_dir.rglob("*") if results_dir.exists() else ()
    for path in paths:
        if "archive" in path.relative_to(results_dir).parts:
            raise ValueError(f"refusing to upload archive path: {path}")


SCORING_SUMMARY_CARD_COLUMNS = (
    ("model_id", "model", None),
    ("language_count", "languages", None),
    ("best_mcc", "MCC @ threshold", "best_mcc_threshold"),
    ("best_f1", "F1 @ threshold", "best_f1_threshold"),
    (
        "best_balanced_accuracy",
        "balanced accuracy @ threshold",
        "best_balanced_accuracy_threshold",
    ),
    ("best_precision", "precision @ threshold", "best_precision_threshold"),
    ("best_recall", "recall @ threshold", "best_recall_threshold"),
    ("roc_auc_macro", "ROC-AUC", None),
    ("throughput_items_per_second_macro", "items/s", None),
    ("peak_vram_bytes_max", "peak VRAM (GiB)", None),
)


def write_viewer_dataset(data_root: Path, output: Path) -> Path:
    """Export the validated multilingual benchmark as one Hub-viewable CSV."""
    import csv

    from landuse_relevance_bench.adapters.translations import (
        load_language_benchmark,
        load_manifest,
    )

    viewer_columns = (
        "item_id",
        "source_item_id",
        "language",
        "sentence",
        "label",
        "polygon_name",
        "h3_cell",
        "latitude",
        "longitude",
        "source",
        "region",
        "source_url",
    )
    manifest = load_manifest(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(viewer_columns), lineterminator="\n")
        writer.writeheader()
        for language in manifest.languages:
            items = load_language_benchmark(data_root, language)
            source_path = data_root / manifest.files[language].path
            with source_path.open(newline="", encoding="utf-8") as source_handle:
                rows = list(csv.DictReader(source_handle))
            if len(rows) != len(items):
                raise ValueError(
                    f"viewer export row mismatch for {language}: {len(rows)} != {len(items)}"
                )
            for item, source_row in zip(items, rows, strict=True):
                row = {column: source_row.get(column, "") or "" for column in viewer_columns}
                row.update(
                    item_id=item.item_id,
                    source_item_id=item.source_item_id,
                    language=item.language,
                    sentence=item.sentence,
                    label=item.label.value,
                )
                writer.writerow(row)
    return output
