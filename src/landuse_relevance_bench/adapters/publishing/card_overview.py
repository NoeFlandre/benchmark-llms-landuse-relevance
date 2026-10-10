"""The card's opening sections: scale, package version, shared settings and the aggregate table."""

from collections.abc import Sequence

from landuse_relevance_bench.adapters.publishing.card_format import metric_rankings, ranked_cell
from landuse_relevance_bench.adapters.publishing.card_validation import uniform
from landuse_relevance_bench.adapters.results_store import aggregate_rows
from landuse_relevance_bench.domain.records import RunResult

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


def package_version_section(results: Sequence[RunResult]) -> str:
    versions = sorted(
        {result.metadata.package_version for result in results if result.metadata.package_version}
    )
    if not versions:
        return ""
    label = "Package version" if len(versions) == 1 else "Package versions"
    values = ", ".join(f"`{version}`" for version in versions)
    missing_note = (
        " (some runs lack version metadata)"
        if any(not result.metadata.package_version for result in results)
        else ""
    )
    return f"\n\n{label} recorded in run metadata: {values}{missing_note}."


def generation_settings(generative: Sequence[RunResult]) -> str:
    """The prompt-and-settings line: decoding, seed, budget, precision, batch, quants."""
    decoding = uniform(generative, "decoding", lambda r: r.metadata.decoding)
    # A GGUF quant has no torch dtype; its precision is its recorded quant label.
    full_precision = [r for r in generative if not r.metadata.quantization]
    dtype = uniform(full_precision or generative, "dtype", lambda r: r.metadata.dtype)
    quantized = sorted(
        {(r.metadata.model_id, r.metadata.quantization) for r in generative}
        - {(r.metadata.model_id, "") for r in generative}
    )
    quant_line = "".join(
        f"\n`{model_id}` runs the `{quant}` GGUF quant through llama.cpp (same prompt, "
        "template, greedy decoding and budget)."
        for model_id, quant in quantized
    )
    seed = uniform(generative, "seed", lambda r: r.metadata.seed)
    max_new_tokens = uniform(generative, "token budget", lambda r: r.metadata.max_new_tokens)
    batch_sizes = sorted({result.metadata.batch_size for result in generative})
    batch_size_line = (
        f"batch {batch_sizes[0]}" if len(batch_sizes) == 1 else "batch varies by model"
    )
    return (
        f"English prompt · {decoding} decoding · seed {seed} · "
        f"`max_new_tokens={max_new_tokens}` ·\n"
        f"`{dtype}` · {batch_size_line}.{quant_line}"
    )


def scale_line(results: Sequence[RunResult], benchmark_name: str) -> str:
    n_items = uniform(results, "items per language", lambda r: r.metrics.n_items)
    languages = sorted({result.metadata.language for result in results})
    language_label = "language" if len(languages) == 1 else "languages"
    item_label = "item" if n_items == 1 else "items"
    return (
        f"`{benchmark_name}` · {len(languages)} {language_label} x {n_items} "
        f"{item_label}/language ·\n"
        f"{len(languages) * n_items:,} items · binary `yes`/`no` labels."
    )


def aggregate_table(generative: Sequence[RunResult]) -> str:
    """The per-model macro table, with best and second-best cells emphasised."""
    header = "| " + " | ".join(CARD_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(CARD_COLUMNS)) + "|"
    rows = aggregate_rows(generative)
    metric_columns = tuple(
        column for column in CARD_COLUMNS if column not in {"model_id", "language_count"}
    )
    rankings = metric_rankings(rows, metric_columns)
    body = "\n".join(
        "| "
        + " | ".join(
            ranked_cell(row, column, str(row[column]), rankings) for column in CARD_COLUMNS
        )
        + " |"
        for row in rows
    )
    return f"{header}\n{divider}\n{body}"
