"""Regenerate the published snapshot status from a merged result tree."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from landuse_relevance_bench.adapters.results_store import read_runs
from landuse_relevance_bench.domain.roster import model_ids
from landuse_relevance_bench.domain.scorers import scorer_ids


def snapshot_status(
    results_dir: Path,
    *,
    benchmark_name: str,
    expected_model_ids: tuple[str, ...] | None = None,
) -> str:
    """Describe the coverage of ``results_dir`` without interpreting its scores."""
    runs = read_runs(results_dir)
    if not runs:
        raise ValueError(f"no run results found in {results_dir}")
    roster = tuple(expected_model_ids or model_ids() + scorer_ids())
    scorer_set = set(scorer_ids())
    generative_roster = tuple(model_id for model_id in roster if model_id not in scorer_set)
    scorers = tuple(model_id for model_id in roster if model_id in scorer_set)
    languages = sorted({run.metadata.language for run in runs})
    per_model = Counter(run.metadata.model_id for run in runs)
    items = {run.metrics.n_items for run in runs}
    if len(items) != 1:
        raise ValueError(f"runs disagree on predictions per run: {sorted(items)}")
    total = len(roster) * len(languages)
    complete = len(runs) == total
    lines = [
        "# Snapshot status",
        "",
        f"- Benchmark: `{benchmark_name}`",
        f"- Status: {'complete' if complete else 'in progress; not the final release'}.",
        f"- Completed model-language runs: {len(runs):,} of {total:,}",
        f"- Models represented: {len(per_model)} of {len(roster)}"
        f" ({len(generative_roster)} generative, {len(scorers)} scoring)",
        f"- Languages represented: {len(languages)}",
        f"- Predictions per run: {items.pop()}",
    ]

    def group(title: str, ids: tuple[str, ...]) -> None:
        if not ids:
            return
        lines.append("")
        lines.append(f"## {title}")
        for model_id in ids:
            count = per_model.get(model_id, 0)
            state = "complete" if count == len(languages) else "in progress"
            lines.append(f"- `{model_id}`: {state}, {count}/{len(languages)} languages")

    group("Generative models", generative_roster)
    group("Scoring models", scorers)
    lines += [
        "",
        "This snapshot contains only valid atomic checkpoints available at publication time.",
        "The result files retain their exact run metadata and provenance.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--benchmark-name", default="v3-multilingual")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help=(
            "Expected release model; repeat for a public roster larger than the local code roster."
        ),
    )
    arguments = parser.parse_args()
    target = arguments.results_dir / "SNAPSHOT_STATUS.md"
    target.write_text(
        snapshot_status(
            arguments.results_dir,
            benchmark_name=arguments.benchmark_name,
            expected_model_ids=tuple(arguments.model_ids) if arguments.model_ids else None,
        ),
        encoding="utf-8",
    )
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
