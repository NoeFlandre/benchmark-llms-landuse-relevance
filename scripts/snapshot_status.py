"""Regenerate the published snapshot status from a merged result tree."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from landuse_relevance_bench.adapters.results_store import read_runs
from landuse_relevance_bench.domain.roster import model_ids
from landuse_relevance_bench.domain.scorers import scorer_ids


def snapshot_status(
    results_dir: Path,
    *,
    benchmark_name: str,
    expected_model_ids: tuple[str, ...] | None = None,
    expected_languages_by_model: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Describe the coverage of ``results_dir`` without interpreting its scores."""
    runs = read_runs(results_dir)
    if not runs:
        raise ValueError(f"no run results found in {results_dir}")
    roster = tuple(expected_model_ids or model_ids() + scorer_ids())
    language_overrides = expected_languages_by_model or {}
    unknown_overrides = set(language_overrides) - set(roster)
    if unknown_overrides:
        raise ValueError(
            "language overrides reference unrostered models: "
            f"{sorted(unknown_overrides)}"
        )
    scorer_set = set(scorer_ids())
    generative_roster = tuple(model_id for model_id in roster if model_id not in scorer_set)
    scorers = tuple(model_id for model_id in roster if model_id in scorer_set)
    languages = sorted({run.metadata.language for run in runs})
    # Count by run name: SGLang, DSpark and log-prob runs share their target's model_id.
    per_model = Counter(run.metadata.name for run in runs)
    observed_languages_by_model: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        observed_languages_by_model[run.metadata.name].add(run.metadata.language)
    expected_languages = {
        model_id: frozenset(language_overrides.get(model_id, languages))
        for model_id in roster
    }
    if any(not model_languages for model_languages in expected_languages.values()):
        raise ValueError("expected languages must not be empty")
    items = {run.metrics.n_items for run in runs}
    if len(items) != 1:
        raise ValueError(f"runs disagree on predictions per run: {sorted(items)}")
    expected_pairs = {
        (model_id, language)
        for model_id, model_languages in expected_languages.items()
        for language in model_languages
    }
    observed_pairs = {(run.metadata.name, run.metadata.language) for run in runs}
    total = len(expected_pairs)
    complete = len(runs) == total and observed_pairs == expected_pairs
    lines = [
        "# Snapshot status",
        "",
        f"- Benchmark: `{benchmark_name}`",
        f"- Status: {'complete' if complete else 'in progress; not the final release'}.",
        f"- Completed model-language runs: {len(runs):,} of {total:,}",
        f"- Models represented: {len(set(per_model) & set(roster))} of {len(roster)}"
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
            expected = expected_languages[model_id]
            observed = observed_languages_by_model.get(model_id, set())
            state = "complete" if observed == expected else "in progress"
            lines.append(
                f"- `{model_id}`: {state}, {len(observed)}/{len(expected)} languages"
            )

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
    parser.add_argument(
        "--model-languages",
        action="append",
        default=[],
        metavar="MODEL=LANG,LANG",
        help="Supported language subset for one model; default is every represented language.",
    )
    arguments = parser.parse_args()
    language_overrides: dict[str, tuple[str, ...]] = {}
    for value in arguments.model_languages:
        model_id, separator, language_list = value.partition("=")
        languages_for_model = tuple(language.strip() for language in language_list.split(","))
        if (
            not separator
            or not model_id
            or not language_list
            or any(not language for language in languages_for_model)
            or len(set(languages_for_model)) != len(languages_for_model)
        ):
            parser.error("--model-languages must be MODEL=LANG,LANG with unique non-empty codes")
        if model_id in language_overrides:
            parser.error(f"--model-languages was repeated for {model_id}")
        language_overrides[model_id] = languages_for_model
    target = arguments.results_dir / "SNAPSHOT_STATUS.md"
    target.write_text(
        snapshot_status(
            arguments.results_dir,
            benchmark_name=arguments.benchmark_name,
            expected_model_ids=tuple(arguments.model_ids) if arguments.model_ids else None,
            expected_languages_by_model=language_overrides,
        ),
        encoding="utf-8",
    )
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
