"""Plan weighted Grid'5000 shards and verify their merged result tree."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from landuse_relevance_bench.adapters.g5k_sites import SiteConfig, validate_site_configs
from landuse_relevance_bench.adapters.results_store import read_run, write_run
from landuse_relevance_bench.adapters.translations import load_manifest
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.roster import model_ids
from landuse_relevance_bench.domain.scorers import scorer_ids
from landuse_relevance_bench.domain.sharding import Pair, model_language_pairs


class CollectionError(ValueError):
    """Raised when site result trees cannot form one complete active sweep."""


@dataclass(frozen=True, slots=True)
class CollectionReport:
    """Evidence produced after a complete validated merge."""

    pairs: tuple[Pair, ...]
    source_commit: str


def allocate_pairs(
    pairs: Sequence[Pair], sites: Sequence[SiteConfig]
) -> dict[str, tuple[Pair, ...]]:
    """Allocate pairs by weighted deterministic round-robin slots."""
    canonical = _canonical_pairs(pairs)
    checked = validate_site_configs(sites)
    slots = [site.name for site in checked for _ in range(site.weight)]
    allocation: dict[str, list[Pair]] = {site.name: [] for site in checked}
    for index, pair in enumerate(canonical):
        allocation[slots[index % len(slots)]].append(pair)
    return {name: tuple(site_pairs) for name, site_pairs in allocation.items()}


def expected_pairs_for_models(models: Sequence[str], languages: Sequence[str]) -> tuple[Pair, ...]:
    """Build a stable collection target for an explicit model subset."""
    return tuple(model_language_pairs(tuple(models), tuple(languages)))


def collect_results(
    site_roots: Mapping[str, Path],
    *,
    expected_pairs: Sequence[Pair],
    output: Path,
    expected_source_commit: str | None = None,
    force: bool = False,
) -> CollectionReport:
    """Validate every site result and atomically merge a complete pair union."""
    expected = _canonical_pairs(expected_pairs)
    found, seen_commit = _read_site_results(
        site_roots, expected=set(expected), expected_source_commit=expected_source_commit
    )
    source_commit = _complete_source_commit(expected, found, seen_commit)
    _install_results(output, found, force=force)
    return CollectionReport(pairs=expected, source_commit=source_commit)


def _read_site_results(
    site_roots: Mapping[str, Path],
    *,
    expected: set[Pair],
    expected_source_commit: str | None,
) -> tuple[dict[Pair, RunResult], str | None]:
    """Read and validate each site's results, returning them with the one commit seen."""
    found: dict[Pair, RunResult] = {}
    source_commit: str | None = None
    for site_name in sorted(site_roots):
        for path in _site_result_paths(site_name, site_roots[site_name]):
            result = _read_site_result(site_name, path)
            pair = (result.metadata.model_id, result.metadata.language)
            _check_pair(site_name, pair, expected=expected, found=found)
            source_commit = _checked_source_commit(
                pair,
                result.metadata.source_commit,
                expected_source_commit=expected_source_commit,
                seen_commit=source_commit,
            )
            found[pair] = result
    return found, source_commit


def _site_result_paths(site_name: str, root: Path) -> Iterator[Path]:
    """Yield the result files of one site, refusing archives and missing roots lazily."""
    if not root.is_dir():
        raise CollectionError(f"site {site_name!r} result root does not exist: {root}")
    for path in sorted(root.rglob("*.json")):
        if "archive" in path.relative_to(root).parts:
            raise CollectionError(f"site {site_name!r} contains an archive result: {path}")
        yield path


def _read_site_result(site_name: str, path: Path) -> RunResult:
    try:
        return read_run(path)
    except ValueError as exc:
        raise CollectionError(f"invalid result at site {site_name!r}: {path}: {exc}") from exc


def _check_pair(
    site_name: str, pair: Pair, *, expected: set[Pair], found: Mapping[Pair, RunResult]
) -> None:
    if pair not in expected:
        raise CollectionError(f"unexpected pair {pair!r} at site {site_name!r}")
    if pair in found:
        raise CollectionError(f"duplicate pair {pair!r} across site result trees")


def _checked_source_commit(
    pair: Pair,
    commit: str,
    *,
    expected_source_commit: str | None,
    seen_commit: str | None,
) -> str:
    """Return the commit every result must share, or raise if this pair breaks that."""
    if not commit:
        raise CollectionError(f"pair {pair!r} has no source commit")
    if expected_source_commit is not None and commit != expected_source_commit:
        raise CollectionError(
            f"pair {pair!r} has source commit {commit!r}; expected {expected_source_commit!r}"
        )
    if seen_commit is not None and commit != seen_commit:
        raise CollectionError(f"mixed source commit values: {seen_commit!r} and {commit!r}")
    return commit


def _complete_source_commit(
    expected: Sequence[Pair], found: Mapping[Pair, RunResult], seen_commit: str | None
) -> str:
    """Require every expected pair to be present, and return the shared source commit."""
    missing = sorted(set(expected) - set(found))
    if missing:
        formatted = ", ".join(f"{model_id}[{language}]" for model_id, language in missing)
        raise CollectionError(f"missing pairs: {formatted}")
    if seen_commit is None:
        raise CollectionError("no result pairs were found")
    return seen_commit


def _install_results(output: Path, results: Mapping[Pair, RunResult], *, force: bool) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_dir():
        raise CollectionError(f"merge output is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not force:
        raise CollectionError(f"merge output is not empty: {output}; pass --force to replace it")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for result in results.values():
            write_run(result, stage)
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _canonical_pairs(pairs: Sequence[Pair]) -> tuple[Pair, ...]:
    canonical = tuple(sorted(pairs))
    if len(set(canonical)) != len(canonical):
        raise CollectionError("expected pair list contains duplicates")
    return canonical


def _parse_site(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("site must have the form NAME=RESULT_DIRECTORY")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", required=True, type=_parse_site)
    parser.add_argument("--data-root", type=Path, default=Path("data/translations"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Collect only this model; repeat for a mixed generative/scoring target.",
    )
    parser.add_argument(
        "--scoring-only",
        action="store_true",
        help="Collect the complete scoring roster instead of the generative roster.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.data_root)
    if args.model_ids and args.scoring_only:
        parser.error("--model-id and --scoring-only cannot be combined")
    selected_models = (
        tuple(args.model_ids)
        if args.model_ids
        else scorer_ids()
        if args.scoring_only
        else model_ids()
    )
    expected_pairs = expected_pairs_for_models(selected_models, manifest.languages)
    site_roots = dict(args.site)
    report = collect_results(
        site_roots,
        expected_pairs=expected_pairs,
        output=args.output,
        expected_source_commit=args.source_commit,
        force=args.force,
    )
    print(f"merged {len(report.pairs)} pairs from source commit {report.source_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
