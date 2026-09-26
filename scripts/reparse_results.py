"""Recompute stored verdicts, parse modes, and metrics from raw generations."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.hf_publish import dataset_card
from landuse_relevance_bench.adapters.results_store import read_runs, write_leaderboard_csv
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.parsing import parse_with_mode

_LEGACY_PACKAGE_VERSIONS = {
    "c6d9085a00003e3668714ff1d4d309e6e34836e7": "0.1.0",
    "2a6b33770a970b29ab8776d892666d41f103524e": "0.1.0",
    "c9ea3d4": "0.1.0",
}


def reparse_directory(directory: Path) -> dict[str, tuple[int, int]]:
    """Reparse every saved run recursively and rebuild committed summary artifacts."""
    if not directory.is_dir():
        raise ValueError(f"results directory does not exist: {directory}")
    changes: dict[str, tuple[int, int]] = {}
    run_paths = sorted(directory.rglob("*.json"))
    for path in run_paths:
        payload = _read_payload(path)
        metadata = payload.setdefault("metadata", {})
        if not metadata.get("package_version"):
            metadata["package_version"] = _LEGACY_PACKAGE_VERSIONS.get(
                metadata.get("source_commit", ""), ""
            )
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError(f"{path} has no predictions list")
        changed = 0
        outcomes = []
        modes: Counter[str] = Counter()
        for prediction in predictions:
            before = prediction.get("predicted")
            parsed = parse_with_mode(prediction["raw_output"])
            selected = None if prediction.get("truncated", False) else parsed.label
            mode = None if prediction.get("truncated", False) else parsed.mode
            after = None if selected is None else selected.value
            changed += before != after
            prediction["predicted"] = after
            prediction["parse_mode"] = mode
            modes[mode or ("truncated" if prediction.get("truncated", False) else "unparsed")] += 1
            outcomes.append(
                (Label(prediction["expected"]), None if after is None else Label(after))
            )
        payload["metrics"] = evaluate(outcomes).to_dict()
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        changes[path.relative_to(directory).as_posix()] = (changed, len(predictions))
        print(
            f"{path.relative_to(directory)}: {changed}/{len(predictions)} verdicts changed; "
            + ", ".join(f"{mode}={count}" for mode, count in sorted(modes.items()))
        )

    groups = [directory, *(path.parent for path in run_paths)]
    for folder in dict.fromkeys(groups):
        direct = read_runs(folder)
        if direct:
            write_leaderboard_csv(direct, folder / "leaderboard.csv")
    all_runs = read_runs(directory, recursive=True)
    if all_runs:
        # The root board and card summarize both current runs and dated historical groups.
        write_leaderboard_csv(all_runs, directory / "leaderboard.csv")
        (directory / "README.md").write_text(
            dataset_card(all_runs, benchmark_name="benchmark.csv", allow_mixed=True),
            encoding="utf-8",
        )
    return changes


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read run file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Root directory of saved run JSON files.")
    args = parser.parse_args()
    reparse_directory(args.results_dir)


if __name__ == "__main__":
    main()
