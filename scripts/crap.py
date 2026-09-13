"""Report the CRAP score of every domain function and fail if any exceeds a limit.

CRAP(m) = complexity(m)^2 * (1 - coverage(m))^3 + complexity(m)

Reads `coverage.json` (written by `pytest --cov-report=json`) and radon's cyclomatic
complexity, joining them on the line ranges of each function.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = "src/landuse_relevance_bench/domain"


def crap(complexity: int, coverage: float) -> float:
    return complexity**2 * (1 - coverage) ** 3 + complexity


def radon_blocks(target: Path) -> dict[str, list[dict]]:
    completed = subprocess.run(
        [sys.executable, "-m", "radon", "cc", "-s", "-j", str(target)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def coverage_by_file(report: Path) -> dict[str, dict[str, set[int]]]:
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {
        _normalise(name): {
            "executed": set(data.get("executed_lines", [])),
            "missing": set(data.get("missing_lines", [])),
        }
        for name, data in payload["files"].items()
    }


def _normalise(name: str) -> str:
    return str(Path(name).resolve().relative_to(PROJECT_ROOT)) if Path(name).is_absolute() else name


def scores(target: Path, report: Path) -> list[tuple[str, int, float, float]]:
    coverage = coverage_by_file(report)
    rows: list[tuple[str, int, float, float]] = []
    for filename, blocks in radon_blocks(target).items():
        lines = coverage.get(_normalise(filename))
        if lines is None:
            continue
        for block in blocks:
            span = range(block["lineno"], block["endline"] + 1)
            executed = len(lines["executed"] & set(span))
            missed = len(lines["missing"] & set(span))
            measurable = executed + missed
            ratio = executed / measurable if measurable else 1.0
            rows.append(
                (
                    f"{filename}:{block['name']}",
                    block["complexity"],
                    ratio,
                    crap(block["complexity"], ratio),
                )
            )
    return sorted(rows, key=lambda row: -row[3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=float, default=6.0, help="Fail above this CRAP score.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--coverage-json", default="coverage.json")
    args = parser.parse_args()

    rows = scores(PROJECT_ROOT / args.target, PROJECT_ROOT / args.coverage_json)
    if not rows:
        print("no functions measured; did the coverage run produce coverage.json?")
        return 1
    width = max(len(name) for name, *_ in rows)
    print(f"{'function'.ljust(width)}  cc   cov     crap")
    for name, complexity, ratio, score in rows:
        print(f"{name.ljust(width)}  {complexity:<3}  {ratio:5.1%}  {score:6.2f}")

    offenders = [row for row in rows if row[3] > args.max]
    if offenders:
        print(f"\nCRAP above {args.max}: " + ", ".join(name for name, *_ in offenders))
        return 1
    print(f"\nall {len(rows)} functions are at or below a CRAP score of {args.max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
