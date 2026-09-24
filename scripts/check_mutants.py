"""Fail the build when mutation testing leaves too many survivors.

`mutmut run` exits non-zero whenever any mutant survives, which makes it unusable as a
gate on its own. This reads the result summary instead, so a run can be accepted with an
explicit, reviewed allowance.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURVIVED = re.compile(r"\bsurvived\b", re.IGNORECASE)


class MutationRunError(RuntimeError):
    """Raised when there is no trustworthy mutation result to gate on."""


def survivors(root: Path = PROJECT_ROOT) -> list[str]:
    """Surviving mutants, refusing to report zero when the run never produced results.

    A crashed `mutmut run` leaves no result metadata, and `mutmut results` then lists no
    survivors; without this check the crash would pass the gate.
    """
    if not any((root / "mutants").rglob("*.meta")):
        raise MutationRunError("no mutation results found; did `mutmut run` crash?")
    completed = subprocess.run(
        [sys.executable, "-m", "mutmut", "results"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise MutationRunError(f"`mutmut results` failed:\n{output}")
    return [line.strip() for line in output.splitlines() if SURVIVED.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-survivors", type=int, default=0)
    args = parser.parse_args()

    try:
        lines = survivors()
    except MutationRunError as exc:
        print(exc)
        return 2
    for line in lines:
        print(line)
    if len(lines) > args.max_survivors:
        print(f"\n{len(lines)} surviving mutant(s), allowance is {args.max_survivors}")
        return 1
    print(f"\n{len(lines)} surviving mutant(s), within the allowance of {args.max_survivors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
