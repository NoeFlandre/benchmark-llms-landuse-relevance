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


def survivors() -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "mutmut", "results"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    return [line.strip() for line in output.splitlines() if SURVIVED.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-survivors", type=int, default=0)
    args = parser.parse_args()

    lines = survivors()
    for line in lines:
        print(line)
    if len(lines) > args.max_survivors:
        print(f"\n{len(lines)} surviving mutant(s), allowance is {args.max_survivors}")
        return 1
    print(f"\n{len(lines)} surviving mutant(s), within the allowance of {args.max_survivors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
