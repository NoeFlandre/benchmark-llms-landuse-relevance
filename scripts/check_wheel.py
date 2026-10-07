"""Fail when the built wheel does not ship the py.typed marker.

The source-tree test only proves the file exists; a packaging exclusion could still
drop it from the archive. This builds a fresh wheel with the project's build backend
(or inspects one given on the command line) and checks the archive itself.
"""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER = "landuse_relevance_bench/py.typed"


def find_wheel(directory: Path) -> Path:
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one wheel in {directory}, found {len(wheels)}")
    return wheels[0]


def require_marker(wheel: Path) -> None:
    """Raise ValueError unless the wheel archive contains the py.typed marker."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    if MARKER not in names:
        raise ValueError(f"{wheel.name} does not contain {MARKER}")


def build_wheel(out_dir: Path) -> Path:
    command = ["uv", "build", "--wheel", "--out-dir", str(out_dir)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)  # noqa: S603 -- fixed args, no shell.
    return find_wheel(out_dir)


def main(argv: list[str]) -> int:
    if argv:
        wheel = Path(argv[0])
        require_marker(wheel)
        print(f"{wheel.name} ships {MARKER}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        wheel = build_wheel(Path(tmp))
        require_marker(wheel)
        print(f"{wheel.name} ships {MARKER}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
