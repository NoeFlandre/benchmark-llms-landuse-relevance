"""Architecture rules, enforced rather than documented."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_import_contracts_hold() -> None:
    # ``python -m importlinter.cli`` has no ``__main__`` hook and exits 0 without
    # linting, so the console-script entry point is invoked directly instead.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlinter.cli import lint_imports_command;"
            " sys.exit(lint_imports_command())",
            "--no-cache",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
