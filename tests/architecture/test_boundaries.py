"""Architecture rules, enforced rather than documented."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN = PROJECT_ROOT / "src" / "landuse_relevance_bench" / "domain"


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


@pytest.mark.parametrize("module", sorted(DOMAIN.glob("*.py")), ids=lambda p: p.name)
def test_no_domain_module_reaches_for_an_io_library(module: Path) -> None:
    source = module.read_text(encoding="utf-8")
    forbidden = ("import csv", "import json", "from pathlib", "import requests", "import torch")
    offenders = [needle for needle in forbidden if needle in source]
    assert not offenders, f"{module.name} imports {offenders}"


def test_the_domain_package_imports_without_any_optional_dependency() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;"
            "import landuse_relevance_bench.domain.orchestration;"
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
