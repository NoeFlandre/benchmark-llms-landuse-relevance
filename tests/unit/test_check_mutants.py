"""The mutation gate refuses to pass a run that produced no results."""

from pathlib import Path

import pytest
from scripts.check_mutants import MutationRunError, survivors


def test_a_crashed_run_with_no_results_is_not_a_clean_pass(tmp_path: Path) -> None:
    with pytest.raises(MutationRunError, match="crash"):
        survivors(tmp_path)
