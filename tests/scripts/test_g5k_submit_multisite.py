import json
import os
import shutil
import subprocess
from pathlib import Path

BASH = shutil.which("bash")
if BASH is None:
    raise RuntimeError("bash is required to test the Grid'5000 submitter")


def test_multisite_dry_run_propagates_a_focused_model(tmp_path: Path) -> None:
    config = tmp_path / "sites.json"
    config.write_text(
        json.dumps({"sites": [{"name": "nancy", "frontend": "nancy", "weight": 1}]}),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "LRB_MODEL_ID": "LiquidAI/LFM2.5-350M",
        "LRB_REMOTE_ROOT": "/remote/benchmark",
        "LRB_RESULTS": "/remote/results",
    }

    completed = subprocess.run(  # noqa: S603 -- local submit script with a test config.
        [
            BASH,
            "scripts/g5k_submit_multisite.sh",
            "--sites-config",
            str(config),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "LRB_MODEL_ID=LiquidAI/LFM2.5-350M" in completed.stdout
