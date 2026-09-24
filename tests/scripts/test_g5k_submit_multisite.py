import json
import os
import subprocess
from pathlib import Path


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

    completed = subprocess.run(
        [
            "bash",
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
