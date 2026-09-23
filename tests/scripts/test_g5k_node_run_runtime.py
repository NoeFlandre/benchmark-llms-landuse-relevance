import os
import shutil
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).parents[2]
GTE_MODEL_ID = "Alibaba-NLP/gte-multilingual-reranker-base"
MXBAI_MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"


def _run_node_script(tmp_path: Path, model_id: str) -> tuple[subprocess.CompletedProcess[str], str]:
    root = tmp_path / "remote-repository"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(REPOSITORY / "scripts/g5k_node_run.sh", scripts / "g5k_node_run.sh")
    (root / "data/translations").mkdir(parents=True)

    home = tmp_path / "home"
    fake_bin = home / ".local/bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (fake_bin / "nvidia-smi").chmod(0o755)
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        """mapfile() {
  local target="$2"
  local line
  eval "$target=()"
  while IFS= read -r line; do eval "$target+=(\\\"\\$line\\\")"; done
}
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "uv.log"
    uv = fake_bin / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\t%s\\n' "${UV_PROJECT_ENVIRONMENT:-default}" "$*" >> "$TEST_UV_LOG"
if [[ "$1" == sync ]]; then
  if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
    printf '%s\\n' '#!/bin/sh' 'printf "5.11.0\\n"' > "$UV_PROJECT_ENVIRONMENT/bin/python"
    chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"
  fi
  exit 0
fi
if [[ "$1" == pip ]]; then exit 0; fi
if [[ "$1" == run ]]; then
  if [[ "$*" == *"lrb languages"* ]]; then printf 'en\\t300\\n'; fi
  if [[ "$*" == *"lrb scorers"* ]]; then printf '%s\\n' "$TEST_MODEL_ID"; fi
  exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    environment = {
        **os.environ,
        "HOME": str(home),
        "USER": "benchmark-test",
        "HF_HOME": str(tmp_path / "hf-cache"),
        "LRB_ROOT": str(root),
        "LRB_DATA_ROOT": str(root / "data/translations"),
        "LRB_RESULTS": str(tmp_path / "results"),
        "LRB_MODEL_ID": model_id,
        "OAR_JOB_ID": "job-42",
        "TEST_MODEL_ID": model_id,
        "TEST_UV_LOG": str(log_path),
        "BASH_ENV": str(bash_env),
    }
    environment.pop("UV_PROJECT_ENVIRONMENT", None)
    completed = subprocess.run(
        ["bash", str(scripts / "g5k_node_run.sh"), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, log_path.read_text(encoding="utf-8")


def test_gte_uses_an_isolated_transformers_511_runtime(tmp_path: Path) -> None:
    completed, uv_log = _run_node_script(tmp_path, GTE_MODEL_ID)

    assert completed.returncode == 0, completed.stderr
    assert ".venv-gte-job-42" in uv_log
    assert "pip install --python" in uv_log
    assert "transformers==5.11.0" in uv_log
    assert "== runtime: Transformers 5.11.0" in completed.stdout


def test_other_scoring_models_keep_the_locked_runtime(tmp_path: Path) -> None:
    completed, uv_log = _run_node_script(tmp_path, MXBAI_MODEL_ID)

    assert completed.returncode == 0, completed.stderr
    assert "UV_PROJECT_ENVIRONMENT" not in uv_log
    assert "pip install" not in uv_log
