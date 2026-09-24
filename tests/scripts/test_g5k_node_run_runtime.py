import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

_BASH_MAJOR = int(
    subprocess.run(
        ["bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True, check=True
    ).stdout
)
# The node script uses `mapfile` and empty arrays under `set -u` (bash >= 4, as on the
# Grid'5000 nodes and CI); macOS ships bash 3.2, where it cannot run at all.
pytestmark = pytest.mark.skipif(_BASH_MAJOR < 4, reason="g5k_node_run.sh needs bash >= 4")

REPOSITORY = Path(__file__).parents[2]
GTE_MODEL_ID = "Alibaba-NLP/gte-multilingual-reranker-base"
MXBAI_MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"
GLICLASS_MODEL_ID = "knowledgator/gliclass-multilang-mini"
GLINER2_MODEL_ID = "fastino/gliner2.5-multi-v1"
GGUF_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS"
GENERATIVE_MODEL_ID = "LiquidAI/LFM2.5-350M"

FAKE_UV = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\t%s\\tLD=%s\\tCMAKE=%s\\n' "${UV_PROJECT_ENVIRONMENT:-default}" "$*" \\
  "${LD_LIBRARY_PATH:-}" "${CMAKE_ARGS:-}" >> "$TEST_UV_LOG"
if [[ "$1" == sync ]]; then
  if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
    printf '%s\\n' '#!/bin/sh' 'printf "5.11.0\\n"' > "$UV_PROJECT_ENVIRONMENT/bin/python"
    chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"
    [[ -z "${TEST_ENV_MARKER:-}" ]] || printf '%s\\n' "$UV_PROJECT_ENVIRONMENT" > "$TEST_ENV_MARKER"
  fi
  exit 0
fi
if [[ "$1" == export ]]; then printf 'llama-cpp-python==0.3.35\\njinja2==3.1.6\\n'; exit 0; fi
if [[ "$1" == pip ]]; then exit 0; fi
if [[ "$1" == run ]]; then
  if [[ "$*" == *"lrb languages"* ]]; then
    printf 'en\\t%s\\nfr\\t%s\\n' "$TEST_ROWS" "$TEST_ROWS"
  fi
  if [[ "$*" == *"lrb scorers"* && "$TEST_ROSTER" == scorers ]]; then
    printf '%s\\tnote\\n' "$TEST_MODEL_ID"
  fi
  if [[ "$*" == *"lrb models"* && "$TEST_ROSTER" == models ]]; then
    printf '%s\\t1B\\tnote\\n' "$TEST_MODEL_ID"
  fi
  exit 0
fi
exit 2
"""

BASH_ENV_SHIM = """mapfile() {
  local target="$2"
  local line
  eval "$target=()"
  while IFS= read -r line; do eval "$target+=(\\\"\\$line\\\")"; done
}
"""


@dataclass
class NodeRun:
    completed: subprocess.CompletedProcess[str]
    uv_log: str
    module_log: str
    environment_path: str

    def uv_lines(self, needle: str) -> list[str]:
        return [line for line in self.uv_log.splitlines() if needle in line]


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_node_script(  # noqa: PLR0913 - one keyword per scenario knob
    tmp_path: Path,
    model_id: str,
    *,
    roster: str = "scorers",
    with_nvcc: bool = True,
    rows: str = "300",
    extra_environment: dict[str, str] | None = None,
) -> NodeRun:
    root = tmp_path / "remote-repository"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(REPOSITORY / "scripts/g5k_node_run.sh", scripts / "g5k_node_run.sh")
    (root / "data/translations").mkdir(parents=True)

    home = tmp_path / "home"
    fake_bin = home / ".local/bin"
    fake_bin.mkdir(parents=True)
    _executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
    module_log = tmp_path / "module.log"
    _executable(
        fake_bin / "module",
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{module_log}"\n'
        '[[ "$2" == cuda-toolkit* ]]\n',
    )
    path_entries = [str(fake_bin)]
    if with_nvcc:
        cuda_bin = tmp_path / "cuda/bin"
        cuda_bin.mkdir(parents=True)
        _executable(cuda_bin / "nvcc", "#!/usr/bin/env bash\nexit 0\n")
        path_entries.append(str(cuda_bin))
    path_entries += ["/usr/bin", "/bin"]

    bash_env = tmp_path / "bash-env"
    bash_env.write_text(BASH_ENV_SHIM, encoding="utf-8")
    log_path = tmp_path / "uv.log"
    log_path.touch()
    _executable(fake_bin / "uv", FAKE_UV)
    marker = tmp_path / "environment-path"

    # Drop exported shell functions (Grid'5000 frontends export `module` this way) so the
    # fake `module` on PATH is the one the script calls.
    environment: dict[str, str] = {
        key: value for key, value in os.environ.items() if not key.startswith("BASH_FUNC_")
    }
    # Never source the host's real Lmod: it would replace the fake `module` below.
    environment["LRB_LMOD_INIT"] = "/dev/null"
    environment.pop("UV_PROJECT_ENVIRONMENT", None)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(
        {
            "PATH": os.pathsep.join(path_entries),
            "HOME": str(home),
            "USER": "benchmark-test",
            "HF_HOME": str(tmp_path / "hf-cache"),
            "LRB_ROOT": str(root),
            "LRB_DATA_ROOT": str(root / "data/translations"),
            "LRB_RESULTS": str(tmp_path / "results"),
            "LRB_MODEL_ID": model_id,
            "OAR_JOB_ID": "job-42",
            "TEST_MODEL_ID": model_id,
            "TEST_ROSTER": roster,
            "TEST_ROWS": rows,
            "TEST_UV_LOG": str(log_path),
            "TEST_ENV_MARKER": str(marker),
            "BASH_ENV": str(bash_env),
        }
    )
    environment.update(extra_environment or {})
    completed = subprocess.run(
        ["bash", str(scripts / "g5k_node_run.sh"), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return NodeRun(
        completed=completed,
        uv_log=log_path.read_text(encoding="utf-8"),
        module_log=module_log.read_text(encoding="utf-8") if module_log.exists() else "",
        environment_path=marker.read_text(encoding="utf-8").strip() if marker.exists() else "",
    )


def test_gte_uses_an_isolated_transformers_511_runtime(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GTE_MODEL_ID)

    assert run.completed.returncode == 0, run.completed.stderr
    assert ".venv-gte-job-42" in run.uv_log
    assert "pip install --python" in run.uv_log
    assert "transformers==5.11.0" in run.uv_log
    assert "== runtime: Transformers 5.11.0" in run.completed.stdout


def test_other_scoring_models_keep_the_locked_runtime(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, MXBAI_MODEL_ID)

    assert run.completed.returncode == 0, run.completed.stderr
    assert ".venv-" not in run.uv_log
    assert "pip install" not in run.uv_log
    assert "sync --extra inference --frozen --no-dev" in run.uv_log


def test_gliclass_syncs_the_scoring_extra_in_its_own_environment(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GLICLASS_MODEL_ID)

    assert run.completed.returncode == 0, run.completed.stderr
    (sync,) = run.uv_lines("\tsync ")
    assert ".venv-gliclass-job-42" in sync
    assert "--extra inference --extra scoring --frozen --no-dev" in sync
    assert "pip install" not in run.uv_log


def test_gliner2_syncs_its_locked_extra_alone(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GLINER2_MODEL_ID)

    assert run.completed.returncode == 0, run.completed.stderr
    (sync,) = run.uv_lines("\tsync ")
    assert ".venv-gliner2-job-42" in sync
    assert "sync --extra gliner2 --frozen --no-dev" in sync
    assert "--extra inference" not in sync
    assert "pip install" not in run.uv_log


def test_gguf_builds_the_locked_llama_cpp_against_cuda(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GGUF_MODEL_ID, roster="models")

    assert run.completed.returncode == 0, run.completed.stderr
    (sync,) = run.uv_lines("\tsync ")
    assert ".venv-gguf-job-42" in sync
    assert "--extra inference --frozen --no-dev" in sync
    (export,) = run.uv_lines("\texport ")
    assert "--extra inference --extra gguf" in export
    (install,) = run.uv_lines("pip install")
    assert "--no-binary llama-cpp-python llama-cpp-python jinja2" in install
    assert "--constraint" in install
    assert "CMAKE=-DGGML_CUDA=on" in install
    cuda_root = tmp_path / "cuda"
    assert f"LD={cuda_root}/lib64:{cuda_root}/lib:" in install
    assert "load cuda-toolkit/12.9.1" in run.module_log.splitlines()


def test_gguf_cuda_module_is_configurable(tmp_path: Path) -> None:
    run = _run_node_script(
        tmp_path,
        GGUF_MODEL_ID,
        roster="models",
        extra_environment={"LRB_CUDA_MODULE": "nvhpc/24.1"},
    )

    assert run.completed.returncode == 0, run.completed.stderr
    assert run.module_log.splitlines() == ["load nvhpc/24.1", "load cuda-toolkit"]


def test_gguf_without_nvcc_exits_before_building(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GGUF_MODEL_ID, roster="models", with_nvcc=False)

    assert run.completed.returncode == 1
    assert "no CUDA toolkit (nvcc)" in run.completed.stderr
    assert "pip install" not in run.uv_log


def test_an_unlisted_quantized_id_is_rejected_before_building(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, "someone/Other-GGUF@Q4_K_M", roster="none")

    assert run.completed.returncode == 2
    assert "unknown LRB_MODEL_ID" in run.completed.stderr
    assert "pip install" not in run.uv_log


def test_generative_models_share_the_locked_runtime(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GENERATIVE_MODEL_ID, roster="models")

    assert run.completed.returncode == 0, run.completed.stderr
    assert ".venv-" not in run.uv_log
    assert run.module_log == ""


def test_isolated_environments_are_removed_on_exit(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GLINER2_MODEL_ID)

    assert run.completed.returncode == 0, run.completed.stderr
    assert run.environment_path.endswith(".venv-gliner2-job-42")
    assert not Path(run.environment_path).exists()


def test_isolated_environments_are_removed_on_failure(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, GGUF_MODEL_ID, roster="models", with_nvcc=False)

    assert run.completed.returncode == 1
    assert run.environment_path.endswith(".venv-gguf-job-42")
    assert not Path(run.environment_path).exists()


def test_row_count_must_match_the_expected_rows(tmp_path: Path) -> None:
    run = _run_node_script(tmp_path, MXBAI_MODEL_ID, rows="299")

    assert run.completed.returncode == 1
    assert "expected 300" in run.completed.stderr


def test_expected_rows_is_configurable(tmp_path: Path) -> None:
    run = _run_node_script(
        tmp_path, MXBAI_MODEL_ID, rows="20", extra_environment={"LRB_EXPECTED_ROWS": "20"}
    )

    assert run.completed.returncode == 0, run.completed.stderr
    assert "rows=40" in run.completed.stdout
