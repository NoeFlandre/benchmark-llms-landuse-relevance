"""How one model's run is sized, checked and named, from its roster spec and the flags.

Pure: the caller supplies the spec and the requested modes. The roster lookup and the
warning for an unlisted model stay in ``RunRequest.for_run``.
"""

from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import SGLANG, TRANSFORMERS, ModelSpec


def resolve_batch_size(spec: ModelSpec, batch_size: int | None, *, throughput_mode: bool) -> int:
    """The requested batch size, else the roster's, else the default."""
    return (
        DEFAULT_BATCH_SIZE
        if throughput_mode and batch_size is None
        else _first_set(batch_size, spec.batch_size, DEFAULT_BATCH_SIZE)
    )


def check_modes(
    spec: ModelSpec,
    resolved_batch_size: int,
    *,
    continuous_batching: bool,
    throughput_mode: bool,
) -> None:
    """Refuse a mode the model's runtime cannot run, or a throughput batch of one or less."""
    if continuous_batching and (spec.runtime != TRANSFORMERS or spec.vision):
        raise ValueError("continuous batching requires the Transformers runtime")
    if throughput_mode and spec.runtime != SGLANG:
        raise ValueError("throughput mode requires the SGLang runtime")
    if throughput_mode and resolved_batch_size <= 1:
        raise ValueError("throughput mode requires batch_size > 1")


def run_id_for(
    spec: ModelSpec,
    resolved_batch_size: int,
    *,
    continuous_batching: bool,
    throughput_mode: bool,
) -> str:
    """The roster's run id, or a variant id that names the mode and its batch size."""
    run_id = spec.run_id
    if continuous_batching:
        run_id = f"{spec.name}@continuous-b{resolved_batch_size}"
    elif throughput_mode:
        run_id = f"{spec.name}-throughput-b{resolved_batch_size}"
    return run_id


def _first_set(*values: int | None) -> int:
    return next(value for value in values if value is not None)
