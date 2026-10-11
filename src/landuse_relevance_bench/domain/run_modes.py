"""How one model's run is sized, checked and named, from its roster spec and the flags.

Pure: the caller supplies the spec and the requested modes. The roster lookup and the
warning for an unlisted model stay in ``RunRequest.for_run``.
"""

from landuse_relevance_bench.domain.orchestration import DEFAULT_BATCH_SIZE
from landuse_relevance_bench.domain.roster import SGLANG, TRANSFORMERS, ModelSpec, spec_for


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


def resolve_mode_options(
    spec: ModelSpec | None,
    batch_size: int | None,
    *,
    continuous_batching: bool,
    throughput: bool,
) -> tuple[int | None, bool, bool]:
    """The batch size and the continuous and throughput flags one run uses.

    ``spec`` is None for an unlisted model. Modes the model's runtime does not support are
    switched off, not refused; the refusing rules are in ``check_modes``.
    """
    runtime = runtime_for(spec)
    return (
        throughput_batch_size(runtime, batch_size, throughput=throughput),
        continuous_batching_applies(spec, runtime, continuous_batching=continuous_batching),
        throughput_applies(runtime, throughput=throughput),
    )


def spec_or_none(model_id: str) -> ModelSpec | None:
    """The roster spec for a model, or None when the model is not rostered."""
    try:
        return spec_for(model_id)
    except KeyError:
        return None


def runtime_for(spec: ModelSpec | None) -> str:
    """An unlisted model runs on Transformers, the default runtime."""
    return TRANSFORMERS if spec is None else spec.runtime


def throughput_batch_size(runtime: str, batch_size: int | None, *, throughput: bool) -> int | None:
    """Throughput mode on SGLang fills an unset batch size with the default."""
    if throughput and runtime == SGLANG and batch_size is None:
        return DEFAULT_BATCH_SIZE
    return batch_size


def continuous_batching_applies(
    spec: ModelSpec | None, runtime: str, *, continuous_batching: bool
) -> bool:
    """Continuous batching needs the Transformers runtime and a model that is not vision."""
    return continuous_batching and runtime == TRANSFORMERS and (spec is None or not spec.vision)


def throughput_applies(runtime: str, *, throughput: bool) -> bool:
    """Throughput mode needs the SGLang runtime."""
    return throughput and runtime == SGLANG


def _first_set(*values: int | None) -> int:
    return next(value for value in values if value is not None)
