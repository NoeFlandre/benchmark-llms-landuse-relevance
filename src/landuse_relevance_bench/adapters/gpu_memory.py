"""Free GPU memory snapshots, logged around model loads to evidence memory release."""

from typing import Any

MIB = 1024 * 1024


def _cuda_memory(torch: Any) -> tuple[int, int] | None:
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except (AttributeError, RuntimeError):
        return None
    return int(free), int(total)


def gpu_memory_line(event: str, run: str) -> str | None:
    """One parseable line of free CUDA memory, or ``None`` without torch or a GPU."""
    try:
        import torch
    except ImportError:
        return None
    memory = _cuda_memory(torch)
    if memory is None:
        return None
    free, total = memory
    return f"gpu_mem event={event} run={run} free_mib={free // MIB} total_mib={total // MIB}"


def free_gpu_memory_bytes() -> int | None:
    """Free CUDA memory in bytes, or ``None`` without torch or a GPU."""
    try:
        import torch
    except ImportError:
        return None
    memory = _cuda_memory(torch)
    return None if memory is None else memory[0]
