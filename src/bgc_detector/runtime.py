"""CPU thread configuration and best-effort mapped-file read-ahead."""

import os
from pathlib import Path

import torch


def configure_cpu_threads() -> int:
    # OMP_NUM_THREADS is shared with the runner and remains usable for direct CLI runs.
    threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    if threads < 1:
        raise ValueError("OMP_NUM_THREADS must be a positive integer")
    torch.set_num_threads(threads)
    return threads


def prefetch_file(path: Path) -> str:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_WILLNEED"):
        return "unsupported"
    try:
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
    except OSError as exc:
        return f"unavailable: {exc}"
    # Advisory only: the kernel may read part of the file or evict pages at any time.
    return "requested (best effort; cache residency not guaranteed)"
