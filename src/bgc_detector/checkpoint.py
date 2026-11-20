"""Checkpoint payloads, provenance helpers and fixed-architecture validation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from bgc_detector.model import ARCHITECTURE

FORMAT_VERSION = 1


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": git_commit(),
    }


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, metadata: dict[str, Any]
) -> None:
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model_state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except AttributeError:
        pass
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or "model_state" not in payload
        or "metadata" not in payload
    ):
        raise ValueError("requires a detector checkpoint with model_state and metadata")
    return payload["model_state"], payload["metadata"]


def validation_threshold(metadata: dict[str, Any]) -> tuple[float, str]:
    """Require the validation-MCC threshold exported by the final training recipe."""
    threshold = metadata.get("validation_threshold")
    if (
        threshold is None
        or not 0 <= float(threshold) <= 1
        or metadata.get("validation_threshold_source") != "validation"
        or metadata.get("validation_threshold_metric") != "mcc"
    ):
        raise ValueError("requires a finite validation-MCC threshold")
    return float(threshold), "validation"


def validate_architecture(metadata: dict[str, Any]) -> None:
    """Reject other architectures instead of rebuilding models from metadata."""
    architecture = metadata.get("architecture", {})
    cli = metadata.get("cli", {})
    if architecture.get("symmetric_local_conv", cli.get("symmetric_local_conv", False)):
        raise ValueError(
            "checkpoint does not match the fixed architecture: symmetric convolution"
        )
    for key, value in ARCHITECTURE.items():
        # Historical final checkpoints recorded normalization/kernel in cli only.
        if architecture.get(key, cli.get(key)) != value:
            raise ValueError(f"checkpoint does not match the fixed architecture: {key}")


def initialize_from_pretraining(
    model: torch.nn.Module, path: str | Path
) -> dict[str, Any]:
    """Transfer the complete pretrained detector, never optimizer or task state."""
    state, metadata = load_checkpoint(path)
    if (
        metadata.get("checkpoint_kind") != "pretraining_best"
        or metadata.get("selection_metric") != "pretraining_val_loss"
    ):
        raise ValueError("requires a validation-selected pretraining_best checkpoint")
    validate_architecture(metadata)
    target = model.state_dict()
    if set(target) != set(state):
        raise ValueError(
            "pretrained weights must contain the complete final detector, including semantic skip"
        )
    for key, value in target.items():
        if value.shape != state[key].shape or not torch.isfinite(state[key]).all():
            raise ValueError(f"incompatible pretrained tensor: {key}")
    if not torch.count_nonzero(state["semantic_skip.weight"]):
        raise ValueError(
            "pretraining checkpoint must contain learned semantic skip weights"
        )
    model.load_state_dict(state, strict=True)
    return {
        "checkpoint": file_fingerprint(path),
        "scope": "detector",
        "stage": "pretraining",
        "source_metadata": metadata,
    }


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
