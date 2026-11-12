"""Checkpoint payloads, provenance helpers and detector settings resolution."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = 1

# Constructor defaults for the final recipe; metadata overrides them.
DETECTOR_DEFAULTS = {
    "d_model": 960,
    "num_layers": 2,
    "nhead": 15,
    "input_norm": True,
    "kernel_size": 5,
    "semantic_skip": True,
    "context_readout": False,
}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
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
        "path": str(path), "exists": True,
        "size": path.stat().st_size, "sha256": sha256_file(path),
    }


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": git_commit(),
    }


def save_checkpoint(path: str | Path, model: torch.nn.Module,
                    metadata: dict[str, Any]) -> None:
    torch.save({
        "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(),
        "metadata": metadata,
    }, path)


def load_checkpoint(path: str | Path,
                    device: str | torch.device = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except AttributeError:
        pass
    payload = torch.load(path, map_location=device, weights_only=True)
    if "model_state" in payload:
        return payload["model_state"], payload.get("metadata", {})
    # Legacy weights have no provenance.
    return payload, {"format_version": 0, "legacy": True}


def validation_threshold(metadata: dict[str, Any]) -> tuple[float, str]:
    """Return the saved validation threshold or the legacy 0.5 default."""
    threshold = metadata.get("validation_threshold")
    if threshold is None:
        return 0.5, "legacy-default"
    return float(threshold), str(metadata.get("validation_threshold_source", "validation"))


def detector_settings(metadata: dict[str, Any], embedding_dim: int) -> dict[str, Any]:
    """Resolve the detector constructor settings recorded in a checkpoint."""
    architecture = metadata.get("architecture", {}) if not metadata.get("legacy") else {}
    settings = dict(DETECTOR_DEFAULTS)
    for key in DETECTOR_DEFAULTS:
        if key in architecture:
            settings[key] = architecture[key]
    settings["input_dim"] = int(architecture.get("input_dim", embedding_dim))
    return settings


def initialize_from_pretraining(model: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    """Transfer verified weak-pretraining detector weights, never task state."""
    state, metadata = load_checkpoint(path)
    if (metadata.get("checkpoint_kind") != "pretraining_best"
            or metadata.get("pretraining_method") != "weak"):
        raise ValueError("requires a completed weak pretraining_best checkpoint")
    target = model.state_dict()
    keys = set(target)
    source_keys = set(state)
    # Branches added after the pretraining checkpoint start at exactly zero, so
    # the transferred detector's initial predictions are unchanged.
    new_keys = keys - source_keys
    branches = {
        "semantic_skip.": {"semantic_skip.weight", "semantic_skip.bias"},
        "context_readout.": {"context_readout.0.weight", "context_readout.0.bias",
                            "context_readout.2.weight", "context_readout.2.bias"},
    }
    zeroed: dict[str, list[str]] = {}
    for prefix, expected in branches.items():
        branch_keys = {k for k in new_keys if k.startswith(prefix)}
        if not branch_keys:
            continue
        if branch_keys != expected:
            raise ValueError(f"unexpected {prefix} keys in target model")
        if not all(torch.isfinite(target[k]).all() for k in branch_keys):
            raise ValueError(f"non-finite {prefix} initialization")
        if not all(torch.count_nonzero(target[k]) == 0 for k in branch_keys if ".2." in k
                   or prefix == "semantic_skip."):
            raise ValueError(f"{prefix} branch must start at zero output")
        keys -= branch_keys
        zeroed[prefix] = sorted(branch_keys)
    if keys != source_keys or not keys:
        raise ValueError("pretrained architecture keys do not match target")
    for key in keys:
        if target[key].shape != state[key].shape or not torch.isfinite(state[key]).all():
            raise ValueError(f"incompatible pretrained tensor: {key}")
    model.load_state_dict(
        {k: state[k] if k in keys else value for k, value in target.items()},
        strict=True,
    )
    return {
        "checkpoint": file_fingerprint(path),
        "scope": "detector",
        "method": "weak",
        "new_zero_initialized_keys": zeroed,
        "source_metadata": metadata,
    }


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
