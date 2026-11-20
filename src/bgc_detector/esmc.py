"""Frozen ESMC-300M (960d) per-protein mean-pooled embeddings.

Loading follows my-bgc-v2/scripts/encode.py, the recipe that produced the frozen
embeddings: the ``esm`` package's ``EsmcModel`` reads the ``biohub/ESMC-300M``
checkpoint directly, in bf16. transformers' own ESMC build does not match this
checkpoint's layout (it produces a 2560-wide model with no shared keys), which is
why the package is used here.

The weights are looked up as: an explicit ``source`` (CLI ``--esmc-path`` or
``BGC_DETECTOR_ESMC``), else the newest local HF snapshot of ``biohub/ESMC-300M``,
else the Hub itself.

Check any change against the frozen NG embeddings with
``uv run python -m scripts.tools.check_esmc``: it must stay at bf16 scale or better.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch

ESMC_MODEL = "biohub/ESMC-300M"
ESMC_MODEL_ENV = "BGC_DETECTOR_ESMC"  # same value the CLI uses for --esmc-path
EMBEDDING_DIM = 960
MAX_TOKS = 8192  # tokens per inference batch; sequences are bucketed by length


def local_snapshot() -> str | None:
    """Most recently written local snapshot of the checkpoint, if any."""
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(home, "hub", "models--biohub--ESMC-300M", "snapshots", "*")
    hits = glob.glob(pattern)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def resolve_sources(explicit: str | Path | None = None) -> list[str]:
    """Where to look for the weights, in order.

    An explicit path (CLI flag or env var) is used alone: if it is wrong, failing
    loudly beats silently falling back to a different set of weights.
    """
    explicit = os.environ.get(ESMC_MODEL_ENV) if explicit is None else explicit
    if explicit:
        return [str(explicit)]
    return [s for s in (local_snapshot(), ESMC_MODEL) if s]


def load_model(device, source: str | Path | None = None):
    """Load ESMC-300M in bf16; returns ``(model, tokenizer, source_used)``."""
    try:
        from esm.models.esmc import EsmcModel, EsmcTokenizer
    except ImportError as exc:  # the CLI reports this without a traceback
        raise ValueError(
            "computing embeddings needs the esm package; run uv sync, or pass "
            "--emb-cache with precomputed embeddings"
        ) from exc

    error: Exception | None = None
    for candidate in resolve_sources(source):
        try:
            model = EsmcModel.from_pretrained(candidate, dtype=torch.bfloat16)
            tokenizer = EsmcTokenizer.from_pretrained(candidate)
            print(f"ESMC loaded from: {candidate}", flush=True)
            return model.to(device).eval(), tokenizer, candidate
        except Exception as exc:
            error = exc
    raise ValueError(f"cannot load ESMC weights: {type(error).__name__}: {error}")


def last_hidden(out: object) -> torch.Tensor:
    """Hidden states from either output shape: last_hidden_state or hidden_states[-1]."""
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        states = getattr(out, "hidden_states", None)
        hidden = states[-1] if isinstance(states, (tuple, list)) else states
    if hidden is None:
        raise ValueError("ESMC output carries no hidden states")
    return hidden


def pool_batch(hidden: torch.Tensor, attn_mask: torch.Tensor) -> np.ndarray:
    """Mean-pool a batch over residues, dropping BOS/EOS and padding."""
    mask = attn_mask.to(hidden.device).bool()
    pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    n_valid = mask.sum(dim=1, keepdim=True)
    w = mask & (pos >= 1) & (pos < n_valid - 1)
    w = w | (~w.any(dim=1, keepdim=True) & mask)  # degenerate: all valid
    w = w.to(hidden.dtype)
    w = w / w.sum(dim=1, keepdim=True)
    return (hidden * w.unsqueeze(-1)).sum(dim=1).float().cpu().numpy()


def embed_sequences(
    sequences: list[str],
    device: torch.device,
    max_toks: int = MAX_TOKS,
    source: str | Path | None = None,
) -> np.ndarray:
    """Load the weights, embed ``sequences``; returns [N, 960] float32."""
    model, tokenizer, _ = load_model(device, source)
    return embed_with(model, tokenizer, sequences, device, max_toks)


def embed_with(
    model: torch.nn.Module,
    tokenizer,
    sequences: list[str],
    device: torch.device,
    max_toks: int = MAX_TOKS,
) -> np.ndarray:
    """Return [N, 960] float32 embeddings with an already loaded model."""
    lens = [len(s) + 2 for s in sequences]  # + BOS/EOS, no truncation
    order = sorted(range(len(sequences)), key=lambda i: lens[i])
    embeddings = np.zeros((len(sequences), EMBEDDING_DIM), dtype=np.float32)
    idx = 0
    while idx < len(order):
        tokens = 0
        batch: list[int] = []
        while idx < len(order) and tokens + lens[order[idx]] <= max_toks:
            batch.append(order[idx])
            tokens += lens[order[idx]]
            idx += 1
        if not batch:  # a single protein exceeds the budget
            batch = [order[idx]]
            idx += 1
        enc = tokenizer(
            [sequences[i] for i in batch], return_tensors="pt", padding=True
        )
        attn = enc["attention_mask"].to(device)
        with torch.inference_mode():
            result = model(input_ids=enc["input_ids"].to(device), attention_mask=attn)
        pooled = pool_batch(last_hidden(result), attn)
        if pooled.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"ESMC produced {pooled.shape[1]}d embeddings, expected {EMBEDDING_DIM}; "
                f"this build does not match the {ESMC_MODEL} checkpoint"
            )
        embeddings[batch] = pooled
        del result
    return embeddings
