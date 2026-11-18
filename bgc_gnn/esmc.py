"""Frozen ESMC-300M (960d) per-protein mean-pooled embeddings.

ESMC is a native transformers architecture (``ESMCModel``), so the weights used to
train this detector load directly from the cached snapshot; the older ``esm``
package expects a different weight layout and produces wrong embeddings.
"""

from __future__ import annotations

import os

import numpy as np
import torch

ESMC_MODEL = "biohub/ESMC-300M"
EMBEDDING_DIM = 960
MAX_TOKS = 8192  # tokens per inference batch; sequence lengths are bucketed


def load_model(device):
    """Load ESMC-300M in bf16, matching the embeddings used for training."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # the CLI reports this without a traceback
        raise ValueError(
            "computing embeddings requires transformers "
            "(uv pip install transformers); pass --emb-cache to score a genome "
            "that is already embedded"
        ) from exc

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    model = (
        AutoModel.from_pretrained(
            ESMC_MODEL, dtype=torch.bfloat16, local_files_only=offline
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(ESMC_MODEL, local_files_only=offline)
    return model, tokenizer


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
    base = attn_mask.to(hidden.device).bool()
    pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    n_valid = base.sum(dim=1, keepdim=True)
    w = base & (pos >= 1) & (pos < n_valid - 1)
    w = w | (~w.any(dim=1, keepdim=True) & base)  # degenerate: all valid
    w = w.to(hidden.dtype)
    w = w / w.sum(dim=1, keepdim=True)
    return (hidden * w.unsqueeze(-1)).sum(dim=1).float().cpu().numpy()


def embed_sequences(
    sequences: list[str],
    device: torch.device,
    max_toks: int = MAX_TOKS,
) -> np.ndarray:
    """Return [N, 960] float32 embeddings, one per sequence."""
    model, tokenizer = load_model(device)
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
        embeddings[batch] = pool_batch(last_hidden(result), attn)
        del result
    return embeddings
