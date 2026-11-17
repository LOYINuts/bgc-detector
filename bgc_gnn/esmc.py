"""Frozen ESMC-300M (960d) per-protein mean-pooled embeddings.

The architecture lives in biohub's ``esm`` package, not in transformers, so it is
imported lazily: scoring an already embedded genome must not require it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ESMC_MODEL = "biohub/ESMC-300M"
EMBEDDING_DIM = 960
MAX_TOKS = 8192  # tokens per inference batch; sequence lengths are bucketed
KERNELS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


def load_model(device):
    """Load ESMC via biohub's esm package."""
    try:
        from esm.models.esmc import EsmcModel, EsmcTokenizer
    except ImportError as exc:  # the CLI reports this without a traceback
        raise ValueError(
            "computing embeddings requires the esm package "
            '(uv add "esm @ git+https://github.com/Biohub/esm.git"); '
            "pass --emb-cache to score a genome that is already embedded"
        ) from exc

    model = EsmcModel.from_pretrained(ESMC_MODEL, dtype=torch.bfloat16).to(device)
    tokenizer = EsmcTokenizer.from_pretrained(ESMC_MODEL)
    model.eval()
    return model, tokenizer


def pool_batch(hidden: torch.Tensor, attn_mask: torch.Tensor) -> np.ndarray:
    """Mean-pool a whole batch over residues (BOS/EOS excluded) in one shot."""
    B, T, _ = hidden.shape
    pos = torch.arange(T, device=hidden.device)
    base = attn_mask.to(hidden.device).bool()
    n_valid = base.sum(dim=1, keepdim=True)  # includes BOS+EOS
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
    out = np.zeros((len(sequences), EMBEDDING_DIM), dtype=np.float32)
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
        with torch.inference_mode(), sdpa_kernel(KERNELS):
            hidden = model(input_ids=enc["input_ids"].to(device), attention_mask=attn)
        out[batch] = pool_batch(hidden.last_hidden_state, attn)
        del hidden
    return out
