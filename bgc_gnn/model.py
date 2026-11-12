"""BGC detector: local channel-gated conv + BiGRU, with a semantic (protein) skip.

One architecture, the frozen final recipe:
  frozen ESM-C 960d gene embeddings
  -> input LayerNorm
  -> zero-initialized linear semantic skip (protein-only logits)
  -> Channel-Gated Conv-BiGRU encoder (2 layers, k=5, gate bias -3.0)
  -> detection MLP 960->128->32->1
  -> optional zero-initialized context readout residual
  -> sigmoid(context logits + semantic logits)

Submodule names and tensor shapes are unchanged from the ablation codebase, so
checkpoints written before this trim still load with ``strict=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW_SIZE = 128
GATE_BIAS = -3.0


class DepthwiseConv1d(nn.Module):
    """Depthwise conv along the gene axis, wrapped for [B, L, D] tensors."""

    def __init__(self, d_model: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2,
                              groups=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class GRUBlock(nn.Module):
    """Bidirectional GRU along the gene axis; padding never enters the recurrence."""

    def __init__(self, d_model: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            d_model, d_model // 2, num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if key_padding_mask is None:
            out, _ = self.gru(x)
        else:
            # pack so the bidirectional recurrence never reads padded steps
            lengths = (~key_padding_mask).sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1))
        return self.norm(x + self.drop(out))


class ChannelGatedConvGRUBlock(nn.Module):
    """BiGRU preceded by position- and channel-wise gated local features."""

    def __init__(self, d_model: int, num_layers: int, dropout: float,
                 kernel: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = DepthwiseConv1d(d_model, kernel)
        self.pointwise = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate.bias, GATE_BIAS)
        self.drop = nn.Dropout(dropout)
        self.gru = GRUBlock(d_model, num_layers, dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask[..., None], 0.0)
        local = self.pointwise(F.gelu(self.depthwise(h)))
        channel_gate = torch.sigmoid(self.gate(h))
        x = x + self.drop(channel_gate * local)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask[..., None], 0.0)
        return self.gru(x, key_padding_mask)


class BGCDetector(nn.Module):
    """Per-gene BGC probability, final recipe (2 layers, 960d, k=5)."""

    def __init__(
        self,
        d_model: int = 960,
        nhead: int = 15,  # recorded for checkpoint provenance; the encoder is recurrent
        num_layers: int = 2,
        window_size: int = WINDOW_SIZE,
        dropout: float = 0.5,
        head_dropout: float = 0.1,
        input_dim: int | None = None,
        input_norm: bool = True,
        kernel_size: int = 5,
        semantic_skip: bool = True,
        context_readout: bool = False,
    ):
        super().__init__()
        if context_readout and not semantic_skip:
            raise ValueError("context_readout requires semantic_skip")
        self.window_size = window_size
        self.kernel_size = kernel_size
        # Identity for input_dim == d_model; the recipe uses matched ESMC 960d.
        self.input_norm = nn.LayerNorm(input_dim or d_model) if input_norm else nn.Identity()
        self.input_proj = (nn.Identity() if input_dim in (None, d_model)
                           else nn.Linear(input_dim, d_model))
        self.encoder = ChannelGatedConvGRUBlock(
            d_model, num_layers, dropout, kernel=kernel_size,
        )
        self.detect_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Identity() if semantic_skip else nn.Sigmoid(),
        )
        self._init_weights()
        # Initialize after the backbone so its seeded weights stay unchanged.
        self.semantic_skip = nn.Linear(input_dim or d_model, 1) if semantic_skip else None
        if self.semantic_skip is not None:
            nn.init.zeros_(self.semantic_skip.weight)
            nn.init.zeros_(self.semantic_skip.bias)
        self.context_readout = None
        if context_readout:
            # Zero output keeps the checkpoint's initial predictions unchanged.
            with torch.random.fork_rng(devices=[]):
                self.context_readout = nn.Sequential(
                    nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, 1),
                )
                nn.init.zeros_(self.context_readout[-1].weight)
                nn.init.zeros_(self.context_readout[-1].bias)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_norm(x)
        semantic_logits = self.semantic_skip(x) if self.semantic_skip is not None else None
        h = self.encoder(self.input_proj(x), key_padding_mask)
        detect = self.detect_head(h)
        if self.context_readout is not None:
            detect = detect + self.context_readout(h)
        if semantic_logits is not None:
            detect = torch.sigmoid(detect + semantic_logits)
        return detect.squeeze(-1)
