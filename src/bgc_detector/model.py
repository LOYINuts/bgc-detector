"""Fixed per-gene detector: normalized ESMC-300M embeddings, Conv-BiGRU, semantic skip."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW_SIZE = 128
EMBEDDING_DIM = 960
DROPOUT = 0.5
HEAD_DROPOUT = 0.1

# Describes the only supported architecture; these are not constructor options.
ARCHITECTURE = {
    "arch": "channel_gated_conv_gru",
    "input_dim": EMBEDDING_DIM,
    "d_model": EMBEDDING_DIM,
    "num_layers": 2,
    "nhead": 15,
    "input_norm": True,
    "kernel_size": 5,
    "semantic_skip": True,
    "context_readout": False,
}


class DepthwiseConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            EMBEDDING_DIM, EMBEDDING_DIM, 5, padding=2, groups=EMBEDDING_DIM
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class GRUBlock(nn.Module):
    """Padding never enters the bidirectional recurrence."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            EMBEDDING_DIM,
            EMBEDDING_DIM // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.norm = nn.LayerNorm(EMBEDDING_DIM)
        self.drop = nn.Dropout(DROPOUT)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if key_padding_mask is None:
            out, _ = self.gru(x)
        else:
            # pack() rejects length 0, i.e. a fully padded row.
            lengths = (~key_padding_mask).sum(1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1)
            )
        return self.norm(x + self.drop(out))


class ChannelGatedConvGRUBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(EMBEDDING_DIM)
        self.depthwise = DepthwiseConv1d()
        self.pointwise = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.gate = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        nn.init.constant_(self.gate.bias, -3.0)
        self.drop = nn.Dropout(DROPOUT)
        self.gru = GRUBlock()

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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
    """The final 10,290,690-parameter model. No architecture switches."""

    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.input_proj = nn.Identity()
        self.encoder = ChannelGatedConvGRUBlock()
        self.detect_head = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(EMBEDDING_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Identity(),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize after the backbone to preserve the verified seeded weights.
        self.semantic_skip = nn.Linear(EMBEDDING_DIM, 1)
        nn.init.zeros_(self.semantic_skip.weight)
        nn.init.zeros_(self.semantic_skip.bias)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_norm(x)
        semantic_logits = self.semantic_skip(x)
        h = self.encoder(self.input_proj(x), key_padding_mask)
        return torch.sigmoid(self.detect_head(h) + semantic_logits).squeeze(-1)
