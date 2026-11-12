"""Weak (supervised-window) pretraining of the detector on frozen ESMC embeddings.

Trains the final detector architecture on audited antiSMASH windows with the
same pos-weighted BCE the detector uses, selecting by pretraining validation
loss. The best checkpoint is transferred into supervised detection training
with ``train.py --pretrained-checkpoint``.

Usage
-----
    uv run python -m bgc_gnn.pretrain \
        --data-dir data/derived/pretraining/antismash_v2_v1 \
        --cache-dir data/derived/pretraining/as566_v3_cache \
        --output-dir runs/pretrain_seed1301 --seed 1301
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from bgc_gnn.checkpoint import file_fingerprint, runtime_metadata, save_checkpoint
from bgc_gnn.model import WINDOW_SIZE, BGCDetector

EMBEDDING_DIM = 960
COLUMNS = ("sequence_ids", "valid_mask", "detect_mask", "loss_mask")


class PretrainingDataset(Dataset):
    def __init__(self, windows, cache_dir):
        self.rows = pq.read_table(windows, columns=list(COLUMNS))
        ids = json.loads((cache_dir / "sequence_ids.json").read_text())
        self.index = {s: i for i, s in enumerate(ids)}
        self.values = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
        if len(ids) != len(self.index) or self.values.ndim != 2 or len(ids) != len(self.values):
            raise ValueError("cache IDs/embedding shape mismatch")
        self.dim = self.values.shape[1]
        self.positive = 0
        self.negative = 0
        for batch in self.rows.to_batches(max_chunksize=512):
            for row in batch.to_pylist():
                valid = row["valid_mask"]
                if any(len(row[k]) != WINDOW_SIZE for k in COLUMNS):
                    raise ValueError(f"windows must have {WINDOW_SIZE} aligned positions")
                length = sum(valid)
                if not length or valid != [True] * length + [False] * (WINDOW_SIZE - length):
                    raise ValueError("windows require non-empty valid prefixes")
                if any(active and sid not in self.index
                       for sid, active in zip(row["sequence_ids"], valid)):
                    raise ValueError("valid window gene missing from cache")
                if any(supervised and not active
                       for supervised, active in zip(row["loss_mask"], valid)):
                    raise ValueError("loss mask includes padding")
                self.positive += sum(t and m for t, m in zip(row["detect_mask"], row["loss_mask"]))
                self.negative += sum(not t and m for t, m in zip(row["detect_mask"], row["loss_mask"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = {name: self.rows[name][i].as_py() for name in self.rows.column_names}
        length = sum(row["valid_mask"])
        x = np.zeros((WINDOW_SIZE, self.dim), dtype=np.float32)
        x[:length] = self.values[[self.index[s] for s in row["sequence_ids"][:length]]]
        return {
            "embeddings": torch.from_numpy(x),
            **{k: torch.tensor(row[k], dtype=torch.bool)
               for k in ("valid_mask", "detect_mask", "loss_mask")},
        }


def weak_loss(detector, x, valid, targets, supervised):
    if (supervised & ~valid).any():
        raise ValueError("weak loss mask includes padding")
    if not supervised.any():
        return None
    probabilities = detector(x, key_padding_mask=~valid)
    per_gene = F.binary_cross_entropy(probabilities, targets.float(), reduction="none")
    weights = torch.where(targets, 2., 1.)
    return (per_gene * weights)[supervised].mean()


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            x = batch["embeddings"].to(device)
            valid = batch["valid_mask"].to(device)
            if not torch.isfinite(x).all():
                raise ValueError("non-finite input embeddings")
            supervised = batch["loss_mask"].to(device)
            loss = weak_loss(model, x, valid, batch["detect_mask"].to(device), supervised)
            n = int(supervised.sum())
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("non-finite pretraining loss")
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            total += loss.item() * n
            count += n
    if not count:
        raise ValueError("split has no supervised positions for this task")
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1301)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.output_dir.exists() or min(a.epochs, a.patience, a.batch_size) < 1 \
            or not math.isfinite(a.lr) or a.lr <= 0:
        p.error("use a fresh output directory and positive finite training settings")

    audit = json.loads((a.data_dir / "audit.json").read_text())
    cache = json.loads((a.cache_dir / "metadata.json").read_text())
    if audit["status"] != "completed" or cache["status"] != "completed":
        p.error("requires completed data and embedding cache")
    inputs = {}
    for role, path, expected in [
        ("train", a.data_dir / "windows_train.parquet", audit["windows_train.parquet"]["sha256"]),
        ("val", a.data_dir / "windows_val.parquet", audit["windows_val.parquet"]["sha256"]),
        ("proteins", a.data_dir / "proteins.parquet", cache["proteins"]["sha256"]),
        ("embeddings", a.cache_dir / "embeddings.npy", cache["embeddings"]["sha256"]),
        ("sequence_ids", a.cache_dir / "sequence_ids.json", cache["sequence_ids"]["sha256"]),
    ]:
        inputs[role] = file_fingerprint(path)
        if inputs[role]["sha256"] != expected:
            p.error(f"input fingerprint mismatch: {role}")
    if cache["model"] != "biohub/ESMC-300M" or cache["dim"] != EMBEDDING_DIM:
        p.error("pretraining requires frozen ESMC-300M 960d")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    train = PretrainingDataset(a.data_dir / "windows_train.parquet", a.cache_dir)
    val = PretrainingDataset(a.data_dir / "windows_val.parquet", a.cache_dir)
    if not len(train) or not len(val) or train.dim != EMBEDDING_DIM or val.dim != EMBEDDING_DIM:
        p.error("invalid data dimensions or empty split")
    for split in (train, val):
        if not split.positive or not split.negative:
            p.error("weak training requires positive and background supervision in both splits")

    detector = BGCDetector(
        d_model=EMBEDDING_DIM, num_layers=2, nhead=15, input_dim=EMBEDDING_DIM,
        # No semantic skip: the transferred checkpoint carries encoder + head only,
        # and the zero-initialized skip is added by the downstream detector.
        semantic_skip=False,
    ).to(device)
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=a.batch_size)
    optimizer = torch.optim.AdamW(
        (p for p in detector.parameters() if p.requires_grad), lr=a.lr, weight_decay=.01,
    )
    metadata = {
        "runtime": runtime_metadata(),
        "pretraining_method": "weak",
        "checkpoint_kind": "pretraining_best",
        "inputs": inputs,
        "cache": cache,
        "architecture": {
            "model": "convgru", "arch": "channel_gated_conv_gru",
            "input_dim": EMBEDDING_DIM, "d_model": EMBEDDING_DIM,
            "num_layers": 2, "nhead": 15, "input_norm": True,
            "kernel_size": 5, "semantic_skip": False, "context_readout": False,
            "window_size": WINDOW_SIZE,
        },
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "selection_metric": "pretraining_val_loss",
        "transfer_scope": "detector",
        "limits": audit["limits"],
    }
    a.output_dir.mkdir(parents=True)
    best = float("inf")
    stale = 0
    history = []
    (a.output_dir / "status.json").write_text(json.dumps({"status": "running"}) + "\n")
    for e in range(1, a.epochs + 1):
        train_loss = epoch(detector, train_loader, device, optimizer)
        # Stable validation, independent of training RNG.
        val_loss = epoch(detector, val_loader, device)
        history.append({"epoch": e, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {e}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            stale = 0
            save_checkpoint(a.output_dir / "pretrained_best.pt", detector,
                            metadata | {"epoch": e, "selection_value": best})
        else:
            stale += 1
        (a.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= a.patience:
            break
    (a.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata | {"best_val_loss": best}, indent=2) + "\n")
    (a.output_dir / "status.json").write_text(
        json.dumps({"status": "completed", "epochs": len(history)}) + "\n")


if __name__ == "__main__":
    main()
