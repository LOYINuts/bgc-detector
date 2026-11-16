"""Pretraining of the detector on labeled windows and frozen ESMC embeddings.

Trains the final detector architecture on audited antiSMASH windows with the
same pos-weighted BCE the detector uses, selecting by pretraining validation
loss. The best checkpoint is transferred into supervised detection training
with ``train.py --pretrained-checkpoint``.

Usage
-----
    uv run python -m bgc_gnn.pretrain \
        --data-dir data/derived/pretraining/antismash_v2_v1 \
        --cache-dir data/derived/pretraining/antismash_v2_v1_restart_v2_cache \
        --output-dir runs/pretrain_seed1301 --seed 1301
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from bgc_gnn.checkpoint import file_fingerprint, runtime_metadata, save_checkpoint
from bgc_gnn.model import (
    ARCHITECTURE,
    DROPOUT,
    EMBEDDING_DIM,
    HEAD_DROPOUT,
    WINDOW_SIZE,
    BGCDetector,
)

EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 32
LR = 1e-4
COLUMNS = ("sequence_ids", "valid_mask", "detect_mask", "loss_mask")


class PretrainingDataset(Dataset):
    def __init__(self, windows, cache_dir, values=None):
        self.rows = pq.read_table(windows, columns=list(COLUMNS))
        ids = json.loads((cache_dir / "sequence_ids.json").read_text())
        self.index = {s: i for i, s in enumerate(ids)}
        # Share a read-only mapping across splits; never allocate a private full cache.
        self.values = (
            np.load(cache_dir / "embeddings.npy", mmap_mode="r")
            if values is None
            else values
        )
        if (
            len(ids) != len(self.index)
            or self.values.ndim != 2
            or len(ids) != len(self.values)
        ):
            raise ValueError("cache IDs/embedding shape mismatch")
        self.dim = self.values.shape[1]
        self.positive = 0
        self.negative = 0
        for batch in self.rows.to_batches(max_chunksize=512):
            for row in batch.to_pylist():
                valid = row["valid_mask"]
                if any(len(row[k]) != WINDOW_SIZE for k in COLUMNS):
                    raise ValueError(
                        f"windows must have {WINDOW_SIZE} aligned positions"
                    )
                length = sum(valid)
                if not length or valid != [True] * length + [False] * (
                    WINDOW_SIZE - length
                ):
                    raise ValueError("windows require non-empty valid prefixes")
                if any(
                    active and sid not in self.index
                    for sid, active in zip(row["sequence_ids"], valid)
                ):
                    raise ValueError("valid window gene missing from cache")
                if any(
                    supervised and not active
                    for supervised, active in zip(row["loss_mask"], valid)
                ):
                    raise ValueError("loss mask includes padding")
                self.positive += sum(
                    t and m for t, m in zip(row["detect_mask"], row["loss_mask"])
                )
                self.negative += sum(
                    not t and m for t, m in zip(row["detect_mask"], row["loss_mask"])
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = {name: self.rows[name][i].as_py() for name in self.rows.column_names}
        length = sum(row["valid_mask"])
        x = np.zeros((WINDOW_SIZE, self.dim), dtype=np.float32)
        x[:length] = self.values[[self.index[s] for s in row["sequence_ids"][:length]]]
        return {
            "embeddings": torch.from_numpy(x),
            **{
                k: torch.tensor(row[k], dtype=torch.bool)
                for k in ("valid_mask", "detect_mask", "loss_mask")
            },
        }


def pretraining_loss(detector, x, valid, targets, supervised):
    if (supervised & ~valid).any():
        raise ValueError("pretraining loss mask includes padding")
    if not supervised.any():
        return None
    probabilities = detector(x, key_padding_mask=~valid)
    per_gene = F.binary_cross_entropy(probabilities, targets.float(), reduction="none")
    weights = torch.where(targets, 2.0, 1.0)
    return (per_gene * weights)[supervised].mean()


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    started = time.perf_counter()
    stage = "train" if optimizer is not None else "validation"
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index, batch in enumerate(loader, 1):
            x = batch["embeddings"].to(device)
            valid = batch["valid_mask"].to(device)
            if not torch.isfinite(x).all():
                raise ValueError("non-finite input embeddings")
            supervised = batch["loss_mask"].to(device)
            loss = pretraining_loss(
                model, x, valid, batch["detect_mask"].to(device), supervised
            )
            n = int(supervised.sum())
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("non-finite pretraining loss")
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * n
            count += n
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                elapsed = time.perf_counter() - started
                print(
                    f"  {stage} batch {batch_index}/{len(loader)} | "
                    f"loss={total / count:.6f} | elapsed={elapsed:.1f}s | "
                    f"batches/s={batch_index / max(elapsed, 1e-9):.2f}",
                    flush=True,
                )
    if not count:
        raise ValueError("split has no supervised positions for this task")
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1301)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    # Match the historical concurrent runs; large CPU pools stall batch collation.
    torch.set_num_threads(2)
    print(
        f"CPU compute threads: {torch.get_num_threads()}; device: {a.device}",
        flush=True,
    )
    if a.output_dir.exists():
        p.error("use a fresh output directory")

    audit = json.loads((a.data_dir / "audit.json").read_text())
    cache = json.loads((a.cache_dir / "metadata.json").read_text())
    if audit["status"] != "completed" or cache["status"] != "completed":
        p.error("requires completed data and embedding cache")
    inputs = {}
    for role, path, expected in [
        (
            "train",
            a.data_dir / "windows_train.parquet",
            audit["windows_train.parquet"]["sha256"],
        ),
        (
            "val",
            a.data_dir / "windows_val.parquet",
            audit["windows_val.parquet"]["sha256"],
        ),
        ("proteins", a.data_dir / "proteins.parquet", cache["proteins"]["sha256"]),
        ("embeddings", a.cache_dir / "embeddings.npy", cache["embeddings"]["sha256"]),
        (
            "sequence_ids",
            a.cache_dir / "sequence_ids.json",
            cache["sequence_ids"]["sha256"],
        ),
    ]:
        print(f"Checking input SHA256: {role} ({path})", flush=True)
        inputs[role] = file_fingerprint(path)
        if inputs[role]["sha256"] != expected:
            p.error(f"input fingerprint mismatch: {role}")
    if cache["model"] != "biohub/ESMC-300M" or cache["dim"] != EMBEDDING_DIM:
        p.error("pretraining requires frozen ESMC-300M 960d")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    print(
        "Opening read-only embedding mapping and validating dataset windows", flush=True
    )
    values = np.load(a.cache_dir / "embeddings.npy", mmap_mode="r")
    train = PretrainingDataset(
        a.data_dir / "windows_train.parquet", a.cache_dir, values
    )
    val = PretrainingDataset(a.data_dir / "windows_val.parquet", a.cache_dir, values)
    if (
        not len(train)
        or not len(val)
        or train.dim != EMBEDDING_DIM
        or val.dim != EMBEDDING_DIM
    ):
        p.error("invalid data dimensions or empty split")
    for split in (train, val):
        if not split.positive or not split.negative:
            p.error(
                "pretraining requires positive and background supervision in both splits"
            )

    detector = BGCDetector().to(device)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=LR, weight_decay=0.01)
    metadata = {
        "runtime": runtime_metadata()
        | {"cpu_compute_threads": torch.get_num_threads()},
        "stage": "pretraining",
        "checkpoint_kind": "pretraining_best",
        "inputs": inputs,
        "cache": cache,
        "architecture": dict(ARCHITECTURE),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
        }
        | {
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
            "dropout": DROPOUT,
            "head_dropout": HEAD_DROPOUT,
        },
        "selection_metric": "pretraining_val_loss",
        "transfer_scope": "detector",
        "limits": audit["limits"],
    }
    a.output_dir.mkdir(parents=True)
    best = float("inf")
    stale = 0
    history = []
    (a.output_dir / "status.json").write_text(json.dumps({"status": "running"}) + "\n")
    for e in range(1, EPOCHS + 1):
        print(f"Epoch {e}/{EPOCHS}: starting", flush=True)
        train_loss = epoch(detector, train_loader, device, optimizer)
        # Stable validation, independent of training RNG.
        val_loss = epoch(detector, val_loader, device)
        history.append({"epoch": e, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {e}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            stale = 0
            save_checkpoint(
                a.output_dir / "pretrained_best.pt",
                detector,
                metadata | {"epoch": e, "selection_value": best},
            )
        else:
            stale += 1
        (a.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= PATIENCE:
            break
    (a.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata | {"best_val_loss": best}, indent=2) + "\n"
    )
    (a.output_dir / "status.json").write_text(
        json.dumps({"status": "completed", "epochs": len(history)}) + "\n"
    )


if __name__ == "__main__":
    main()
