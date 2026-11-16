"""Train the BGC detector: frozen ESMC embeddings -> Channel-Gated Conv-BiGRU.

Fixed recipe (weights, EMA, schedule and selection all fixed):
  weighted BCE (pos_weight 2), batch 64, AdamW lr 5e-4 wd 0.01,
  dropout 0.5 / head 0.1, EMA 0.999 evaluated and exported,
  ExponentialLR gamma 0.95 starting at epoch 5 (no warmup),
  checkpoint selection by validation loss, patience 20,
  validation-MCC threshold saved with the checkpoint.

Pretrained detector transfer is required. The semantic skip is trained during both
stages; transferred encoder weights always use learning-rate multiplier 0.2.

Usage
-----
    uv run python -m bgc_gnn.train \
        --fixed-dataset data/derived/train_test/dataset_train.parquet \
        --fixed-val-dataset data/derived/train_test/dataset_val.parquet \
        --emb-path data/derived/catalog/gene_embeddings_960d.parquet \
        --checkpoint-dir runs/seed1301 --seed 1301 \
        --pretrained-checkpoint pretrain/seed1301/pretrained_best.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from bgc_gnn.checkpoint import (
    file_fingerprint,
    initialize_from_pretraining,
    runtime_metadata,
    save_checkpoint,
    write_metadata,
)
from bgc_gnn.dataset import EmbeddingStore, FixedBGCDataset, collate_detect_batch
from bgc_gnn.eval import compute_detect_metrics, select_detection_threshold
from bgc_gnn.model import (
    ARCHITECTURE,
    DROPOUT,
    EMBEDDING_DIM,
    HEAD_DROPOUT,
    BGCDetector,
)
from bgc_gnn.runtime import configure_cpu_threads

BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 0.01
POS_WEIGHT = 2.0
EMA_DECAY = 0.999
LR_GAMMA = 0.95
LR_START_EPOCH = 5
MIN_LR = 5e-6
EPOCHS = 100
PATIENCE = 20
THRESHOLD_METRIC = "mcc"
ENCODER_LR_SCALE = 0.2
ENCODER_PREFIXES = ("input_norm.", "input_proj.", "encoder.")


def weighted_bce(
    outputs: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, pos_weight: float
) -> torch.Tensor:
    """Pos-weight-weighted BCE averaged over real (non-padded) genes."""
    per_pos = F.binary_cross_entropy(
        outputs,
        targets,
        reduction="none",
    ) * torch.where(
        targets > 0, torch.as_tensor(pos_weight, device=targets.device), 1.0
    )
    return (per_pos * valid).sum() / valid.sum().clamp(min=1.0)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    ema_model: AveragedModel,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    n_batches = len(dataloader)
    print_every = max(1, min(10, n_batches // 20))

    for batch in dataloader:
        embs = batch["embeddings"].to(device)
        targets = batch["detect_mask"].float().to(device)
        valid = batch["valid_mask"].float().to(device)
        outputs = model(embs, key_padding_mask=(valid == 0))
        loss = weighted_bce(outputs, targets, valid, POS_WEIGHT)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_model.update_parameters(model)
        total_loss += loss.item()
        n += 1
        if n % print_every == 0 or n == n_batches:
            print(
                f"  [E{epoch}/{total_epochs}] batch {n:4d}/{n_batches} | "
                f"loss={total_loss / n:.4f}",
                flush=True,
            )

    return {"loss": total_loss / n}


@torch.no_grad()
def val_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_scores: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []

    for batch in dataloader:
        embs = batch["embeddings"].to(device)
        targets = batch["detect_mask"].float().to(device)
        valid = batch["valid_mask"].float().to(device)
        outputs = model(embs, key_padding_mask=(valid == 0))
        total_loss += weighted_bce(outputs, targets, valid, POS_WEIGHT).item()
        # Padded positions are not genes and must not count as negatives.
        keep = (valid != 0).cpu()
        all_scores.append(outputs.cpu()[keep])
        all_masks.append(targets.cpu()[keep])

    y_score = torch.cat(all_scores).numpy().ravel()
    y_true = torch.cat(all_masks).numpy().ravel().astype(int)
    threshold, threshold_value = select_detection_threshold(
        y_true,
        y_score,
        THRESHOLD_METRIC,
    )
    metrics = compute_detect_metrics(y_true, y_score, threshold=threshold)
    metrics["validation_threshold"] = threshold
    metrics["validation_threshold_metric"] = THRESHOLD_METRIC
    metrics["validation_threshold_value"] = threshold_value
    metrics["val_loss"] = total_loss / len(dataloader)
    return metrics


def build_optimizer(model: torch.nn.Module) -> AdamW:
    """Fixed encoder LR 1e-4 and detector/semantic head LR 5e-4."""
    encoder_parameters = [
        p for name, p in model.named_parameters() if name.startswith(ENCODER_PREFIXES)
    ]
    head_parameters = [
        p
        for name, p in model.named_parameters()
        if not name.startswith(ENCODER_PREFIXES)
    ]
    if not encoder_parameters or not head_parameters:
        raise ValueError(
            "pretrained encoder learning-rate groups must both be nonempty"
        )
    return AdamW(
        [
            {"params": encoder_parameters, "lr": LR * ENCODER_LR_SCALE},
            {"params": head_parameters, "lr": LR},
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dataset", type=str, required=True)
    parser.add_argument("--fixed-val-dataset", type=str, required=True)
    parser.add_argument(
        "--emb-path",
        type=str,
        required=True,
        help="Frozen ESMC-300M 960d embedding parquet",
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Semantic-enabled pretraining_best checkpoint",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    try:
        configure_cpu_threads()
    except ValueError as exc:
        parser.error(str(exc))
    print(f"CPU compute threads: {torch.get_num_threads()}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        parser.error("checkpoint directory is not empty; use a fresh output directory")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.emb_path} ...")
    t0 = time.time()
    store = EmbeddingStore(args.emb_path)
    embedding_dim = store.emb_dim
    if embedding_dim != EMBEDDING_DIM:
        parser.error("the final model requires ESMC-300M 960d embeddings")
    print(f"  {len(store):,} embeddings ({embedding_dim}d) in {time.time() - t0:.0f}s")

    train_ds = FixedBGCDataset(args.fixed_dataset, store, emb_dim=embedding_dim)
    val_ds = FixedBGCDataset(args.fixed_val_dataset, store, emb_dim=embedding_dim)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_detect_batch,
        num_workers=args.num_workers,
        pin_memory=bool(args.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_detect_batch,
        num_workers=args.num_workers,
        pin_memory=bool(args.num_workers),
    )
    print(f"  Train samples/epoch: {len(train_ds):,}  Val: {len(val_ds):,}")

    model = BGCDetector().to(device)
    print(
        f"BGCDetector: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
    )

    metadata = {
        "metadata_version": 1,
        "cli": vars(args).copy(),
        "runtime": runtime_metadata(),
        "embedding": file_fingerprint(args.emb_path),
        "architecture": dict(ARCHITECTURE),
        "training": {
            "loss": "weighted_bce",
            "pos_weight": POS_WEIGHT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "head_dropout": HEAD_DROPOUT,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "encoder_lr_scale": ENCODER_LR_SCALE,
            "ema_decay": EMA_DECAY,
            "lr_schedule": "exponential",
            "lr_gamma": LR_GAMMA,
            "lr_start_epoch": LR_START_EPOCH,
        },
        "data_train": file_fingerprint(args.fixed_dataset),
        "data_val": file_fingerprint(args.fixed_val_dataset),
        "weight_source": "ema",
    }
    metadata["pretraining_initialization"] = initialize_from_pretraining(
        model,
        args.pretrained_checkpoint,
    )
    if (
        metadata["pretraining_initialization"]["source_metadata"]["settings"]["seed"]
        != args.seed
    ):
        raise ValueError("pretraining checkpoint seed must match the finetuning seed")
    print(
        f"Initialized from {args.pretrained_checkpoint} (complete pretrained detector transfer)"
    )
    write_metadata(checkpoint_dir / "run_metadata.json", metadata)

    optimizer = build_optimizer(model)
    scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    ema_model.requires_grad_(False)
    print(f"EMA decay={EMA_DECAY}; validation and exported checkpoints use EMA weights")

    best_val = float("inf")
    best_epoch = 0
    best_threshold = 0.5
    best_threshold_value = 0.0
    patience_counter = 0
    prefix = f"detector_{embedding_dim}d"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---", flush=True)
        print(
            f"Training learning rates: {[g['lr'] for g in optimizer.param_groups]}",
            flush=True,
        )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            EPOCHS,
            ema_model,
        )
        val_metrics = val_epoch(ema_model.module, val_loader, device)
        # Fixed exponential schedule: first decay follows epoch 5.
        if epoch >= LR_START_EPOCH and max(scheduler.get_last_lr()) > MIN_LR:
            scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} f1={val_metrics['f1']:.4f} "
            f"auroc={val_metrics['auroc']:.4f} auprc={val_metrics['auprc']:.4f} "
            f"mcc={val_metrics['mcc']:.4f} | {time.time() - t0:.0f}s",
            flush=True,
        )

        metadata["ema_updates"] = int(ema_model.n_averaged.item())
        save_checkpoint(
            checkpoint_dir / f"{prefix}_epoch{epoch:03d}.pt",
            ema_model.module,
            metadata | {"epoch": epoch, "checkpoint_kind": "epoch"},
        )
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            best_epoch = epoch
            best_threshold = val_metrics["validation_threshold"]
            best_threshold_value = val_metrics["validation_threshold_value"]
            patience_counter = 0
            save_checkpoint(
                checkpoint_dir / f"{prefix}_best.pt",
                ema_model.module,
                metadata
                | {
                    "epoch": epoch,
                    "checkpoint_kind": "best",
                    "selection_metric": "val_loss",
                    "selection_value": best_val,
                    "validation_threshold": best_threshold,
                    "validation_threshold_metric": THRESHOLD_METRIC,
                    "validation_threshold_value": best_threshold_value,
                    "validation_threshold_source": "validation",
                },
            )
            print(
                f"  -> best (val_loss={best_val:.4f}, threshold="
                f"{best_threshold:.4f} by {THRESHOLD_METRIC})"
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} (val_loss not improving "
                f"for {PATIENCE} epochs)"
            )
            break

    print(
        f"\nDone. Best val_loss: {best_val:.4f} (epoch {best_epoch}, "
        f"threshold {best_threshold:.4f} by {THRESHOLD_METRIC})"
    )
    metadata |= {
        "best_validation_threshold": best_threshold,
        "threshold_metric": THRESHOLD_METRIC,
        "selected_epoch": best_epoch,
        "best_validation_threshold_value": best_threshold_value,
        "validation_threshold_source": "validation",
    }
    write_metadata(checkpoint_dir / "run_metadata.json", metadata)


if __name__ == "__main__":
    main()
