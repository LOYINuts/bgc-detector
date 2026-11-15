"""Detection datasets and embedding lookup.

Training reads pre-generated fixed 128-gene windows (``build_dataset.py``) and
looks up per-gene ESMC embeddings by gene ID.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

WINDOW_SIZE = 128


def normalize_translation(translation: object) -> str:
    """Remove whitespace and trailing stop markers from a translation."""
    return str(translation or "").strip().rstrip("*")


def normalize_strand(strand: object) -> str | None:
    """Normalize catalog and GenBank strand values to ``+`` or ``-``."""
    if strand is None:
        return None
    if strand in ("+", 1, "1"):
        return "+"
    if strand in ("-", -1, "-1"):
        return "-"
    raise ValueError(f"unknown strand value: {strand!r}")


class EmbeddingStore:
    """In-memory mapping from gene_id to float32 embedding."""

    def __init__(self, path: str | Path):
        table = pq.read_table(path)
        self._map: dict[str, np.ndarray] = {
            gid: np.array(emb, dtype=np.float32)
            for gid, emb in zip(
                table.column("gene_id").to_pylist(),
                table.column("embedding").to_pylist(),
            )
        }
        self.emb_dim: int = next(iter(self._map.values())).shape[0] if self._map else 0

    def get(self, gene_ids: list[str]) -> np.ndarray:
        """Return [L, D] float32 array."""
        return np.stack([self._map[gid] for gid in gene_ids]).astype(np.float32)

    def __contains__(self, gene_id: str) -> bool:
        return gene_id in self._map

    def __len__(self) -> int:
        return len(self._map)


def _source_runs(entity: dict, namespace: str) -> list[list[str]]:
    """Split an entity's genes into runs of contiguous source positions."""
    gene_ids = entity.get("source_gene_ids", entity["gene_ids"])
    if not gene_ids:
        return []
    entity_id = entity.get(
        "source_entity", entity.get("bgc_id", entity.get("genome_id"))
    )
    expected_prefix = f"{namespace}:{entity_id}" if entity_id is not None else None
    positions: list[tuple[str, int] | None] = []
    for gene_id in gene_ids:
        prefix, separator, suffix = gene_id.rpartition(":")
        positions.append(
            (prefix, int(suffix)) if separator and suffix.isdigit() else None
        )
    if all(position is None for position in positions):
        return [gene_ids]
    if any(position is None for position in positions):
        return []
    if expected_prefix is not None and any(
        prefix != expected_prefix for prefix, _ in positions
    ):
        return []
    runs: list[list[str]] = []
    previous_prefix = ""
    previous_position = -2
    contigs = entity.get("source_contigs")
    if contigs is not None and len(contigs) != len(gene_ids):
        raise ValueError("source contigs must align with source gene IDs")
    previous_contig = None
    for i, (gene_id, (prefix, position)) in enumerate(zip(gene_ids, positions)):
        contig = contigs[i] if contigs is not None else None
        if (
            not runs
            or prefix != previous_prefix
            or position != previous_position + 1
            or contig != previous_contig
        ):
            runs.append([])
        runs[-1].append(gene_id)
        previous_prefix, previous_position = prefix, position
        previous_contig = contig
    return runs


def _positive_source_window(
    bgc: dict,
    bgc_genes: list[str],
    window_size: int,
) -> tuple[list[str], torch.Tensor] | None:
    bgc_set = set(bgc_genes)
    for run in _source_runs(bgc, "bgc"):
        positions = {gene_id: i for i, gene_id in enumerate(run)}
        if not bgc_set.issubset(positions) or len(run) < window_size:
            continue
        bgc_positions = [positions[gene_id] for gene_id in bgc_genes]
        low = max(0, max(bgc_positions) - window_size + 1)
        high = min(min(bgc_positions), len(run) - window_size)
        if low > high:
            continue
        start = random.randint(low, high)
        gene_ids = run[start : start + window_size]
        mask = torch.tensor([gene_id in bgc_set for gene_id in gene_ids])
        return gene_ids, mask
    return None


class BGCDetectDataset(Dataset):
    """Random positive/negative windows; used by ``build_dataset.py``.

    Positive: a real BGC-source window when it fits, otherwise the BGC placed
    among contiguous GTDB context genes. Negative: contiguous GTDB window.
    """

    def __init__(
        self,
        bgc_index: list[dict],
        gtdb_index: list[dict],
        negative_pool: list[str],
        embedding_store: EmbeddingStore,
        window_size: int = WINDOW_SIZE,
        pos_ratio: float = 0.5,
        augment: int = 5,
    ):
        if not (0 < pos_ratio <= 1):
            raise ValueError(f"pos_ratio must be in (0, 1], got {pos_ratio}")
        self.bgc_index = bgc_index
        self.gtdb_index = gtdb_index
        self.negative_pool = negative_pool
        self.store = embedding_store
        self.window_size = window_size
        self.pos_ratio = pos_ratio
        self.augment = augment
        n_pos = len(bgc_index) * augment
        n_neg = round(n_pos / pos_ratio - n_pos) if pos_ratio < 1 else 0
        self.n_samples = n_pos + n_neg

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        n_pos = len(self.bgc_index) * self.augment
        if idx < n_pos:
            return self._make_positive(idx)
        return self._make_negative(idx - n_pos)

    def _make_positive(self, idx: int) -> dict:
        bgc = self.bgc_index[idx % len(self.bgc_index)]
        bgc_genes = bgc["gene_ids"]
        L = self.window_size

        if len(bgc_genes) > L:
            start = random.randint(0, len(bgc_genes) - L)
            bgc_genes = bgc_genes[start : start + L]

        source_window = _positive_source_window(bgc, bgc_genes, L)
        if source_window is not None:
            gene_ids, mask = source_window
        else:
            n_bgc = len(bgc_genes)
            pos = random.randint(0, L - n_bgc)
            n_pad = L - n_bgc
            pad_runs = [
                run
                for genome in self.gtdb_index
                for run in _source_runs(genome, "gtdb")
            ]
            long_runs = [run for run in pad_runs if len(run) >= n_pad]
            if n_pad == 0:
                pad_genes = []
            elif long_runs:
                run = random.choice(long_runs)
                start = random.randint(0, len(run) - n_pad)
                pad_genes = run[start : start + n_pad]
            else:
                short_runs = [run for run in pad_runs if run]
                genome_genes = random.choice(short_runs) if short_runs else []
                pad_genes = genome_genes + random.sample(
                    self.negative_pool,
                    k=n_pad - len(genome_genes),
                )
            gene_ids = pad_genes[:pos] + bgc_genes + pad_genes[pos:]
            mask = torch.zeros(L, dtype=torch.bool)
            mask[pos : pos + n_bgc] = True

        return {
            "embeddings": torch.from_numpy(self.store.get(gene_ids)),
            "detect_mask": mask,
            "valid_mask": torch.ones(L, dtype=torch.bool),  # store has all real genes
            "is_positive": True,
            "gene_ids": gene_ids,
        }

    def _make_negative(self, idx: int) -> dict:
        L = self.window_size
        source_runs = [
            run for genome in self.gtdb_index for run in _source_runs(genome, "gtdb")
        ]
        long_runs = [run for run in source_runs if len(run) >= L]
        if long_runs:
            genes = random.choice(long_runs)
            start = random.randint(0, len(genes) - L)
            gene_ids = genes[start : start + L]
        else:
            genome_runs = _source_runs(
                self.gtdb_index[idx % len(self.gtdb_index)],
                "gtdb",
            )
            base_runs = [run for run in genome_runs if run]
            genes = random.choice(base_runs) if base_runs else []
            extra_runs = [run for run in source_runs if run]
            extra = random.choice(extra_runs) if extra_runs else []
            remaining = L - len(genes)
            if len(extra) > remaining:
                start = random.randint(0, len(extra) - remaining)
                gene_ids = genes + extra[start : start + remaining]
            else:
                gene_ids = (
                    genes
                    + extra
                    + random.sample(
                        self.negative_pool,
                        k=remaining - len(extra),
                    )
                )

        return {
            "embeddings": torch.from_numpy(self.store.get(gene_ids)),
            "detect_mask": torch.zeros(L, dtype=torch.bool),
            "valid_mask": torch.ones(L, dtype=torch.bool),
            "is_positive": False,
            "gene_ids": gene_ids,
        }


class FixedBGCDataset(Dataset):
    """Pre-generated fixed windows from parquet, same windows every epoch."""

    def __init__(self, path: str | Path, store: EmbeddingStore, emb_dim: int):
        table = pq.read_table(path)
        self.gene_ids = table.column("gene_ids").to_pylist()
        self.masks = table.column("detect_mask").to_pylist()
        self.is_pos = table.column("is_positive").to_pylist()
        self.valid_masks = (
            table.column("valid_mask").to_pylist()
            if "valid_mask" in table.column_names
            else None
        )
        self.store = store
        self.emb_dim = emb_dim
        missing = set()
        for i, gids in enumerate(self.gene_ids):
            valid = (
                self.valid_masks[i]
                if self.valid_masks is not None
                else [True] * len(gids)
            )
            if len(valid) != len(gids) or len(self.masks[i]) != len(gids):
                raise ValueError(f"window {i} has misaligned gene IDs and masks")
            missing.update(
                gid for gid, active in zip(gids, valid) if active and gid not in store
            )
        if missing:
            examples = ", ".join(sorted(missing)[:5])
            raise ValueError(
                f"{len(missing)} real genes lack embeddings in {path}: {examples}. "
                "Complete the embedding cache before training; missing real genes cannot be zero padded."
            )

    def __len__(self) -> int:
        return len(self.gene_ids)

    def __getitem__(self, idx: int) -> dict:
        gids = self.gene_ids[idx]
        # Only explicit padding positions may remain zero.
        embs = np.zeros((len(gids), self.emb_dim), dtype=np.float32)
        for i, gid in enumerate(gids):
            if gid in self.store:
                embs[i] = self.store._map[gid]
        valid = (
            torch.tensor(self.valid_masks[idx], dtype=torch.bool)
            if self.valid_masks is not None
            else torch.ones(len(gids), dtype=torch.bool)
        )
        return {
            "embeddings": torch.from_numpy(embs),
            "detect_mask": torch.tensor(self.masks[idx], dtype=torch.bool),
            "valid_mask": valid,
            "is_positive": self.is_pos[idx],
        }


def load_index(path: str | Path) -> list[dict]:
    """Load bgc_index or gtdb_index parquet into list of dicts."""
    return pq.read_table(path).to_pylist()


def collate_detect_batch(batch: list[dict]) -> dict:
    """Collate detection samples into batched tensors."""
    return {
        "embeddings": torch.stack([s["embeddings"] for s in batch]),
        "detect_mask": torch.stack([s["detect_mask"] for s in batch]),
        "valid_mask": torch.stack([s["valid_mask"] for s in batch]),
        "is_positive": torch.tensor([s["is_positive"] for s in batch]),
    }
