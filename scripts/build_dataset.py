"""Pre-generate fixed BGC detection windows (train/val parquet).

BGC IDs and GTDB genomes are split before augmentation, so one entity cannot
cross the split. Writes five positive windows per BGC plus the requested number
of negative windows; only gene IDs are stored, embeddings are reused as-is.

Usage
-----
    uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure project root is on path so `bgc_gnn` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

from bgc_gnn.dataset import BGCDetectDataset, EmbeddingStore, load_index

WINDOW_SIZE = 128
AUGMENT = 5
N_NEGATIVE = 20000
SEED = 42


def split_entities(
    items: list[dict], val_fraction: float, seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split entities before augmentation so one entity cannot cross splits."""
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0:
        return list(items), []
    if len(items) < 2:
        raise ValueError("grouped split requires at least two entities")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_val = min(len(items) - 1, max(1, int(len(items) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def entity_gene_pool(index: list[dict]) -> list[str]:
    return [gene_id for entity in index for gene_id in entity["gene_ids"]]


def generate_samples(
    dataset: BGCDetectDataset, n_negative: int, sample_offset: int = 0,
) -> list[dict]:
    n_pos = len(dataset.bgc_index) * dataset.augment
    samples: list[dict] = []
    for i in range(n_pos):
        sample = dataset._make_positive(i)
        sample["sample_idx"] = sample_offset + len(samples)
        samples.append(sample)
        if (i + 1) % 1000 == 0:
            print(f"  pos: {i + 1:,}/{n_pos:,}")

    for i in range(n_negative):
        sample = dataset._make_negative(i)
        sample["sample_idx"] = sample_offset + len(samples)
        samples.append(sample)
        if (i + 1) % 2000 == 0:
            print(f"  neg: {i + 1:,}/{n_negative:,}")
    return samples


def _to_table(samples: list[dict]) -> pa.Table:
    return pa.table({
        "sample_idx": pa.array([s["sample_idx"] for s in samples], pa.int32()),
        "gene_ids": pa.array([s["gene_ids"] for s in samples], pa.list_(pa.string())),
        "detect_mask": pa.array([s["detect_mask"].tolist() for s in samples],
                                pa.list_(pa.bool_())),
        "is_positive": pa.array([s["is_positive"] for s in samples], pa.bool_()),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-dir", type=str, default="data/derived/catalog")
    parser.add_argument("--emb-path", type=str, default="",
                        help="Embedding parquet used to validate generated gene IDs")
    parser.add_argument("--output", type=str, default="data/derived/train_test/dataset.parquet")
    parser.add_argument("--pos_ratio", type=float, default=0.5,
                        help="Detection positive sample ratio")
    parser.add_argument("--n_negative", type=int, default=N_NEGATIVE,
                        help=f"Number of negative windows (default {N_NEGATIVE})")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of entities for val (0=no split, output one file)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Random seed for the entity split")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog_dir)
    random.seed(SEED)

    print("Loading indices ...")
    bgc_index = load_index(catalog_dir / "bgc_index.parquet")
    gtdb_index = load_index(catalog_dir / "gtdb_index.parquet")
    neg_pool = pq.read_table(catalog_dir / "negative_pool.parquet") \
                 .column("gene_id").to_pylist()
    print(f"  BGCs: {len(bgc_index)}  Genomes: {len(gtdb_index)}  "
          f"Negative pool: {len(neg_pool):,}")

    emb_path = Path(args.emb_path) if args.emb_path else \
        catalog_dir / "gene_embeddings_960d.parquet"
    if not emb_path.is_file():
        raise FileNotFoundError(f"embedding parquet not found: {emb_path}")
    print(f"Loading embeddings from {emb_path} ...")
    store = EmbeddingStore(emb_path)
    print(f"  {len(store):,} embeddings loaded")

    def build(bgc: list[dict], gtdb: list[dict], n_neg: int,
              offset: int, seed: int) -> list[dict]:
        dataset = BGCDetectDataset(
            bgc, gtdb, entity_gene_pool(gtdb), store,
            window_size=WINDOW_SIZE, augment=AUGMENT, pos_ratio=args.pos_ratio,
        )
        print(f"\nGenerating {len(bgc) * AUGMENT:,} positive, {n_neg:,} negative windows ...")
        random.seed(seed)
        return generate_samples(dataset, n_neg, sample_offset=offset)

    n_neg = args.n_negative
    if args.val_split > 0:
        train_bgc, val_bgc = split_entities(bgc_index, args.val_split, args.split_seed)
        train_gtdb, val_gtdb = split_entities(
            gtdb_index, args.val_split, args.split_seed + 1,
        )
        n_val_neg = int(n_neg * args.val_split)
        print(f"Grouped split: BGCs {len(train_bgc):,}/{len(val_bgc):,}, "
              f"genomes {len(train_gtdb):,}/{len(val_gtdb):,} (train/val)")
        train_samples = build(train_bgc, train_gtdb, n_neg - n_val_neg, 0, SEED)
        val_samples = build(val_bgc, val_gtdb, n_val_neg, len(train_samples), SEED + 1)
    else:
        train_samples = build(bgc_index, gtdb_index, n_neg, 0, SEED)
        val_samples = []

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if val_samples:
        paths = [
            (out.with_name(out.stem + "_train.parquet"), train_samples, "train"),
            (out.with_name(out.stem + "_val.parquet"), val_samples, "val"),
        ]
    else:
        paths = [(out, train_samples, "all")]

    for path, samples, label in paths:
        pq.write_table(_to_table(samples), path)
        n_pos = sum(s["is_positive"] for s in samples)
        print(f"\nSaved {label} ({len(samples):,}, {n_pos:,} pos) -> {path}")
        print(f"  Size: {path.stat().st_size / 1024**2:.1f} MB")

    avg_bgc_genes = sum(sum(s["detect_mask"]) for s in train_samples) / max(
        1, sum(s["is_positive"] for s in train_samples))
    print(f"  Avg BGC genes per positive window: {avg_bgc_genes:.1f}")


if __name__ == "__main__":
    main()
