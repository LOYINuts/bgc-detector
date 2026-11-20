"""Build fixed 128-gene windows using the final experiment's sample split.

Generation seed 42, five positive windows per BGC, 20,000 negative windows,
then an 80/20 sample split with seed 42. Reuse the recorded training artifacts
for exact experiment reproduction; rebuilding data does not establish identical fingerprints.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bgc_detector.dataset import BGCDetectDataset, EmbeddingStore, load_index

WINDOW_SIZE = 128
AUGMENT = 5
N_NEGATIVE = 20000
SEED = 42


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Match the original sample split while preserving stored row order."""
    indices = list(range(len(samples)))
    random.Random(SEED).shuffle(indices)
    val_set = set(indices[: int(len(samples) * 0.2)])
    return (
        [s for i, s in enumerate(samples) if i not in val_set],
        [s for i, s in enumerate(samples) if i in val_set],
    )


def generate_samples(dataset: BGCDetectDataset, n_negative: int) -> list[dict]:
    samples = []
    for i in range(len(dataset.bgc_index) * dataset.augment):
        sample = dataset._make_positive(i)
        sample["sample_idx"] = len(samples)
        samples.append(sample)
    for i in range(n_negative):
        sample = dataset._make_negative(i)
        sample["sample_idx"] = len(samples)
        samples.append(sample)
    return samples


def _to_table(samples: list[dict]) -> pa.Table:
    return pa.table(
        {
            "sample_idx": pa.array([s["sample_idx"] for s in samples], pa.int32()),
            "gene_ids": pa.array(
                [s["gene_ids"] for s in samples], pa.list_(pa.string())
            ),
            "detect_mask": pa.array(
                [s["detect_mask"].tolist() for s in samples], pa.list_(pa.bool_())
            ),
            "is_positive": pa.array([s["is_positive"] for s in samples], pa.bool_()),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-dir", type=Path, default=Path("data/derived/catalog")
    )
    parser.add_argument("--emb-path", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("data/derived/train_test/dataset.parquet")
    )
    args = parser.parse_args()
    paths = [
        args.output.with_name(args.output.stem + suffix)
        for suffix in ["_train.parquet", "_val.parquet"]
    ]
    if any(p.exists() for p in paths):
        parser.error(
            "output already exists; preserve existing datasets and use a fresh output path"
        )
    bgc_index = load_index(args.catalog_dir / "bgc_index.parquet")
    gtdb_index = load_index(args.catalog_dir / "gtdb_index.parquet")
    neg_pool = pq.read_table(args.catalog_dir / "negative_pool.parquet")[
        "gene_id"
    ].to_pylist()
    store = EmbeddingStore(
        args.emb_path or args.catalog_dir / "gene_embeddings_960d.parquet"
    )
    dataset = BGCDetectDataset(
        bgc_index, gtdb_index, neg_pool, store, window_size=WINDOW_SIZE, augment=AUGMENT
    )
    random.seed(SEED)
    train, val = split_samples(generate_samples(dataset, N_NEGATIVE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, samples in zip(paths, [train, val]):
        pq.write_table(_to_table(samples), path)
        print(f"Saved {len(samples):,} samples -> {path}")


if __name__ == "__main__":
    main()
