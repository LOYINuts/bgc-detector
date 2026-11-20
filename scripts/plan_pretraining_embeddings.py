"""Plan v2 embedding reuse with a disk index, without loading protein sequences at once."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pyarrow.parquet as pq

from bgc_detector.checkpoint import file_fingerprint, runtime_metadata


def plan(data_dir, reuse_cache, output_dir):
    if output_dir.exists():
        raise ValueError("use a fresh output directory")
    audit = json.loads((data_dir / "audit.json").read_text())
    metadata = json.loads((reuse_cache / "metadata.json").read_text())
    if audit["status"] != "completed" or metadata["status"] != "completed":
        raise ValueError("data and reuse cache must be completed")
    if metadata["model"] != "biohub/ESMC-300M" or metadata["dim"] != 960:
        raise ValueError("reuse cache requires ESMC-300M 960d")
    proteins = file_fingerprint(data_dir / "proteins.parquet")
    if proteins["sha256"] != audit["proteins.parquet"]["sha256"]:
        raise ValueError("protein catalog changed after data preparation")
    fingerprints = {}
    for key, name in [
        ("embeddings", "embeddings.npy"),
        ("sequence_ids", "sequence_ids.json"),
    ]:
        fingerprints[key] = file_fingerprint(reuse_cache / name)
        if fingerprints[key]["sha256"] != metadata[key]["sha256"]:
            raise ValueError(f"reuse cache fingerprint mismatch: {key}")
    ids = json.loads((reuse_cache / "sequence_ids.json").read_text())
    array = np.load(reuse_cache / "embeddings.npy", mmap_mode="r")
    if array.shape != (len(ids), 960) or array.dtype != np.float32:
        raise ValueError("reuse cache dimensions or dtype mismatch")
    del array
    output_dir.mkdir(parents=True)
    (output_dir / "status.json").write_text(json.dumps({"status": "running"}) + "\n")
    index_path = output_dir / "inventory.sqlite"
    with closing(sqlite3.connect(index_path)) as index:
        index.execute(
            "CREATE TABLE proteins (id TEXT PRIMARY KEY, length INTEGER NOT NULL, reused INTEGER NOT NULL DEFAULT 0)"
        )
        for batch in pq.ParquetFile(data_dir / "proteins.parquet").iter_batches(
            batch_size=8192, columns=["gene_id", "target_aa"]
        ):
            index.executemany(
                "INSERT INTO proteins(id,length) VALUES (?,?)",
                ((r["gene_id"], len(r["target_aa"])) for r in batch.to_pylist()),
            )
            index.commit()
        index.executemany(
            "UPDATE proteins SET reused=1 WHERE id=?", ((sid,) for sid in ids)
        )
        index.commit()
        total, reused = index.execute(
            "SELECT COUNT(*), COALESCE(SUM(reused),0) FROM proteins"
        ).fetchone()
        missing, longest, long_count, residues = index.execute(
            "SELECT COUNT(*), COALESCE(MAX(length),0), COALESCE(SUM(length>4096),0), COALESCE(SUM(length),0) FROM proteins WHERE reused=0"
        ).fetchone()
    index_path.unlink()
    result = {
        "status": "completed",
        "stage": "inventory_only_no_encoding",
        "runtime": runtime_metadata(),
        "proteins": proteins,
        "reuse_cache": str(reuse_cache),
        "reuse_fingerprints": fingerprints,
        "unique_proteins": total,
        "reusable_exact_sequence_embeddings": reused,
        "missing_to_encode": missing,
        "missing_AA_total": residues,
        "max_missing_protein_length": longest,
        "missing_over_4096_AA": long_count,
        "cache_bytes_float32": total * 960 * 4,
        "limits": [
            "Exact sequence reuse; no homology deduplication.",
            "Inventory does not encode proteins or prove downstream improvement.",
        ],
    }
    (output_dir / "embedding_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--reuse-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(plan(a.data_dir, a.reuse_cache, a.output_dir), indent=2), flush=True
    )


if __name__ == "__main__":
    main()
