"""Build a sequence-deduplicated ESMC cache, reusing known exact protein embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from bgc_gnn.checkpoint import file_fingerprint, runtime_metadata
from scripts.build_pretraining_data import protein_sequence, sequence_id


def reuse_embeddings(proteins, reuse_catalog, reuse_embeddings_path, array):
    index = {sid: i for i, sid in enumerate(proteins)}
    old_ids = {}
    for batch in pq.ParquetFile(reuse_catalog).iter_batches(
        columns=["gene_id", "target_aa"], batch_size=8192
    ):
        for row in batch.to_pylist():
            seq = protein_sequence(row["target_aa"])
            if seq is not None and (sid := sequence_id(seq)) in index:
                old_ids[row["gene_id"]] = index[sid]
    found = set()
    for batch in pq.ParquetFile(reuse_embeddings_path).iter_batches(
        columns=["gene_id", "embedding"], batch_size=2048
    ):
        for row in batch.to_pylist():
            target = old_ids.get(row["gene_id"])
            if target is None or target in found:
                continue
            value = np.asarray(row["embedding"], dtype=np.float32)
            if value.shape != (array.shape[1],) or not np.isfinite(value).all():
                raise ValueError(
                    "reused embeddings have invalid dimension or non-finite values"
                )
            array[target] = value
            found.add(target)
    return found


def reuse_cached_embeddings(proteins, cache_dir, array):
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if (
        metadata["status"] != "completed"
        or metadata["model"] != "biohub/ESMC-300M"
        or metadata["dim"] != array.shape[1]
    ):
        raise ValueError(
            "reuse cache must be completed ESMC-300M with matching dimensions"
        )
    for key, name in [
        ("embeddings", "embeddings.npy"),
        ("sequence_ids", "sequence_ids.json"),
    ]:
        if file_fingerprint(cache_dir / name)["sha256"] != metadata[key]["sha256"]:
            raise ValueError(f"reuse cache fingerprint mismatch: {key}")
    old_ids = json.loads((cache_dir / "sequence_ids.json").read_text())
    if len(old_ids) != len(set(old_ids)):
        raise ValueError("duplicate sequence IDs in reuse cache")
    index = {sid: i for i, sid in enumerate(proteins)}
    found = set()
    old = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    if old.shape != (len(old_ids), array.shape[1]) or old.dtype != np.float32:
        raise ValueError("reuse cache dimensions or dtype mismatch")
    for offset in range(0, len(old_ids), 2048):
        pairs = [
            (offset + j, index[sid])
            for j, sid in enumerate(old_ids[offset : offset + 2048])
            if sid in index
        ]
        if not pairs:
            continue
        source, target = zip(*pairs)
        values = old[list(source)]
        if not np.isfinite(values).all():
            raise ValueError("reuse cache contains non-finite values")
        array[list(target)] = values
        found.update(target)
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reuse-catalog", type=Path)
    p.add_argument("--reuse-embeddings", type=Path)
    p.add_argument(
        "--reuse-cache",
        type=Path,
        help="Completed sequence-deduplicated cache; alternative to catalog/parquet reuse",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Padded token budget; longer single proteins remain untruncated",
    )
    a = p.parse_args()
    if a.output_dir.exists():
        p.error("output directory exists; use a fresh cache directory")
    if bool(a.reuse_catalog) != bool(a.reuse_embeddings) or a.max_tokens < 1:
        p.error("provide both reuse paths and a positive token budget")
    if a.reuse_cache and a.reuse_catalog:
        p.error("choose either --reuse-cache or catalog/embedding reuse")
    audit = json.loads((a.data_dir / "audit.json").read_text())
    if audit["status"] != "completed":
        p.error("requires a completed data audit")
    if (
        file_fingerprint(a.data_dir / "proteins.parquet")["sha256"]
        != audit["proteins.parquet"]["sha256"]
    ):
        p.error("proteins file changed after audit")
    rows = pq.read_table(
        a.data_dir / "proteins.parquet", columns=["gene_id", "target_aa"]
    )
    ids = rows["gene_id"].to_pylist()
    sequences = rows["target_aa"]
    lengths = pc.utf8_length(sequences).to_numpy()
    if not len(rows) or len(ids) != len(set(ids)):
        p.error("protein catalog must contain distinct sequences")
    for batch in rows.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            if sequence_id(row["target_aa"]) != row["gene_id"]:
                p.error("protein sequence hash mismatch")
    a.output_dir.mkdir(parents=True)
    metadata = {
        "status": "incomplete",
        "runtime": runtime_metadata(),
        "model": "biohub/ESMC-300M",
        "dim": 960,
        "pool": "mean excluding BOS/EOS",
        "dtype": "bf16 model, float32 cache",
        "proteins": file_fingerprint(a.data_dir / "proteins.parquet"),
        "reuse_assumption": "Reuse supplied cache as ESMC-300M mean-pooled protein embeddings; sequence equality is verified, old model identity is a caller assertion.",
    }
    if a.reuse_catalog:
        metadata["reuse_catalog"] = file_fingerprint(a.reuse_catalog)
        metadata["reuse_embeddings"] = file_fingerprint(a.reuse_embeddings)
    if a.reuse_cache:
        metadata["reuse_cache"] = file_fingerprint(a.reuse_cache / "metadata.json")
    (a.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    array = np.lib.format.open_memmap(
        a.output_dir / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 960),
    )
    found = (
        reuse_embeddings(ids, a.reuse_catalog, a.reuse_embeddings, array)
        if a.reuse_catalog
        else set()
    )
    if a.reuse_cache:
        found = reuse_cached_embeddings(ids, a.reuse_cache, array)
    pending = sorted(
        (i for i in range(len(rows)) if i not in found), key=lambda i: lengths[i]
    )
    print(
        f"Reused {len(found)}/{len(rows)} exact protein embeddings; missing {len(pending)}",
        flush=True,
    )
    if pending:
        from scripts.encode import load_model, pool_batch

        device = torch.device(a.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model, tokenizer = load_model(device)
        from importlib.metadata import distribution

        native = distribution("esm")
        metadata["encoder_package"] = {
            "version": native.version,
            "source": native.read_text("direct_url.json"),
        }
        # Compare representative reused proteins before mixing old and new representations.
        sample = sorted(
            (i for i in found if 50 <= lengths[i] <= 1024),
            key=lambda i: (lengths[i], i),
        )
        sample = (
            [
                sample[i]
                for i in np.linspace(0, len(sample) - 1, min(8, len(sample)), dtype=int)
            ]
            if sample
            else []
        )
        checks = []
        for i in sample:
            enc = tokenizer([sequences[i].as_py()], return_tensors="pt", padding=True)
            with torch.inference_mode():
                hidden = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).last_hidden_state
                actual = pool_batch(hidden, enc["attention_mask"])[0]
            expected = np.asarray(array[i])
            relative_error = float(
                np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
            )
            if not np.isfinite(relative_error) or relative_error > 0.02:
                raise ValueError(
                    f"ESMC reuse consistency check failed: relative L2 error {relative_error}"
                )
            checks.append({"sequence_id": ids[i], "relative_l2_error": relative_error})
        if found and not sample:
            raise ValueError(
                "no representative reused proteins available for consistency check"
            )
        metadata["reuse_consistency_checks"] = checks
        (a.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(
            f"Validated {len(checks)} reused proteins against native ESMC", flush=True
        )
        start = 0
        while start < len(pending):
            batch = [pending[start]]
            start += 1
            while (
                start < len(pending)
                and (lengths[pending[start]] + 2) * (len(batch) + 1) <= a.max_tokens
            ):
                batch.append(pending[start])
                start += 1
            enc = tokenizer(
                [sequences[i].as_py() for i in batch], return_tensors="pt", padding=True
            )
            with torch.inference_mode():
                hidden = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).last_hidden_state
                values = pool_batch(hidden, enc["attention_mask"])
            if values.shape != (len(batch), 960) or not np.isfinite(values).all():
                raise ValueError("ESMC returned invalid embeddings")
            array[batch] = values
            print(f"Encoded missing {start}/{len(pending)}", flush=True)
    array.flush()
    # Bound validation memory instead of materializing the full multi-GB cache.
    for offset in range(0, len(rows), 4096):
        if not np.isfinite(array[offset : offset + 4096]).all():
            raise ValueError("cache contains non-finite values")
    (a.output_dir / "sequence_ids.json").write_text(json.dumps(ids) + "\n")
    metadata.update(
        status="completed",
        reused=len(found),
        newly_encoded=len(pending),
        embeddings=file_fingerprint(a.output_dir / "embeddings.npy"),
        sequence_ids=file_fingerprint(a.output_dir / "sequence_ids.json"),
    )
    (a.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
