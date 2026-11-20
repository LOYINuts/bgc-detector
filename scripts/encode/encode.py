"""Encode genes through frozen ESMC-300M (960d) per-protein mean pooling.

Sources
-------
  catalog : read gene_catalog.parquet, write <catalog>/gene_embeddings_960d.parquet
  ng      : parse the 9-genome ClusterFinder GBK, write
            data/derived/ng_embeddings/ng_gene_embeddings_960d.parquet

Usage
-----
    uv run python -m scripts.encode.encode --source catalog --catalog-dir data/derived/catalog
    uv run python -m scripts.encode.encode --source ng
"""

from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bgc_detector.dataset import normalize_strand, normalize_translation
from bgc_detector.esmc import (
    EMBEDDING_DIM,
    ESMC_MODEL,
    last_hidden,
    load_model,
    pool_batch,
)

NG_GBK = Path("data/raw/ng/ClusterFinder_Annotated_Contigs.full.gbk")
NG_OUT_DIR = Path("data/derived/ng_embeddings")
SOURCES = ("catalog", "ng")


def load_catalog_rows(catalog_dir: Path) -> list[dict]:
    table = pq.read_table(catalog_dir / "gene_catalog.parquet")
    return [
        {
            "gene_id": r["gene_id"],
            "strand": r["strand"],
            "target_aa": normalize_translation(r["target_aa"]),
        }
        for r in table.to_pylist()
    ]


def load_ng_rows(ng_gbk: Path) -> list[dict]:
    """Parse 9-genome GBK → one row per CDS (locus_tag-based dedup)."""
    from Bio import SeqIO

    rows: list[dict] = []
    seen: set[str] = set()
    for record in SeqIO.parse(str(ng_gbk), "genbank"):
        acc = record.annotations.get("accessions", [record.id])[0]
        for feat in record.features:
            if feat.type != "CDS":
                continue
            locus = feat.qualifiers.get("locus_tag", [None])[0]
            aa = normalize_translation(feat.qualifiers.get("translation", [""])[0])
            if not locus or not aa:
                continue
            gid = f"{acc}:{locus}"
            if gid in seen:
                continue
            seen.add(gid)
            rows.append(
                {
                    "gene_id": gid,
                    "target_aa": aa,
                    "strand": normalize_strand(feat.location.strand),
                }
            )
    return rows


def output_path(source: str, emb_dim: int, catalog_dir: Path, ng_out_dir: Path) -> Path:
    if source == "catalog":
        return catalog_dir / f"gene_embeddings_{emb_dim}d.parquet"
    return ng_out_dir / f"ng_gene_embeddings_{emb_dim}d.parquet"


def run(
    source: str,
    catalog_dir: Path,
    ng_gbk: Path,
    ng_out_dir: Path,
    max_toks: int,
    max_genes: int | None,
    device_str: str,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # default device is cuda:0; running on cuda:N makes every op pay a
        # device-context switch
        torch.cuda.set_device(device)
    print(f"Device: {device}", flush=True)

    model_name, emb_dim = ESMC_MODEL, EMBEDDING_DIM
    print(f"Model: {model_name} (esmc, {emb_dim}d)", flush=True)
    model_obj, tokenizer = load_model(device)

    rows = (
        load_catalog_rows(catalog_dir) if source == "catalog" else load_ng_rows(ng_gbk)
    )
    if max_genes:
        rows = rows[:max_genes]
    n_genes = len(rows)
    print(f"  {n_genes:,} genes", flush=True)

    # ESMC tokenizer does NOT ignore spaces; residues only, no separators.
    inputs = [r["target_aa"] for r in rows]
    lens = [len(s) + 2 for s in inputs]  # + BOS/EOS, no truncation

    order = sorted(range(n_genes), key=lambda i: lens[i])
    rows = [rows[i] for i in order]
    inputs = [inputs[i] for i in order]
    lens = [lens[i] for i in order]

    out_path = output_path(source, emb_dim, catalog_dir, ng_out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        out_path,
        pa.schema(
            [
                ("gene_id", pa.string()),
                ("embedding", pa.list_(pa.float32(), emb_dim)),
            ]
        ),
    )

    flush_every = 5000
    log_secs = 30  # time-based: batch sizes vary, gene-count gates print irregularly
    buffer_gids: list[str] = []
    buffer_embs: list[list[float]] = []
    total = 0
    n_batches = 0
    t0 = time.time()
    last_log = t0 - log_secs
    last_total = 0
    idx = 0

    while idx < n_genes:
        # Pack bucket up to max_toks
        toks = 0
        batch_rows: list[dict] = []
        batch_inputs: list[str] = []
        while idx < n_genes and toks + lens[idx] <= max_toks:
            batch_rows.append(rows[idx])
            batch_inputs.append(inputs[idx])
            toks += lens[idx]
            idx += 1
        if not batch_rows:
            batch_rows = [rows[idx]]
            batch_inputs = [inputs[idx]]
            toks = lens[idx]
            idx += 1

        # rotary positions: no length cap; transformers sdpa handles the mask
        enc = tokenizer(batch_inputs, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.inference_mode():
            out = model_obj(input_ids=input_ids, attention_mask=attn)
        hidden = last_hidden(out)  # [B, T, D]

        embs = pool_batch(hidden, enc["attention_mask"])
        buffer_gids.extend(r["gene_id"] for r in batch_rows)
        buffer_embs.extend(embs.tolist())

        del out, hidden
        n_batches += 1
        if device.type == "cuda" and n_batches % 10 == 0:
            torch.cuda.empty_cache()  # dynamic batching fragments the allocator
        total += len(batch_rows)

        if len(buffer_gids) >= flush_every:
            writer.write_table(
                pa.table(
                    {
                        "gene_id": pa.array(buffer_gids, pa.string()),
                        "embedding": pa.array(
                            buffer_embs, pa.list_(pa.float32(), emb_dim)
                        ),
                    }
                )
            )
            buffer_gids.clear()
            buffer_embs.clear()

        if time.time() - last_log >= log_secs or total >= n_genes:
            now = time.time()
            dt = now - last_log
            inst = (total - last_total) / dt if dt > 0 else 0
            last_log = now
            last_total = total
            elapsed = now - t0
            avg = total / elapsed if elapsed > 0 else 0
            # eta from the instantaneous rate: the length-sorted ramp makes the
            # cumulative average unrepresentative early on
            eta = (
                str(timedelta(seconds=int((n_genes - total) / inst)))
                if inst > 0
                else "?"
            )
            mem = (
                f"  gpu={torch.cuda.memory_allocated(device) / 2**30:.1f}/"
                f"{torch.cuda.memory_reserved(device) / 2**30:.1f}G"
                if device.type == "cuda"
                else ""
            )
            print(
                f"  [{total}/{n_genes} {100 * total / n_genes:.1f}%]"
                f"  {inst:.1f} genes/s (avg {avg:.1f})"
                f"  elapsed={timedelta(seconds=int(elapsed))}"
                f"  eta={eta}{mem}",
                flush=True,
            )

    if buffer_gids:
        writer.write_table(
            pa.table(
                {
                    "gene_id": pa.array(buffer_gids, pa.string()),
                    "embedding": pa.array(buffer_embs, pa.list_(pa.float32(), emb_dim)),
                }
            )
        )
    writer.close()
    print(
        f"\nDone - {total:,} embeddings in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    print(f"  Output: {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Encode genes through frozen ESMC-300M.")
    p.add_argument(
        "--source",
        required=True,
        choices=list(SOURCES),
        help="catalog = read gene_catalog.parquet; "
        "ng = parse 9-genome GBK (NG test set)",
    )
    p.add_argument("--catalog-dir", default="data/derived/catalog")
    p.add_argument("--ng-gbk", default=str(NG_GBK))
    p.add_argument("--ng-out-dir", default=str(NG_OUT_DIR))
    p.add_argument(
        "--max-toks", type=int, default=40960, help="Max tokens per dynamic batch"
    )
    p.add_argument(
        "--max-genes", type=int, default=None, help="Cap number of genes (debug)"
    )
    p.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    args = p.parse_args()
    run(
        args.source,
        Path(args.catalog_dir),
        Path(args.ng_gbk),
        Path(args.ng_out_dir),
        args.max_toks,
        args.max_genes,
        args.device,
    )


if __name__ == "__main__":
    main()
