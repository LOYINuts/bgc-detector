"""Command line entry point: inspect a checkpoint and predict genes from GenBank."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bgc_gnn.checkpoint import file_fingerprint, git_commit, validation_threshold
from bgc_gnn.esmc import EMBEDDING_DIM, MAX_TOKS, embed_sequences
from bgc_gnn.predict import STRIDE, gene_order, load_detector, mean_scores, parse_gbk


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_embeddings(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {
        gid: np.array(emb, dtype=np.float32)
        for gid, emb in zip(
            table.column("gene_id").to_pylist(), table.column("embedding").to_pylist()
        )
    }


def _write_embeddings(path: Path, gene_ids: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "gene_id": pa.array(gene_ids, pa.string()),
                "embedding": pa.array(
                    matrix.tolist(), pa.list_(pa.float32(), EMBEDDING_DIM)
                ),
            }
        ),
        path,
    )


def _gene_ids_with_translation(genes: list[dict]) -> tuple[list[str], list[str]]:
    with_seq = [g["gene_id"] for g in genes if g["translation"]]
    without = [g["gene_id"] for g in genes if not g["translation"]]
    return with_seq, without


def info(args: argparse.Namespace) -> None:
    _, metadata = load_detector(args.checkpoint, torch.device("cpu"))
    threshold, source = validation_threshold(metadata)
    print(f"checkpoint: {args.checkpoint}")
    print(
        f"  kind: {metadata.get('checkpoint_kind')}  weights: {metadata.get('weight_source')}"
    )
    print(
        f"  epoch: {metadata.get('epoch')}  selection: {metadata.get('selection_metric')} = {metadata.get('selection_value')}"
    )
    print(
        f"  threshold: {threshold:.6f} (source={source}, metric={metadata.get('validation_threshold_metric')})"
    )
    runtime = metadata.get("runtime", {})
    print(
        f"  trained on: torch {runtime.get('torch')}, commit {runtime.get('git_commit')}"
    )


def predict(args: argparse.Namespace) -> None:
    device = _device(args.device)
    model, metadata = load_detector(args.checkpoint, device)
    threshold, threshold_source = validation_threshold(metadata)

    genes = parse_gbk(args.genome)
    if not genes:
        raise ValueError(f"no CDS features found in {args.genome}")

    emb_map: dict[str, np.ndarray] = {}
    reused_embeddings = bool(args.emb_cache and args.emb_cache.is_file())
    if reused_embeddings:
        print(f"Reusing embeddings: {args.emb_cache}", flush=True)
        emb_map.update(_read_embeddings(args.emb_cache))
    gene_ids, without_translation = _gene_ids_with_translation(genes)
    if not gene_ids:
        raise ValueError("no gene has a translation; cannot embed")
    missing = [gid for gid in gene_ids if gid not in emb_map]
    if missing:
        print(f"Embedding {len(missing):,} proteins with ESMC-300M ...", flush=True)
        by_id = {g["gene_id"]: g for g in genes}
        matrix = embed_sequences(
            [by_id[gid]["translation"] for gid in missing], device, args.max_toks
        )
        emb_map.update(dict(zip(missing, matrix)))
        if args.emb_cache:
            _write_embeddings(
                args.emb_cache, list(emb_map), np.stack(list(emb_map.values()))
            )
            print(f"  embeddings saved -> {args.emb_cache}", flush=True)
    if without_translation:
        warnings.warn(
            f"{len(without_translation)} genes have no translation and were skipped",
            RuntimeWarning,
        )

    order = gene_order(genes, set(emb_map))
    scores, windows = mean_scores(model, emb_map, order, device)
    excluded = [g for g in genes if g["gene_id"] not in scores]
    if excluded:
        warnings.warn(
            f"{len(excluded)} genes were not scored (no embedding)",
            RuntimeWarning,
        )

    suffix = ""
    if args.name:
        suffix = f".{args.name}"
    genes_path = args.outdir / f"predict{suffix}.genes.tsv"
    provenance_path = args.outdir / f"predict{suffix}.json"
    args.outdir.mkdir(parents=True, exist_ok=True)
    by_id = {g["gene_id"]: g for g in genes}
    rows = [(accession, gid) for accession, ordered in order.items() for gid in ordered]
    with genes_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "contig\tgene_id\tlocus_tag\tstart\tend\tstrand\tscore\tpredicted\n"
        )
        for accession, gid in rows:
            gene = by_id[gid]
            score = scores[gid]
            stream.write(
                f"{accession}\t{gid}\t{gene['locus_tag']}\t{gene['start']}\t{gene['end']}\t"
                f"{gene['strand']}\t{score!r}\t{int(score >= threshold)}\n"
            )

    provenance = {
        "genome": file_fingerprint(args.genome),
        "checkpoint": file_fingerprint(args.checkpoint),
        "checkpoint_epoch": metadata.get("epoch"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_metric": metadata.get("validation_threshold_metric"),
        "window": 128,
        "stride": STRIDE,
        "aggregate": "mean",
        "postprocess": "none",
        "esmc_model": "biohub/ESMC-300M",
        "embedding_dim": EMBEDDING_DIM,
        "embeddings_from": "cache" if reused_embeddings else "computed",
        "n_genes": len(rows),
        "n_scored": len(scores),
        "n_windows": windows,
        "n_predicted": sum(int(scores[gid] >= threshold) for _, gid in rows),
        "code_commit": git_commit(),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"  {len(rows):,} genes scored in {windows} windows", flush=True)
    print(
        f"  {provenance['n_predicted']:,} genes called BGC at threshold {threshold:.4f}",
        flush=True,
    )
    print(f"  -> {genes_path}", flush=True)
    print(f"  -> {provenance_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bgc-detector",
        description="Per-gene BGC detection from frozen ESMC-300M embeddings",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    info_parser = commands.add_parser("info", help="show checkpoint provenance")
    info_parser.add_argument("--checkpoint", type=Path, required=True)
    info_parser.set_defaults(handler=info)

    predict_parser = commands.add_parser(
        "predict", help="predict per-gene BGC probability for a GenBank file"
    )
    predict_parser.add_argument("--genome", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--outdir", type=Path, default=Path("output"))
    predict_parser.add_argument("--name", default="", help="output file suffix")
    predict_parser.add_argument("--device", default=None, help="cuda:0 or cpu")
    predict_parser.add_argument(
        "--emb-cache",
        type=Path,
        default=None,
        help="reuse this gene embedding parquet, and write it when computing",
    )
    predict_parser.add_argument("--max-toks", type=int, default=MAX_TOKS)
    predict_parser.set_defaults(handler=predict)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
