"""Command line entry point: inspect a checkpoint, predict genes, merge regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bgc_detector.checkpoint import file_fingerprint, git_commit, validation_threshold
from bgc_detector.esmc import EMBEDDING_DIM, ESMC_MODEL, MAX_TOKS, embed_sequences
from bgc_detector.predict import (
    STRIDE,
    gene_order,
    load_detector,
    mean_scores,
    parse_gbk,
)

GENES_HEADER = "contig\tgene_id\tlocus_tag\tstart\tend\tstrand\tscore\tpredicted"
REGIONS_HEADER = "contig\tstart\tend\tn_genes\tmean_score\tmax_score"


def _device(name: str | None) -> torch.device:
    if name:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available; pass --device cpu")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_embeddings(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {
        gid: np.array(emb, dtype=np.float32)
        for gid, emb in zip(
            table.column("gene_id").to_pylist(),
            table.column("embedding").to_pylist(),
            strict=True,
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


def info(args: argparse.Namespace) -> None:
    _, metadata = load_detector(args.checkpoint, torch.device("cpu"))
    threshold, source = validation_threshold(metadata)
    runtime = metadata.get("runtime", {})
    print(f"checkpoint: {args.checkpoint}")
    print(
        f"  kind: {metadata.get('checkpoint_kind')}  weights: {metadata.get('weight_source')}"
    )
    print(
        f"  epoch: {metadata.get('epoch')}  selection: "
        f"{metadata.get('selection_metric')} = {metadata.get('selection_value')}"
    )
    print(
        f"  threshold: {threshold:.6f} (source={source}, "
        f"metric={metadata.get('validation_threshold_metric')})"
    )
    print(
        f"  trained on: torch {runtime.get('torch')}, commit {runtime.get('git_commit')}"
    )


def _threshold(value: str, metadata: dict) -> tuple[float, str]:
    """Default 0.5; 'checkpoint' uses the validation-MCC threshold of the checkpoint."""
    if value == "checkpoint":
        return validation_threshold(metadata)
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be within [0, 1], got {value}")
    return threshold, "fixed"


def _embeddings_for(
    genes: list[dict], args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Gene embeddings for one genome, reusing ``--emb-cache`` when present."""
    gene_ids = [g["gene_id"] for g in genes if g["translation"]]
    if not gene_ids:
        raise ValueError("no gene has a translation; cannot embed")

    emb_map: dict[str, np.ndarray] = {}
    reused = bool(args.emb_cache and args.emb_cache.is_file())
    if reused:
        print(f"Reusing embeddings: {args.emb_cache}", flush=True)
        emb_map.update(_read_embeddings(args.emb_cache))

    missing = [gid for gid in gene_ids if gid not in emb_map]
    if missing:
        print(f"Embedding {len(missing):,} proteins with ESMC-300M ...", flush=True)
        by_id = {g["gene_id"]: g for g in genes}
        matrix = embed_sequences(
            [by_id[gid]["translation"] for gid in missing], device, args.max_toks
        )
        emb_map.update(dict(zip(missing, matrix, strict=True)))
        if args.emb_cache and not reused:
            # Never rewrite a cache that already existed: it may be shared input.
            _write_embeddings(
                args.emb_cache, list(emb_map), np.stack(list(emb_map.values()))
            )
            print(f"  embeddings saved -> {args.emb_cache}", flush=True)
        elif args.emb_cache:
            print(
                f"  {args.emb_cache} kept as is; {len(missing)} new embeddings "
                "were not written",
                flush=True,
            )

    if not reused:
        source = "computed"
    elif missing:
        source = "cache+computed"
    else:
        source = "cache"
    counts = {
        "embeddings_from": source,
        "n_embedded_computed": len(missing),
        "n_embedded_reused": len(emb_map) - len(missing),
    }
    return emb_map, counts


def _score_genome(
    genome: Path,
    stem: str,
    args: argparse.Namespace,
    model: torch.nn.Module,
    metadata: dict,
    device: torch.device,
    threshold: float,
    threshold_source: str,
    checkpoint_fingerprint: dict,
) -> None:
    genes = parse_gbk(genome)
    if not genes:
        raise ValueError(f"no CDS features found in {genome}")

    without_translation = [g["gene_id"] for g in genes if not g["translation"]]
    if without_translation:
        print(
            f"  warning: {len(without_translation):,} genes have no translation and "
            "were skipped",
            flush=True,
        )

    emb_map, embedding_counts = _embeddings_for(genes, args, device)
    order = gene_order(genes, set(emb_map))
    scores, windows = mean_scores(model, emb_map, order, device)

    by_id = {g["gene_id"]: g for g in genes}
    rows = [(accession, gid) for accession, ordered in order.items() for gid in ordered]
    args.outdir.mkdir(parents=True, exist_ok=True)
    genes_path = args.outdir / f"{stem}.genes.tsv"
    provenance_path = args.outdir / f"{stem}.json"

    lines = [GENES_HEADER]
    for accession, gid in rows:
        gene = by_id[gid]
        score = scores[gid]
        lines.append(
            f"{accession}\t{gid}\t{gene['locus_tag']}\t{gene['start']}\t{gene['end']}\t"
            f"{gene['strand']}\t{score!r}\t{int(score >= threshold)}"
        )
    genes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_predicted = sum(int(scores[gid] >= threshold) for _, gid in rows)
    provenance = {
        "genome": file_fingerprint(genome),
        "checkpoint": checkpoint_fingerprint,
        "checkpoint_epoch": metadata.get("epoch"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_metric": (
            metadata.get("validation_threshold_metric")
            if threshold_source == "validation"
            else None
        ),
        "window": 128,
        "stride": STRIDE,
        "aggregate": "mean",
        "postprocess": "none",
        "esmc_model": ESMC_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "max_toks": args.max_toks,
        **embedding_counts,
        "n_genes": len(genes),
        "n_scored": len(scores),
        "n_excluded": len(without_translation),
        "n_windows": windows,
        "n_predicted": n_predicted,
        "code_commit": git_commit(),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"  {len(rows):,} genes scored in {windows} windows", flush=True)
    print(
        f"  {n_predicted:,} genes called BGC at threshold {threshold:.4f}", flush=True
    )
    print(f"  -> {genes_path}", flush=True)
    print(f"  -> {provenance_path}", flush=True)


def predict(args: argparse.Namespace) -> None:
    if args.name and len(args.genome) > 1:
        raise ValueError("--name with several genomes is ambiguous; drop --name")
    device = _device(args.device)
    model, metadata = load_detector(args.checkpoint, device)
    threshold, threshold_source = _threshold(args.threshold, metadata)
    checkpoint_fingerprint = file_fingerprint(args.checkpoint)
    for genome in args.genome:
        stem = args.name or genome.stem
        print(f"Genome: {genome} -> {stem}.genes.tsv", flush=True)
        _score_genome(
            genome,
            stem,
            args,
            model,
            metadata,
            device,
            threshold,
            threshold_source,
            checkpoint_fingerprint,
        )


def _merge_regions(rows: list[list[str]], max_gap: int, min_count: int) -> list[dict]:
    """Group consecutive predicted genes, tolerating ``max_gap`` uncalled genes."""
    regions: list[dict] = []
    current: dict | None = None
    gap = 0
    for row in rows:
        contig, start, end, score, predicted = (
            row[0],
            int(row[3]),
            int(row[4]),
            float(row[6]),
            int(row[7]),
        )
        if predicted:
            if current is not None and current["contig"] == contig and gap <= max_gap:
                current["end"] = end
                current["scores"].append(score)
            else:
                if current is not None:
                    regions.append(current)
                current = {
                    "contig": contig,
                    "start": start,
                    "end": end,
                    "scores": [score],
                }
            gap = 0
        elif current is not None:
            gap += 1
    if current is not None:
        regions.append(current)
    return [
        {
            "contig": region["contig"],
            "start": region["start"],
            "end": region["end"],
            "scores": region["scores"],
        }
        for region in regions
        if len(region["scores"]) >= min_count
    ]


def _regions_path(genes: Path) -> Path:
    """run.genes.tsv -> run.regions.tsv; anything else -> <stem>.regions.tsv."""
    if genes.name.endswith(".genes.tsv"):
        return genes.with_name(genes.name[: -len(".genes.tsv")] + ".regions.tsv")
    return genes.with_name(genes.stem + ".regions.tsv")


def regions(args: argparse.Namespace) -> None:
    lines = args.genes.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != GENES_HEADER:
        raise ValueError(f"{args.genes} is not a {GENES_HEADER.split(chr(9))[0]} file")
    rows = [line.split("\t") for line in lines[1:] if line]
    merged = _merge_regions(rows, args.max_gap, args.min_count)

    output = args.output or _regions_path(args.genes)
    output.parent.mkdir(parents=True, exist_ok=True)
    out = [REGIONS_HEADER]
    for region in merged:
        scores = np.array(region["scores"], dtype=np.float64)
        out.append(
            f"{region['contig']}\t{region['start']}\t{region['end']}\t{len(scores)}\t"
            f"{float(scores.mean())!r}\t{float(scores.max())!r}"
        )
    output.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(
        f"  {len(merged)} regions from {sum(1 for r in rows if r[7] == '1'):,} predicted "
        f"genes (max_gap {args.max_gap}, min_count {args.min_count})",
        flush=True,
    )
    print(f"  -> {output}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bgc-detector",
        description=(
            "Per-gene BGC detection. Scores every CDS of a GenBank file with the "
            "trained BGCDetector, then optionally merges the called genes into regions."
        ),
        epilog=(
            "examples:\n"
            "  bgc-detector info --checkpoint runs/final_seed1506/detector_960d_best.pt\n"
            "  bgc-detector predict --genome genome.gbk \\\n"
            "      --checkpoint runs/final_seed1506/detector_960d_best.pt \\\n"
            "      --threshold checkpoint\n"
            "  bgc-detector regions --genes output/genome.genes.tsv --max-gap 3\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    info_parser = commands.add_parser(
        "info",
        help="show checkpoint provenance: epoch, selection metric, threshold",
    )
    info_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PT",
        help="detector checkpoint to inspect",
    )
    info_parser.set_defaults(handler=info)

    predict_parser = commands.add_parser(
        "predict",
        help="score every gene of one or more GenBank files",
        description=(
            "Score every CDS of a GenBank file in fixed 128-gene windows (stride 64) "
            "and write per-gene BGC probabilities. Genes without a /translation "
            "qualifier are skipped with a warning. Several genomes run in one pass and "
            "share the loaded model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    predict_parser.add_argument(
        "--genome",
        type=Path,
        required=True,
        nargs="+",
        metavar="GBK",
        help="one or more GenBank files whose CDS features carry /translation",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PT",
        help="validation-selected EMA detector checkpoint",
    )
    predict_parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output"),
        metavar="DIR",
        help="output directory (default: output)",
    )
    predict_parser.add_argument(
        "--name",
        default="",
        metavar="STEM",
        help="output stem; writes STEM.genes.tsv and STEM.json "
        "(default: the genome file name, no extension)",
    )
    predict_parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="cuda:0 or cpu (default: cuda when available)",
    )
    predict_parser.add_argument(
        "--emb-cache",
        type=Path,
        default=None,
        metavar="PARQUET",
        help="reuse this gene embedding parquet; written only when it does not exist "
        "and only for a single genome",
    )
    predict_parser.add_argument(
        "--max-toks",
        type=int,
        default=MAX_TOKS,
        metavar="N",
        help=f"tokens per ESMC batch (default: {MAX_TOKS}); lower it if the GPU "
        "runs out of memory",
    )
    predict_parser.add_argument(
        "--threshold",
        default="0.5",
        metavar="T",
        help="decision threshold in [0, 1], or 'checkpoint' for the validation-MCC "
        "threshold stored in the checkpoint (default: 0.5)",
    )
    predict_parser.set_defaults(handler=predict)

    regions_parser = commands.add_parser(
        "regions",
        help="merge called genes into BGC regions",
        description=(
            "Group consecutive genes flagged predicted=1 into regions, bridging gaps "
            "of up to --max-gap uncalled genes and dropping regions with fewer than "
            "--min-count genes. This is a reporting step; it does not change the "
            "per-gene predictions or the evaluated protocol."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    regions_parser.add_argument(
        "--genes",
        type=Path,
        required=True,
        metavar="TSV",
        help="the .genes.tsv written by predict",
    )
    regions_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="TSV",
        help="region table to write (default: replace .genes.tsv with .regions.tsv)",
    )
    regions_parser.add_argument(
        "--max-gap",
        type=int,
        default=3,
        metavar="N",
        help="bridge up to N uncalled genes inside a region (default: 3)",
    )
    regions_parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        metavar="N",
        help="drop regions with fewer called genes (default: 2)",
    )
    regions_parser.set_defaults(handler=regions)

    args = parser.parse_args(argv)
    if args.command == "predict" and args.emb_cache and len(args.genome) > 1:
        parser.error("--emb-cache supports a single genome; run genomes separately")
    try:
        args.handler(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
