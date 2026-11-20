"""Audit fixed detection windows and optional catalog/NG sequence overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest}


def source_position(gene_id: str) -> tuple[str, int]:
    entity, separator, ordinal = gene_id.rpartition(":")
    if (
        not separator
        or not ordinal.isdigit()
        or not entity.startswith(("bgc:", "gtdb:"))
    ):
        raise ValueError(f"expected bgc/gtdb source ID with ordinal: {gene_id!r}")
    return entity, int(ordinal)


def audit_windows(path: Path) -> tuple[dict, set[str], Counter]:
    parquet = pq.ParquetFile(path)
    columns = ["gene_ids", "is_positive", "detect_mask"]
    if "adjacency_breaks" in parquet.schema_arrow.names:
        columns.append("adjacency_breaks")
    genes: set[str] = set()
    windows: Counter = Counter()
    sources: Counter = Counter()
    breaks: Counter = Counter()
    lengths: Counter = Counter()
    positive = total = repeated_genes = mismatched_breaks = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=512):
        for row in batch.to_pylist():
            ids, mask = row["gene_ids"], row["detect_mask"]
            if not ids or len(ids) != 128 or len(mask) != len(ids):
                raise ValueError(
                    f"{path}: window {total} must contain 128 genes and labels"
                )
            if any(not isinstance(value, bool) for value in mask):
                raise ValueError(f"{path}: window {total} has non-binary labels")
            if row["is_positive"] != any(mask):
                raise ValueError(f"{path}: window {total} has inconsistent is_positive")
            positions = [source_position(gid) for gid in ids]
            inferred = {
                i
                for i in range(1, len(ids))
                if positions[i][0] != positions[i - 1][0]
                or positions[i][1] != positions[i - 1][1] + 1
            }
            if "adjacency_breaks" in row:
                declared = row["adjacency_breaks"]
                if declared is None or any(not 1 <= b < len(ids) for b in declared):
                    raise ValueError(
                        f"{path}: invalid adjacency_breaks at window {total}"
                    )
                mismatched_breaks += set(declared) != inferred
            namespaces = {entity.split(":", 1)[0] for entity, _ in positions}
            if namespaces == {"bgc"} and not inferred:
                kind = "bgc_source_contiguous"
            elif namespaces == {"gtdb"} and not inferred:
                kind = "gtdb_source_contiguous"
            elif namespaces == {"bgc", "gtdb"}:
                kind = "bgc_gtdb_join"
            else:
                kind = "other_source_or_ordinal_breaks"
            label = "positive" if row["is_positive"] else "negative"
            sources[f"{label}:{kind}"] += 1
            breaks[f"{label}:{len(inferred)}"] += 1
            lengths[str(sum(mask))] += 1
            positive += row["is_positive"]
            repeated_genes += len(set(ids)) != len(ids)
            windows[tuple(ids)] += 1
            genes.update(ids)
            total += 1
    entities = {source_position(gid)[0] for gid in genes}
    report = {
        "file": fingerprint(path),
        "windows": total,
        "positive_windows": positive,
        "negative_windows": total - positive,
        "unique_genes": len(genes),
        "entities": {
            ns: sum(e.startswith(ns + ":") for e in entities) for ns in ("bgc", "gtdb")
        },
        "duplicate_window_excess": sum(count - 1 for count in windows.values()),
        "windows_with_repeated_gene_ids": repeated_genes,
        "source_window_counts": dict(sorted(sources.items())),
        "inferred_break_counts": dict(sorted(breaks.items())),
        "declared_break_mismatch_windows": mismatched_breaks
        if "adjacency_breaks" in columns
        else None,
        "positive_gene_count_histogram": dict(
            sorted(lengths.items(), key=lambda item: int(item[0]))
        ),
    }
    return report, genes, windows


def overlap_summary(left: set, right: set) -> dict:
    overlap = left & right
    return {"count": len(overlap), "examples": sorted(overlap)[:10]}


def sequence_digest(sequence: str) -> bytes | None:
    normalized = "".join((sequence or "").split()).upper().rstrip("*")
    return hashlib.sha256(normalized.encode("ascii")).digest() if normalized else None


def audit_sequences(
    catalog: Path, train_genes: set[str], val_genes: set[str], ng_gbk: Path | None
) -> dict:
    print("Checking catalog protein sequences ...", flush=True)
    hashes = {"train": set(), "val": set()}
    genes_by_hash = {"train": Counter(), "val": Counter()}
    missing = {"train": set(train_genes), "val": set(val_genes)}
    seen: set[str] = set()
    for batch in pq.ParquetFile(catalog).iter_batches(
        columns=["gene_id", "target_aa"], batch_size=8192
    ):
        for row in batch.to_pylist():
            gid = row["gene_id"]
            splits = [
                name
                for name, ids in (("train", train_genes), ("val", val_genes))
                if gid in ids
            ]
            if not splits:
                continue
            if gid in seen:
                raise ValueError(f"duplicate catalog gene ID: {gid}")
            seen.add(gid)
            digest = sequence_digest(row["target_aa"])
            if digest is not None:
                for name in splits:
                    hashes[name].add(digest)
                    genes_by_hash[name][digest] += 1
                    missing[name].discard(gid)
    report = {
        "catalog_file": fingerprint(catalog),
        "normalization": "remove whitespace, uppercase, strip trailing stop markers",
        "missing_or_empty_catalog_genes": {
            name: overlap_summary(ids, ids) for name, ids in missing.items()
        },
        "unique_sequences": {name: len(values) for name, values in hashes.items()},
    }
    pairs = [("train", "val")]
    if ng_gbk is not None:
        from Bio import SeqIO

        print("Checking NG protein sequences ...", flush=True)
        hashes["ng"] = set()
        genes_by_hash["ng"] = Counter()
        empty = 0
        # ponytail: exact sequence hashes only. Use homology clustering for identity-based splits.
        with ng_gbk.open(encoding="utf-8") as stream:
            for record in SeqIO.parse(stream, "genbank"):
                for feature in record.features:
                    if feature.type != "CDS":
                        continue
                    digest = sequence_digest(
                        feature.qualifiers.get("translation", [""])[0]
                    )
                    if digest is None:
                        empty += 1
                    else:
                        hashes["ng"].add(digest)
                        genes_by_hash["ng"][digest] += 1
        report["ng_file"] = fingerprint(ng_gbk)
        report["ng_empty_translations"] = empty
        report["unique_sequences"]["ng"] = len(hashes["ng"])
        pairs += [("train", "ng"), ("val", "ng")]
    report["overlap"] = {}
    for left, right in pairs:
        shared = hashes[left] & hashes[right]
        report["overlap"][f"{left}/{right}"] = {
            "unique_sequences": len(shared),
            f"{left}_genes": sum(genes_by_hash[left][digest] for digest in shared),
            f"{right}_genes": sum(genes_by_hash[right][digest] for digest in shared),
        }
    return report


def audit_split(
    train: Path, val: Path, catalog: Path | None = None, ng_gbk: Path | None = None
) -> dict:
    if ng_gbk is not None and catalog is None:
        raise ValueError("NG sequence audit requires --catalog")
    train_report, train_genes, train_windows = audit_windows(train)
    val_report, val_genes, val_windows = audit_windows(val)
    entities = [
        {source_position(gid)[0] for gid in ids} for ids in (train_genes, val_genes)
    ]
    report = {
        "train": train_report,
        "val": val_report,
        "gene_overlap": overlap_summary(train_genes, val_genes),
        "entity_overlap": {
            ns: overlap_summary(
                *[{e for e in values if e.startswith(ns + ":")} for values in entities]
            )
            for ns in ("bgc", "gtdb")
        },
        "cross_split_duplicate_windows": len(train_windows.keys() & val_windows.keys()),
        "source_note": "Source-local ordinal continuity is not proof of biological contig continuity or true flanks.",
    }
    for name in ("train", "val"):
        r = report[name]
        print(
            f"{name}: {r['windows']:,} windows ({r['positive_windows']:,} positive), {r['unique_genes']:,} genes",
            flush=True,
        )
        print(f"  source windows: {r['source_window_counts']}", flush=True)
    print(f"train/val gene overlap: {report['gene_overlap']['count']:,}", flush=True)
    print(f"train/val entity overlap: {report['entity_overlap']}", flush=True)
    print(
        f"cross-split duplicate windows: {report['cross_split_duplicate_windows']}",
        flush=True,
    )
    if catalog is not None:
        report["sequences"] = audit_sequences(catalog, train_genes, val_genes, ng_gbk)
        print(f"sequence overlap: {report['sequences']['overlap']}", flush=True)
        print(
            f"missing catalog genes: {report['sequences']['missing_or_empty_catalog_genes']}",
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="gene_catalog.parquet for exact protein-sequence checks",
    )
    parser.add_argument(
        "--ng-gbk",
        type=Path,
        help="Optional NG GenBank file for exact sequence overlap",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON audit output")
    args = parser.parse_args()
    if args.ng_gbk and not args.catalog:
        parser.error("--ng-gbk requires --catalog")
    report = audit_split(args.train, args.val, args.catalog, args.ng_gbk)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Audit saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
