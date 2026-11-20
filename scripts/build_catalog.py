"""Build protein catalogs directly from MIBiG and antiSMASH GenBank files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO

from bgc_detector.checkpoint import file_fingerprint, write_metadata
from bgc_detector.dataset import normalize_strand, normalize_translation


def read_genes(
    path: Path, namespace: str, entity: str, exclude_regions: bool
) -> tuple[list[dict], dict]:
    """Keep original CDS ordinals, including gaps left by exclusions."""
    rows, ids, contigs = [], [], []
    ordinal = 0
    seen = set()
    for record in SeqIO.parse(path, "genbank"):
        if record.id in seen:
            raise ValueError(f"duplicate contig ID in {path}: {record.id}")
        seen.add(record.id)
        regions = [f for f in record.features if f.type == "region"]
        for feature in sorted(
            (f for f in record.features if f.type == "CDS"),
            key=lambda f: int(f.location.start),
        ):
            gene_id = f"{namespace}:{entity}:{ordinal}"
            ordinal += 1
            start, end = int(feature.location.start), int(feature.location.end)
            if len(feature.location.parts) != 1:
                continue
            aa = normalize_translation(feature.qualifiers.get("translation", [""])[0])
            if not aa and namespace == "bgc":
                table = int(feature.qualifiers.get("transl_table", ["11"])[0])
                aa = normalize_translation(
                    str(
                        feature.extract(record.seq).translate(table=table, to_stop=True)
                    )
                )
            if not aa:
                continue
            if exclude_regions and any(
                max(
                    0, min(end, int(r.location.end)) - max(start, int(r.location.start))
                )
                / max(1, end - start)
                >= 0.5
                for r in regions
            ):
                continue
            ids.append(gene_id)
            contigs.append(record.id)
            rows.append(
                {
                    "gene_id": gene_id,
                    "target_aa": aa,
                    "strand": normalize_strand(feature.location.strand),
                    "contig": record.id,
                    "start": start,
                    "end": end,
                }
            )
    if not rows:
        raise ValueError(f"no usable proteins in {path}")
    key = "bgc_id" if namespace == "bgc" else "genome_id"
    return rows, {
        key: entity,
        "source_entity": entity,
        "gene_ids": ids,
        "source_gene_ids": ids,
        "source_contigs": contigs,
        "gene_count": len(ids),
    }


def build_catalog(mibig_dir: Path, antismash_dir: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("catalog output exists; use a fresh directory")
    positives = sorted(mibig_dir.glob("BGC*.gbk"))
    backgrounds = []
    for folder in sorted(p for p in antismash_dir.iterdir() if p.is_dir()):
        paths = sorted(p for p in folder.glob("*.gbk") if ".region" not in p.name)
        if len(paths) != 1:
            raise ValueError(f"{folder.name}: expected one complete GenBank file")
        backgrounds.append((folder.name, paths[0]))
    if not positives or not backgrounds:
        raise ValueError("requires MIBiG BGC files and antiSMASH background genomes")
    output.mkdir(parents=True)
    metadata = {"status": "incomplete", "inputs": []}
    write_metadata(output / "metadata.json", metadata)
    schema = pa.schema(
        [
            ("gene_id", pa.string()),
            ("target_aa", pa.string()),
            ("strand", pa.string()),
            ("contig", pa.string()),
            ("start", pa.int64()),
            ("end", pa.int64()),
        ]
    )
    bgc_index, genome_index, negative = [], [], []
    with pq.ParquetWriter(output / "gene_catalog.parquet", schema) as writer:
        for entity, path, namespace in [(p.stem, p, "bgc") for p in positives] + [
            (e, p, "gtdb") for e, p in backgrounds
        ]:
            rows, index = read_genes(
                path, namespace, entity, exclude_regions=namespace == "gtdb"
            )
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            (bgc_index if namespace == "bgc" else genome_index).append(index)
            if namespace == "gtdb":
                negative.extend(index["gene_ids"])
            metadata["inputs"].append(file_fingerprint(path))
    if not negative:
        raise ValueError("no background genes remain after antiSMASH exclusions")
    pq.write_table(pa.Table.from_pylist(bgc_index), output / "bgc_index.parquet")
    pq.write_table(pa.Table.from_pylist(genome_index), output / "gtdb_index.parquet")
    pq.write_table(
        pa.table({"gene_id": sorted(negative)}), output / "negative_pool.parquet"
    )
    metadata.update(status="completed", bgcs=len(bgc_index), genomes=len(genome_index))
    write_metadata(output / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mibig-dir", type=Path, required=True)
    parser.add_argument("--antismash-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/catalog"))
    args = parser.parse_args()
    build_catalog(args.mibig_dir, args.antismash_dir, args.output_dir)


if __name__ == "__main__":
    main()
