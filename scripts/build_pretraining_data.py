"""Build isolated pretraining windows from complete antiSMASH genomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO

from bgc_gnn.checkpoint import file_fingerprint, runtime_metadata


def protein_sequence(value):
    sequence = "".join(str(value or "").split()).upper().rstrip("*")
    if not sequence or any(c not in "ACDEFGHIKLMNPQRSTVWYBXZOU" for c in sequence):
        return None
    return sequence


def sequence_id(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def record_genes(record, excluded_proteins, boundary_margin=2):
    """Preserve missing/excluded CDS as breaks; labels never enter model inputs."""
    regions = [f for f in record.features if f.type == "region"]
    for region in regions:
        if (
            not 0
            <= int(region.location.start)
            < int(region.location.end)
            <= len(record.seq)
        ):
            raise ValueError(f"invalid region coordinates: {record.id}")
    cds = sorted(
        (f for f in record.features if f.type == "CDS"),
        key=lambda f: (int(f.location.start), int(f.location.end)),
    )
    rows = []
    for ordinal, feature in enumerate(cds):
        start, end = int(feature.location.start), int(feature.location.end)
        if not 0 <= start < end <= len(record.seq):
            raise ValueError(f"invalid CDS coordinates: {record.id}")
        if len(feature.location.parts) != 1:
            # Multipart/origin-spanning CDS has no single linear gene interval.
            rows.append(None)
            continue
        sequence = protein_sequence(feature.qualifiers.get("translation", [""])[0])
        if sequence is None or sequence_id(sequence) in excluded_proteins:
            rows.append(None)
            continue
        hits = [
            r
            for r in regions
            if max(0, min(end, int(r.location.end)) - max(start, int(r.location.start)))
            / (end - start)
            >= 0.5
        ]
        edge = any(
            r.qualifiers.get("contig_edge", ["False"])[0].lower() == "true"
            for r in hits
        )
        rows.append(
            {
                "sequence_id": sequence_id(sequence),
                "target_aa": sequence,
                "ordinal": ordinal,
                "start": start,
                "end": end,
                "positive": bool(hits),
                "supervised": not edge,
            }
        )
    # Ignore genes around both sides of each predicted boundary, including edge regions.
    for region in regions:
        inside = [
            i
            for i, f in enumerate(cds)
            if max(
                0,
                min(int(f.location.end), int(region.location.end))
                - max(int(f.location.start), int(region.location.start)),
            )
            > 0
        ]
        if inside:
            for boundary in (inside[0], inside[-1] + 1):
                for i in range(
                    max(0, boundary - boundary_margin),
                    min(len(rows), boundary + boundary_margin),
                ):
                    if rows[i] is not None:
                        rows[i]["supervised"] = False
    return rows


def contiguous_windows(rows, width=128, stride=64):
    """Never wrap a contig or bridge a missing CDS. Tail windows are right padded."""
    if width < 1 or not 1 <= stride <= width:
        raise ValueError("stride must be between 1 and window width")
    runs = [[]]
    for row in rows:
        if row is None:
            if runs[-1]:
                runs.append([])
        else:
            runs[-1].append(row)
    for run in runs:
        for start in range(0, len(run), stride):
            chunk = run[start : start + width]
            padding = width - len(chunk)
            yield {
                "sequence_ids": [r["sequence_id"] for r in chunk] + [""] * padding,
                "detect_mask": [r["positive"] for r in chunk] + [False] * padding,
                "valid_mask": [True] * len(chunk) + [False] * padding,
                "loss_mask": [r["supervised"] for r in chunk] + [False] * padding,
                "start_ordinal": chunk[0]["ordinal"],
            }
            if start + width >= len(run):
                break


def build(
    source,
    ng_gbk,
    output,
    split_seed=42,
    val_fraction=0.1,
    boundary_margin=2,
    audit_dir=None,
):
    if output.exists():
        raise ValueError("output directory exists; use a fresh directory")
    if not 0 < val_fraction < 1 or boundary_margin < 0:
        raise ValueError("invalid validation fraction or boundary margin")
    folders = sorted(p for p in source.iterdir() if p.is_dir())
    audited_inputs = {}
    if audit_dir is not None:
        summary = json.loads((audit_dir / "summary.json").read_text())
        if (
            summary.get("status") != "completed"
            or Path(summary["source"]).resolve() != source.resolve()
        ):
            raise ValueError("requires completed audit for the same source directory")
        usable = set(summary["usable_assemblies"])
        with (audit_dir / "assemblies.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if item["assembly"] in usable:
                    if item["errors"] or item["assembly"] in audited_inputs:
                        raise ValueError("inconsistent or duplicate audited assembly")
                    audited_inputs[item["assembly"]] = item["gbk"]
        if usable != set(audited_inputs):
            raise ValueError("usable assembly list differs from detailed audit")
        folders = [p for p in folders if p.name in usable]
        if {p.name for p in folders} != usable:
            raise ValueError("audited source assemblies are missing")
    if len(folders) < 2:
        raise ValueError("at least two genome directories required")
    with ng_gbk.open(encoding="utf-8") as stream:
        ng_records = list(SeqIO.parse(stream, "genbank"))
    if not ng_records:
        raise ValueError("NG exclusion file contains no records")
    ng_ids = {r.id.split(".")[0] for r in ng_records}
    ng_dna = {
        hashlib.sha256(str(r.seq).upper().encode()).hexdigest() for r in ng_records
    }
    excluded = {
        sequence_id(s)
        for r in ng_records
        for f in r.features
        if f.type == "CDS"
        if (s := protein_sequence(f.qualifiers.get("translation", [""])[0]))
    }
    random.Random(split_seed).shuffle(folders)
    n_val = min(len(folders) - 1, max(1, round(len(folders) * val_fraction)))
    val_assemblies = {p.name for p in folders[:n_val]}
    schema = pa.schema(
        [
            ("assembly", pa.string()),
            ("contig", pa.string()),
            ("start_ordinal", pa.int32()),
            ("sequence_ids", pa.list_(pa.string())),
            ("detect_mask", pa.list_(pa.bool_())),
            ("valid_mask", pa.list_(pa.bool_())),
            ("loss_mask", pa.list_(pa.bool_())),
        ]
    )
    output.mkdir(parents=True)
    unique_proteins = 0
    counts = Counter()
    inputs = []
    # NG is used only for exclusions, never optimization or model selection.
    audit = {
        "status": "incomplete",
        "runtime": runtime_metadata(),
        "source": str(source),
        "ng_exclusion": file_fingerprint(ng_gbk),
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "boundary_margin_genes": boundary_margin,
        "window_size": 128,
        "stride": 64,
        "source_audit": file_fingerprint(audit_dir / "summary.json")
        if audit_dir is not None
        else None,
        "source_audit_details": file_fingerprint(audit_dir / "assemblies.jsonl")
        if audit_dir is not None
        else None,
        "limits": [
            "antiSMASH-derived labels are imperfect; outside regions is candidate background.",
            "Genome-held-out split may share exact proteins; overlap is reported.",
            "Sample-split finetuning may share BGCs or genomes between splits.",
            "Multipart CDS are excluded as continuity breaks rather than labeled by their bounding span.",
        ],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    index_path = output / "protein_index.sqlite"
    with (
        closing(sqlite3.connect(index_path)) as index,
        pq.ParquetWriter(output / "windows_train.parquet", schema) as train_writer,
        pq.ParquetWriter(output / "windows_val.parquet", schema) as val_writer,
    ):
        index.execute(
            "CREATE TABLE proteins (gene_id TEXT PRIMARY KEY, sequence TEXT NOT NULL, splits INTEGER NOT NULL)"
        )
        for i, folder in enumerate(folders):
            gbks = sorted(p for p in folder.glob("*.gbk") if ".region" not in p.name)
            if len(gbks) != 1:
                raise ValueError(
                    f"{folder.name}: expected one complete genome GBK, found {len(gbks)}"
                )
            fingerprint = file_fingerprint(gbks[0])
            if (
                audited_inputs
                and fingerprint["sha256"] != audited_inputs[folder.name]["sha256"]
            ):
                raise ValueError(f"{folder.name}: GBK changed after integrity audit")
            inputs.append(fingerprint)
            split = "val" if folder.name in val_assemblies else "train"
            split_bit = 2 if split == "val" else 1
            batch = []
            with gbks[0].open(encoding="utf-8") as stream:
                records = list(SeqIO.parse(stream, "genbank"))
            if not records:
                raise ValueError(f"{folder.name}: empty GenBank")
            seen_contigs = set()
            for record in records:
                if record.id in seen_contigs:
                    raise ValueError(
                        f"duplicate contig ID in {folder.name}: {record.id}"
                    )
                seen_contigs.add(record.id)
                counts["contigs"] += 1
                if (
                    record.id.split(".")[0] in ng_ids
                    or hashlib.sha256(str(record.seq).upper().encode()).hexdigest()
                    in ng_dna
                ):
                    counts["excluded_ng_contigs"] += 1
                    continue
                rows = record_genes(record, excluded, boundary_margin)
                counts["cds"] += len(rows)
                multipart = sum(
                    f.type == "CDS" and len(f.location.parts) != 1
                    for f in record.features
                )
                counts["discontinuous_cds_breaks"] += multipart
                counts["missing_or_ng_protein_breaks"] += (
                    sum(r is None for r in rows) - multipart
                )
                counts["regions"] += sum(f.type == "region" for f in record.features)
                protein_batch = []
                for row in rows:
                    if row is not None:
                        protein_batch.append(
                            (row["sequence_id"], row["target_aa"], split_bit)
                        )
                        counts["positive_genes"] += row["positive"]
                        counts["supervised_genes"] += row["supervised"]
                before = index.total_changes
                index.executemany(
                    "INSERT OR IGNORE INTO proteins VALUES (?, ?, ?)", protein_batch
                )
                unique_proteins += index.total_changes - before
                index.executemany(
                    "UPDATE proteins SET splits = splits | ? WHERE gene_id = ? AND (splits & ?) = 0",
                    ((split_bit, sid, split_bit) for sid, _, _ in protein_batch),
                )
                for window in contiguous_windows(rows):
                    window.update(assembly=folder.name, contig=record.id)
                    batch.append(window)
            if batch:
                (val_writer if split == "val" else train_writer).write_table(
                    pa.Table.from_pylist(batch, schema=schema)
                )
                counts[split + "_windows"] += len(batch)
            counts["assemblies"] += 1
            index.commit()
            print(
                f"Audited {i + 1}/{len(folders)} genomes; {unique_proteins} unique proteins",
                flush=True,
            )
        overlap = index.execute(
            "SELECT COUNT(*) FROM proteins WHERE splits = 3"
        ).fetchone()[0]
        protein_schema = pa.schema(
            [
                ("gene_id", pa.string()),
                ("target_aa", pa.string()),
                ("strand", pa.string()),
            ]
        )
        with pq.ParquetWriter(
            output / "proteins.parquet", protein_schema
        ) as protein_writer:
            cursor = index.execute(
                "SELECT gene_id, sequence FROM proteins ORDER BY rowid"
            )
            while records := cursor.fetchmany(4096):
                protein_writer.write_table(
                    pa.table(
                        {
                            "gene_id": [r[0] for r in records],
                            "target_aa": [r[1] for r in records],
                            "strand": ["+"] * len(records),
                        },
                        schema=protein_schema,
                    )
                )
    if not counts["train_windows"] or not counts["val_windows"]:
        raise ValueError("NG exclusions left an empty split")
    index_path.unlink()
    audit.update(
        status="completed",
        counts=dict(counts),
        unique_proteins=unique_proteins,
        inputs=inputs,
        val_assemblies=sorted(val_assemblies),
        exact_protein_overlap_train_val=overlap,
    )
    for name in ("windows_train.parquet", "windows_val.parquet", "proteins.parquet"):
        audit[name] = file_fingerprint(output / name)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--antismash-dir", type=Path, required=True)
    p.add_argument(
        "--ng-gbk",
        type=Path,
        default=Path("data/raw/ng/ClusterFinder_Annotated_Contigs.full.gbk"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help="Completed audit_antismash.py report; verifies eligible assemblies and GBK hashes",
    )
    a = p.parse_args()
    build(a.antismash_dir, a.ng_gbk, a.output_dir, audit_dir=a.audit_dir)


if __name__ == "__main__":
    main()
