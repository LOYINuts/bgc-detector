"""Audit every antiSMASH assembly without modifying source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Bio import SeqIO

from bgc_detector.checkpoint import file_fingerprint, runtime_metadata
from scripts.build_pretraining_data import protein_sequence, sequence_id


def audit_assembly(folder, ng_ids, ng_dna, ng_proteins):
    result = {"assembly": folder.name, "errors": [], "warnings": [], "counts": {}}
    counts = Counter()
    gbks = sorted(p for p in folder.glob("*.gbk") if ".region" not in p.name)
    result["complete_gbks"] = [p.name for p in gbks]
    if len(gbks) != 1:
        result["errors"].append(f"expected one complete GBK, found {len(gbks)}")
        return result
    path = gbks[0]
    result["gbk"] = file_fingerprint(path)
    json_path = path.with_suffix(".json")
    record_ids = set()
    record_sequences = {}
    try:
        with (
            path.open(encoding="utf-8") as stream,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            for record in SeqIO.parse(stream, "genbank"):
                counts["contigs"] += 1
                if record.id in record_ids:
                    result["errors"].append(f"duplicate contig ID: {record.id}")
                record_ids.add(record.id)
                dna = hashlib.sha256(str(record.seq).upper().encode()).hexdigest()
                record_sequences[record.id] = dna
                if record.id.split(".")[0] in ng_ids or dna in ng_dna:
                    counts["ng_matching_contigs"] += 1
                for feature in record.features:
                    if feature.type not in ("CDS", "region"):
                        continue
                    start, end = int(feature.location.start), int(feature.location.end)
                    if not 0 <= start < end <= len(record.seq):
                        result["errors"].append(
                            f"invalid {feature.type} coordinates: {record.id}:{start}-{end}"
                        )
                    if feature.type == "region":
                        counts["regions"] += 1
                        if (
                            feature.qualifiers.get("contig_edge", ["False"])[0].lower()
                            == "true"
                        ):
                            counts["edge_regions"] += 1
                    else:
                        counts["cds"] += 1
                        sequence = protein_sequence(
                            feature.qualifiers.get("translation", [""])[0]
                        )
                        if sequence is None:
                            counts["invalid_or_missing_translation"] += 1
                        else:
                            counts["translated_cds"] += 1
                            counts["ng_matching_protein_occurrences"] += (
                                sequence_id(sequence) in ng_proteins
                            )
            result["warnings"].extend(str(w.message) for w in caught)
    except Exception as exc:
        result["errors"].append(f"GBK parse failed: {type(exc).__name__}: {exc}")
    if not counts["contigs"] or not counts["cds"]:
        result["errors"].append("complete GBK has no contigs or CDS")
    try:
        result["json"] = file_fingerprint(json_path)
        with json_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        result["antismash_version"] = data.get("version", "")
        json_ids = [r["id"] for r in data["records"]]
        if len(json_ids) != len(set(json_ids)) or set(json_ids) != record_ids:
            result["errors"].append("JSON and full GBK contig IDs differ")
        for record in data["records"]:
            sequence = record.get("seq")
            if isinstance(sequence, str):
                digest = hashlib.sha256(sequence.upper().encode()).hexdigest()
                if digest != record_sequences.get(record["id"]):
                    result["errors"].append(
                        f"JSON and GBK sequence differ: {record['id']}"
                    )
            else:
                result["warnings"].append(f"JSON sequence not checked: {record['id']}")
    except Exception as exc:
        result["errors"].append(f"JSON parse failed: {type(exc).__name__}: {exc}")
    region_files = sorted(folder.glob("*.region*.gbk"))
    counts["region_files"] = len(region_files)
    if len(region_files) != counts["regions"]:
        result["errors"].append(
            "region GBK count differs from full GBK region feature count"
        )
    for region_path in region_files:
        try:
            with region_path.open(encoding="utf-8") as stream:
                records = list(SeqIO.parse(stream, "genbank"))
            if len(records) != 1 or not any(
                f.type == "region" for f in records[0].features
            ):
                raise ValueError("expected one record with a region feature")
        except Exception as exc:
            result["errors"].append(f"{region_path.name}: {type(exc).__name__}: {exc}")
    if not counts["regions"]:
        result["warnings"].append(
            "no predicted BGC regions; may be valid background genome"
        )
    result["counts"] = dict(counts)
    result["usable_with_exclusions"] = not result["errors"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--original-source",
        type=Path,
        help="Optional earlier assembly directory for overlap reporting",
    )
    p.add_argument(
        "--ng-gbk",
        type=Path,
        default=Path("data/raw/ng/ClusterFinder_Annotated_Contigs.full.gbk"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    with args.ng_gbk.open(encoding="utf-8") as stream:
        ng = list(SeqIO.parse(stream, "genbank"))
    if not ng:
        p.error("NG exclusion file contains no records")
    ng_ids = {r.id.split(".")[0] for r in ng}
    ng_dna = {hashlib.sha256(str(r.seq).upper().encode()).hexdigest() for r in ng}
    ng_proteins = {
        sequence_id(s)
        for r in ng
        for f in r.features
        if f.type == "CDS"
        if (s := protein_sequence(f.qualifiers.get("translation", [""])[0]))
    }
    folders = sorted(p for p in args.source.iterdir() if p.is_dir())
    original = (
        {p.name for p in args.original_source.iterdir() if p.is_dir()}
        if args.original_source
        else set()
    )
    summary = {
        "status": "running",
        "runtime": runtime_metadata(),
        "source": str(args.source),
        "ng_input": file_fingerprint(args.ng_gbk),
        "assembly_directories": len(folders),
        "original_source": str(args.original_source) if args.original_source else None,
        "exact_assembly_overlap_original": (
            len(original & {p.name for p in folders}) if args.original_source else None
        ),
        "counts": {},
        "usable_assemblies": [],
        "failed_assemblies": [],
        "versions": {},
    }
    totals = Counter()
    versions = Counter()
    with (args.output_dir / "assemblies.jsonl").open("w", encoding="utf-8") as output:
        for i, folder in enumerate(folders, 1):
            try:
                row = audit_assembly(folder, ng_ids, ng_dna, ng_proteins)
            except Exception as exc:
                row = {
                    "assembly": folder.name,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "counts": {},
                }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            totals.update(row["counts"])
            versions[row.get("antismash_version", "unknown")] += 1
            summary[
                "failed_assemblies" if row["errors"] else "usable_assemblies"
            ].append(folder.name)
            summary.update(counts=dict(totals), versions=dict(versions), audited=i)
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Audited {i}/{len(folders)}; usable={len(summary['usable_assemblies'])}; failed={len(summary['failed_assemblies'])}",
                flush=True,
            )
    summary["status"] = "completed"
    summary["limits"] = [
        "Completeness is structural, not biological label accuracy.",
        "Exact NG matches require exclusion before pretraining; homology is not checked.",
        "More assemblies do not guarantee better downstream AUROC.",
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
