"""Build gLM2 mixed-modality strings from antiSMASH GBK output.

Reads antiSMASH's own CDS predictions and intergenic DNA from the
output GBK, producing gLM2 strings in the same format.

Output: one .txt file per genome → data/derived/glm2_gtdb/

Usage
-----
    uv run python scripts/gbk_as_to_glm2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Bio import SeqIO


def parse_args():
    p = argparse.ArgumentParser(description="Convert antiSMASH GBK → gLM2 strings")
    p.add_argument("--antismash-dir", default="data/raw/gtdb/antismash_results",
                   help="antiSMASH output directory")
    p.add_argument("--outdir", default="data/derived/glm2_gtdb",
                   help="Output directory for gLM2 strings")
    return p.parse_args()


def build_glm2_string(gbk_path: Path) -> tuple[str, dict]:
    """Convert one antiSMASH GBK → gLM2 string.

    Returns (glm2_string, stats_dict).
    """
    elements: list[tuple[str, str, str]] = []  # (tag, seq, kind)

    for record in SeqIO.parse(str(gbk_path), "genbank"):
        full_seq = str(record.seq).upper()

        # Collect CDS features, sorted by position
        cds_feats = [
            (int(f.location.start), int(f.location.end),
             "+" if f.location.strand in (1, None) else "-",
             f.qualifiers.get("translation", [""])[0].rstrip("*"))
            for f in record.features if f.type == "CDS"
        ]
        cds_feats.sort(key=lambda x: x[0])

        prev_end = 0
        for start, end, strand, aa in cds_feats:
            # IGS before this CDS
            if start > prev_end:
                igs = full_seq[prev_end:start].lower()
                if igs:
                    elements.append(("<+>", igs, "IGS"))
            # CDS
            tag = "<->" if strand == "-" else "<+>"
            elements.append((tag, aa.upper(), "CDS"))
            prev_end = end

        # Trailing IGS after last CDS
        if len(full_seq) > prev_end:
            igs = full_seq[prev_end:].lower()
            if igs:
                elements.append(("<+>", igs, "IGS"))

    glm2_str = "".join(tag + s for tag, s, _ in elements)

    n_cds = sum(1 for _, _, k in elements if k == "CDS")
    n_igs = sum(1 for _, _, k in elements if k == "IGS")
    return glm2_str, {
        "cds": n_cds, "igs": n_igs,
        "tokens": len(glm2_str),
    }


def main():
    args = parse_args()
    as_dir = Path(args.antismash_dir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genome_dirs = sorted(as_dir.glob("GC*")) + sorted(as_dir.glob("GB_*"))
    ok, skipped, failed = 0, 0, 0

    for gdir in genome_dirs:
        genome_id = gdir.name
        main_gbks = sorted(gdir.glob("*genomic.gbk"))
        if not main_gbks:
            continue

        out_path = out_dir / f"{genome_id}_glm2.txt"
        if out_path.exists():
            skipped += 1
            continue

        try:
            glm2_str, stats = build_glm2_string(main_gbks[0])
            if stats["cds"] < 10:
                print(f"SKIP {genome_id}: only {stats['cds']} genes")
                continue
            out_path.write_text(glm2_str)
            print(f"{genome_id}: {stats['cds']} CDS, {stats['tokens']} tokens", flush=True)
            ok += 1
        except Exception as e:
            print(f"FAIL {genome_id}: {e}")
            failed += 1

    print(f"\nDone: {ok} OK, {skipped} skipped, {failed} failed")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
