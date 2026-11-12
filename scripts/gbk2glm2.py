#!/usr/bin/env python3
"""Convert MIBiG GBK files to gLM2 mixed-modality input format.

Parses a GenBank file containing a BGC, extracts CDS features (with their
amino-acid translations) and intergenic (IGS) DNA, then assembles them into
a single string compatible with gLM2's tokenizer.

Usage:
    python gbk2glm2.py BGC0000016.gbk
    python gbk2glm2.py BGC0000016.gbk -o sequence.txt --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

from Bio import SeqIO
from Bio.Seq import Seq

# gLM2 context window size (tokens)
CONTEXT_WINDOW = 4096


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def parse_mibig_gbk(
    path: str,
    *,
    drop_edge_igs: bool = False,
    min_igs_len: int = 0,
) -> Iterator[dict]:
    """Yield one dict per BGC in *path*.

    In practice MIBiG files contain a single record, but ``SeqIO.parse`` is
    used for robustness.

    Each yielded dict::

        {
            "id":          str   – accession or record.id
            "organism":    str   – source organism
            "elements":    list  – see below
            "glm2_string": str   – ready-to-tokenize input
            "stats":       dict  – token counts
        }

    An *element* is ``(strand_tag: "<+>"|"<->", seq: str, kind: "CDS"|"IGS")``
    ordered 5'→3' along the contig.
    """
    for record in SeqIO.parse(path, "genbank"):
        # -- collect CDS features sorted by start position ----------------
        cds_list: list[tuple[int, int, int, str]] = []  # start, end, strand(+1|-1), translation
        for feat in record.features:
            if feat.type != "CDS":
                continue

            loc = feat.location
            start = int(loc.start)
            end = int(loc.end)
            strand = loc.strand  # 1 → +, -1 → complement

            if "translation" in feat.qualifiers and feat.qualifiers["translation"][0]:
                aa = str(feat.qualifiers["translation"][0])
            else:
                # fallback – translate the extracted nucleotide sequence
                trans_table = int(feat.qualifiers.get("transl_table", ["11"])[0])
                cds_seq: Seq = loc.extract(record.seq)
                aa = str(cds_seq.translate(table=trans_table, to_stop=True))

            cds_list.append((start, end, strand, aa))

        cds_list.sort(key=lambda x: x[0])  # sort by start 5'→3'

        # -- build element list ------------------------------------------
        elements: list[tuple[str, str, str]] = []
        contig_len = len(record.seq)
        prev_end = 0

        for start, end, strand, aa in cds_list:
            # IGS before this CDS
            if start > prev_end:
                igs = str(record.seq[prev_end:start]).lower()
                if len(igs) > min_igs_len:
                    elements.append(("<+>", igs, "IGS"))

            # CDS itself
            strand_tag = "<->" if strand == -1 else "<+>"
            elements.append((strand_tag, aa.upper(), "CDS"))
            prev_end = end

        # trailing IGS
        if contig_len > prev_end:
            igs = str(record.seq[prev_end:contig_len]).lower()
            if len(igs) > min_igs_len:
                elements.append(("<+>", igs, "IGS"))

        # -- optional edge trimming (mimics gLM2 preprocessing) ----------
        if drop_edge_igs:
            while elements and elements[0][2] == "IGS":
                elements.pop(0)
            while elements and elements[-1][2] == "IGS":
                elements.pop(-1)

        # -- assemble ----------------------------------------------------
        glm2_string = "".join(tag + seq for tag, seq, _ in elements)

        # -- stats -------------------------------------------------------
        n_cds = sum(1 for _, _, k in elements if k == "CDS")
        n_igs = sum(1 for _, _, k in elements if k == "IGS")
        aa_tokens = sum(len(s) for _, s, k in elements if k == "CDS")
        nt_tokens = sum(len(s) for _, s, k in elements if k == "IGS")
        strand_tokens = n_cds + n_igs
        total = aa_tokens + nt_tokens + strand_tokens

        yield {
            "id":          str(record.id),
            "organism":    str(record.annotations.get("organism", "?")),
            "elements":    elements,
            "glm2_string": glm2_string,
            "stats": {
                "cds":         n_cds,
                "igs":         n_igs,
                "aa_tokens":   aa_tokens,
                "nt_tokens":   nt_tokens,
                "strand_tokens": strand_tokens,
                "total":       total,
                "fits_4k":     total <= CONTEXT_WINDOW,
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert MIBiG GBK to gLM2 mixed-modality input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s BGC0000016.gbk\n"
            "  %(prog)s BGC0000016.gbk -o gLM2_input.txt\n"
            "  %(prog)s BGC0000016.gbk --stats --drop-edge-igs\n"
            "  %(prog)s ../mibig_gbk/*.gbk --outdir converted/\n"
        ),
    )
    p.add_argument("gbk", type=str, help="MIBiG GBK file or glob pattern")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output file (single-record mode only; default stdout)")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (for multi-file mode; "
                        "one .txt per input)")
    p.add_argument("--drop-edge-igs", action="store_true",
                   help="Drop leading/trailing IGS (mimics gLM2 preprocessing)")
    p.add_argument("--min-igs-len", type=int, default=0,
                   help="Skip IGS segments shorter than N bases (default: 0)")
    p.add_argument("--stats", action="store_true",
                   help="Print per-file element/token statistics to stderr")
    p.add_argument("--summary", action="store_true", default=None,
                   help="Print final batch summary (auto-enabled for multiple files)")
    p.add_argument("--no-summary", action="store_false", dest="summary",
                   help="Suppress final batch summary")
    return p


def main() -> None:
    args = build_parser().parse_args()
    gbk_path = Path(args.gbk)

    # --- collect input files ------------------------------------------------
    if gbk_path.is_dir():
        files = sorted(gbk_path.glob("*.gbk"))
    elif "*" in str(gbk_path) or "?" in str(gbk_path):
        files = sorted(Path().glob(str(gbk_path)))
    else:
        files = [gbk_path]

    if not files:
        print("No GBK files found.", file=sys.stderr)
        sys.exit(1)

    # --- process each file --------------------------------------------------
    multiple = len(files) > 1

    if multiple and args.output and not args.outdir:
        print("Multiple inputs — use --outdir instead of -o.", file=sys.stderr)
        sys.exit(1)

    # default: show summary when multiple files, unless --no-summary
    show_summary = args.summary if args.summary is not None else multiple
    all_stats = [] if (show_summary or args.stats) else None

    for fp in files:
        for entry in parse_mibig_gbk(
            str(fp),
            drop_edge_igs=args.drop_edge_igs,
            min_igs_len=args.min_igs_len,
        ):
            s = entry["stats"]
            if show_summary or args.stats:
                all_stats.append((entry["id"], entry["organism"], s))

            # per-file stats
            if args.stats:
                tag = "OK" if s["fits_4k"] else "EXCEEDED"
                print(
                    f"[{entry['id']}] {entry['organism']}\n"
                    f"  CDS={s['cds']}  IGS={s['igs']}  "
                    f"AA={s['aa_tokens']}  NT={s['nt_tokens']}  "
                    f"total={s['total']}  (≤{CONTEXT_WINDOW})={tag}",
                    file=sys.stderr,
                )

            # output
            out_content = entry["glm2_string"]
            if args.outdir:
                out_dir = Path(args.outdir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{entry['id']}_glm2.txt"
                out_path.write_text(out_content)
            elif args.output and not multiple:
                Path(args.output).write_text(out_content)
            else:
                sys.stdout.write(out_content)
                sys.stdout.write("\n")

    # --- batch summary -------------------------------------------------------
    if show_summary and all_stats:
        n_total = len(all_stats)
        n_ok = sum(1 for _, _, s in all_stats if s["fits_4k"])
        n_exceed = n_total - n_ok
        totals = {k: sum(s[k] for _, _, s in all_stats)
                  for k in ("cds", "igs", "aa_tokens", "nt_tokens", "total")}
        print(
            f"\n{'='*50}\n"
            f"BATCH SUMMARY  ({n_total} files)\n"
            f"{'='*50}\n"
            f"  Within {CONTEXT_WINDOW}:  {n_ok:>5} ({n_ok/n_total*100:5.1f}%)\n"
            f"  Over {CONTEXT_WINDOW}:    {n_exceed:>5} ({n_exceed/n_total*100:5.1f}%)\n"
            f"  ───────────────────────────\n"
            f"  Total CDS:       {totals['cds']:>5}\n"
            f"  Total IGS:       {totals['igs']:>5}\n"
            f"  Total AA tokens: {totals['aa_tokens']:>9}\n"
            f"  Total NT tokens: {totals['nt_tokens']:>9}\n"
            f"  Total tokens:    {totals['total']:>9}\n"
            f"  Avg tokens/file: {totals['total']/n_total:>8.0f}\n"
            f"{'='*50}",
            file=sys.stderr,
        )



if __name__ == "__main__":
    main()
