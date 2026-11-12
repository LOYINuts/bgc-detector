"""Score antiSMASH's per-gene calls on the NG test set with the eval_ng metrics.

antiSMASH is a rule-based cluster caller, not a ranking model: it emits binary
region/protocluster calls. A single binary score would make AUROC degenerate, so
this script ranks the tiers antiSMASH itself annotates, plus the plain binary
membership variants:

    score  tier                                    source in antiSMASH output
    0.90   core biosynthetic                       CDS /gene_kind="biosynthetic"
    0.80   biosynthetic, did not define the core    CDS /gene_kind="biosynthetic-additional"
    0.70   other / transport / regulatory          CDS /gene_kind="other|transport|regulatory"
    0.60   inside a region, outside its protoclusters   inside a /region span only
    0.00   outside every region

Every tier above 0.5 is a gene antiSMASH puts inside a region, so the 0.5 cut is
the tool's own binary decision ("in a region"); the finer tiers only order genes
inside its calls. A gene the tool calls can therefore never be scored as "not BGC".

AUROC/AUPRC are computed on that 5-level score, so they rank tiers, not genes:
within a tier all genes tie. The `binary_variants` block gives the honest
deployment numbers, because that is what antiSMASH actually decides.

Blocks mirror scripts/eval_deepbgc.py so numbers stay comparable:

  unique_genes         one score per gene (unit of the overlap stride64/mean entry)
  stride128_positions  the exact gene-position multiset of the stride128/none entry
  in_region_subset     only genes inside an antiSMASH region (coverage block)

Usage (on the server, project venv):
    uv run python scripts/eval_antismash.py \
        --antismash-glob '~/bgc_antismash/out/ng_full/*.gbk' \
        --out ~/bgc_antismash/out/antismash_ng_metrics.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import glob
import json

import numpy as np
from Bio import SeqIO

from eval_deepbgc import (
    evaluate_block,
    per_contig_blocks,
    score_vectors,
    unique_gene_order,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from eval_ng import CSV_PATH, GBK_PATH, load_labels, parse_ng_gbk

GENE_KIND_SCORES = {
    "biosynthetic": 0.90,
    "biosynthetic-additional": 0.80,
    "other": 0.70,
    "transport": 0.70,
    "regulatory": 0.70,
}
# Inside a region is antiSMASH's own binary call, so every tier above sits > 0.5:
# a gene the tool puts in a region can never be scored as "not BGC" at 0.5.
REGION_ONLY_SCORE = 0.60
FIXED_THRESHOLD = 0.5  # == "inside a region", the tool's actual decision
PROTOCLUSTER_SCORE = 0.70


def parse_antismash(paths: list[str]) -> dict[str, dict]:
    """Parse antiSMASH output GenBank files → tier per CDS plus diagnostics.

    Tiers are returned twice, keyed by locus_tag (used when antiSMASH was given the
    NG CDS via ``--genefinding-tool none``) and as coordinate records (used when
    antiSMASH ran its own prodigal gene calling).
    """
    tiers: dict[str, float] = {}
    gene_kinds: dict[str, str] = {}
    in_region: set[str] = set()
    coordinates: list[tuple[str, int, int, float]] = []
    diagnostics = {"n_records": 0, "n_regions": 0, "n_protoclusters": 0,
                   "n_cds": 0, "gene_kind_counts": {}}

    for path in paths:
        for record in SeqIO.parse(path, "genbank"):
            diagnostics["n_records"] += 1
            accession = _base_accession(
                record.annotations.get("accessions", [record.id])[0])
            regions = [(int(f.location.start), int(f.location.end))
                       for f in record.features if f.type == "region"]
            diagnostics["n_regions"] += len(regions)
            diagnostics["n_protoclusters"] += sum(1 for f in record.features
                                                  if f.type == "protocluster")
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                locus_tag = feature.qualifiers.get("locus_tag", [""])[0]
                diagnostics["n_cds"] += 1
                kind = feature.qualifiers.get("gene_kind", [""])[0]
                start, end = int(feature.location.start), int(feature.location.end)
                # Region borders are offsets, not gene boundaries; a CDS that
                # straddles a border still belongs to the region.
                inside = any(start < r_end and end > r_start for r_start, r_end in regions)
                if kind:
                    if locus_tag:
                        gene_kinds[locus_tag] = kind
                    diagnostics["gene_kind_counts"][kind] = (
                        diagnostics["gene_kind_counts"].get(kind, 0) + 1)
                # A gene_kind outside every region (e.g. a full-hmmer SecMet domain)
                # is an annotation, not a cluster call, so it cannot score.
                tier = GENE_KIND_SCORES.get(kind, REGION_ONLY_SCORE) if inside else 0.0
                coordinates.append((accession, start, end, tier))
                if locus_tag and tier > 0.0:
                    tiers[locus_tag] = tier
                    if inside:
                        in_region.add(locus_tag)

    diagnostics["n_in_region"] = len(in_region)
    diagnostics["n_scored"] = len(tiers)
    return {"scores": tiers, "gene_kinds": gene_kinds, "in_region": in_region,
            "coordinates": coordinates, "diagnostics": diagnostics}


def _base_accession(accession: str) -> str:
    """NG gene IDs use the bare accession, antiSMASH records may keep the version."""
    return accession.split(".")[0]


def map_by_coordinates(genes: list[dict],
                       coordinates: list[tuple[str, int, int, float]]) -> dict[str, float]:
    """Transfer antiSMASH CDS tiers onto NG genes by interval overlap.

    Used when antiSMASH ran its own gene calling, so no locus_tag matches. A CDS
    is assigned to an NG gene when they overlap by at least half of the shorter
    span; an NG gene covered by several CDS keeps the highest tier. NG genes with
    no overlapping antiSMASH CDS are absent from the result and score 0.0, which
    is the coverage penalty of running the tool's own gene prediction.
    """
    from bisect import bisect_left
    from collections import defaultdict

    by_accession: dict[str, list[dict]] = defaultdict(list)
    for gene in genes:
        by_accession[_base_accession(gene["accession"])].append(gene)
    starts: dict[str, list[int]] = {}
    for accession, ordered in by_accession.items():
        ordered.sort(key=lambda gene: gene["start"])
        starts[accession] = [gene["start"] for gene in ordered]

    scores: dict[str, float] = {}
    for accession, start, end, tier in coordinates:
        accession = _base_accession(accession)
        if tier <= 0.0 or accession not in by_accession:
            continue
        ordered = by_accession[accession]
        index = max(0, bisect_left(starts[accession], start) - 1)
        while index < len(ordered) and ordered[index]["start"] < end:
            gene = ordered[index]
            index += 1
            overlap = min(end, gene["end"]) - max(start, gene["start"])
            shorter = min(end - start, gene["end"] - gene["start"])
            if overlap > 0 and 2 * overlap >= shorter:
                gene_id = gene["gene_id"]
                if tier > scores.get(gene_id, -1.0):
                    scores[gene_id] = tier
    return scores


def binary_variants(y_true: np.ndarray, scores: np.ndarray,
                    in_region: np.ndarray) -> dict[str, dict]:
    """Plain binary calls antiSMASH actually makes, per gene."""
    variants = {
        "in_region": in_region > 0,
        "in_protocluster": scores >= PROTOCLUSTER_SCORE,
        "core_biosynthetic_only": scores >= 0.85,
    }
    blocks = {}
    for name, mask in variants.items():
        y_pred = mask.astype(np.int32)
        blocks[name] = {
            "n_predicted": int(y_pred.sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate antiSMASH calls on NG")
    parser.add_argument("--antismash-glob", nargs="+", required=True,
                        help="antiSMASH output *.gbk files (glob supported)")
    parser.add_argument("--out", required=True, help="Metrics JSON output path")
    parser.add_argument("--our-scores", nargs="*", default=[],
                        help="Optional eval_ng --dump-gene-scores .npz files; adds a paired "
                             "comparison on the same genes")
    parser.add_argument("--map-by-coordinates", action="store_true",
                        help="Match antiSMASH CDS to NG genes by interval overlap instead of "
                             "locus_tag (needed when antiSMASH ran its own gene calling)")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.antismash_glob:
        paths.extend(sorted(glob.glob(str(Path(pattern).expanduser()))))
    if not paths:
        raise SystemExit("no antiSMASH output GenBank files matched")

    genes = parse_ng_gbk(GBK_PATH)
    labels = load_labels(CSV_PATH, genes)
    order = unique_gene_order(genes)
    locus_tag_to_gene_id = {gene["locus_tag"]: gene["gene_id"] for gene in genes}
    assert len(locus_tag_to_gene_id) == len(genes), "duplicate locus_tag in NG"

    parsed = parse_antismash(paths)
    if args.map_by_coordinates:
        scores = map_by_coordinates(genes, parsed["coordinates"])
        in_region_ids = {gene_id for gene_id, score in scores.items()
                         if score >= REGION_ONLY_SCORE}
        unmatched = parsed["diagnostics"]["n_cds"] - len(scores)
        print(f"  antiSMASH CDS mapped to {len(scores):,} / {len(genes):,} NG genes by "
              f"coordinate overlap; {unmatched:,} CDS without an NG gene", flush=True)
        mapping = "coordinates"
    else:
        scores = {locus_tag_to_gene_id[lt]: score
                  for lt, score in parsed["scores"].items() if lt in locus_tag_to_gene_id}
        unmatched = len(parsed["scores"]) - len(scores)
        in_region_ids = {locus_tag_to_gene_id[lt] for lt in parsed["in_region"]
                         if lt in locus_tag_to_gene_id}
        print(f"  antiSMASH scored {len(scores):,} / {len(genes):,} genes; "
              f"{unmatched:,} scored tags not in the NG gene set", flush=True)
        mapping = "locus_tag"

    result: dict = {
        "source": ("antiSMASH 8.0.4, --taxon bacteria"
                   + (", own gene calling (prodigal)" if args.map_by_coordinates
                      else ", --genefinding-tool none (NG CDS reused)")),
        "gene_mapping": mapping,
        "score_tiers": {"biosynthetic": GENE_KIND_SCORES["biosynthetic"],
                        "biosynthetic-additional": GENE_KIND_SCORES["biosynthetic-additional"],
                        "other|transport|regulatory": PROTOCLUSTER_SCORE,
                        "in_region_only": REGION_ONLY_SCORE, "outside": 0.0},
        "fixed_threshold": FIXED_THRESHOLD,
        "n_genes_total": len(genes),
        "n_genes_scored": len(scores),
        "antismash_diagnostics": parsed["diagnostics"],
    }

    for mode in ("unique_genes", "stride128_positions"):
        y_true, y_score, n_scored = score_vectors(order, labels, scores, mode)
        block = evaluate_block(y_true, y_score)
        block["n_scored_units"] = int(n_scored)
        block["missing_score_imputed"] = 0.0
        result[mode] = block

    unique_ids = [gene_id for ordered in order.values() for gene_id in ordered]
    region_ids = [gene_id for gene_id in unique_ids if gene_id in in_region_ids]
    result["in_region_subset"] = evaluate_block(
        np.array([int(labels[gene_id]) for gene_id in region_ids]),
        np.array([scores[gene_id] for gene_id in region_ids]),
    )

    y_true = np.array([int(labels[gene_id]) for gene_id in unique_ids])
    y_score = np.array([scores.get(gene_id, 0.0) for gene_id in unique_ids])
    y_in_region = np.array([int(gene_id in in_region_ids) for gene_id in unique_ids])
    result["binary_variants"] = binary_variants(y_true, y_score, y_in_region)
    result["per_contig_unique_genes"] = per_contig_blocks(order, labels, scores)

    for path in args.our_scores:
        data = np.load(Path(path).expanduser(), allow_pickle=False)
        ours = {str(g): float(s) for g, s in zip(data["gene_ids"], data["scores"])}
        common = [gene_id for gene_id in unique_ids if gene_id in ours]
        if not common:
            print(f"  {path}: no overlapping genes, skipped", flush=True)
            continue
        y_true = np.array([int(labels[gene_id]) for gene_id in common])
        dump_path = Path(path).expanduser()
        ours_block = evaluate_block(y_true, np.array([ours[g] for g in common]))
        theirs_block = evaluate_block(y_true, np.array([scores.get(g, 0.0) for g in common]))
        ours_block.pop("threshold_curve", None)
        theirs_block.pop("threshold_curve", None)
        result.setdefault("paired_subset", {})[f"{dump_path.parent.name}/{dump_path.name}"] = {
            "n_genes": len(common),
            "n_pos": int(y_true.sum()),
            "antismash": theirs_block,
            "ours": ours_block,
        }

    print("\n" + "=" * 100, flush=True)
    print(f"  {'block':22s} {'genes':>8s} {'pos':>6s} {'AUROC':>8s} {'AUPRC':>8s} "
          f"{'AUPRC/preval':>12s} {'Acc@0.5':>8s} {'F1@0.5':>8s} {'MCC@0.5':>8s} "
          f"{'NGbest':>8s} {'F1/MCC@best':>14s}", flush=True)
    for name in ("unique_genes", "stride128_positions", "in_region_subset"):
        block = result.get(name)
        if not block:
            continue
        print(f"  {name:22s} {block['n_genes']:>8,} {block['n_pos']:>6,} "
              f"{block['auroc']:>8.4f} {block['auprc']:>8.4f} "
              f"{block['auprc_lift']:>12.2f} {block['accuracy']:>8.4f} "
              f"{block['f1']:>8.4f} {block['mcc']:>8.4f} "
              f"{block['ng_best_threshold']:>8.2f} "
              f"{block['ng_best_f1']:>6.3f}/{block['ng_best_mcc']:<6.3f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  {'binary variant':22s} {'pred':>8s} {'Acc':>8s} {'P':>8s} {'R':>8s} "
          f"{'F1':>8s} {'MCC':>8s}", flush=True)
    for name, block in result["binary_variants"].items():
        print(f"  {name:22s} {block['n_predicted']:>8,} {block['accuracy']:>8.4f} "
              f"{block['precision']:>8.4f} {block['recall']:>8.4f} "
              f"{block['f1']:>8.4f} {block['mcc']:>8.4f}", flush=True)
    print("=" * 100, flush=True)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  Metrics saved → {out}", flush=True)


if __name__ == "__main__":
    main()