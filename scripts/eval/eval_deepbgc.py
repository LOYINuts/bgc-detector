"""Score DeepBGC's per-gene predictions on the NG test set with the eval_ng metrics.

DeepBGC scores each protein once (mean of its Pfam-domain scores), so it has no
window/stride protocol. To stay comparable with the project ledger we report:

  unique_genes        per-gene scores; unit of the overlap (stride64/mean) entry
  pfam_subset         only genes DeepBGC scored; it skips proteins without Pfam
                      domains, so unscored genes cannot enter this block

In the full unique-gene block an unscored gene is imputed 0.0, which is what DeepBGC
effectively does when it decodes regions from scores.

Usage (on the server, project venv):
    uv run python scripts/eval/eval_deepbgc.py \
        --deepbgc-glob '~/bgc_deepbgc/out/*/*.full.gbk' \
        --out ~/bgc_deepbgc/out/deepbgc_ng_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.eval.eval_ng import CSV_PATH, GBK_PATH, load_labels, parse_ng_gbk

SCORE_QUALIFIER = "deepbgc_score"
FIXED_THRESHOLD = 0.5
THRESHOLD_GRID = np.arange(0.10, 0.95, 0.05)


def load_deepbgc_scores(paths: list[str]) -> dict[str, float]:
    """Parse DeepBGC output GenBank files → {locus_tag: per-protein score}."""
    scores: dict[str, float] = {}
    for path in paths:
        for record in SeqIO.parse(path, "genbank"):
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                locus_tag = feature.qualifiers.get("locus_tag", [""])[0]
                raw = feature.qualifiers.get(SCORE_QUALIFIER)
                if locus_tag and raw:
                    scores[locus_tag] = float(raw[0])
    return scores


def remap_locus_tags(
    raw_scores: dict[str, float], locus_tag_to_gene_id: dict[str, str]
) -> dict[str, float]:
    """DeepBGC keys scores by locus_tag; NG gene IDs are "<accession>:<locus_tag>"."""
    return {
        locus_tag_to_gene_id[lt]: score
        for lt, score in raw_scores.items()
        if lt in locus_tag_to_gene_id
    }


def threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict:
    y_pred = (y_score >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_block(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Ranking, calibration-free threshold and NG-best threshold metrics."""
    n_pos = int(y_true.sum())
    prevalence = n_pos / len(y_true) if len(y_true) else 0.0
    block = {
        "n_genes": len(y_true),
        "n_pos": n_pos,
        "n_neg": int(len(y_true) - n_pos),
        "auroc": float(roc_auc_score(y_true, y_score))
        if 0 < n_pos < len(y_true)
        else 0.5,
        "auprc": float(average_precision_score(y_true, y_score))
        if 0 < n_pos < len(y_true)
        else 0.0,
        "prevalence": prevalence,
    }
    block["auprc_lift"] = block["auprc"] / prevalence if prevalence else 0.0
    block.update(threshold_metrics(y_true, y_score, FIXED_THRESHOLD))

    best = {"mcc": -2.0}
    curve = []
    for t in THRESHOLD_GRID:
        metrics = threshold_metrics(y_true, y_score, float(t))
        curve.append(metrics)
        if metrics["mcc"] > best["mcc"]:
            best = metrics
    block["ng_best_threshold"] = best["threshold"]
    block["ng_best_f1"] = best["f1"]
    block["ng_best_mcc"] = best["mcc"]
    block["ng_best_precision"] = best["precision"]
    block["ng_best_recall"] = best["recall"]
    block["threshold_curve"] = curve
    return block


def per_contig_blocks(
    order: dict[str, list[str]],
    labels: dict[str, bool],
    scores: dict[str, float],
) -> dict[str, dict]:
    """Same metrics restricted to one contig, unique genes only."""
    blocks = {}
    for accession, ordered in order.items():
        y_true = np.array([int(labels[gene_id]) for gene_id in ordered])
        y_score = np.array([scores.get(gene_id, 0.0) for gene_id in ordered])
        block = evaluate_block(y_true, y_score)
        block.pop("threshold_curve", None)
        blocks[accession] = block
    return blocks


def unique_gene_order(genes: list[dict]) -> dict[str, list[str]]:
    """Contig → gene IDs in genomic order (eval_ng's gene_order without filters)."""
    order: dict[str, list[str]] = defaultdict(list)
    for gene in genes:
        order[gene["accession"]].append(gene["gene_id"])
    start = {gene["gene_id"]: gene["start"] for gene in genes}
    for accession in order:
        order[accession].sort(key=lambda gene_id: start[gene_id])
    return dict(order)


def score_vectors(
    order: dict[str, list[str]],
    labels: dict[str, bool],
    scores: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return one label/score per gene, matching overlap mean evaluation units."""
    gene_ids = [gene_id for ordered in order.values() for gene_id in ordered]
    y_true = np.array([int(labels[gene_id]) for gene_id in gene_ids])
    y_score = np.array([scores.get(gene_id, 0.0) for gene_id in gene_ids])
    n_scored = sum(1 for gene_id in gene_ids if gene_id in scores)
    return y_true, y_score, n_scored


def subset_vectors(
    gene_ids: list[str],
    labels: dict[str, bool],
    scores: dict[str, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    kept = [gene_id for gene_id in gene_ids if gene_id in scores]
    if not kept:
        return None
    y_true = np.array([int(labels[gene_id]) for gene_id in kept])
    y_score = np.array([scores[gene_id] for gene_id in kept])
    return y_true, y_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepBGC per-gene scores on NG"
    )
    parser.add_argument(
        "--deepbgc-glob",
        nargs="+",
        required=True,
        help="DeepBGC *.full.gbk output files (glob supported)",
    )
    parser.add_argument("--out", required=True, help="Metrics JSON output path")
    parser.add_argument(
        "--our-scores",
        nargs="*",
        default=[],
        help="Optional eval_ng --dump-gene-scores .npz files; adds a paired "
        "comparison on the genes DeepBGC scored",
    )
    parser.add_argument(
        "--gene-table",
        type=str,
        default="",
        help="Also write a per-gene CSV (labels, DeepBGC score, each our-scores file)",
    )
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.deepbgc_glob:
        paths.extend(sorted(glob.glob(str(Path(pattern).expanduser()))))
    if not paths:
        raise SystemExit("no DeepBGC output GenBank files matched")

    genes = parse_ng_gbk(GBK_PATH)
    labels = load_labels(CSV_PATH, genes)
    order = unique_gene_order(genes)
    # DeepBGC output is keyed by locus_tag; the NG gene IDs are "<accession>:<locus_tag>".
    locus_tag_to_gene_id = {gene["locus_tag"]: gene["gene_id"] for gene in genes}
    raw_scores = load_deepbgc_scores(paths)
    scores = remap_locus_tags(raw_scores, locus_tag_to_gene_id)
    unmatched = len(raw_scores) - len(scores)
    print(
        f"  DeepBGC scored {len(scores):,} / {len(genes):,} genes; "
        f"{len(genes) - len(scores):,} genes have no Pfam-derived score; "
        f"{unmatched:,} score tags not in the NG gene set",
        flush=True,
    )

    result: dict = {
        "source": "DeepBGC 0.1.23 deepbgc detector (official pickle, v0.1.0 data release)",
        "score_qualifier": SCORE_QUALIFIER,
        "n_deepbgc_gbk": len(paths),
        "n_genes_total": len(genes),
        "n_genes_scored": len(scores),
    }

    scored_ids = set(scores)
    our_columns: dict[str, dict[str, float]] = {}
    y_true, y_score, n_scored = score_vectors(order, labels, scores)
    block = evaluate_block(y_true, y_score)
    block["n_scored_units"] = int(n_scored)
    block["missing_score_imputed"] = 0.0
    result["unique_genes"] = block

    unique_ids = [gene_id for ordered in order.values() for gene_id in ordered]
    subset = subset_vectors(unique_ids, labels, scores)
    if subset is not None:
        y_true, y_score = subset
        result["pfam_subset"] = evaluate_block(y_true, y_score)

    result["per_contig_unique_genes"] = per_contig_blocks(order, labels, scores)

    for path in args.our_scores:
        data = np.load(Path(path).expanduser(), allow_pickle=False)
        ours = {
            str(g): float(s)
            for g, s in zip(data["gene_ids"], data["scores"], strict=True)
        }
        our_columns[Path(path).expanduser().parent.name] = ours
        common = [
            gene_id
            for gene_id in unique_ids
            if gene_id in scored_ids and gene_id in ours
        ]
        if not common:
            print(f"  {path}: no overlapping scored genes, skipped", flush=True)
            continue
        y_true = np.array([int(labels[gene_id]) for gene_id in common])
        dump_path = Path(path).expanduser()
        deepbgc_block = evaluate_block(y_true, np.array([scores[g] for g in common]))
        ours_block = evaluate_block(y_true, np.array([ours[g] for g in common]))
        deepbgc_block.pop("threshold_curve", None)
        ours_block.pop("threshold_curve", None)
        result.setdefault("paired_subset", {})[
            f"{dump_path.parent.name}/{dump_path.name}"
        ] = {
            "n_genes": len(common),
            "n_pos": int(y_true.sum()),
            "deepbgc": deepbgc_block,
            "ours": ours_block,
        }

    print("\n" + "=" * 100)
    print(
        f"  {'block':22s} {'genes':>8s} {'pos':>6s} {'AUROC':>8s} {'AUPRC':>8s} "
        f"{'AUPRC/preval':>12s} {'Acc@0.5':>8s} {'F1@0.5':>8s} {'MCC@0.5':>8s} "
        f"{'NGbest':>8s} {'F1/MCC@best':>14s}"
    )
    for name in ("unique_genes", "pfam_subset"):
        block = result.get(name)
        if not block:
            continue
        print(
            f"  {name:22s} {block['n_genes']:>8,} {block['n_pos']:>6,} "
            f"{block['auroc']:>8.4f} {block['auprc']:>8.4f} "
            f"{block['auprc_lift']:>12.2f} {block['accuracy']:>8.4f} "
            f"{block['f1']:>8.4f} {block['mcc']:>8.4f} "
            f"{block['ng_best_threshold']:>8.2f} "
            f"{block['ng_best_f1']:>6.3f}/{block['ng_best_mcc']:<6.3f}"
        )
    print("=" * 100)

    if args.gene_table:
        gene_table = Path(args.gene_table).expanduser()
        gene_table.parent.mkdir(parents=True, exist_ok=True)
        with open(gene_table, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "gene_id",
                    "contig",
                    "locus_tag",
                    "label",
                    "deepbgc_score",
                    *list(our_columns),
                ]
            )
            for gene_id in unique_ids:
                contig, locus_tag = gene_id.split(":", 1)
                writer.writerow(
                    [
                        gene_id,
                        contig,
                        locus_tag,
                        int(labels[gene_id]),
                        f"{scores[gene_id]:.6f}" if gene_id in scores else "",
                    ]
                    + [
                        f"{column[gene_id]:.6f}" if gene_id in column else ""
                        for column in our_columns.values()
                    ]
                )
        print(f"  Per-gene table saved → {gene_table}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  Metrics saved → {out}")


if __name__ == "__main__":
    main()
