"""Evaluate a trained detector on the Nine-Genome (NG) test set.

Fixed 128-gene windows over each contig, per-gene BGC scores against the
ClusterFinder labels in ``291BGC.csv``. The primary number uses the validation
threshold saved with the checkpoint; the NG-best threshold scan is reported
for diagnostics only and must never select a model or a seed.

Requires NG embeddings from ``scripts/encode.py --source ng``.

Usage
-----
    uv run python scripts/eval_ng.py \
        --checkpoint runs/seed1301/detector_960d_best.pt \
        --metrics-path runs/seed1301/ng_validation_selected.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import torch
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
from tqdm import tqdm

from bgc_gnn.checkpoint import detector_settings, load_checkpoint, validation_threshold
from bgc_gnn.dataset import normalize_strand, normalize_translation
from bgc_gnn.eval import bgc_prophet_postprocess, window_aggregation_weight
from bgc_gnn.model import BGCProphetDetector, WINDOW_SIZE

# ── paths ───────────────────────────────────────────────────────────────────
GBK_PATH = Path("data/raw/ng/ClusterFinder_Annotated_Contigs.full.gbk")
CSV_PATH = Path("data/raw/ng/291BGC.csv")
OUT_DIR = Path("data/derived/ng_embeddings")
EMBEDDING_DIM = 960


def _embedding_path() -> Path:
    return OUT_DIR / f"ng_gene_embeddings_{EMBEDDING_DIM}d.parquet"


def _metrics_path() -> Path:
    return OUT_DIR / f"ng_metrics_{EMBEDDING_DIM}d.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_ng_gbk(gbk_path: Path) -> list[dict]:
    """Parse the 9-genome GBK → list of CDS genes with metadata."""
    print("[1] Parsing GBK ...", flush=True)
    genes: list[dict] = []
    for record in tqdm(SeqIO.parse(str(gbk_path), "genbank"), desc="  contigs"):
        acc = record.annotations.get("accessions", [record.id])[0]
        org = record.annotations.get("organism", "?")
        for feature_index, feat in enumerate(record.features):
            if feat.type != "CDS":
                continue
            lt = feat.qualifiers.get("locus_tag", [""])[0]
            if not lt:
                lt = f"__cds_{feature_index}"
            genes.append({
                "gene_id": f"{acc}:{lt}",
                "accession": acc,
                "organism": org,
                "locus_tag": lt,
                "start": int(feat.location.start),
                "end": int(feat.location.end),
                "strand": normalize_strand(feat.location.strand),
                "translation": normalize_translation(
                    feat.qualifiers.get("translation", [""])[0]),
            })
    print(f"  → {len(genes):,} CDS genes", flush=True)
    return genes


def load_labels(csv_path: Path, genes: list[dict]) -> dict[str, bool]:
    """Cross-reference CSV BGC annotations → {gene_id: in_bgc}."""
    print("[2] Loading BGC labels ...", flush=True)
    lt2cluster: dict[str, str] = {}
    with open(csv_path) as f:
        import csv
        for row in csv.DictReader(f, delimiter="\t"):
            lt = row["Locus Tag"].strip()
            cluster = row["Cluster ID"].strip()
            if lt and cluster:
                lt2cluster[lt] = cluster

    labels: dict[str, bool] = {}
    n_bgc = 0
    for g in genes:
        in_bgc = g["locus_tag"] in lt2cluster
        labels[g["gene_id"]] = in_bgc
        n_bgc += in_bgc

    print(f"  → {n_bgc:,} BGC genes / {len(genes):,} total ({n_bgc/len(genes)*100:.1f}%)")
    return labels


# ── evaluation ───────────────────────────────────────────────────────────────

def load_detector(checkpoint: str, device: torch.device):
    state, metadata = load_checkpoint(checkpoint, device)
    settings = detector_settings(metadata, EMBEDDING_DIM)
    model = BGCProphetDetector(**settings).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata, settings


def evaluate_model(
    checkpoint: str, labels: dict[str, bool], device: str = "cuda",
    stride: int = WINDOW_SIZE, aggregate: str = "none",
    postprocess: bool = False, max_gap: int = 2, min_genes: int = 3,
    longest_only: bool = False, metrics_path: str = "",
    fixed_threshold_metrics_path: str = "", dump_gene_scores: str = "",
) -> None:
    """Evaluate using fixed 128-gene windows (same as training)."""
    print("[5] Evaluating model (fixed-window) ...", flush=True)
    if not 1 <= stride <= WINDOW_SIZE:
        raise ValueError(f"stride must be in [1, {WINDOW_SIZE}]")
    if postprocess and aggregate == "none":
        raise ValueError("BGC-Prophet postprocessing requires overlap aggregation")
    selected_output = Path(metrics_path) if metrics_path else _metrics_path()
    if fixed_threshold_metrics_path and Path(fixed_threshold_metrics_path).resolve() == selected_output.resolve():
        raise ValueError("fixed-threshold output must differ from validation-selected output")

    print("  Loading embeddings ...", flush=True)
    emb_map: dict[str, np.ndarray] = {}
    table = pq.read_table(_embedding_path())
    for gid, emb in zip(table.column("gene_id").to_pylist(),
                        table.column("embedding").to_pylist()):
        if gid in labels:
            emb_map[gid] = np.array(emb, dtype=np.float32)
    print(f"  → {len(emb_map):,} gene embeddings loaded", flush=True)

    print("  Loading model ...", flush=True)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, metadata, _ = load_detector(checkpoint, torch_device)
    calibrated_threshold, threshold_source = validation_threshold(metadata)

    genes = parse_ng_gbk(GBK_PATH)
    full_gene_order: dict[str, list[str]] = defaultdict(list)
    for g in genes:
        full_gene_order[g["accession"]].append(g["gene_id"])
    gene_start = {g["gene_id"]: g["start"] for g in genes}
    gene_order: dict[str, list[str]] = defaultdict(list)
    for acc, ordered in full_gene_order.items():
        full_gene_order[acc].sort(key=lambda gid: gene_start[gid])
        gene_order[acc] = [gid for gid in full_gene_order[acc]
                           if gid in emb_map and gid in labels]

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_gene_ids: list[str] = []
    scores_by_contig: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    score_sums: dict[str, float] = defaultdict(float)
    score_weights: dict[str, float] = defaultdict(float)

    window_count = 0
    for acc in tqdm(gene_order, desc="  contigs"):
        ordered = gene_order[acc]
        n = len(ordered)
        # Tail wrap-around keeps the fixed window protocol.
        for start in range(0, n, stride):
            positions = list(range(start, min(start + WINDOW_SIZE, n)))
            if len(positions) < WINDOW_SIZE:
                # Keep the model window size, but never invent contig adjacency.
                positions.extend(i % n for i in range(WINDOW_SIZE - len(positions)))
            window_ids = [ordered[position] for position in positions]

            embs = torch.tensor(
                np.stack([emb_map[gid] for gid in window_ids]), dtype=torch.float32,
            ).unsqueeze(0).to(torch_device)  # [1, 128, 960]
            with torch.no_grad():
                scores = model(embs)[0].cpu().numpy()

            for i, gid in enumerate(window_ids):
                s = float(scores[i])
                if aggregate == "none":
                    all_scores.append(s)
                    all_labels.append(int(labels[gid]))
                    all_gene_ids.append(gid)
                    scores_by_contig[acc][0].append(s)
                    scores_by_contig[acc][1].append(int(labels[gid]))
                else:
                    weight = window_aggregation_weight(i, len(window_ids), aggregate)
                    score_sums[gid] += weight * s
                    score_weights[gid] += weight
            window_count += 1

    if aggregate != "none":
        unique_ids = [gid for ordered in gene_order.values() for gid in ordered]
        all_scores = [score_sums[gid] / score_weights[gid] for gid in unique_ids]
        all_labels = [int(labels[gid]) for gid in unique_ids]
        all_gene_ids = unique_ids
        for acc, ordered in gene_order.items():
            scores_by_contig[acc] = (
                [score_sums[gid] / score_weights[gid] for gid in ordered],
                [int(labels[gid]) for gid in ordered],
            )
    print(f"  → {window_count} windows evaluated ({len(all_labels):,} genes; "
          f"stride={stride}, aggregate={aggregate})", flush=True)

    if dump_gene_scores:
        dump_path = Path(dump_gene_scores).expanduser()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_path,
            gene_ids=np.array(all_gene_ids),
            scores=np.array(all_scores, dtype=np.float32),
            labels=np.array(all_labels, dtype=np.int8),
            stride=np.int64(stride),
            aggregate=np.array(aggregate),
        )
        print(f"  → per-gene scores saved → {dump_path}", flush=True)

    def predictions_at(threshold: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        by_contig = {}
        for acc, (scores, _) in scores_by_contig.items():
            predicted = (np.asarray(scores) >= threshold).astype(np.int32)
            if postprocess:
                predicted = bgc_prophet_postprocess(
                    predicted, max_gap=max_gap, min_genes=min_genes,
                    longest_only=longest_only,
                )
            by_contig[acc] = predicted
        return np.concatenate(list(by_contig.values())), by_contig

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    y_pred, predictions_by_contig = predictions_at(calibrated_threshold)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    metrics: dict = {
        "n_genes": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_windows": window_count,
        "stride": stride,
        "aggregate": aggregate,
        "postprocess": ("bgc_prophet_longest" if longest_only
                        else "bgc_prophet") if postprocess else "none",
        "max_gap": max_gap if postprocess else None,
        "min_genes": min_genes if postprocess else None,
        "threshold": calibrated_threshold,
        "threshold_source": threshold_source,
        "threshold_metric": metadata.get("validation_threshold_metric", "legacy-default"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "checkpoint": str(checkpoint),
        "selected_epoch": metadata.get("epoch"),
    }
    if n_pos > 0 and n_neg > 0:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = 0.0

    if fixed_threshold_metrics_path:
        fixed_pred, _ = predictions_at(0.5)
        fixed_metrics = metrics | {
            "threshold": 0.5, "threshold_source": "fixed", "threshold_metric": "fixed",
            "accuracy": float(accuracy_score(y_true, fixed_pred)),
            "precision": float(precision_score(y_true, fixed_pred, zero_division=0)),
            "recall": float(recall_score(y_true, fixed_pred, zero_division=0)),
            "f1": float(f1_score(y_true, fixed_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, fixed_pred)),
        }
        fixed_output = Path(fixed_threshold_metrics_path)
        fixed_output.parent.mkdir(parents=True, exist_ok=True)
        fixed_output.write_text(json.dumps(fixed_metrics, indent=2) + "\n", encoding="utf-8")
        print(f"  Fixed 0.5: F1={fixed_metrics['f1']:.4f} "
              f"MCC={fixed_metrics['mcc']:.4f}; saved to {fixed_output}", flush=True)

    # Exploratory NG-best threshold scan: diagnostics only, never model selection.
    ng_best_mcc, ng_best_threshold, ng_best_pred = -1.0, 0.5, y_pred
    for t in np.arange(0.10, 0.95, 0.05):
        p, _ = predictions_at(float(t))
        m = float(matthews_corrcoef(y_true, p))
        if m > ng_best_mcc:
            ng_best_mcc, ng_best_threshold, ng_best_pred = m, t, p
    metrics["ng_best_threshold"] = ng_best_threshold
    metrics["ng_best_f1"] = f1_score(y_true, ng_best_pred, zero_division=0)
    metrics["ng_best_mcc"] = ng_best_mcc
    metrics["best_threshold"] = ng_best_threshold
    metrics["f1_best"] = metrics["ng_best_f1"]
    metrics["mcc_best"] = ng_best_mcc

    print("\n[6] Results", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Metric':18s} {'Global':>10s}", flush=True)
    print("-" * 70, flush=True)
    for k in ["n_genes", "n_pos", "n_neg", "n_windows"]:
        print(f"  {k:18s} {metrics[k]:>10}", flush=True)
    for k in ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"]:
        print(f"  {k:18s} {metrics[k]:>10.4f}", flush=True)
    print(f"  {'validation threshold':18s} {calibrated_threshold:>10.4f} ({threshold_source})", flush=True)
    print(f"  {'NG-best threshold':18s} {ng_best_threshold:>10.4f}", flush=True)
    print(f"  {'NG-best F1':18s} {metrics['ng_best_f1']:>10.4f}", flush=True)
    print(f"  {'NG-best MCC':18s} {ng_best_mcc:>10.4f}", flush=True)
    print("-" * 70, flush=True)

    print(f"\n  {'Genome':40s} {'Genes':>8s} {'F1':>8s} {'AUROC':>8s}", flush=True)
    print("  " + "-" * 66, flush=True)
    for acc in sorted(scores_by_contig):
        sc, lb = scores_by_contig[acc]
        yt = np.array(lb)
        ys = np.array(sc)
        if yt.sum() > 0 and len(yt) > yt.sum():
            f1_g = f1_score(yt, predictions_by_contig[acc], zero_division=0)
            auroc_g = roc_auc_score(yt, ys)
        else:
            f1_g = 0.0
            auroc_g = 0.5
        print(f"  {acc[:38]:40s} {len(yt):>8,} {f1_g:>8.4f} {auroc_g:>8.4f}", flush=True)
    print("=" * 70, flush=True)

    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\n  Metrics saved → {selected_output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BGC detector on the NG set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metrics-path", type=str, default="",
                        help="Override metrics output path (default: auto)")
    parser.add_argument("--fixed-threshold-metrics-path", type=str, default="",
                        help="Also save fixed 0.5 metrics from the same scores")
    parser.add_argument("--dump-gene-scores", type=str, default="",
                        help="Save per-gene (gene_id, score, label) npz for paired comparisons")
    parser.add_argument("--stride", type=int, default=WINDOW_SIZE,
                        help="Gene stride between inference windows (1-128)")
    parser.add_argument("--aggregate", choices=["none", "mean", "center"], default="none",
                        help="Overlap aggregation; none preserves legacy per-window metrics")
    parser.add_argument("--bgc-prophet-postprocess", action="store_true",
                        help="Merge gaps up to 2 genes and remove regions shorter than 3 genes")
    parser.add_argument("--bgc-prophet-longest-only", action="store_true",
                        help="Retain only the longest eligible region per contig")
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-genes", type=int, default=3)
    args = parser.parse_args()
    if args.max_gap < 0 or args.min_genes < 1:
        parser.error("--max-gap must be non-negative and --min-genes must be positive")
    if args.bgc_prophet_longest_only and not args.bgc_prophet_postprocess:
        parser.error("--bgc-prophet-longest-only requires --bgc-prophet-postprocess")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels(CSV_PATH, parse_ng_gbk(GBK_PATH))
    evaluate_model(
        args.checkpoint, labels, device=args.device, stride=args.stride,
        aggregate=args.aggregate, postprocess=args.bgc_prophet_postprocess,
        max_gap=args.max_gap, min_genes=args.min_genes,
        longest_only=args.bgc_prophet_longest_only,
        metrics_path=args.metrics_path,
        fixed_threshold_metrics_path=args.fixed_threshold_metrics_path,
        dump_gene_scores=args.dump_gene_scores,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
