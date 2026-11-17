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
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bgc_gnn.checkpoint import validation_threshold
from bgc_gnn.predict import (
    STRIDE,
    load_detector,
    mean_scores,
)
from bgc_gnn.predict import (
    gene_order as contig_order,
)
from bgc_gnn.predict import (
    parse_gbk as parse_ng_gbk,
)

# ── paths ───────────────────────────────────────────────────────────────────
GBK_PATH = Path("data/raw/ng/ClusterFinder_Annotated_Contigs.full.gbk")
CSV_PATH = Path("data/raw/ng/291BGC.csv")
OUT_DIR = Path("data/derived/ng_embeddings")
EMBEDDING_DIM = 960


def _embedding_path() -> Path:
    return OUT_DIR / f"ng_gene_embeddings_{EMBEDDING_DIM}d.parquet"


# ── helpers ──────────────────────────────────────────────────────────────────


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

    print(
        f"  → {n_bgc:,} BGC genes / {len(genes):,} total ({n_bgc / len(genes) * 100:.1f}%)"
    )
    return labels


# ── evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    checkpoint: str,
    labels: dict[str, bool],
    device: str = "cuda",
    metrics_path: str = "",
    fixed_threshold_metrics_path: str = "",
    dump_gene_scores: str = "",
    ng_gbk: Path = GBK_PATH,
    emb_path: Path | None = None,
) -> None:
    """Evaluate using fixed 128-gene windows (same as training)."""
    print("[5] Evaluating model (fixed-window) ...", flush=True)
    stride, aggregate = STRIDE, "mean"
    selected_output = (
        Path(metrics_path)
        if metrics_path
        else Path(checkpoint).parent / "ng_metrics.json"
    )
    if (
        fixed_threshold_metrics_path
        and Path(fixed_threshold_metrics_path).resolve() == selected_output.resolve()
    ):
        raise ValueError(
            "fixed-threshold output must differ from validation-selected output"
        )

    print("  Loading embeddings ...", flush=True)
    emb_map: dict[str, np.ndarray] = {}
    table = pq.read_table(emb_path if emb_path is not None else _embedding_path())
    for gid, emb in zip(
        table.column("gene_id").to_pylist(), table.column("embedding").to_pylist()
    ):
        if gid in labels:
            emb_map[gid] = np.array(emb, dtype=np.float32)
    missing = set(labels) - emb_map.keys()
    if missing:
        warnings.warn(
            f"NG evaluation excludes {len(missing)} genes without embeddings "
            f"(including {sum(labels[g] for g in missing)} positives); "
            "reported metrics cover the remaining genes only.",
            RuntimeWarning,
        )
    if not emb_map:
        raise ValueError("NG evaluation requires at least one gene embedding")
    print(f"  → {len(emb_map):,} gene embeddings loaded", flush=True)

    print("  Loading model ...", flush=True)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, metadata = load_detector(checkpoint, torch_device)
    calibrated_threshold, threshold_source = validation_threshold(metadata)

    genes = parse_ng_gbk(ng_gbk)
    gene_order = contig_order(genes, set(emb_map) & set(labels))

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_gene_ids: list[str] = []
    scores_by_contig: dict[str, tuple[list[float], list[int]]] = defaultdict(
        lambda: ([], [])
    )

    scores, window_count = mean_scores(model, emb_map, gene_order, torch_device)
    unique_ids = [gid for ordered in gene_order.values() for gid in ordered]
    all_scores = [scores[gid] for gid in unique_ids]
    all_labels = [int(labels[gid]) for gid in unique_ids]
    all_gene_ids = unique_ids
    for acc, ordered in gene_order.items():
        if ordered:
            scores_by_contig[acc] = (
                [scores[gid] for gid in ordered],
                [int(labels[gid]) for gid in ordered],
            )
    print(
        f"  → {window_count} windows evaluated ({len(all_labels):,} genes; "
        f"stride={stride}, aggregate={aggregate})",
        flush=True,
    )

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
            by_contig[acc] = predicted
        return np.concatenate(list(by_contig.values())), by_contig

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    y_pred, predictions_by_contig = predictions_at(calibrated_threshold)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    metrics: dict = {
        "missing_embeddings": len(missing),
        "missing_positive_genes": sum(labels[g] for g in missing),
        "n_genes": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_windows": window_count,
        "stride": stride,
        "aggregate": aggregate,
        "postprocess": "none",
        "threshold": calibrated_threshold,
        "threshold_source": threshold_source,
        "threshold_metric": metadata["validation_threshold_metric"],
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
            "threshold": 0.5,
            "threshold_source": "fixed",
            "threshold_metric": "fixed",
            "accuracy": float(accuracy_score(y_true, fixed_pred)),
            "precision": float(precision_score(y_true, fixed_pred, zero_division=0)),
            "recall": float(recall_score(y_true, fixed_pred, zero_division=0)),
            "f1": float(f1_score(y_true, fixed_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, fixed_pred)),
        }
        fixed_output = Path(fixed_threshold_metrics_path)
        fixed_output.parent.mkdir(parents=True, exist_ok=True)
        fixed_output.write_text(
            json.dumps(fixed_metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"  Fixed 0.5: F1={fixed_metrics['f1']:.4f} "
            f"MCC={fixed_metrics['mcc']:.4f}; saved to {fixed_output}",
            flush=True,
        )

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

    print("\n[6] Results", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Metric':18s} {'Global':>10s}", flush=True)
    print("-" * 70, flush=True)
    for k in ["n_genes", "n_pos", "n_neg", "n_windows"]:
        print(f"  {k:18s} {metrics[k]:>10}", flush=True)
    for k in ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"]:
        print(f"  {k:18s} {metrics[k]:>10.4f}", flush=True)
    print(
        f"  {'validation threshold':18s} {calibrated_threshold:>10.4f} ({threshold_source})",
        flush=True,
    )
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
        print(
            f"  {acc[:38]:40s} {len(yt):>8,} {f1_g:>8.4f} {auroc_g:>8.4f}", flush=True
        )
    print("=" * 70, flush=True)

    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\n  Metrics saved → {selected_output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a BGC detector on the NG set"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Read-only data directory containing raw/ and derived/",
    )
    parser.add_argument("--ng-gbk", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--emb-path", type=Path)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="",
        help="Override metrics output path (default: auto)",
    )
    parser.add_argument(
        "--fixed-threshold-metrics-path",
        type=str,
        default="",
        help="Also save fixed 0.5 metrics from the same scores",
    )
    parser.add_argument(
        "--dump-gene-scores",
        type=str,
        default="",
        help="Save per-gene (gene_id, score, label) npz for paired comparisons",
    )
    args = parser.parse_args()

    ng_gbk = (
        args.ng_gbk
        or args.data_root / "raw/ng/ClusterFinder_Annotated_Contigs.full.gbk"
    )
    labels_path = args.labels or args.data_root / "raw/ng/291BGC.csv"
    emb_path = (
        args.emb_path
        or args.data_root / "derived/ng_embeddings/ng_gene_embeddings_960d.parquet"
    )
    for path in [ng_gbk, labels_path, emb_path]:
        if not path.is_file():
            parser.error(f"missing input: {path}")
    labels = load_labels(labels_path, parse_ng_gbk(ng_gbk))
    evaluate_model(
        args.checkpoint,
        labels,
        device=args.device,
        metrics_path=args.metrics_path,
        fixed_threshold_metrics_path=args.fixed_threshold_metrics_path,
        dump_gene_scores=args.dump_gene_scores,
        ng_gbk=ng_gbk,
        emb_path=emb_path,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
