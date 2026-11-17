"""Genome-level prediction: GenBank genes scored in fixed 128-gene windows.

Shared by ``bgc-detector predict`` and the NG evaluation, so the CLI cannot drift
from the evaluated protocol: 128-gene windows, stride 64, mean over windows that
cover a gene, decision threshold taken from the checkpoint.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

from bgc_gnn.checkpoint import load_checkpoint, validate_architecture
from bgc_gnn.dataset import normalize_strand, normalize_translation
from bgc_gnn.model import WINDOW_SIZE, BGCDetector

STRIDE = 64


def parse_gbk(gbk_path: Path) -> list[dict]:
    """Parse a clusterfinder-annotated GenBank into one row per CDS."""
    print("Parsing GenBank ...", flush=True)
    genes: list[dict] = []
    with gbk_path.open() as stream:
        for record in tqdm(SeqIO.parse(stream, "genbank"), desc="  contigs"):
            acc = record.annotations.get("accessions", [record.id])[0]
            org = record.annotations.get("organism", "?")
            for feature_index, feat in enumerate(record.features):
                if feat.type != "CDS":
                    continue
                lt = feat.qualifiers.get("locus_tag", [""])[0]
                if not lt:
                    lt = f"__cds_{feature_index}"
                genes.append(
                    {
                        "gene_id": f"{acc}:{lt}",
                        "accession": acc,
                        "organism": org,
                        "locus_tag": lt,
                        "start": int(feat.location.start),
                        "end": int(feat.location.end),
                        "strand": normalize_strand(feat.location.strand),
                        "translation": normalize_translation(
                            feat.qualifiers.get("translation", [""])[0]
                        ),
                    }
                )
    print(f"  -> {len(genes):,} CDS genes", flush=True)
    return genes


def gene_order(genes: list[dict], keep: set[str]) -> dict[str, list[str]]:
    """Contig -> gene IDs sorted by start, restricted to genes in ``keep``."""
    by_contig: dict[str, list[dict]] = defaultdict(list)
    for gene in genes:
        if gene["gene_id"] in keep:
            by_contig[gene["accession"]].append(gene)
    order: dict[str, list[str]] = {}
    for accession, contig_genes in by_contig.items():
        contig_genes.sort(key=lambda gene: gene["start"])
        order[accession] = [gene["gene_id"] for gene in contig_genes]
    return order


def load_detector(checkpoint: str | Path, device: torch.device):
    """Require the validation-selected EMA detector checkpoint."""
    state, metadata = load_checkpoint(checkpoint, device)
    validate_architecture(metadata)
    if (
        metadata.get("checkpoint_kind") != "best"
        or metadata.get("weight_source") != "ema"
    ):
        raise ValueError(
            "prediction requires the validation-selected EMA detector checkpoint"
        )
    model = BGCDetector().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata


def mean_scores(
    model: torch.nn.Module,
    emb_map: dict[str, np.ndarray],
    order: dict[str, list[str]],
    device: torch.device,
    stride: int = STRIDE,
) -> tuple[dict[str, float], int]:
    """Per-gene mean probability over the windows that cover it, plus window count."""
    sums: dict[str, float] = defaultdict(float)
    weights: dict[str, int] = defaultdict(int)
    windows = 0
    for ordered in tqdm(order.values(), desc="  contigs", total=len(order)):
        n = len(ordered)
        for start in range(0, n, stride):
            positions = list(range(start, min(start + WINDOW_SIZE, n)))
            # Tail wrap-around keeps the fixed window protocol.
            if len(positions) < WINDOW_SIZE:
                positions.extend(i % n for i in range(WINDOW_SIZE - len(positions)))
            window_ids = [ordered[position] for position in positions]
            embs = (
                torch.tensor(
                    np.stack([emb_map[gid] for gid in window_ids]),
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                scores = model(embs)[0].cpu().numpy()
            for i, gid in enumerate(window_ids):
                sums[gid] += float(scores[i])
                weights[gid] += 1
            windows += 1
    return {gid: sums[gid] / weights[gid] for gid in sums}, windows
