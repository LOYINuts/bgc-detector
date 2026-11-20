"""Check that the repo's ESMC path reproduces the frozen training embeddings.

Runs `bgc_detector.esmc.embed_sequences` -- the same function `bgc-detector predict`
uses -- on a few NG genes and compares against the frozen parquet. Any change to
the loader must keep max |delta| near 4e-3.

    uv run scripts/check_esmc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bgc_detector.esmc import embed_sequences  # noqa: E402

EMB = "/home/gonghang/codes/my-bgc-v2/data/derived/ng_embeddings/ng_gene_embeddings_960d.parquet"
GBK = "/home/gonghang/codes/my-bgc-v2/data/raw/ng/ClusterFinder_Annotated_Contigs.full.gbk"
N_GENES = 8
TOLERANCE = 0.02


def main() -> int:
    table = pq.ParquetFile(EMB).read_row_group(0)
    frozen = {
        gene_id: np.array(vec, dtype=np.float32)
        for gene_id, vec in zip(
            table.column("gene_id").to_pylist(), table.column("embedding").to_pylist()
        )
    }

    wanted = sorted(frozen)[:N_GENES]
    proteins: dict[str, str] = {}
    for record in SeqIO.parse(GBK, "genbank"):
        accession = record.annotations.get("accessions", [record.id])[0]
        for feature in record.features:
            if feature.type != "CDS":
                continue
            gene_id = f"{accession}:{feature.qualifiers.get('locus_tag', [''])[0]}"
            if gene_id in wanted and gene_id not in proteins:
                protein = (
                    feature.qualifiers.get("translation", [""])[0].strip().rstrip("*")
                )
                if protein:
                    proteins[gene_id] = protein
        if len(proteins) == len(wanted):
            break

    kept = [gene_id for gene_id in wanted if gene_id in proteins]
    if not kept:
        print("no gene from the parquet was found in the GenBank; check both paths")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    computed = embed_sequences([proteins[gene_id] for gene_id in kept], device)

    diffs = []
    for row, gene_id in enumerate(kept):
        reference = frozen[gene_id]
        delta = float(np.abs(computed[row] - reference).max())
        diffs.append(delta)
        print(f"  {gene_id} max|d|={delta:.6g}")

    values = np.array(diffs)
    print(
        f"summary: genes={len(values)} max|d|={values.max():.6g} mean|d|={values.mean():.6g}"
    )
    print("reference from training: max|d| about 4e-3")
    usable = values.max() <= TOLERANCE
    print("VERDICT:", "usable" if usable else "not usable")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
