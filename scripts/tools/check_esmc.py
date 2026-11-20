"""Check that the repo's ESMC path reproduces the frozen training embeddings.

Runs `bgc_detector.esmc.embed_sequences` -- the same function `bgc-detector predict`
uses -- on a few NG genes and compares against the frozen parquet. Any change to
the loader must keep max |delta| near 4e-3.

    uv run python -m scripts.tools.check_esmc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from Bio import SeqIO

from bgc_detector.esmc import embed_sequences

DATA_ROOT = Path("/home/gonghang/codes/my-bgc-v2/data")
EMB = DATA_ROOT / "derived/ng_embeddings/ng_gene_embeddings_960d.parquet"
GBK = DATA_ROOT / "raw/ng/ClusterFinder_Annotated_Contigs.full.gbk"
N_GENES = 8
TOLERANCE = 0.02


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare freshly computed ESMC embeddings against the frozen NG "
            "embeddings used for training"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--emb",
        type=Path,
        default=EMB,
        metavar="PARQUET",
        help="frozen gene embedding parquet (gene_id + 960d embedding)",
    )
    parser.add_argument(
        "--gbk",
        type=Path,
        default=GBK,
        metavar="GBK",
        help="GenBank the parquet was built from, for the protein sequences",
    )
    parser.add_argument(
        "--genes",
        type=int,
        default=N_GENES,
        metavar="N",
        help="how many genes to compare",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="cuda:0 or cpu (default: cuda when available)",
    )
    args = parser.parse_args()
    for path in (args.emb, args.gbk):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    table = pq.ParquetFile(args.emb).read_row_group(0)
    frozen = {
        gene_id: np.array(vec, dtype=np.float32)
        for gene_id, vec in zip(
            table.column("gene_id").to_pylist(),
            table.column("embedding").to_pylist(),
            strict=True,
        )
    }

    wanted = sorted(frozen)[: args.genes]
    proteins: dict[str, str] = {}
    for record in SeqIO.parse(args.gbk.open(encoding="utf-8"), "genbank"):
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

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
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
    print("expected: exact match on the machine that produced the parquet; "
          "other GPUs may differ at bf16 scale (about 4e-3)")
    usable = values.max() <= TOLERANCE
    print("VERDICT:", "usable" if usable else "not usable")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
