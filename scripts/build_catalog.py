"""Build gene catalog and index files from gLM2 converted data.

Parses all BGC and GTDB gLM2 strings, builds per-gene local context windows
(≤4096 tokens), and writes Parquet files for downstream use.

Output
------
data/derived/catalog/
  gene_catalog.parquet    gene_id → context_glm2_str, target_aa, strand,
                                   token_start, token_end
  bgc_index.parquet       bgc_id, source_entity, class_vector, gene_ids,
                          source_gene_ids, gene_count
  gtdb_index.parquet      genome_id, source_entity, gene_ids, source_gene_ids,
                          gene_count
  negative_pool.parquet   gene_id
  Source gene IDs retain source-local CDS order and ordinal gaps.
  Current BGC source files are not assumed to contain full contigs.

Usage
-----
    uv run python scripts/build_catalog.py
    uv run python scripts/build_catalog.py --exclude data/gtdb/antismash_bgc_genes.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = 4096

CLASSES = [
    "Polyketide", "NRP", "RiPP",
    "Saccharide", "Terpene", "Alkaloid", "Other",
]

BGC_DIR = Path("data/derived/glm2_mibig")
GTDB_DIR = Path("data/derived/glm2_gtdb")
JSON_DIR = Path("data/raw/mibig_json")
out_dir = Path("data/derived/catalog")

GTDB_BATCH = 50


# ---------------------------------------------------------------------------
# gLM2 string parsing
# ---------------------------------------------------------------------------

def parse_elements(glm2_str: str) -> list[tuple[str, str, str]]:
    """Parse a gLM2 string into ``(tag, seq, kind)`` elements.

    tag  = ``<+>`` or ``<->``
    seq  = AA sequence (uppercase) or IGS DNA (lowercase)
    kind = ``"CDS"`` or ``"IGS"``
    """
    elements: list[tuple[str, str, str]] = []
    i = 0
    n = len(glm2_str)

    while i < n:
        lt = glm2_str.find("<", i)
        if lt == -1:
            break
        tag = glm2_str[lt:lt + 3]
        i = lt + 3
        next_lt = glm2_str.find("<", i)
        seq = glm2_str[i:next_lt] if next_lt != -1 else glm2_str[i:]
        kind = "CDS" if seq and seq[0].isupper() else "IGS"
        elements.append((tag, seq, kind))
        i = next_lt if next_lt != -1 else n

    return elements


def build_context_window(
    elements: list[tuple[str, str, str]],
    cds_idx: int,
    max_tokens: int = CONTEXT_WINDOW,
) -> tuple[str, int, str]:
    """Build a local-context gLM2 string around a target CDS.

    Returns
    -------
    context_str : str
        gLM2-formatted window string.
    token_start : int
        Token offset of the target CDS within *context_str*.
        gLM2 tokenizer maps every AA/DNA character 1:1, and each
        ``<+>``/``<->`` tag to a single token.
    strand : str
        ``"+"`` or ``"-"``
    """
    target_tag, target_seq, _ = elements[cds_idx]
    window: list[int] = [cds_idx]
    tokens = 1 + len(target_seq)

    left = cds_idx - 1
    right = cds_idx + 1
    n = len(elements)

    while tokens < max_tokens and (left >= 0 or right < n):
        added = False
        if left >= 0:
            cost = 1 + len(elements[left][1])
            if tokens + cost <= max_tokens:
                window.insert(0, left)
                tokens += cost
                left -= 1
                added = True
        if right < n:
            cost = 1 + len(elements[right][1])
            if tokens + cost <= max_tokens:
                window.append(right)
                tokens += cost
                right += 1
                added = True
        if not added:
            break

    # Reconstruct context string
    context_str = "".join(elements[i][0] + elements[i][1] for i in window)

    # Compute token offset of the target CDS
    token_start = 0
    for i in window:
        if i == cds_idx:
            token_start += 1  # skip the target's own tag (<+> or <->)
            break
        token_start += 1 + len(elements[i][1])

    strand = "+" if target_tag == "<+>" else "-"

    return context_str, token_start, strand


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def process_source_files(
    txt_files: list[Path],
    *,
    source: str,
    exclude: set[str] | None = None,
) -> tuple[dict[str, list[str]], list[dict]]:
    """Process gLM2 text files → (index, gene_rows).

    index
        entity_id → [gene_id, ...]
    gene_rows
        [ {gene_id, context_glm2_str, target_aa, strand, token_start, token_end}, ... ]

    If ``exclude`` is provided, gene_ids in that set are skipped.
    """
    index: dict[str, list[str]] = {}
    gene_rows: list[dict] = []

    for txt in txt_files:
        entity_id = txt.stem.removesuffix("_glm2")
        glm2_str = txt.read_text().strip()
        elements = parse_elements(glm2_str)

        cds_positions = [
            j for j, (_, _, kind) in enumerate(elements) if kind == "CDS"
        ]

        gene_ids: list[str] = []
        for k, cds_idx in enumerate(cds_positions):
            gid = f"{source}:{entity_id}:{k}"
            if exclude and gid in exclude:
                continue
            ctx_str, tok_start, strand = build_context_window(elements, cds_idx)
            target_aa = elements[cds_idx][1]
            gene_rows.append({
                "gene_id":          gid,
                "context_glm2_str": ctx_str,
                "target_aa":        target_aa,
                "strand":           strand,
                "token_start":      tok_start,
                "token_end":        tok_start + len(target_aa),
            })
            gene_ids.append(gid)

        index[entity_id] = gene_ids

    return index, gene_rows


def write_gene_catalog(path: Path, batches: list[list[dict]]) -> None:
    """Write gene_catalog.parquet incrementally."""
    schema = pa.schema([
        ("gene_id",          pa.string()),
        ("context_glm2_str", pa.string()),
        ("target_aa",        pa.string()),
        ("strand",           pa.string()),
        ("token_start",      pa.int32()),
        ("token_end",        pa.int32()),
    ])
    writer = pq.ParquetWriter(path, schema)
    for batch in batches:
        table = pa.table({
            "gene_id":          pa.array([r["gene_id"]          for r in batch], pa.string()),
            "context_glm2_str": pa.array([r["context_glm2_str"] for r in batch], pa.string()),
            "target_aa":        pa.array([r["target_aa"]        for r in batch], pa.string()),
            "strand":           pa.array([r["strand"]           for r in batch], pa.string()),
            "token_start":      pa.array([r["token_start"]      for r in batch], pa.int32()),
            "token_end":        pa.array([r["token_end"]        for r in batch], pa.int32()),
        })
        writer.write_table(table)
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", type=str, default=None,
                        help="Path to antiSMASH exclusion list (one gene_id per line)")
    parser.add_argument("--antismash-dir", type=str, default=None,
                        help="antiSMASH results dir (auto-builds exclusion set from GBK)")
    parser.add_argument("--gtdb-dir", type=str, default="data/derived/glm2_gtdb",
                        help="Directory with GTDB gLM2 files")
    parser.add_argument("--catalog-dir", type=str, default="data/derived/catalog",
                        help="Output catalog directory")
    args = parser.parse_args()

    out_dir = Path(args.catalog_dir)
    gtdb_dir = Path(args.gtdb_dir)

    # Load exclusion set
    exclude_set: set[str] | None = None
    if args.exclude:
        exclude_path = Path(args.exclude)
        if exclude_path.exists():
            exclude_set = set(exclude_path.read_text().strip().splitlines())
            print(f"Loaded {len(exclude_set):,} excluded gene IDs from {args.exclude}")
        else:
            print(f"WARNING: exclusion list not found: {args.exclude}")

    if args.antismash_dir:
        from Bio import SeqIO
        as_dir = Path(args.antismash_dir)
        exclude_set = set()
        for gdir in sorted(as_dir.glob("GC*")) + sorted(as_dir.glob("GB_*")):
            genome_id = gdir.name
            gbks = sorted(gdir.glob("*genomic.gbk"))
            if not gbks:
                continue
            # Collect per-contig CDS + regions
            cds_per: list[list[tuple[int, int]]] = []
            reg_per: list[list[tuple[int, int]]] = []
            for record in SeqIO.parse(str(gbks[0]), "genbank"):
                cds_per.append(sorted(
                    [(int(f.location.start), int(f.location.end))
                     for f in record.features if f.type == "CDS"]))
                reg_per.append([
                    (int(f.location.start), int(f.location.end))
                    for f in record.features if f.type == "region"])
            k = 0
            for ci in range(len(cds_per)):
                for gs, ge in cds_per[ci]:
                    for rs, re in reg_per[ci]:
                        ov = max(0, min(ge, re) - max(gs, rs))
                        if ov / max(1, ge - gs) >= 0.5:
                            exclude_set.add(f"gtdb:{genome_id}:{k}")
                            break
                    k += 1
        print(f"Built exclusion set: {len(exclude_set):,} BGC genes from antiSMASH")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load BGC class labels ----
    print("Loading class labels from MIBiG JSON ...")
    label_map: dict[str, list[str]] = {}
    for jf in sorted(JSON_DIR.glob("BGC*.json")):
        bgc_id = jf.stem
        data = json.loads(jf.read_text())
        biosyn = data.get("cluster", {}).get("biosyn_class") or []
        label_map[bgc_id] = biosyn
    print(f"  {len(label_map)} BGCs with labels")

    # ---- 2. Process BGC files ----
    print("\nProcessing BGC gLM2 files ...")
    bgc_files = sorted(BGC_DIR.glob("BGC*_glm2.txt"))
    bgc_index, bgc_rows = process_source_files(bgc_files, source="bgc")
    print(f"  {len(bgc_index)} BGCs, {len(bgc_rows):,} genes")

    # ---- 3. Process GTDB files ----
    print("\nProcessing GTDB gLM2 files ...")
    gtdb_files = sorted(gtdb_dir.glob("*_glm2.txt"))
    gtdb_index: dict[str, list[str]] = {}
    gtdb_batches: list[list[dict]] = []
    negative_pool: list[str] = []

    for start in range(0, len(gtdb_files), GTDB_BATCH):
        batch_files = gtdb_files[start:start + GTDB_BATCH]
        idx, rows = process_source_files(batch_files, source="gtdb", exclude=exclude_set)
        gtdb_index.update(idx)
        gtdb_batches.append(rows)
        negative_pool.extend(r["gene_id"] for r in rows)
        print(f"  batch {start // GTDB_BATCH + 1}: "
              f"{len(batch_files)} files → {len(rows):,} genes")

    total_gtdb = sum(len(r) for r in gtdb_batches)
    print(f"  Total: {len(gtdb_index)} genomes, {total_gtdb:,} genes")

    # ---- 4. Write gene_catalog.parquet ----
    print("\nWriting gene_catalog.parquet ...")
    all_batches = [bgc_rows] + gtdb_batches
    write_gene_catalog(out_dir / "gene_catalog.parquet", all_batches)
    total_genes = len(bgc_rows) + total_gtdb
    print(f"  {total_genes:,} rows written")

    # ---- 5. Write bgc_index.parquet ----
    print("\nWriting bgc_index.parquet ...")
    bgc_ids: list[str] = []
    class_vecs: list[list[float]] = []
    gene_id_lists: list[list[str]] = []
    gene_counts: list[int] = []

    for bgc_id in sorted(bgc_index):
        labels = label_map.get(bgc_id, [])
        vec = [1.0 if c in labels else 0.0 for c in CLASSES]
        gids = bgc_index[bgc_id]
        bgc_ids.append(bgc_id)
        class_vecs.append(vec)
        gene_id_lists.append(gids)
        gene_counts.append(len(gids))

    pq.write_table(
        pa.table({
            "bgc_id":       pa.array(bgc_ids,      pa.string()),
            "source_entity": pa.array(bgc_ids, pa.string()),
            "class_vector": pa.array(class_vecs,    pa.list_(pa.float32())),
            "gene_ids":     pa.array(gene_id_lists, pa.list_(pa.string())),
            "source_gene_ids": pa.array(gene_id_lists, pa.list_(pa.string())),
            "gene_count":   pa.array(gene_counts,   pa.int32()),
        }),
        out_dir / "bgc_index.parquet",
    )
    print(f"  {len(bgc_ids)} rows written")

    # ---- 6. Write gtdb_index.parquet ----
    print("\nWriting gtdb_index.parquet ...")
    genome_ids: list[str] = []
    gid_lists: list[list[str]] = []
    g_counts: list[int] = []

    for genome_id in sorted(gtdb_index):
        gids = gtdb_index[genome_id]
        genome_ids.append(genome_id)
        gid_lists.append(gids)
        g_counts.append(len(gids))

    pq.write_table(
        pa.table({
            "genome_id":  pa.array(genome_ids,  pa.string()),
            "source_entity": pa.array(genome_ids, pa.string()),
            "gene_ids":   pa.array(gid_lists,   pa.list_(pa.string())),
            "source_gene_ids": pa.array(gid_lists, pa.list_(pa.string())),
            "gene_count": pa.array(g_counts,    pa.int32()),
        }),
        out_dir / "gtdb_index.parquet",
    )
    print(f"  {len(genome_ids)} rows written")

    # ---- 7. Write negative_pool.parquet ----
    print("\nWriting negative_pool.parquet ...")
    pq.write_table(
        pa.table({"gene_id": pa.array(sorted(negative_pool), pa.string())}),
        out_dir / "negative_pool.parquet",
    )
    print(f"  {len(negative_pool):,} gene_ids")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("Done — data/derived/catalog/")
    print(f"  gene_catalog.parquet   {total_genes:>10,} genes")
    print(f"  bgc_index.parquet      {len(bgc_ids):>10,} BGCs")
    print(f"  gtdb_index.parquet     {len(genome_ids):>10,} genomes")
    print(f"  negative_pool.parquet  {len(negative_pool):>10,} gene IDs")


if __name__ == "__main__":
    main()
