import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from bgc_detector.checkpoint import save_checkpoint
from bgc_detector.cli import main
from bgc_detector.model import ARCHITECTURE, BGCDetector
from bgc_detector.predict import load_detector


def write_genome(path: Path, contigs: list[tuple[str, int]]) -> None:
    records = []
    for accession, n_genes in contigs:
        features = [
            SeqFeature(
                FeatureLocation(i * 9, i * 9 + 9, strand=1),
                type="CDS",
                qualifiers={
                    "locus_tag": [f"g{i}"],
                    "translation": ["MKV"],
                },
            )
            for i in range(n_genes)
        ]
        record = SeqRecord(
            Seq("ATG" * (n_genes * 3)),
            id=accession,
            name=accession,
            annotations={
                "accessions": [accession],
                "organism": "test",
                "molecule_type": "DNA",
            },
        )
        record.features = features
        records.append(record)
    SeqIO.write(records, str(path), "genbank")


def write_checkpoint(path: Path, threshold: float | None = 0.7) -> None:
    metadata = {
        "checkpoint_kind": "best",
        "weight_source": "ema",
        "epoch": 7,
        "selection_metric": "val_loss",
        "selection_value": 0.01,
        "architecture": dict(ARCHITECTURE),
        "runtime": {"torch": "test", "git_commit": "test"},
    }
    if threshold is not None:
        metadata |= {
            "validation_threshold": threshold,
            "validation_threshold_source": "validation",
            "validation_threshold_metric": "mcc",
        }
    save_checkpoint(path, BGCDetector(), metadata)


def write_embeddings(path: Path, gene_ids: list[str], dim: int = 960) -> None:
    rng = np.random.default_rng(0)
    matrix = rng.random((len(gene_ids), dim)).astype(np.float32)
    pq.write_table(
        pa.table(
            {
                "gene_id": pa.array(gene_ids, pa.string()),
                "embedding": pa.array(matrix.tolist(), pa.list_(pa.float32(), dim)),
            }
        ),
        path,
    )


class CommandLineTest(unittest.TestCase):
    def test_info_reports_the_stored_validation_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "detector.pt"
            write_checkpoint(checkpoint, threshold=0.6901)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                main(["info", "--checkpoint", str(checkpoint)])
            out = buffer.getvalue()
            self.assertIn("threshold: 0.690100", out)
            self.assertIn("metric=mcc", out)

    def test_predict_writes_genes_tsv_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome = root / "genome.gbk"
            write_genome(genome, [("contigA", 3), ("contigB", 130)])
            genes = [f"contigA:g{i}" for i in range(3)]
            genes += [f"contigB:g{i}" for i in range(130)]
            embeddings = root / "emb.parquet"
            write_embeddings(embeddings, genes)
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint, threshold=0.7)
            outdir = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(outdir),
                        "--emb-cache",
                        str(embeddings),
                        "--device",
                        "cpu",
                        "--name",
                        "run1",
                    ]
                )

            rows = (outdir / "run1.genes.tsv").read_text().splitlines()
            self.assertEqual(len(rows), 1 + 133)
            self.assertEqual(
                rows[0].split("\t"),
                [
                    "contig",
                    "gene_id",
                    "locus_tag",
                    "start",
                    "end",
                    "strand",
                    "score",
                    "predicted",
                ],
            )
            self.assertEqual([r.split("\t")[0] for r in rows[1:4]], ["contigA"] * 3)
            for row in rows[1:]:
                score, predicted = row.split("\t")[6:8]
                self.assertEqual(int(predicted), int(float(score) >= 0.5))

            provenance = json.loads((outdir / "run1.json").read_text())
            self.assertEqual(provenance["threshold"], 0.5)
            self.assertEqual(provenance["threshold_source"], "fixed")
            self.assertIsNone(provenance["threshold_metric"])
            self.assertEqual(provenance["stride"], 64)
            self.assertEqual(provenance["aggregate"], "mean")
            self.assertEqual(provenance["n_genes"], 133)
            self.assertEqual(provenance["n_excluded"], 0)
            self.assertEqual(provenance["n_scored"], 133)
            self.assertEqual(provenance["embeddings_from"], "cache")
            self.assertEqual(
                provenance["n_windows"], 1 + 3
            )  # ceil(3/64) + ceil(130/64)

    def test_predict_computes_embeddings_and_caches_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome = root / "genome.gbk"
            write_genome(genome, [("ctg", 4)])
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint)
            cache = root / "cache.parquet"

            def fake_embed(sequences, device, max_toks):
                self.assertEqual(len(sequences), 4)
                return np.full((len(sequences), 960), 0.9, dtype=np.float32)

            with (
                patch("bgc_detector.cli.embed_sequences", fake_embed),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out"),
                        "--emb-cache",
                        str(cache),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertTrue(cache.is_file())
            provenance = json.loads((root / "out/genome.json").read_text())
            self.assertEqual(provenance["embeddings_from"], "computed")
            self.assertEqual(provenance["n_scored"], 4)

            with (
                patch("bgc_detector.cli.embed_sequences") as embed,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out2"),
                        "--emb-cache",
                        str(cache),
                        "--device",
                        "cpu",
                    ]
                )
            embed.assert_not_called()

    def test_predict_can_use_the_checkpoint_validation_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome = root / "genome.gbk"
            write_genome(genome, [("ctg", 3)])
            embeddings = root / "emb.parquet"
            write_embeddings(embeddings, [f"ctg:g{i}" for i in range(3)])
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint, threshold=0.7)
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out"),
                        "--emb-cache",
                        str(embeddings),
                        "--device",
                        "cpu",
                        "--threshold",
                        "checkpoint",
                    ]
                )
            provenance = json.loads((root / "out/genome.json").read_text())
            self.assertEqual(provenance["threshold"], 0.7)
            self.assertEqual(provenance["threshold_source"], "validation")
            self.assertEqual(provenance["threshold_metric"], "mcc")

    def test_predict_rejects_a_threshold_outside_the_unit_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome = root / "genome.gbk"
            write_genome(genome, [("ctg", 2)])
            embeddings = root / "emb.parquet"
            write_embeddings(embeddings, ["ctg:g0", "ctg:g1"])
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint)
            with (
                contextlib.redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit),
            ):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out"),
                        "--emb-cache",
                        str(embeddings),
                        "--device",
                        "cpu",
                        "--threshold",
                        "1.5",
                    ]
                )
            self.assertIn("within [0, 1]", err.getvalue())

    def test_predict_without_esm_and_without_cache_reports_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome = root / "genome.gbk"
            write_genome(genome, [("ctg", 2)])
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint)
            with (
                patch.dict(sys.modules, {"esm": None, "esm.models.esmc": None}),
                contextlib.redirect_stderr(io.StringIO()) as err,
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out"),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertIn("needs the esm package", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_predict_refuses_checkpoint_threshold_when_it_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome = root / "genome.gbk"
            write_genome(genome, [("ctg", 2)])
            embeddings = root / "emb.parquet"
            write_embeddings(embeddings, ["ctg:g0", "ctg:g1"])
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint, threshold=None)
            with (
                contextlib.redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit),
            ):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out"),
                        "--emb-cache",
                        str(embeddings),
                        "--device",
                        "cpu",
                        "--threshold",
                        "checkpoint",
                    ]
                )
            self.assertIn("validation-MCC threshold", err.getvalue())

    def test_existing_cache_is_never_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome = root / "genome.gbk"
            write_genome(genome, [("ctg", 2)])
            cache = root / "cache.parquet"
            write_embeddings(cache, ["ctg:g0"])  # only one of the two genes
            before = cache.read_bytes()
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint)
            with (
                patch(
                    "bgc_detector.cli.embed_sequences",
                    lambda sequences, device, max_toks: np.full(
                        (len(sequences), 960), 0.5, dtype=np.float32
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(
                    [
                        "predict",
                        "--genome",
                        str(genome),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out"),
                        "--emb-cache",
                        str(cache),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(cache.read_bytes(), before)
            provenance = json.loads((root / "out/genome.json").read_text())
            self.assertEqual(provenance["n_scored"], 2)
            self.assertEqual(provenance["embeddings_from"], "cache+computed")
            self.assertEqual(provenance["n_embedded_reused"], 1)
            self.assertEqual(provenance["n_embedded_computed"], 1)

    def test_help_documents_both_commands(self):
        for argv in (["--help"], ["predict", "--help"], ["info", "--help"]):
            with (
                contextlib.redirect_stdout(io.StringIO()) as out,
                self.assertRaises(SystemExit) as exit_info,
            ):
                main(argv)
            self.assertEqual(exit_info.exception.code, 0)
            self.assertIn("--checkpoint", out.getvalue())
        with (
            contextlib.redirect_stdout(io.StringIO()) as out,
            self.assertRaises(SystemExit),
        ):
            main(["predict", "--help"])
        text = out.getvalue()
        self.assertIn("STEM.genes.tsv", text)
        self.assertIn("checkpoint", text)

    def test_a_second_genome_reuses_the_loaded_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "a.gbk", root / "b.gbk"
            write_genome(first, [("ctg", 2)])
            write_genome(second, [("ctg", 3)])
            checkpoint = root / "detector.pt"
            write_checkpoint(checkpoint)
            loader = Mock(wraps=load_detector)
            with (
                patch(
                    "bgc_detector.cli.embed_sequences",
                    lambda sequences, device, max_toks: np.full(
                        (len(sequences), 960), 0.5, dtype=np.float32
                    ),
                ),
                patch("bgc_detector.cli.load_detector", loader),
                patch(
                    "bgc_detector.cli.mean_scores",
                    Mock(
                        side_effect=lambda model, emb, order, device: (
                            {gid: 0.7 for gids in order.values() for gid in gids},
                            1,
                        )
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(
                    [
                        "predict",
                        "--genome",
                        str(first),
                        str(second),
                        "--checkpoint",
                        str(checkpoint),
                        "--outdir",
                        str(root / "out"),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(loader.call_count, 1)
            for stem in ("a", "b"):
                self.assertTrue((root / f"out/{stem}.genes.tsv").is_file())
                self.assertTrue((root / f"out/{stem}.json").is_file())

    def test_regions_merge_called_genes_and_apply_the_filters(self):
        tsv = (
            "\n".join(
                [
                    "contig\tgene_id\tlocus_tag\tstart\tend\tstrand\tscore\tpredicted",
                    "ctg\tctg:g0\tg0\t0\t100\t+\t0.9\t1",
                    "ctg\tctg:g1\tg1\t100\t200\t+\t0.8\t1",
                    "ctg\tctg:g2\tg2\t200\t300\t+\t0.1\t0",
                    "ctg\tctg:g3\tg3\t300\t400\t+\t0.95\t1",
                    "ctg\tctg:g4\tg4\t400\t500\t+\t0.1\t0",
                    "ctg\tctg:g5\tg5\t500\t600\t+\t0.1\t0",
                    "ctg\tctg:g6\tg6\t600\t700\t+\t0.1\t0",
                    "ctg\tctg:g7\tg7\t700\t800\t+\t0.1\t0",
                    "ctg\tctg:g8\tg8\t800\t900\t+\t0.97\t1",
                    "ctg\tctg:g9\tg9\t900\t1000\t+\t0.2\t0",
                ]
            )
            + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            genes = Path(tmp) / "run.genes.tsv"
            genes.write_text(tsv, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["regions", "--genes", str(genes)])
            out = (Path(tmp) / "run.regions.tsv").read_text().splitlines()
            self.assertEqual(
                out[0], "contig\tstart\tend\tn_genes\tmean_score\tmax_score"
            )
            # g0,g1 and g3 join across one uncalled gene; g8 is far away and alone
            self.assertEqual(len(out), 2)
            self.assertEqual(out[1].split("\t")[:4], ["ctg", "0", "400", "3"])
            self.assertEqual(out[1].split("\t")[5:], [repr(0.95)])

    def test_regions_respect_max_gap_and_min_count(self):
        tsv = (
            "\n".join(
                [
                    "contig\tgene_id\tlocus_tag\tstart\tend\tstrand\tscore\tpredicted",
                    "ctg\tctg:g0\tg0\t0\t100\t+\t0.9\t1",
                    "ctg\tctg:g1\tg1\t100\t200\t+\t0.1\t0",
                    "ctg\tctg:g2\tg2\t200\t300\t+\t0.1\t0",
                    "ctg\tctg:g3\tg3\t300\t400\t+\t0.8\t1",
                ]
            )
            + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            genes = Path(tmp) / "run.genes.tsv"
            genes.write_text(tsv, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["regions", "--genes", str(genes), "--max-gap", "1"])
            # gap of two uncalled genes exceeds --max-gap 1 -> no merged region,
            # and each single gene is below --min-count 2
            self.assertEqual(
                len((Path(tmp) / "run.regions.tsv").read_text().splitlines()), 1
            )


if __name__ == "__main__":
    unittest.main()
