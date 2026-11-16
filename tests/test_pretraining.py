import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from bgc_gnn.checkpoint import file_fingerprint, load_checkpoint
from bgc_gnn.pretrain import PretrainingDataset, pretraining_loss
from scripts.build_pretraining_data import (
    build,
    contiguous_windows,
    protein_sequence,
    record_genes,
    sequence_id,
)
from scripts.encode_pretraining import main as encode_main
from scripts.encode_pretraining import reuse_cached_embeddings, reuse_embeddings


class PretrainingTest(unittest.TestCase):
    def record(self, identifier="contig1"):
        record = SeqRecord(
            Seq("ACGT" * 200),
            id=identifier,
            description="test",
            annotations={"molecule_type": "DNA"},
        )
        record.features = [
            SeqFeature(
                FeatureLocation(i * 60, i * 60 + 54, strand=1),
                type="CDS",
                qualifiers={"translation": ["M" + ("ACDEFGHIK"[i] * 17)]},
            )
            for i in range(8)
        ]
        record.features.append(
            SeqFeature(
                FeatureLocation(120, 300),
                type="region",
                qualifiers={"contig_edge": ["False"], "product": ["terpene"]},
            )
        )
        return record

    def test_multipart_cds_is_a_break_not_a_bounding_span_label(self):
        record = self.record()
        record.features[2].location = CompoundLocation(
            [FeatureLocation(700, 750), FeatureLocation(0, 50)]
        )
        rows = record_genes(record, set(), boundary_margin=0)
        self.assertIsNone(rows[1])
        self.assertEqual(sum(r is None for r in rows), 1)
        windows = list(contiguous_windows(rows, width=128, stride=64))
        self.assertEqual([sum(w["valid_mask"]) for w in windows], [1, 6])

    def test_continuity_exclusions_boundary_and_ng(self):
        record = self.record()
        record.features[3].qualifiers = {}
        excluded = {
            sequence_id(
                protein_sequence(record.features[5].qualifiers["translation"][0])
            )
        }
        rows = record_genes(record, excluded, boundary_margin=1)
        self.assertIsNone(rows[3])
        self.assertIsNone(rows[5])
        self.assertTrue(rows[2]["positive"])
        self.assertFalse(rows[1]["supervised"])
        self.assertFalse(rows[2]["supervised"])
        windows = list(contiguous_windows(rows, width=4, stride=2))
        self.assertEqual([w["start_ordinal"] for w in windows], [0, 4, 6])
        self.assertEqual([sum(w["valid_mask"]) for w in windows], [3, 1, 2])
        for w in windows:
            self.assertFalse(
                any(m and not v for m, v in zip(w["loss_mask"], w["valid_mask"]))
            )
        record.features[-1].qualifiers["contig_edge"] = ["True"]
        rows = record_genes(record, set(), boundary_margin=0)
        self.assertFalse(rows[2]["supervised"])
        self.assertTrue(rows[0]["supervised"])
        self.assertEqual(protein_sequence(" MA C* "), "MAC")
        self.assertIsNone(protein_sequence("MA*C"))
        with self.assertRaisesRegex(ValueError, "stride"):
            list(contiguous_windows(rows, width=4, stride=5))

    def test_data_build_fingerprints_splits_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for n in range(4):
                folder = source / f"genome{n}"
                folder.mkdir()
                SeqIO.write(self.record(f"ctg{n}"), folder / "whole.gbk", "genbank")
            ng = root / "ng.gbk"
            SeqIO.write(
                SeqRecord(
                    Seq("TTTT"),
                    id="NG",
                    description="",
                    annotations={"molecule_type": "DNA"},
                ),
                ng,
                "genbank",
            )
            output = root / "prepared"
            with contextlib.redirect_stdout(io.StringIO()):
                audit = build(source, ng, output, val_fraction=0.25, boundary_margin=0)
            self.assertEqual(audit["status"], "completed")
            self.assertEqual(audit["counts"]["assemblies"], 4)
            self.assertEqual(audit["counts"]["train_windows"], 3)
            self.assertEqual(audit["counts"]["val_windows"], 1)
            self.assertEqual(audit["unique_proteins"], 8)
            train = pq.read_table(output / "windows_train.parquet").to_pylist()
            val = pq.read_table(output / "windows_val.parquet").to_pylist()
            self.assertFalse(
                {r["assembly"] for r in train} & {r["assembly"] for r in val}
            )
            self.assertEqual(audit["exact_protein_overlap_train_val"], 8)
            self.assertEqual(
                file_fingerprint(output / "proteins.parquet")["sha256"],
                audit["proteins.parquet"]["sha256"],
            )
            with self.assertRaisesRegex(ValueError, "exists"):
                build(source, ng, output)
            # Exact NG protein matches create gaps, not artificial adjacent neighbors.
            ng_record = self.record("NG")
            ng_record.seq = Seq("TTTT")
            ng_record.features = ng_record.features[:1]
            SeqIO.write(ng_record, ng, "genbank")
            with contextlib.redirect_stdout(io.StringIO()):
                audit2 = build(
                    source, ng, root / "excluded", val_fraction=0.25, boundary_margin=0
                )
            self.assertEqual(audit2["counts"]["missing_or_ng_protein_breaks"], 4)
            (source / "genome0" / "extra.gbk").write_text("bad")
            with self.assertRaisesRegex(ValueError, "one complete"):
                build(source, ng, root / "bad")
            self.assertEqual(
                json.loads((root / "bad/audit.json").read_text())["status"],
                "incomplete",
            )

    def test_audit_selection_excludes_failed_assemblies_and_rejects_changed_gbk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            integrity = root / "integrity"
            integrity.mkdir()
            rows = []
            for n in range(3):
                folder = source / f"genome{n}"
                folder.mkdir()
                path = folder / "whole.gbk"
                SeqIO.write(self.record(f"ctg{n}"), path, "genbank")
                rows.append(
                    {
                        "assembly": folder.name,
                        "errors": [],
                        "gbk": file_fingerprint(path),
                    }
                )
            (source / "failed").mkdir()
            (integrity / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "source": str(source),
                        "usable_assemblies": [r["assembly"] for r in rows],
                        "failed_assemblies": ["failed"],
                    }
                )
            )
            (integrity / "assemblies.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            ng = root / "ng.gbk"
            SeqIO.write(
                SeqRecord(Seq("TTTT"), id="NG", annotations={"molecule_type": "DNA"}),
                ng,
                "genbank",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                audit = build(source, ng, root / "data", audit_dir=integrity)
            self.assertEqual(audit["counts"]["assemblies"], 3)
            self.assertEqual(audit["unique_proteins"], 8)
            self.assertEqual(
                audit["source_audit"]["sha256"],
                file_fingerprint(integrity / "summary.json")["sha256"],
            )
            path = source / "genome0/whole.gbk"
            path.write_text(path.read_text() + "\n")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "changed after integrity audit"),
            ):
                build(source, ng, root / "changed", audit_dir=integrity)
            self.assertEqual(
                json.loads((root / "changed/audit.json").read_text())["status"],
                "incomplete",
            )

    def test_embedding_inventory_counts_reuse_and_missing_without_encoding(self):
        from scripts.plan_pretraining_embeddings import plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            cache = root / "cache"
            cache.mkdir()
            sequences = ["MAC", "MDEFG", "M" + "A" * 5000]
            ids = [sequence_id(s) for s in sequences]
            pq.write_table(
                pa.table({"gene_id": ids, "target_aa": sequences}),
                data / "proteins.parquet",
            )
            (data / "audit.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "proteins.parquet": file_fingerprint(data / "proteins.parquet"),
                    }
                )
            )
            np.save(cache / "embeddings.npy", np.ones((1, 960), dtype=np.float32))
            (cache / "sequence_ids.json").write_text(json.dumps(ids[:1]))
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "model": "biohub/ESMC-300M",
                        "dim": 960,
                        "embeddings": file_fingerprint(cache / "embeddings.npy"),
                        "sequence_ids": file_fingerprint(cache / "sequence_ids.json"),
                    }
                )
            )
            output = np.zeros((3, 960), dtype=np.float32)
            found = reuse_cached_embeddings(ids, cache, output)
            self.assertEqual(found, {0})
            np.testing.assert_equal(output[0], np.ones(960, dtype=np.float32))
            np.testing.assert_equal(output[1:], np.zeros((2, 960), dtype=np.float32))
            result = plan(data, cache, root / "inventory")
            self.assertEqual(result["unique_proteins"], 3)
            self.assertEqual(result["reusable_exact_sequence_embeddings"], 1)
            self.assertEqual(result["missing_to_encode"], 2)
            self.assertEqual(result["missing_over_4096_AA"], 1)
            self.assertEqual(result["cache_bytes_float32"], 3 * 960 * 4)
            (cache / "sequence_ids.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                plan(data, cache, root / "bad_inventory")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                reuse_cached_embeddings(ids, cache, output)

    def test_pretraining_loss_ignores_unsupervised_positions(self):
        import torch

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.probabilities = torch.nn.Parameter(
                    torch.tensor([[0.2, 0.4, 0.7, 0.8]])
                )

            def forward(self, x, key_padding_mask=None):
                return self.probabilities

        detector = Detector()
        valid = torch.tensor([[True, True, True, False]])
        targets = torch.tensor([[False, True, True, False]])
        supervised = torch.tensor([[True, False, True, False]])
        loss = pretraining_loss(
            detector, torch.zeros(1, 4, 8), valid, targets, supervised
        )
        expected = (
            -torch.log(torch.tensor(0.8)) - 2 * torch.log(torch.tensor(0.7))
        ) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(detector.probabilities.grad[0, 1], 0)
        self.assertEqual(detector.probabilities.grad[0, 3], 0)
        self.assertIsNone(
            pretraining_loss(
                detector, torch.zeros(1, 4, 8), valid, targets, torch.zeros_like(valid)
            )
        )

    def test_exact_sequence_reuse_from_catalog_and_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [sequence_id(s) for s in ("MAC", "MDE", "MFG")]
            pq.write_table(
                pa.table({"gene_id": ["old1", "old2"], "target_aa": ["MAC", "MDE"]}),
                root / "catalog.parquet",
            )
            pq.write_table(
                pa.table(
                    {"gene_id": ["old2", "old1"], "embedding": [[2.0] * 8, [1.0] * 8]}
                ),
                root / "emb.parquet",
            )
            array = np.zeros((3, 8), dtype=np.float32)
            found = reuse_embeddings(
                ids, root / "catalog.parquet", root / "emb.parquet", array
            )
            self.assertEqual(found, {0, 1})
            np.testing.assert_equal(array[0], [1.0] * 8)
            np.testing.assert_equal(array[1], [2.0] * 8)

    def test_encoding_checks_reused_vectors_before_encoding_missing(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                import torch

                root = Path(tmp)
                data = root / "data"
                data.mkdir()
                sequences = ["M" + "A" * 60, "M" + "C" * 60]
                pq.write_table(
                    pa.table(
                        {
                            "gene_id": [sequence_id(s) for s in sequences],
                            "target_aa": sequences,
                        }
                    ),
                    data / "proteins.parquet",
                )
                (data / "audit.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "proteins.parquet": file_fingerprint(
                                data / "proteins.parquet"
                            ),
                        }
                    )
                )
                pq.write_table(
                    pa.table({"gene_id": ["old"], "target_aa": sequences[:1]}),
                    root / "catalog.parquet",
                )
                pq.write_table(
                    pa.table({"gene_id": ["old"], "embedding": [[1.0] * 960]}),
                    root / "emb.parquet",
                )

                def tokenizer(seqs, **kwargs):
                    return {
                        "input_ids": torch.ones(len(seqs), 63, dtype=torch.long),
                        "attention_mask": torch.ones(len(seqs), 63, dtype=torch.long),
                    }

                def model(**kwargs):
                    return SimpleNamespace(
                        last_hidden_state=torch.full(
                            (*kwargs["input_ids"].shape, 960), 2.0 if mismatch else 1.0
                        )
                    )

                argv = [
                    "encode",
                    "--data-dir",
                    str(data),
                    "--output-dir",
                    str(root / "cache"),
                    "--device",
                    "cpu",
                    "--reuse-catalog",
                    str(root / "catalog.parquet"),
                    "--reuse-embeddings",
                    str(root / "emb.parquet"),
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch("scripts.encode.load_model", return_value=(model, tokenizer)),
                    patch(
                        "importlib.metadata.distribution",
                        return_value=SimpleNamespace(
                            version="test", read_text=lambda _: "{}"
                        ),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if mismatch:
                        with self.assertRaisesRegex(
                            ValueError, "consistency check failed"
                        ):
                            encode_main()
                    else:
                        encode_main()
                metadata = json.loads((root / "cache/metadata.json").read_text())
                self.assertEqual(
                    metadata["status"], "incomplete" if mismatch else "completed"
                )
                if not mismatch:
                    self.assertEqual(metadata["newly_encoded"], 1)
                    self.assertEqual(len(metadata["reuse_consistency_checks"]), 1)

    def test_encoding_reuses_complete_cache_without_loading_esmc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            sid = sequence_id("MAC")
            pq.write_table(
                pa.table({"gene_id": [sid], "target_aa": ["MAC"]}),
                data / "proteins.parquet",
            )
            (data / "audit.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "proteins.parquet": file_fingerprint(data / "proteins.parquet"),
                    }
                )
            )
            pq.write_table(
                pa.table({"gene_id": ["old"], "target_aa": ["MAC"]}),
                root / "old_catalog.parquet",
            )
            pq.write_table(
                pa.table({"gene_id": ["old"], "embedding": [[0.25] * 960]}),
                root / "old_embeddings.parquet",
            )
            out = root / "cache"
            argv = [
                "encode",
                "--data-dir",
                str(data),
                "--output-dir",
                str(out),
                "--reuse-catalog",
                str(root / "old_catalog.parquet"),
                "--reuse-embeddings",
                str(root / "old_embeddings.parquet"),
            ]
            with (
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                encode_main()
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["reused"], 1)
            self.assertEqual(metadata["newly_encoded"], 0)
            np.testing.assert_equal(
                np.load(out / "embeddings.npy"),
                np.full((1, 960), 0.25, dtype=np.float32),
            )
            cached_argv = [
                "encode",
                "--data-dir",
                str(data),
                "--output-dir",
                str(root / "cache2"),
                "--reuse-cache",
                str(out),
            ]
            with (
                patch.object(sys, "argv", cached_argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                encode_main()
            cached = json.loads((root / "cache2/metadata.json").read_text())
            self.assertEqual(cached["status"], "completed")
            self.assertEqual(cached["reused"], 1)
            self.assertEqual(cached["newly_encoded"], 0)
            np.testing.assert_equal(
                np.load(root / "cache2/embeddings.npy"), np.load(out / "embeddings.npy")
            )
            with (
                patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                encode_main()

    def test_pretraining_cli_completes_and_transfers(self):
        from bgc_gnn import pretrain
        from bgc_gnn.checkpoint import initialize_from_pretraining
        from bgc_gnn.model import BGCDetector

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            ids = [sequence_id("M" + "A" * i) for i in range(1, 9)]
            values = np.random.default_rng(42).normal(size=(8, 960)).astype(np.float32)
            np.save(cache / "embeddings.npy", values)
            (cache / "sequence_ids.json").write_text(json.dumps(ids))
            pq.write_table(pa.table({"gene_id": ids}), root / "proteins.parquet")
            row = {
                "sequence_ids": ids + [""] * 120,
                "valid_mask": [True] * 8 + [False] * 120,
                "detect_mask": [False] * 4 + [True] * 4 + [False] * 120,
                "loss_mask": [True] * 8 + [False] * 120,
            }
            audit = {"status": "completed", "limits": ["fixture"]}
            for split in ("train", "val"):
                path = root / f"windows_{split}.parquet"
                pq.write_table(pa.Table.from_pylist([row]), path)
                audit[path.name] = file_fingerprint(path)
            (root / "audit.json").write_text(json.dumps(audit))
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "model": "biohub/ESMC-300M",
                        "dim": 960,
                        "proteins": file_fingerprint(root / "proteins.parquet"),
                        "embeddings": file_fingerprint(cache / "embeddings.npy"),
                        "sequence_ids": file_fingerprint(cache / "sequence_ids.json"),
                    }
                )
            )
            dataset = PretrainingDataset(root / "windows_train.parquet", cache)
            self.assertIsInstance(dataset.values, np.memmap)
            self.assertFalse(dataset.values.flags.writeable)
            validation = PretrainingDataset(
                root / "windows_val.parquet", cache, dataset.values
            )
            self.assertIs(validation.values, dataset.values)
            np.testing.assert_array_equal(validation.values, values)
            del validation
            self.assertEqual(dataset[0]["embeddings"].shape, (128, 960))
            import torch

            torch.testing.assert_close(
                dataset[0]["embeddings"][:8], torch.from_numpy(values)
            )
            self.assertEqual(dataset.positive, 4)
            self.assertEqual(dataset.negative, 4)
            del dataset

            dest = root / "pretraining"
            argv = [
                "pretrain",
                "--data-dir",
                str(root),
                "--cache-dir",
                str(cache),
                "--output-dir",
                str(dest),
                "--device",
                "cpu",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(pretrain, "EPOCHS", 1),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                pretrain.main()
            self.assertEqual(
                json.loads((dest / "status.json").read_text())["status"], "completed"
            )
            state, saved = load_checkpoint(dest / "pretrained_best.pt")
            self.assertEqual(saved["stage"], "pretraining")
            self.assertEqual(saved["selection_metric"], "pretraining_val_loss")
            self.assertTrue(saved["architecture"]["semantic_skip"])
            self.assertFalse(saved["architecture"]["context_readout"])
            self.assertEqual(saved["settings"]["batch_size"], 32)
            self.assertEqual(saved["runtime"]["cpu_compute_threads"], 2)
            self.assertGreater(float(state["semantic_skip.weight"].abs().sum()), 0)
            target = BGCDetector()
            provenance = initialize_from_pretraining(
                target, dest / "pretrained_best.pt"
            )
            self.assertEqual(provenance["stage"], "pretraining")
            torch.testing.assert_close(
                target.semantic_skip.weight,
                state["semantic_skip.weight"],
                rtol=0,
                atol=0,
            )
            with (
                patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                pretrain.main()


if __name__ == "__main__":
    unittest.main()
