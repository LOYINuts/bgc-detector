import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bgc_gnn.checkpoint import save_checkpoint
from bgc_gnn.model import ARCHITECTURE, BGCDetector
from scripts import eval_ng


class ScoreDetector(torch.nn.Module):
    def forward(self, x):
        return x[..., 0]


class NGEvaluationTest(unittest.TestCase):
    def test_cli_reads_external_data_root_and_defaults_output_to_checkpoint_directory(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "read only data"
            raw = data / "raw/ng"
            raw.mkdir(parents=True)
            gbk = raw / "ClusterFinder_Annotated_Contigs.full.gbk"
            csv = raw / "291BGC.csv"
            gbk.touch()
            csv.touch()
            emb = data / "derived/ng_embeddings/ng_gene_embeddings_960d.parquet"
            emb.parent.mkdir(parents=True)
            emb.touch()
            checkpoint = root / "outputs/selected.pt"
            checkpoint.parent.mkdir()
            genes = [
                {"gene_id": "g0", "accession": "c", "start": 0},
                {"gene_id": "g1", "accession": "c", "start": 1},
            ]
            metadata = {
                "validation_threshold": 0.5,
                "validation_threshold_source": "validation",
                "validation_threshold_metric": "mcc",
                "epoch": 1,
            }
            pq.write_table(
                pa.table(
                    {"gene_id": ["g0", "g1"], "embedding": [[0.9] * 960, [0.1] * 960]}
                ),
                emb,
            )
            before = {
                p.relative_to(data): p.read_bytes()
                for p in data.rglob("*")
                if p.is_file()
            }
            with (
                patch(
                    "sys.argv",
                    [
                        "eval",
                        "--data-root",
                        str(data),
                        "--checkpoint",
                        str(checkpoint),
                        "--device",
                        "cpu",
                    ],
                ),
                patch.object(eval_ng, "parse_ng_gbk", return_value=genes) as parse,
                patch.object(
                    eval_ng, "load_labels", return_value={"g0": True, "g1": False}
                ) as labels,
                patch.object(
                    eval_ng, "load_detector", return_value=(ScoreDetector(), metadata)
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_ng.main()
            labels.assert_called_once_with(csv, genes)
            self.assertTrue(all(call.args[0] == gbk for call in parse.call_args_list))
            self.assertTrue((checkpoint.parent / "ng_metrics.json").is_file())
            self.assertEqual(
                before,
                {
                    p.relative_to(data): p.read_bytes()
                    for p in data.rglob("*")
                    if p.is_file()
                },
            )

    def test_repeated_tail_occurrences_use_arithmetic_mean(self):
        class PositionDetector(torch.nn.Module):
            def forward(self, x):
                return torch.arange(128, device=x.device).float().unsqueeze(0) / 128

        genes = [{"gene_id": f"g{i}", "accession": "c", "start": i} for i in range(5)]
        labels = {f"g{i}": i == 0 for i in range(5)}
        metadata = {
            "validation_threshold": 0.4,
            "validation_threshold_source": "validation",
            "validation_threshold_metric": "mcc",
            "epoch": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedding = root / "emb.parquet"
            pq.write_table(
                pa.table({"gene_id": list(labels), "embedding": [[0.0] * 960] * 5}),
                embedding,
            )
            with (
                patch.object(eval_ng, "_embedding_path", return_value=embedding),
                patch.object(eval_ng, "parse_ng_gbk", return_value=genes),
                patch.object(
                    eval_ng,
                    "load_detector",
                    return_value=(PositionDetector(), metadata),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_ng.evaluate_model(
                    "fixture",
                    labels,
                    device="cpu",
                    metrics_path=str(root / "metrics.json"),
                    dump_gene_scores=str(root / "scores.npz"),
                )
            import numpy as np

            values = np.arange(128) / 128
            expected = [values[np.arange(128) % 5 == i].mean() for i in range(5)]
            np.testing.assert_allclose(
                np.load(root / "scores.npz")["scores"], expected, rtol=1e-6
            )

    def test_overlap_preserves_unique_gene_means(self):
        genes = [
            {"gene_id": f"g{i}", "accession": "contig", "start": i} for i in range(130)
        ]
        labels = {g["gene_id"]: i % 4 == 0 for i, g in enumerate(genes)}
        metadata = {
            "epoch": 7,
            "validation_threshold": 0.4,
            "validation_threshold_source": "validation",
            "validation_threshold_metric": "mcc",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "emb.parquet"
            pq.write_table(
                pa.table(
                    {
                        "gene_id": list(labels),
                        "embedding": [
                            [0.9 if labels[g] else 0.1] * 960 for g in labels
                        ],
                    }
                ),
                embeddings,
            )
            for n_genes, n_pos, n_windows in [(130, 33, 3)]:
                path = root / "metrics.json"
                with (
                    patch.object(eval_ng, "_embedding_path", return_value=embeddings),
                    patch.object(eval_ng, "parse_ng_gbk", return_value=genes),
                    patch.object(
                        eval_ng,
                        "load_detector",
                        return_value=(ScoreDetector(), metadata),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    eval_ng.evaluate_model(
                        "fixture", labels, device="cpu", metrics_path=str(path)
                    )
                result = json.loads(path.read_text())
                self.assertEqual(
                    (result["n_genes"], result["n_pos"], result["n_windows"]),
                    (n_genes, n_pos, n_windows),
                )
                self.assertEqual(result["postprocess"], "none")
                self.assertEqual(result["threshold_source"], "validation")
                for key in ["auroc", "auprc", "f1", "mcc"]:
                    self.assertEqual(result[key], 1.0)

    def test_missing_ng_embeddings_warn_and_record_excluded_genes(self):
        genes = [
            {"gene_id": f"g{i}", "accession": "contig", "start": i} for i in range(3)
        ]
        labels = {"g0": True, "g1": False, "g2": True}
        metadata = {
            "epoch": 1,
            "validation_threshold": 0.4,
            "validation_threshold_source": "validation",
            "validation_threshold_metric": "mcc",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "emb.parquet"
            pq.write_table(
                pa.table(
                    {"gene_id": ["g0", "g1"], "embedding": [[0.9] * 960, [0.1] * 960]}
                ),
                embeddings,
            )
            with (
                patch.object(eval_ng, "_embedding_path", return_value=embeddings),
                patch.object(eval_ng, "parse_ng_gbk", return_value=genes),
                patch.object(
                    eval_ng, "load_detector", return_value=(ScoreDetector(), metadata)
                ),
                self.assertWarnsRegex(RuntimeWarning, "excludes 1 genes"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_ng.evaluate_model(
                    "fixture",
                    labels,
                    device="cpu",
                    metrics_path=str(root / "metrics.json"),
                )
            result = json.loads((root / "metrics.json").read_text())
            self.assertEqual(result["missing_embeddings"], 1)
            self.assertEqual(result["missing_positive_genes"], 1)
            self.assertEqual(result["n_genes"], 2)

    def test_loads_only_fixed_validation_selected_ema_checkpoint(self):
        model = BGCDetector().eval()
        metadata = {
            "architecture": dict(ARCHITECTURE),
            "checkpoint_kind": "best",
            "weight_source": "ema",
            "validation_threshold": 0.42,
            "validation_threshold_source": "validation",
            "validation_threshold_metric": "mcc",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            save_checkpoint(path, model, metadata)
            restored, _ = eval_ng.load_detector(str(path), torch.device("cpu"))
            x = torch.randn(1, 4, 960)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), restored(x)))
            historical = metadata | {
                "architecture": {
                    k: v
                    for k, v in ARCHITECTURE.items()
                    if k not in ["input_norm", "kernel_size"]
                },
                "cli": {"input_norm": True, "kernel_size": 5},
            }
            save_checkpoint(path, model, historical)
            restored, _ = eval_ng.load_detector(str(path), torch.device("cpu"))
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), restored(x)))
            save_checkpoint(path, model, metadata | {"weight_source": "raw"})
            with self.assertRaisesRegex(ValueError, "validation-selected EMA"):
                eval_ng.load_detector(str(path), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
