import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bgc_detector.dataset import BGCDetectDataset, FixedBGCDataset


class _Embeddings:
    def get(self, gene_ids):
        return np.zeros((len(gene_ids), 4), dtype=np.float32)


def genome(prefix: str, count: int) -> dict:
    return {
        "genome_id": prefix,
        "gene_ids": [f"gtdb:{prefix}:{i}" for i in range(count)],
    }


class DetectionWindowSamplingTest(unittest.TestCase):
    def dataset(
        self, genomes: list[dict], negative_pool: list[str]
    ) -> BGCDetectDataset:
        return BGCDetectDataset(
            [{"bgc_id": "cluster", "gene_ids": ["bgc:cluster:0", "bgc:cluster:1"]}],
            genomes,
            negative_pool,
            _Embeddings(),
            window_size=128,
            augment=1,
        )

    def test_positive_padding_prefers_genome_long_enough_for_padding(self) -> None:
        short = genome("short", 100)
        long = genome("long", 130)
        dataset = self.dataset([short, long], [f"gtdb:pool:{i}" for i in range(200)])
        with (
            patch(
                "bgc_detector.dataset.random.choice",
                side_effect=lambda values: values[0],
            ),
            patch("bgc_detector.dataset.random.randint", return_value=0),
        ):
            sample = dataset._make_positive(0)

        padding = sample["gene_ids"][2:]
        self.assertEqual(len(sample["gene_ids"]), 128)
        self.assertTrue(all(gene_id.startswith("gtdb:long:") for gene_id in padding))
        self.assertEqual(int(sample["detect_mask"].sum()), 2)

    def test_negative_window_prefers_one_long_genome(self) -> None:
        short = genome("short", 50)
        long = genome("long", 130)
        dataset = self.dataset([short, long], [f"gtdb:pool:{i}" for i in range(200)])
        with (
            patch(
                "bgc_detector.dataset.random.choice",
                side_effect=lambda values: values[0],
            ),
            patch("bgc_detector.dataset.random.randint", return_value=0),
        ):
            sample = dataset._make_negative(0)

        self.assertEqual(len(sample["gene_ids"]), 128)
        self.assertTrue(
            all(gene_id.startswith("gtdb:long:") for gene_id in sample["gene_ids"])
        )
        self.assertFalse(sample["detect_mask"].any())

    def test_positive_uses_known_same_entity_flanks(self) -> None:
        source_ids = [f"bgc:cluster:{i}" for i in range(200)]
        bgc_ids = source_ids[70:75]
        bgc = {
            "bgc_id": "cluster",
            "source_gene_ids": source_ids,
            "gene_ids": bgc_ids,
        }
        dataset = BGCDetectDataset(
            [bgc],
            [genome("unused", 128)],
            [],
            _Embeddings(),
            window_size=128,
            augment=1,
        )
        with patch("bgc_detector.dataset.random.randint", return_value=40):
            sample = dataset._make_positive(0)

        self.assertEqual(sample["gene_ids"], source_ids[40:168])
        self.assertEqual(
            {
                gene_id
                for gene_id, is_positive in zip(
                    sample["gene_ids"], sample["detect_mask"].tolist()
                )
                if is_positive
            },
            set(bgc_ids),
        )
        self.assertEqual(int(sample["detect_mask"].sum()), len(bgc_ids))

    def test_fallback_splices_runs_in_order(self) -> None:
        first = genome("first", 40)
        second = genome("second", 30)
        pool = [f"gtdb:pool:{i}" for i in range(100)]
        dataset = self.dataset([first, second], pool)
        with patch(
            "bgc_detector.dataset.random.choice", side_effect=lambda values: values[-1]
        ):
            sample = dataset._make_negative(0)

        self.assertEqual(len(sample["gene_ids"]), 128)
        self.assertEqual(sample["gene_ids"][:40], first["gene_ids"])
        self.assertEqual(sample["gene_ids"][40:70], second["gene_ids"])
        self.assertTrue(
            all(gene_id.startswith("gtdb:pool:") for gene_id in sample["gene_ids"][70:])
        )

    def test_positive_fallback_surrounds_the_cluster(self) -> None:
        short = genome("short", 20)
        dataset = self.dataset([short], [f"gtdb:pool:{i}" for i in range(200)])
        with (
            patch(
                "bgc_detector.dataset.random.choice",
                side_effect=lambda values: values[0],
            ),
            patch("bgc_detector.dataset.random.randint", return_value=0),
        ):
            sample = dataset._make_positive(0)

        self.assertEqual(len(sample["gene_ids"]), 128)
        self.assertEqual(int(sample["detect_mask"].sum()), 2)


class FixedDatasetTest(unittest.TestCase):
    def test_missing_real_gene_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.parquet"
            pq.write_table(
                pa.table(
                    {
                        "gene_ids": [["g0", "missing"]],
                        "detect_mask": [[True, False]],
                        "is_positive": [True],
                    }
                ),
                path,
            )
            with self.assertRaisesRegex(
                ValueError, "1 real genes lack embeddings.*missing"
            ):
                FixedBGCDataset(path, self.Store({"g0"}), emb_dim=4)

    class Store:
        def __init__(self, present):
            self.present = present
            self._map = {gene_id: np.ones(4, dtype=np.float32) for gene_id in present}

        def __getitem__(self, gene_id):
            return self._map[gene_id]

        def __contains__(self, gene_id):
            return gene_id in self.present

    def test_missing_genes_are_zero_padded_and_valid_mask_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.parquet"
            pq.write_table(
                pa.table(
                    {
                        "gene_ids": pa.array(
                            [["g0", "g1", "missing"]], pa.list_(pa.string())
                        ),
                        "detect_mask": pa.array(
                            [[False, True, False]], pa.list_(pa.bool_())
                        ),
                        "is_positive": pa.array([True]),
                        "valid_mask": pa.array(
                            [[True, True, False]], pa.list_(pa.bool_())
                        ),
                    }
                ),
                path,
            )
            dataset = FixedBGCDataset(path, self.Store({"g0", "g1"}), emb_dim=4)
            sample = dataset[0]

        self.assertEqual(sample["valid_mask"].tolist(), [True, True, False])
        torch.testing.assert_close(sample["embeddings"][2], torch.zeros(4))
        self.assertEqual(sample["detect_mask"].tolist(), [False, True, False])
        self.assertTrue(sample["is_positive"])


if __name__ == "__main__":
    unittest.main()
