import unittest

import numpy as np

from scripts.eval_deepbgc import (
    evaluate_block,
    per_contig_blocks,
    remap_locus_tags,
    score_vectors,
)


class DeepBGCScoreVectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.order = {"acc": ["acc:g1", "acc:g2", "acc:g3"]}
        self.labels = {"acc:g1": True, "acc:g2": False, "acc:g3": True}
        self.scores = {"acc:g1": 0.9, "acc:g3": 0.1}

    def test_unique_genes_imputes_missing_score_zero(self) -> None:
        y_true, y_score, n_scored = score_vectors(self.order, self.labels, self.scores)
        np.testing.assert_array_equal(y_true, [1, 0, 1])
        np.testing.assert_allclose(y_score, [0.9, 0.0, 0.1])
        self.assertEqual(n_scored, 2)


class DeepBGCRemapTest(unittest.TestCase):
    def test_locus_tags_become_gene_ids(self) -> None:
        remapped = remap_locus_tags(
            {"SSDG_00780": 0.7, "UNKNOWN": 0.2}, {"SSDG_00780": "CM000950:SSDG_00780"}
        )
        self.assertEqual(remapped, {"CM000950:SSDG_00780": 0.7})


class DeepBGCBlockTest(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        block = evaluate_block(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
        self.assertEqual(block["auroc"], 1.0)
        self.assertEqual(block["f1"], 1.0)
        self.assertEqual(block["mcc"], 1.0)

    def test_constant_score_is_chance(self) -> None:
        block = evaluate_block(np.array([1, 0, 1, 0]), np.full(4, 0.3))
        self.assertEqual(block["auroc"], 0.5)
        self.assertEqual(block["n_pos"], 2)

    def test_extra_metrics_present(self) -> None:
        block = evaluate_block(np.array([1, 0, 1, 0]), np.array([0.9, 0.4, 0.7, 0.2]))
        self.assertAlmostEqual(block["prevalence"], 0.5)
        self.assertAlmostEqual(block["auprc_lift"], block["auprc"] / 0.5)
        self.assertEqual(len(block["threshold_curve"]), 17)
        for key in ("accuracy", "ng_best_precision", "ng_best_recall"):
            self.assertIn(key, block)

    def test_per_contig_split(self) -> None:
        order = {"a": ["a:g1", "a:g2"], "b": ["b:g1"]}
        labels = {"a:g1": True, "a:g2": False, "b:g1": False}
        scores = {"a:g1": 0.9, "a:g2": 0.1}
        blocks = per_contig_blocks(order, labels, scores)
        self.assertEqual(sorted(blocks), ["a", "b"])
        self.assertEqual(blocks["a"]["n_genes"], 2)
        self.assertEqual(blocks["b"]["n_pos"], 0)
        self.assertNotIn("threshold_curve", blocks["a"])

    def test_single_class_guarded(self) -> None:
        block = evaluate_block(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        self.assertEqual(block["auroc"], 0.5)
        self.assertEqual(block["auprc"], 0.0)


if __name__ == "__main__":
    unittest.main()
