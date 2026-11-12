import unittest

import numpy as np

from bgc_gnn.eval import bgc_prophet_postprocess


class BGCProphetPostprocessTest(unittest.TestCase):
    def test_merges_short_gaps_and_removes_short_regions(self) -> None:
        predicted = np.array([
            0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
        ])
        actual = bgc_prophet_postprocess(predicted, max_gap=2, min_genes=3)
        expected = np.array([
            0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_keeps_a_region_of_exact_minimum_length(self) -> None:
        predicted = np.array([0, 1, 1, 1, 0])
        actual = bgc_prophet_postprocess(predicted, max_gap=2, min_genes=3)
        np.testing.assert_array_equal(actual, predicted)

    def test_longest_only_keeps_first_longest_region(self) -> None:
        predicted = np.array([
            1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
        ])
        actual = bgc_prophet_postprocess(
            predicted, max_gap=2, min_genes=3, longest_only=True,
        )
        expected = np.array([
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_does_not_change_input(self) -> None:
        predicted = np.array([1, 0, 1])
        original = predicted.copy()
        bgc_prophet_postprocess(predicted)
        np.testing.assert_array_equal(predicted, original)


if __name__ == "__main__":
    unittest.main()
