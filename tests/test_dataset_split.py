import random
import unittest

from scripts.data.build_dataset import split_samples


class SampleSplitTest(unittest.TestCase):
    def test_matches_recorded_sample_split_and_preserves_row_order(self):
        samples = [{"sample_idx": i} for i in range(100)]
        indices = list(range(100))
        random.Random(42).shuffle(indices)
        expected_val = set(indices[:20])
        train, val = split_samples(samples)
        self.assertEqual([s["sample_idx"] for s in val], sorted(expected_val))
        self.assertEqual(
            [s["sample_idx"] for s in train], sorted(set(range(100)) - expected_val)
        )
        self.assertEqual(split_samples(samples), (train, val))


if __name__ == "__main__":
    unittest.main()
