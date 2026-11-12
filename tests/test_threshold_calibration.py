import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bgc_gnn.checkpoint import load_checkpoint, validation_threshold
from bgc_gnn.eval import select_detection_threshold


class ThresholdCalibrationTest(unittest.TestCase):
    def test_selects_validation_mcc_or_f1(self) -> None:
        truth = np.array([0, 1, 1, 1])
        scores = np.array([0.35, 0.3, 0.4, 0.9])
        mcc_threshold, mcc = select_detection_threshold(truth, scores, "mcc")
        f1_threshold, f1 = select_detection_threshold(truth, scores, "f1")
        self.assertEqual(mcc_threshold, 0.4)
        self.assertAlmostEqual(mcc, 2 / np.sqrt(12))
        self.assertEqual(f1_threshold, 0.3)
        self.assertAlmostEqual(f1, 6 / 7)

    def test_old_checkpoint_uses_half_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.pt"
            torch.save({"weight": torch.ones(1)}, path)
            _, metadata = load_checkpoint(path)
            self.assertEqual(validation_threshold(metadata), (0.5, "legacy-default"))


if __name__ == "__main__":
    unittest.main()
