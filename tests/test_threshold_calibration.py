import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bgc_detector.checkpoint import load_checkpoint, validation_threshold
from bgc_detector.eval import select_detection_threshold


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

    def test_rejects_unannotated_weights_and_missing_validation_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unannotated.pt"
            torch.save({"weight": torch.ones(1)}, path)
            with self.assertRaisesRegex(ValueError, "model_state and metadata"):
                load_checkpoint(path)
            for metadata in [
                {},
                {"validation_threshold": float("nan")},
                {"validation_threshold": 0.5, "validation_threshold_source": "test"},
            ]:
                with self.assertRaisesRegex(ValueError, "validation-MCC"):
                    validation_threshold(metadata)
            self.assertEqual(
                validation_threshold(
                    {
                        "validation_threshold": 0.42,
                        "validation_threshold_source": "validation",
                        "validation_threshold_metric": "mcc",
                    }
                ),
                (0.42, "validation"),
            )


if __name__ == "__main__":
    unittest.main()
