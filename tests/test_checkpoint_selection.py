import tempfile
import unittest
from pathlib import Path

import torch

from bgc_gnn.checkpoint import load_checkpoint, save_checkpoint
from bgc_gnn.eval import window_aggregation_weight
from bgc_gnn.model import BGCProphetDetector


class CheckpointRoundTripTest(unittest.TestCase):
    def test_center_aggregation_weights(self) -> None:
        self.assertEqual(window_aggregation_weight(0, 128, "mean"), 1.0)
        self.assertAlmostEqual(
            window_aggregation_weight(0, 128, "center"),
            window_aggregation_weight(127, 128, "center"),
        )
        self.assertGreater(
            window_aggregation_weight(63, 128, "center"),
            window_aggregation_weight(0, 128, "center"),
        )
        with self.assertRaises(ValueError):
            window_aggregation_weight(128, 128, "mean")

    def test_checkpoint_round_trip_keeps_weights_and_metadata(self) -> None:
        model = BGCProphetDetector(input_dim=16, d_model=16, nhead=2, num_layers=1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "detector_16d_best.pt"
            save_checkpoint(path, model, {"epoch": 7, "validation_threshold": 0.42})
            state, metadata = load_checkpoint(path, "cpu")
            restored = BGCProphetDetector(input_dim=16, d_model=16, nhead=2, num_layers=1)
            restored.load_state_dict(state, strict=True)
            self.assertEqual(metadata["epoch"], 7)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
