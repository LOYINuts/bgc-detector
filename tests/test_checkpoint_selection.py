import tempfile
import unittest
from pathlib import Path

import torch

from bgc_gnn.checkpoint import load_checkpoint, save_checkpoint
from bgc_gnn.model import BGCDetector


class CheckpointRoundTripTest(unittest.TestCase):
    def test_checkpoint_round_trip_keeps_weights_and_metadata(self) -> None:
        model = BGCDetector()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "detector_960d_best.pt"
            save_checkpoint(path, model, {"epoch": 7, "validation_threshold": 0.42})
            state, metadata = load_checkpoint(path, "cpu")
            restored = BGCDetector()
            restored.load_state_dict(state, strict=True)
            self.assertEqual(metadata["epoch"], 7)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
