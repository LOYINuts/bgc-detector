import tempfile
import unittest
from pathlib import Path

import torch

from bgc_gnn import train
from bgc_gnn.checkpoint import (
    initialize_from_pretraining,
    save_checkpoint,
)
from bgc_gnn.model import ARCHITECTURE, BGCDetector


class FinalModelTest(unittest.TestCase):
    def test_fixed_structure_semantic_gradient_and_padding(self):
        model = BGCDetector().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10290690)
        self.assertFalse(hasattr(model, "context_readout"))
        for kwargs in [
            {"semantic_skip": False},
            {"context_readout": True},
            {"d_model": 8},
        ]:
            with self.assertRaises(TypeError):
                BGCDetector(**kwargs)
        x = torch.randn(2, 8, 960)
        padding = torch.zeros(2, 8, dtype=torch.bool)
        padding[0, 6:] = True
        with torch.no_grad():
            baseline = model(x, key_padding_mask=padding)
            model.semantic_skip.weight[0, 0] = 0.25
            model.semantic_skip.bias.fill_(0.1)
            expected = torch.sigmoid(
                torch.logit(baseline)
                + model.semantic_skip(model.input_norm(x)).squeeze(-1)
            )
        actual = model(x, key_padding_mask=padding)
        torch.testing.assert_close(actual, expected)
        actual[~padding].mean().backward()
        self.assertGreater(float(model.semantic_skip.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            torch.testing.assert_close(
                actual[~padding],
                model(x.masked_fill(padding[..., None], 1e4), key_padding_mask=padding)[
                    ~padding
                ],
            )

    def test_learned_semantic_transfer_is_exact_and_rejects_old_variants(self):
        source = BGCDetector().eval()
        with torch.no_grad():
            source.semantic_skip.weight.fill_(0.01)
            source.semantic_skip.bias.fill_(0.2)
        target = BGCDetector().eval()
        metadata = {
            "checkpoint_kind": "pretraining_best",
            "stage": "pretraining",
            "selection_metric": "pretraining_val_loss",
            "architecture": dict(ARCHITECTURE),
            "settings": {"seed": 1506},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pretrained.pt"
            save_checkpoint(path, source, metadata)
            initialize_from_pretraining(target, path)
            for key, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, target.state_dict()[key]))
            x = torch.randn(1, 8, 960)
            with torch.no_grad():
                self.assertTrue(torch.equal(source(x), target(x)))
            for field, value in [
                ("semantic_skip", False),
                ("context_readout", True),
                ("d_model", 320),
                ("symmetric_local_conv", True),
            ]:
                incompatible = metadata | {
                    "architecture": ARCHITECTURE | {field: value}
                }
                save_checkpoint(path, source, incompatible)
                with self.assertRaisesRegex(ValueError, "fixed architecture"):
                    initialize_from_pretraining(target, path)
            state = source.state_dict()
            del state["semantic_skip.weight"]
            torch.save({"model_state": state, "metadata": metadata}, path)
            with self.assertRaisesRegex(ValueError, "complete final detector"):
                initialize_from_pretraining(target, path)
            with torch.no_grad():
                source.semantic_skip.weight.zero_()
            save_checkpoint(path, source, metadata)
            with self.assertRaisesRegex(ValueError, "learned semantic"):
                initialize_from_pretraining(target, path)

    def test_fixed_optimizer_groups(self):
        model = BGCDetector()
        optimizer = train.build_optimizer(model)
        encoder, head = optimizer.param_groups
        self.assertEqual(encoder["lr"], 0.0001)
        self.assertEqual(head["lr"], 0.0005)
        expected = {
            id(p)
            for n, p in model.named_parameters()
            if n.startswith(train.ENCODER_PREFIXES)
        }
        self.assertEqual({id(p) for p in encoder["params"]}, expected)
        self.assertIn(id(model.semantic_skip.weight), {id(p) for p in head["params"]})
        self.assertEqual(
            {id(p) for p in encoder["params"]} | {id(p) for p in head["params"]},
            {id(p) for p in model.parameters()},
        )


if __name__ == "__main__":
    unittest.main()
