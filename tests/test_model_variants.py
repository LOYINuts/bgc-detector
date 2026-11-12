import tempfile
import unittest
from pathlib import Path

import torch

from bgc_gnn import train
from bgc_gnn.checkpoint import detector_settings, initialize_from_pretraining, load_checkpoint, save_checkpoint
from bgc_gnn.model import BGCDetector


def small_model(**kwargs) -> BGCDetector:
    torch.manual_seed(7)
    return BGCDetector(input_dim=8, d_model=8, nhead=2, num_layers=2, **kwargs)


class SemanticSkipTest(unittest.TestCase):
    def test_zero_initialization_matches_seeded_baseline(self):
        torch.manual_seed(7)
        baseline = BGCDetector(input_dim=8, d_model=8, nhead=2, num_layers=2,
                                      semantic_skip=False)
        torch.manual_seed(7)
        skip = BGCDetector(input_dim=8, d_model=8, nhead=2, num_layers=2)
        baseline.eval(); skip.eval()
        for name, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, skip.state_dict()[name], rtol=0, atol=0)
        inputs = torch.randn(2, 12, 8)
        with torch.no_grad():
            torch.testing.assert_close(baseline(inputs), skip(inputs), rtol=0, atol=0)
        self.assertEqual(
            sum(p.numel() for p in skip.parameters())
            - sum(p.numel() for p in baseline.parameters()), 9)
        self.assertTrue(torch.equal(skip.semantic_skip.weight,
                                    torch.zeros_like(skip.semantic_skip.weight)))

    def test_adds_normalized_protein_logits_and_receives_gradient(self):
        model = small_model().eval()
        inputs = torch.randn(2, 12, 8)
        with torch.no_grad():
            base = model(inputs)
            model.semantic_skip.weight[0, 0] = 0.25
            model.semantic_skip.bias.fill_(0.1)
            expected = torch.sigmoid(
                torch.logit(base)
                + model.semantic_skip(model.input_norm(inputs)).squeeze(-1))
        actual = model(inputs)
        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(model.semantic_skip.weight.grad).all())
        self.assertGreater(model.semantic_skip.weight.grad.abs().sum().item(), 0)

        padding = torch.zeros(2, 12, dtype=torch.bool)
        padding[0, 9:] = True
        noisy = inputs.clone()
        noisy[padding] = 1e4
        with torch.no_grad():
            torch.testing.assert_close(
                model(inputs, key_padding_mask=padding)[~padding],
                model(noisy, key_padding_mask=padding)[~padding])


class ContextReadoutTest(unittest.TestCase):
    def test_initial_equivalence_rng_gradient_and_padding(self):
        torch.manual_seed(1301)
        base = small_model()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(1301)
        new = small_model(context_readout=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)

        x = torch.randn(2, 8, 8)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, 6:] = True
        base.eval(); new.eval()
        torch.testing.assert_close(base(x, key_padding_mask=mask),
                                   new(x, key_padding_mask=mask))
        y = new(x, key_padding_mask=mask)
        y[~mask].mean().backward()
        self.assertGreater(float(new.context_readout[2].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            new.context_readout[2].weight.fill_(.01)
        noisy = x.masked_fill(mask[..., None], 1000.)
        torch.testing.assert_close(new(x, key_padding_mask=mask)[~mask],
                                   new(noisy, key_padding_mask=mask)[~mask])

    def test_checkpoint_metadata_restores_the_readout_model(self):
        model = small_model(context_readout=True).eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'best.pt'
            save_checkpoint(path, model, {
                'architecture': {
                    'd_model': 8, 'nhead': 2, 'num_layers': 2, 'input_dim': 8,
                    'input_norm': True, 'kernel_size': 5, 'semantic_skip': True,
                    'context_readout': True,
                },
            })
            state, metadata = load_checkpoint(path, 'cpu')
            restored = BGCDetector(**detector_settings(metadata, 8)).eval()
            restored.load_state_dict(state, strict=True)
            x = torch.randn(2, 8, 8)
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_weak_transfer_allows_only_new_zero_output_branch(self):
        source = small_model(semantic_skip=False).eval()
        target = small_model(context_readout=True).eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'weak.pt'
            save_checkpoint(path, source, {
                'checkpoint_kind': 'pretraining_best', 'pretraining_method': 'weak',
            })
            provenance = initialize_from_pretraining(target, path)
            self.assertEqual(len(provenance['new_zero_initialized_keys']['context_readout.']), 4)
            x = torch.randn(2, 8, 8)
            with torch.no_grad():
                torch.testing.assert_close(source(x), target(x))
            with torch.no_grad():
                target.context_readout[2].bias.fill_(1.)
            with self.assertRaisesRegex(ValueError, 'zero output'):
                initialize_from_pretraining(target, path)


class OptimizerGroupTest(unittest.TestCase):
    def test_transfer_scales_only_encoder_parameters(self):
        model = small_model()
        optimizer = train.build_optimizer(model, 0.2)
        self.assertEqual(len(optimizer.param_groups), 2)
        encoder = {id(p) for p in optimizer.param_groups[0]['params']}
        head = {id(p) for p in optimizer.param_groups[1]['params']}
        expected_encoder = {
            id(p) for name, p in model.named_parameters()
            if name.startswith(train.ENCODER_PREFIXES)
        }
        self.assertEqual(encoder, expected_encoder)
        self.assertFalse(encoder & head)
        self.assertEqual(encoder | head, {id(p) for p in model.parameters()})
        self.assertAlmostEqual(
            optimizer.param_groups[0]['lr'] / optimizer.param_groups[1]['lr'], 0.2)

    def test_scratch_uses_one_group(self):
        optimizer = train.build_optimizer(small_model(), 1.0)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], train.LR)


if __name__ == '__main__':
    unittest.main()