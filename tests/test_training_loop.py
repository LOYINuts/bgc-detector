import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from bgc_gnn import train
from bgc_gnn.checkpoint import load_checkpoint


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))
        self.drop = torch.nn.Dropout(0.3)
        self.register_buffer('fixed', torch.tensor(3))

    def forward(self, x, key_padding_mask=None):
        # constant per gene; keeps dropout in the graph to exercise the RNG stream
        return torch.sigmoid(self.logit) + 0 * self.drop(x[..., 0])


def batch(embeddings=None):
    return {
        'embeddings': torch.zeros(1, 3, 1) if embeddings is None else embeddings,
        'detect_mask': torch.tensor([[1., 0., 1.]]),
        'valid_mask': torch.tensor([[1., 1., 0.]]),
        'is_positive': torch.tensor([True]),
    }


class WeightedBCETest(unittest.TestCase):
    def test_pos_weight_is_applied_and_padding_ignored(self):
        outputs = torch.tensor([[0.5, 0.5, 0.5]])
        targets = torch.tensor([[1., 0., 0.]])
        valid = torch.tensor([[1., 1., 0.]])
        loss = train.weighted_bce(outputs, targets, valid, pos_weight=2.0)
        per_pos = torch.log(torch.tensor(2.0))
        expected = (2 * per_pos + per_pos) / 2
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)

    def test_calls_the_model_with_a_padding_mask(self):
        seen = {}

        class Recorder(TinyDetector):
            def forward(self, x, key_padding_mask=None):
                seen['mask'] = key_padding_mask
                return super().forward(x, key_padding_mask)

        loader = [batch()]
        train.train_epoch(Recorder(), loader, torch.optim.SGD(Recorder().parameters(), lr=.1),
                          torch.device('cpu'), 1, 1,
                          AveragedModel(Recorder(), multi_avg_fn=get_ema_multi_avg_fn(.5)))
        self.assertEqual(seen['mask'].tolist(), [[False, False, True]])


class EMATrainingTest(unittest.TestCase):
    def test_ema_moves_towards_raw_weights_and_preserves_raw_rng(self):
        baseline, source = TinyDetector(), TinyDetector()
        ema = AveragedModel(source, multi_avg_fn=get_ema_multi_avg_fn(.5))
        ema.requires_grad_(False)
        raw_optimizer = torch.optim.SGD(baseline.parameters(), lr=.2)
        optimizer = torch.optim.SGD(source.parameters(), lr=.2)
        previous = None
        for i in range(3):
            torch.manual_seed(77 + i)
            train.train_epoch(baseline, [batch()], raw_optimizer, torch.device('cpu'),
                              1, 1, AveragedModel(baseline, multi_avg_fn=get_ema_multi_avg_fn(.5)))
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(77 + i)
            train.train_epoch(source, [batch()], optimizer, torch.device('cpu'), 1, 1, ema)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            torch.testing.assert_close(source.logit, baseline.logit, rtol=0, atol=0)
            expected = (source.logit.detach().clone() if previous is None
                        else .5 * previous + .5 * source.logit.detach())
            torch.testing.assert_close(ema.module.logit, expected)
            self.assertEqual(ema.n_averaged.item(), i + 1)
            self.assertIsNone(ema.module.logit.grad)
            self.assertFalse(ema.module.logit.requires_grad)
            previous = expected
        source.fixed.fill_(9)
        ema.update_parameters(source)
        self.assertEqual(ema.module.fixed.item(), 9)

    def test_validation_reads_the_ema_weights(self):
        torch.manual_seed(3)
        model = TinyDetector()
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.999))
        ema.requires_grad_(False)
        with torch.no_grad():
            ema.module.logit.fill_(5.)
            model.logit.fill_(-5.)
        metrics = train.val_epoch(ema.module, [batch()], torch.device('cpu'))
        raw_metrics = train.val_epoch(model, [batch()], torch.device('cpu'))
        # targets are [1, 0] at the two valid positions; the confident EMA weights
        # must beat the inverted raw weights on exactly that metric
        self.assertLess(metrics['val_loss'], raw_metrics['val_loss'])


class TrainingRunTest(unittest.TestCase):
    def _dataset(self, root: Path, name: str) -> Path:
        path = root / name
        pq.write_table(pa.table({
            'sample_idx': pa.array([0, 1], pa.int32()),
            'gene_ids': pa.array([['g0', 'g1'], ['g2', 'g3']], pa.list_(pa.string())),
            'detect_mask': pa.array([[True, True], [False, False]], pa.list_(pa.bool_())),
            'is_positive': pa.array([True, False], pa.bool_()),
        }), path)
        return path

    def test_run_exports_ema_weights_and_validation_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            embeddings = root / 'emb.parquet'
            pq.write_table(pa.table({
                'gene_id': pa.array([f'g{i}' for i in range(4)], pa.string()),
                'embedding': pa.array([[0.1] * 8] * 4, pa.list_(pa.float32(), 8)),
            }), embeddings)
            train_ds = self._dataset(root, 'train.parquet')
            val_ds = self._dataset(root, 'val.parquet')
            checkpoints = root / 'run'
            argv = [
                'train', '--fixed-dataset', str(train_ds),
                '--fixed-val-dataset', str(val_ds),
                '--emb-path', str(embeddings),
                '--checkpoint-dir', str(checkpoints),
                '--seed', '5', '--epochs', '2', '--patience', '0',
                '--num-workers', '0', '--device', 'cpu',
            ]
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                train.main()
            state, metadata = load_checkpoint(checkpoints / 'detector_8d_best.pt', 'cpu')
            self.assertEqual(metadata['checkpoint_kind'], 'best')
            self.assertEqual(metadata['selection_metric'], 'val_loss')
            self.assertEqual(metadata['weight_source'], 'ema')
            self.assertEqual(metadata['architecture']['context_readout'], False)
            self.assertIn('validation_threshold', metadata)
            self.assertIn('semantic_skip.weight', state)
            self.assertFalse(any(key.startswith('context_readout.') for key in state))
            final = (checkpoints / 'run_metadata.json').read_text()
            self.assertIn('"selected_epoch"', final)

    def test_cli_requires_the_pretrained_scale_checkpoint(self):
        argv = ['train', '--fixed-dataset', 'a', '--fixed-val-dataset', 'b',
                '--emb-path', 'c', '--checkpoint-dir', 'd',
                '--pretrained-encoder-lr-scale', '0.2']
        with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit):
            train.main()


if __name__ == '__main__':
    unittest.main()