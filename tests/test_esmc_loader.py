import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from bgc_detector.esmc import (
    ESMC_MODEL,
    embed_sequences,
    last_hidden,
    load_model,
    local_snapshot,
    pool_batch,
)


def esm_modules():
    esmc = types.ModuleType("esm.models.esmc")
    esmc.EsmcModel = Mock()
    esmc.EsmcTokenizer = Mock()
    models = types.ModuleType("esm.models")
    package = types.ModuleType("esm")
    return {"esm": package, "esm.models": models, "esm.models.esmc": esmc}, esmc


class EsmcLoaderTest(unittest.TestCase):
    def test_model_loads_in_bf16_from_the_cached_snapshot(self):
        modules, esmc = esm_modules()
        with (
            patch.dict(sys.modules, modules),
            patch("bgc_detector.esmc.local_snapshot", return_value="/snap/abc"),
        ):
            model, tokenizer = load_model("cpu")
        esmc.EsmcModel.from_pretrained.assert_called_once_with(
            "/snap/abc", dtype=torch.bfloat16
        )
        esmc.EsmcTokenizer.from_pretrained.assert_called_once_with("/snap/abc")
        esmc.EsmcModel.from_pretrained.return_value.to.assert_called_once_with("cpu")
        loaded = esmc.EsmcModel.from_pretrained.return_value.to.return_value
        loaded.eval.assert_called_once_with()
        self.assertIs(model, loaded.eval.return_value)
        self.assertIs(tokenizer, esmc.EsmcTokenizer.from_pretrained.return_value)

    def test_the_hub_is_tried_when_no_snapshot_is_cached(self):
        modules, esmc = esm_modules()
        with (
            patch.dict(sys.modules, modules),
            patch("bgc_detector.esmc.local_snapshot", return_value=None),
        ):
            load_model("cpu")
        esmc.EsmcModel.from_pretrained.assert_called_once_with(
            ESMC_MODEL, dtype=torch.bfloat16
        )

    def test_a_failing_snapshot_falls_back_to_the_hub(self):
        modules, esmc = esm_modules()
        esmc.EsmcModel.from_pretrained.side_effect = [OSError("broken"), Mock()]
        with (
            patch.dict(sys.modules, modules),
            patch("bgc_detector.esmc.local_snapshot", return_value="/snap/old"),
        ):
            load_model("cpu")
        self.assertEqual(esmc.EsmcModel.from_pretrained.call_count, 2)
        self.assertEqual(esmc.EsmcModel.from_pretrained.call_args.args[0], ESMC_MODEL)

    def test_snapshot_lookup_prefers_the_most_recent_one(self):
        with (
            patch.dict("os.environ", {"HF_HOME": "/hf"}),
            patch(
                "glob.glob",
                return_value=["/hf/hub/x/snapshots/a", "/hf/hub/x/snapshots/b"],
            ),
            patch(
                "os.path.getmtime",
                side_effect=lambda path: 2.0 if path.endswith("a") else 1.0,
            ),
        ):
            self.assertTrue(local_snapshot().endswith("snapshots/a"))

    def test_missing_esm_package_becomes_a_clear_error(self):
        with (
            patch.dict(sys.modules, {"esm": None, "esm.models.esmc": None}),
            self.assertRaisesRegex(ValueError, "needs the esm package"),
        ):
            load_model("cpu")

    def test_last_hidden_accepts_both_output_shapes(self):
        hidden = torch.zeros(2, 5, 960)
        self.assertIs(
            last_hidden(types.SimpleNamespace(last_hidden_state=hidden)), hidden
        )
        self.assertIs(
            last_hidden(types.SimpleNamespace(hidden_states=(torch.zeros(1), hidden))),
            hidden,
        )
        with self.assertRaisesRegex(ValueError, "no hidden states"):
            last_hidden(types.SimpleNamespace())

    def test_pooling_drops_padding_and_the_bos_eos_tokens(self):
        hidden = torch.zeros(1, 5, 960)
        hidden[0, 1, :] = 2.0  # BOS at 0, one residue at 1, EOS at 2
        hidden[0, 2, :] = 4.0
        pooled = pool_batch(hidden, torch.tensor([[True, True, True, False, False]]))
        self.assertEqual(pooled.shape, (1, 960))
        np.testing.assert_allclose(pooled[0], 2.0)

    def test_embed_sequences_pools_every_sequence_once(self):
        class Blocking(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 960)
                return types.SimpleNamespace(last_hidden_state=hidden)

        tokenizer = Mock()
        tokenizer.side_effect = lambda batch, **_: {
            "input_ids": torch.ones(len(batch), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(batch), 4, dtype=torch.long),
        }
        with patch(
            "bgc_detector.esmc.load_model", return_value=(Blocking(), tokenizer)
        ):
            matrix = embed_sequences(["MKVA", "MKVAG"], torch.device("cpu"))
        self.assertEqual(matrix.shape, (2, 960))
        np.testing.assert_allclose(matrix, 1.0)

    def test_a_wrong_width_is_refused(self):
        class Wide(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 2560)
                return types.SimpleNamespace(last_hidden_state=hidden)

        tokenizer = Mock()
        tokenizer.side_effect = lambda batch, **_: {
            "input_ids": torch.ones(len(batch), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(batch), 4, dtype=torch.long),
        }
        with (
            patch("bgc_detector.esmc.load_model", return_value=(Wide(), tokenizer)),
            self.assertRaisesRegex(ValueError, "2560d embeddings, expected 960"),
        ):
            embed_sequences(["MKVA"], torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
