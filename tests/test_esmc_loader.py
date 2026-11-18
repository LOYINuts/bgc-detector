import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from bgc_gnn.esmc import (
    ESMC_MODEL,
    embed_sequences,
    last_hidden,
    load_model,
    pool_batch,
)


def transformers_modules():
    auto_model = Mock()
    auto_tokenizer = Mock()
    module = types.ModuleType("transformers")
    module.AutoModel = auto_model
    module.AutoTokenizer = auto_tokenizer
    config = auto_model.from_pretrained.return_value.to.return_value.config
    config.hidden_size = None  # older ESMC schema only carries d_model
    config.d_model = 960
    return module, auto_model, auto_tokenizer


class EsmcLoaderTest(unittest.TestCase):
    def test_model_loads_in_bf16_on_the_requested_device(self):
        module, auto_model, auto_tokenizer = transformers_modules()
        with patch.dict(sys.modules, {"transformers": module}):
            model, tokenizer = load_model("cpu")
        auto_model.from_pretrained.assert_called_once_with(
            ESMC_MODEL, dtype=torch.bfloat16, local_files_only=False
        )
        auto_model.from_pretrained.return_value.to.assert_called_once_with("cpu")
        loaded = auto_model.from_pretrained.return_value.to.return_value
        loaded.eval.assert_called_once_with()
        auto_tokenizer.from_pretrained.assert_called_once_with(
            ESMC_MODEL, local_files_only=False
        )
        self.assertIs(model, loaded.eval.return_value)
        self.assertIs(tokenizer, auto_tokenizer.from_pretrained.return_value)

    def test_offline_environment_keeps_the_loader_off_the_network(self):
        module, auto_model, auto_tokenizer = transformers_modules()
        with (
            patch.dict(sys.modules, {"transformers": module}),
            patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}),
        ):
            load_model("cpu")
        self.assertTrue(auto_model.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertTrue(
            auto_tokenizer.from_pretrained.call_args.kwargs["local_files_only"]
        )

    def test_last_hidden_accepts_both_output_shapes(self):
        hidden = torch.zeros(2, 5, 960)
        legacy = types.SimpleNamespace(last_hidden_state=hidden)
        current = types.SimpleNamespace(hidden_states=(torch.zeros(1), hidden))
        self.assertIs(last_hidden(legacy), hidden)
        self.assertIs(last_hidden(current), hidden)
        with self.assertRaisesRegex(ValueError, "no hidden states"):
            last_hidden(types.SimpleNamespace())

    def test_pooling_drops_padding_and_the_bos_eos_tokens(self):
        hidden = torch.zeros(1, 5, 960)
        hidden[0, 1, :] = 2.0  # BOS at 0, only residue 1, EOS at 2
        hidden[0, 2, :] = 4.0
        mask = torch.tensor([[True, True, True, False, False]])
        pooled = pool_batch(hidden, mask)
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
        with patch("bgc_gnn.esmc.load_model", return_value=(Blocking(), tokenizer)):
            matrix = embed_sequences(["MKVA", "MKVAG"], torch.device("cpu"))
        self.assertEqual(matrix.shape, (2, 960))
        np.testing.assert_allclose(matrix, 1.0)

    def test_missing_transformers_becomes_a_clear_error(self):
        with (
            patch.dict(sys.modules, {"transformers": None}),
            self.assertRaisesRegex(ValueError, "needs a transformers build"),
        ):
            load_model("cpu")

    def test_a_mismatched_esmc_schema_is_refused(self):
        module, auto_model, _ = transformers_modules()
        auto_model.from_pretrained.return_value.to.return_value.config.hidden_size = (
            2560
        )
        with (
            patch.dict(sys.modules, {"transformers": module}),
            self.assertRaisesRegex(ValueError, "2560-wide instead of 960"),
        ):
            load_model("cpu")


if __name__ == "__main__":
    unittest.main()
