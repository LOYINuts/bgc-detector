import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from bgc_gnn.esmc import last_hidden, load_model


class EsmcLoaderTest(unittest.TestCase):
    def test_native_model_and_tokenizer_are_loaded_in_eval_mode(self):
        native = types.ModuleType("esm.models.esmc")
        native.EsmcModel = Mock()
        native.EsmcTokenizer = Mock()
        model = native.EsmcModel.from_pretrained.return_value.to.return_value
        with patch.dict(sys.modules, {"esm.models.esmc": native}):
            result, tokenizer = load_model("cpu")
        native.EsmcModel.from_pretrained.assert_called_once_with(
            "biohub/ESMC-300M", dtype=torch.bfloat16
        )
        native.EsmcModel.from_pretrained.return_value.to.assert_called_once_with("cpu")
        native.EsmcTokenizer.from_pretrained.assert_called_once_with("biohub/ESMC-300M")
        model.eval.assert_called_once_with()
        self.assertIs(result, model)
        self.assertIs(tokenizer, native.EsmcTokenizer.from_pretrained.return_value)

    def test_current_esm_package_names_are_supported(self):
        native = types.ModuleType("esm.models.esmc")
        native.ESMC = Mock()
        tokenizer_module = types.ModuleType("esm.tokenization")
        tokenizer_module.EsmSequenceTokenizer = Mock()
        with patch.dict(
            sys.modules,
            {"esm.models.esmc": native, "esm.tokenization": tokenizer_module},
        ):
            result, tokenizer = load_model("cpu")
        native.ESMC.from_pretrained.assert_called_once_with(
            "biohub/ESMC-300M", dtype=torch.bfloat16
        )
        tokenizer_module.EsmSequenceTokenizer.from_pretrained.assert_called_once_with(
            "biohub/ESMC-300M"
        )
        self.assertIs(result, native.ESMC.from_pretrained.return_value.to.return_value)
        self.assertIs(
            tokenizer,
            tokenizer_module.EsmSequenceTokenizer.from_pretrained.return_value,
        )

    def test_last_hidden_accepts_both_output_shapes(self):
        hidden = torch.zeros(2, 5, 960)
        legacy = types.SimpleNamespace(last_hidden_state=hidden)
        current = types.SimpleNamespace(hidden_states=(torch.zeros(1), hidden))
        self.assertIs(last_hidden(legacy), hidden)
        self.assertIs(last_hidden(current), hidden)
        with self.assertRaisesRegex(ValueError, "no hidden states"):
            last_hidden(types.SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
