import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.encode import load_model


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


if __name__ == "__main__":
    unittest.main()
