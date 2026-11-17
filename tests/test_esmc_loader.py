import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from bgc_gnn.esmc import ESMC_MODEL, ESMC_REGISTRY_NAME, last_hidden, load_model


def esm_modules(model_cls_name):
    native = types.ModuleType("esm.models.esmc")
    setattr(native, model_cls_name, Mock())
    tokenizer_class = Mock()
    tokenization = types.ModuleType("esm.tokenization")
    tokenization.EsmSequenceTokenizer = tokenizer_class
    return native, tokenization, getattr(native, model_cls_name), tokenizer_class


class EsmcLoaderTest(unittest.TestCase):
    def test_current_release_loads_by_registry_name_and_sets_eval(self):
        native, tokenization, model_cls, tokenizer_cls = esm_modules("ESMC")
        with patch.dict(
            sys.modules,
            {"esm.models.esmc": native, "esm.tokenization": tokenization},
        ):
            result, tokenizer = load_model("cpu")
        model_cls.from_pretrained.assert_called_once_with(
            ESMC_REGISTRY_NAME, device=torch.device("cpu")
        )
        tokenizer_cls.from_pretrained.assert_called_once_with(ESMC_REGISTRY_NAME)
        loaded = model_cls.from_pretrained.return_value
        loaded.to.assert_called_once_with("cpu")
        model = loaded.to.return_value
        model.eval.assert_called_once_with()
        self.assertIs(result, model)
        self.assertIs(tokenizer, tokenizer_cls.from_pretrained.return_value)

    def test_older_release_falls_back_to_hf_id_and_dtype(self):
        native, tokenization, model_cls, tokenizer_cls = esm_modules("ESMC")
        fallback = Mock()
        model_cls.from_pretrained.side_effect = [TypeError("dtype"), fallback]
        with patch.dict(
            sys.modules,
            {"esm.models.esmc": native, "esm.tokenization": tokenization},
        ):
            result, tokenizer = load_model("cpu")
        self.assertEqual(
            model_cls.from_pretrained.call_args_list[1].kwargs,
            {"dtype": torch.bfloat16},
        )
        self.assertEqual(
            model_cls.from_pretrained.call_args_list[1].args, (ESMC_MODEL,)
        )
        tokenizer_cls.from_pretrained.assert_called_once_with(ESMC_MODEL)
        self.assertIs(result, fallback.to.return_value)
        self.assertIs(tokenizer, tokenizer_cls.from_pretrained.return_value)

    def test_legacy_class_names_are_still_accepted(self):
        native = types.ModuleType("esm.models.esmc")
        native.EsmcModel = Mock()
        native.EsmcTokenizer = Mock()
        with patch.dict(sys.modules, {"esm.models.esmc": native}):
            result, tokenizer = load_model("cpu")
        native.EsmcModel.from_pretrained.assert_called_once_with(
            ESMC_REGISTRY_NAME, device=torch.device("cpu")
        )
        native.EsmcTokenizer.from_pretrained.assert_called_once_with(ESMC_REGISTRY_NAME)
        self.assertIs(tokenizer, native.EsmcTokenizer.from_pretrained.return_value)

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
