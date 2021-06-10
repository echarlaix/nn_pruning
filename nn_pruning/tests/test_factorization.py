import unittest
from unittest import TestCase

from transformers import BertConfig, BertForQuestionAnswering

from nn_pruning.model_structure import BertStructure
from nn_pruning.modules.masked_nn import (
    FactorizationModulePatcher,
    ChannelPruningModulePatcher,
    FactorizationArgs,
    LinearPruningArgs,
    FactorizationPruningArgs,
)
from nn_pruning.training_patcher import LinearModelPatcher, PatcherContext


class TestFun(TestCase):
    MODEL_STRUCTURE = BertStructure

    def test_patch_factorization(self):
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertForQuestionAnswering(config)

        parameters = FactorizationArgs(
            method="disabled",
            submethod="factorization",
            rank=200,
        )

        context = PatcherContext()

        p = FactorizationModulePatcher(context, parameters, self.MODEL_STRUCTURE)

        module_patchers = dict(
            query=p,
            key=p,
            value=p,
            att_dense=p,
            interm_dense=p,
            output_dense=p,
        )

        patcher = LinearModelPatcher(module_patchers, self.MODEL_STRUCTURE)
        patcher.patch(model)

        self.assertEqual(patcher.stats["patched"], 72)
        key_sizes = {k: len(v) for k, v in context.context_modules.items()}

        self.assertEqual(key_sizes, {})

    def test_patch_factorization_pruning(self):
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertForQuestionAnswering(config)

        parameters = FactorizationPruningArgs(
            method="sigmoied_threshold",
            submethod="1d_factorization",
            ampere_method="disabled",
            block_rows=1,
            block_cols=1,
            min_elements=0.0,
            rank=200,
        )

        context = PatcherContext()

        p = FactorizationModulePatcher(context, parameters, self.MODEL_STRUCTURE)

        module_patchers = dict(
            query=p,
            key=p,
            value=p,
            att_dense=p,
            interm_dense=p,
            output_dense=p,
        )

        patcher = LinearModelPatcher(module_patchers, self.MODEL_STRUCTURE)
        patcher.patch(model)

        self.assertEqual(patcher.stats["patched"], 72)
        key_sizes = {k: len(v) for k, v in context.context_modules.items()}

        self.assertEqual(key_sizes, {"mask_1d": 72})

    def test_patch_hybrid_factorization_pruning(self):
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertForQuestionAnswering(config)

        parameters_attention = FactorizationPruningArgs(
            method="sigmoied_threshold",
            submethod="1d_factorization",
            ampere_method="disabled",
            block_rows=1,
            block_cols=1,
            min_elements=0.0,
            rank=200,
        )

        parameters_dense = LinearPruningArgs(
            method="sigmoied_threshold",
            submethod="1d_alt",
            ampere_method="disabled",
            block_rows=1,
            block_cols=1,
            min_elements=0.0,
        )

        context = PatcherContext()

        p_attention = FactorizationModulePatcher(context, parameters_attention, self.MODEL_STRUCTURE)
        p_dense = ChannelPruningModulePatcher(context, parameters_dense, self.MODEL_STRUCTURE, suffix="dense")

        module_patchers = dict(
            query=p_attention,
            key=p_attention,
            value=p_attention,
            att_dense=p_attention,
            interm_dense=p_dense,
            output_dense=p_dense,
        )

        patcher = LinearModelPatcher(module_patchers, self.MODEL_STRUCTURE)
        patcher.patch(model)

        self.assertEqual(patcher.stats["patched"], 72)
        key_sizes = {k: len(v) for k, v in context.context_modules.items()}

        for k, v in key_sizes.items():
            print(k, v)

        for k, v in context.context_modules.items():
            print(k, v)
        self.assertEqual(key_sizes, {"mask_1d": 60})


if __name__ == "__main__":
    unittest.main()
