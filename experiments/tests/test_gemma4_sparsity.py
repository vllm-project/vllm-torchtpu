import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from sparsity.gemma4 import find_layers, gemma4_sequential
from sparsity.lrc import make_lrc_pruner


def _make_mock_gemma4():

    class MockDecoderLayer(nn.Module):

        def __init__(self):
            super().__init__()
            self.self_attn_q = nn.Linear(32, 32, bias=False)
            self.mlp_down = nn.Linear(32, 32, bias=False)
            self.self_attn_q.weight.data = torch.randn(32, 32)
            self.mlp_down.weight.data = torch.randn(32, 32)

        def forward(self, hidden_states, *args, **kwargs):
            return (self.mlp_down(self.self_attn_q(hidden_states)), )

    class MockLanguageModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [MockDecoderLayer(), MockDecoderLayer()])
            self.embed_tokens = nn.Embedding(100, 32)
            self.embed_tokens.weight.data = torch.randn(100, 32)
            self.unique_layer_types = ["full_attention"]
            self.config = types.SimpleNamespace(
                layer_types=["full_attention", "full_attention"],
                text_config=types.SimpleNamespace(hidden_size=32))

        def rotary_emb(self, *args, **kwargs):
            return torch.zeros((1, 32, 32))

    class MockInnerModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.language_model = MockLanguageModel()

    class MockGemmaModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = MockInnerModel()
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(hidden_size=32))
            self.seqlen = 32

        def forward(self, input_ids):
            x = self.model.language_model.embed_tokens(input_ids)
            return self.model.language_model.layers[0](x)

    return MockGemmaModel()


class TestGemma4Sparsity(unittest.TestCase):
    """Tests Gemma 4 model traversal and SparseGPT / LRC sequential pruning."""

    def test_find_layers_identifies_linear_projections(self):

        class DummyBlock(nn.Module):

            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.k_proj = nn.Linear(16, 16)
                self.norm = nn.LayerNorm(16)
                self.act = nn.GELU()

        found = find_layers(DummyBlock())
        self.assertEqual(set(found.keys()), {"q_proj", "k_proj"})

    def test_gemma4_sequential_pruning_execution(self):
        model = _make_mock_gemma4()
        dataloader = [[torch.arange(0, 32).unsqueeze(0)]]
        args = types.SimpleNamespace(prunen=2,
                                     prunem=4,
                                     sparsity=0,
                                     minlayer=0,
                                     maxlayer=2,
                                     nsamples=1,
                                     prune_only=None,
                                     percdamp=0.01,
                                     blocksize=32)

        with patch("transformers.models.gemma4.modeling_gemma4.create_causal_mask", return_value=None), \
             patch("transformers.models.gemma4.modeling_gemma4.create_sliding_window_causal_mask", return_value=None):
            gemma4_sequential(model,
                              dataloader,
                              dev=torch.device("cpu"),
                              args=args)

        for layer in model.model.language_model.layers:
            w = layer.self_attn_q.weight.data
            zero_ratio = (w == 0).sum().item() / w.numel()
            self.assertAlmostEqual(zero_ratio, 0.5, delta=0.05)

    def test_gemma4_sequential_lrc_execution(self):
        model = _make_mock_gemma4()
        dataloader = [[torch.arange(0, 32).unsqueeze(0)]]
        args = types.SimpleNamespace(prunen=2,
                                     prunem=4,
                                     sparsity=0,
                                     minlayer=0,
                                     maxlayer=2,
                                     nsamples=1,
                                     prune_only=None,
                                     percdamp=0.01,
                                     blocksize=32)
        prune_fn = make_lrc_pruner(prunen=2,
                                   prunem=4,
                                   blocksize=32,
                                   percdamp=0.01,
                                   T=2,
                                   k=4)

        with patch("transformers.models.gemma4.modeling_gemma4.create_causal_mask", return_value=None), \
             patch("transformers.models.gemma4.modeling_gemma4.create_sliding_window_causal_mask", return_value=None):
            gemma4_sequential(model,
                              dataloader,
                              dev=torch.device("cpu"),
                              args=args,
                              prune_fn=prune_fn)

        for layer in model.model.language_model.layers:
            self.assertFalse(torch.isnan(layer.self_attn_q.weight.data).any())
            self.assertFalse(torch.isinf(layer.self_attn_q.weight.data).any())


if __name__ == "__main__":
    unittest.main()
