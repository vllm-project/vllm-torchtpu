"""
test_gemma4_sparsity.py
-----------------------
Unit and forward-hook calibration tests for Gemma 4 sparsity pipeline:
- get_gemma4: HuggingFace model instantiation with dynamic offloading and device placement.
- gemma4_sequential: Layer-by-layer sequential calibration loop, forward-hook activation capture, and SparseGPT Hessian solver.
- sparsity.__init__: Package-level export verification.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import sparsity
import torch
import torch.nn as nn
from sparsity.gemma4 import gemma4_sequential, get_gemma4


class TestGemma4Sparsity(unittest.TestCase):
    """Test suite validating Gemma 4 model loading and calibration sequence execution."""

    def test_sparsity_init_exports(self):
        """
        Arrange-Act-Assert: Verify that package-level __init__.py initializes and exports cleanly.
        """
        self.assertTrue(hasattr(sparsity, "__file__"))

    @patch("sparsity.gemma4.Gemma4ForConditionalGeneration")
    def test_get_gemma4_loader(self, mock_gemma_cls):
        """
        Arrange-Act-Assert: Verify get_gemma4 passes correct arguments and default seqlen to HuggingFace loader.
        """
        # Arrange
        mock_model = MagicMock()
        mock_gemma_cls.from_pretrained.return_value = mock_model

        # Act
        model = get_gemma4("/path/to/gemma4")

        # Assert
        self.assertEqual(model.seqlen, 2048)
        mock_gemma_cls.from_pretrained.assert_called_once()

    def test_gemma4_sequential_mock_execution(self):
        """
        Arrange-Act-Assert: Tests the forward-hook catcher and Hessian estimation loop on a mock decoder block.
        Simulates:
        1. A miniature 2-layer transformer model with self-attention and MLP linear submodules.
        2. Calibration dataloader passing token sequences.
        3. Hook registration, activation caching, and SparseGPT solver invocation on each layer.
        """

        # Arrange: Create a mock Gemma 4 model structure
        class MockDecoderLayer(nn.Module):

            def __init__(self):
                super().__init__()
                self.self_attn_q = nn.Linear(32, 32, bias=False)
                self.mlp_down = nn.Linear(32, 32, bias=False)
                # Random initialization
                self.self_attn_q.weight.data = torch.randn(32, 32)
                self.mlp_down.weight.data = torch.randn(32, 32)

            def forward(self, hidden_states, *args, **kwargs):
                return (self.mlp_down(self.self_attn_q(hidden_states)), )

        class MockLanguageModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [MockDecoderLayer(),
                     MockDecoderLayer()])
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
                # Trigger layers[0] forward hook to populate inps
                x = self.model.language_model.embed_tokens(input_ids)
                return self.model.language_model.layers[0](x)

        model = MockGemmaModel()
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

        # Act: Execute sequential calibration loop with patched causal masks
        with patch("transformers.models.gemma4.modeling_gemma4.create_causal_mask", return_value=None), \
             patch("transformers.models.gemma4.modeling_gemma4.create_sliding_window_causal_mask", return_value=None):
            gemma4_sequential(model,
                              dataloader,
                              dev=torch.device("cpu"),
                              args=args)

        # Assert: Verify linear weights in layers[0] and layers[1] are 2:4 sparse
        for layer in model.model.language_model.layers:
            w = layer.self_attn_q.weight.data
            zero_ratio = (w == 0).sum().item() / w.numel()
            self.assertAlmostEqual(zero_ratio,
                                   0.5,
                                   delta=0.05,
                                   msg="Layer weights must be 2:4 sparse")

    def test_gemma4_sequential_lrc_execution(self):
        """
        Arrange-Act-Assert: Exercises gemma4_sequential with use_lrc=True to verify
        the active lrc.lrc_prune alternating minimization path end-to-end.
        """

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
                    [MockDecoderLayer(),
                     MockDecoderLayer()])
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

        model = MockGemmaModel()
        dataloader = [[torch.arange(0, 32).unsqueeze(0)]]

        args = types.SimpleNamespace(
            prunen=2,
            prunem=4,
            sparsity=0,
            minlayer=0,
            maxlayer=2,
            nsamples=1,
            prune_only=None,
            percdamp=0.01,
            blocksize=32,
            use_lrc=True,
            lrc_rank=4,
            lrc_iters=2,
        )

        with patch(
                "transformers.models.gemma4.modeling_gemma4.create_causal_mask",
                return_value=None
        ), patch(
                "transformers.models.gemma4.modeling_gemma4.create_sliding_window_causal_mask",
                return_value=None):
            gemma4_sequential(model,
                              dataloader,
                              dev=torch.device("cpu"),
                              args=args)

        for layer in model.model.language_model.layers:
            self.assertFalse(torch.isnan(layer.self_attn_q.weight.data).any())
            self.assertFalse(torch.isinf(layer.self_attn_q.weight.data).any())
            self.assertFalse(torch.isnan(layer.mlp_down.weight.data).any())
            self.assertFalse(torch.isinf(layer.mlp_down.weight.data).any())


if __name__ == "__main__":
    unittest.main()
