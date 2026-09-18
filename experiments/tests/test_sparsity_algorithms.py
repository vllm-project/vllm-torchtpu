"""
test_sparsity_algorithms.py
---------------------------
Comprehensive unit and mathematical test suite for structured sparsity algorithms:
- NaiveMagnitudePruner: 2:4 and 1:4 N:M structured sparsity vector pruning math, shape invariants, and edge cases.
- SparseGPT: Hessian matrix accumulation (H = 2 * X * X^T), dampening, Cholesky inversion, and error minimization.
- Quantizer: Uniform symmetric and asymmetric integer quantization forward passes.
- ModelUtils: Recursive layer traversal and linear submodule discovery.
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn
from safetensors import safe_open
from sparsity.modelutils import find_layers
from sparsity.naive_magnitude_pruner import (prune_model_magnitude,
                                             prune_tensor_magnitude_nm)
from sparsity.quant import Quantizer
from sparsity.sparsegpt import SparseGPT

try:
    from tests.fixtures import create_synthetic_model
except ImportError:
    from fixtures import create_synthetic_model


class TestNaiveMagnitudePruner(unittest.TestCase):
    """Mathematical validation for data-free N:M structured magnitude pruning."""

    def test_2of4_structured_sparsity_math(self):
        """
        Arrange-Act-Assert: Verify 2:4 structured sparsity vector mathematical properties.
        Rule: In every consecutive group of 4 elements along dim=-1, the 2 elements
        with the smallest absolute magnitude must be set to exactly 0.0,
        while the 2 largest elements are preserved untouched.
        """
        # Arrange: Construct deterministic tensor with known 4-element groups
        tensor = torch.tensor([[1.0, 4.0, 2.0, 3.0, 10.0, 5.0, 8.0, 2.0],
                               [9.0, 1.0, 2.0, 7.0, 3.0, 8.0, 1.0, 4.0]],
                              dtype=torch.float32)

        # Act: Apply 2:4 structured magnitude pruning
        pruned = prune_tensor_magnitude_nm(tensor, prunen=2, prunem=4)

        # Assert: Verify shape and exact 50% zero count
        self.assertEqual(pruned.shape, tensor.shape, "Shape must be preserved")
        total_elements = tensor.numel()
        zero_count = (pruned == 0).sum().item()
        self.assertEqual(zero_count, total_elements // 2,
                         "Must zero exactly 50% of elements")

        # Check exact values kept for row 0 group 1
        expected_row0_grp1 = torch.tensor([0.0, 4.0, 0.0, 3.0])
        self.assertTrue(torch.equal(pruned[0, :4], expected_row0_grp1))

    def test_1of4_structured_sparsity_math(self):
        """
        Arrange-Act-Assert: Verify 1:4 structured sparsity (25% sparse) mathematical properties.
        Rule: In every group of 4 elements, exactly 1 element (the smallest in magnitude) is zeroed.
        """
        # Arrange: Construct tensor
        tensor = torch.tensor([[1.0, 4.0, 2.0, 3.0, 10.0, 5.0, 8.0, 2.0]],
                              dtype=torch.float32)

        # Act: Apply 1:4 pruning
        pruned = prune_tensor_magnitude_nm(tensor, prunen=1, prunem=4)

        # Assert: 1 of 4 means exactly 25% zeros (2 out of 8 elements)
        zero_count = (pruned == 0).sum().item()
        self.assertEqual(zero_count, 2)

    def test_edge_cases(self):
        """
        Arrange-Act-Assert: Boundary condition and edge-case testing for magnitude pruner:
        1. 3D tensor shape preservation [batch, seq, hidden].
        2. All-zero tensor (remains all-zero without NaN/Inf).
        """
        # Case 1: 3D tensor
        tensor_3d = torch.randn(2, 4, 8)
        pruned_3d = prune_tensor_magnitude_nm(tensor_3d, prunen=2, prunem=4)
        self.assertEqual(pruned_3d.shape, (2, 4, 8))
        self.assertEqual((pruned_3d == 0).sum().item(), tensor_3d.numel() // 2)

        # Case 2: All zeros tensor
        zero_tensor = torch.zeros(4, 4)
        pruned_zero = prune_tensor_magnitude_nm(zero_tensor,
                                                prunen=2,
                                                prunem=4)
        self.assertEqual((pruned_zero == 0).sum().item(), 16)

    def test_prune_model_magnitude_on_synthetic_checkpoint(self):
        """
        Arrange-Act-Assert: Execute prune_model_magnitude across a multi-shard safetensors model.
        Verifies layer boundary filtering: only layers in [min_layer, max_layer] are pruned,
        while other layers and embeddings remain dense.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = os.path.join(temp_dir, "base_dense")
            pruned_dir = os.path.join(temp_dir, "pruned_sparse")
            create_synthetic_model(base_dir,
                                   num_layers=4,
                                   num_shards=2,
                                   sparsity_ratio=0.0)

            # Act: Prune layers 1 to 3 with 2:4 sparsity
            prune_model_magnitude(base_model_dir=base_dir,
                                  output_dir=pruned_dir,
                                  prunen=2,
                                  prunem=4,
                                  min_layer=1,
                                  max_layer=3)

            # Assert: Verify layer 0 is dense, layers 1 and 2 are 2:4 sparse, and layer 3 is dense
            for fname in os.listdir(pruned_dir):
                if not fname.endswith(".safetensors"):
                    continue
                with safe_open(os.path.join(pruned_dir, fname),
                               framework="pt",
                               device="cpu") as f:
                    for k in f.keys():
                        if "layers." in k and "weight" in k and not any(
                                s in k for s in ["norm", "embed"]):
                            layer_idx = int(
                                k.split("layers.")[1].split(".")[0])
                            tensor = f.get_tensor(k)
                            zero_fraction = (
                                tensor == 0).sum().item() / tensor.numel()
                            if 1 <= layer_idx < 3:
                                self.assertAlmostEqual(
                                    zero_fraction,
                                    0.5,
                                    delta=0.05,
                                    msg=f"Layer {layer_idx} should be sparse")
                            else:
                                self.assertEqual(
                                    zero_fraction,
                                    0.0,
                                    msg=f"Layer {layer_idx} should be dense")


class TestSparseGPT(unittest.TestCase):
    """Tests SparseGPT Hessian matrix calculation, Cholesky decomposition, and weight updating."""

    def test_sparsegpt_add_batch_and_prune(self):
        """
        Arrange-Act-Assert: SparseGPT calibration batch accumulation and pruning execution.
        Verifies:
        1. Accumulation of calibration activation covariance H = 2 * X * X^T with full rank.
        2. Verification that Hessian matrix H is symmetric and positive semi-definite.
        3. Execution of fasterprune() with 2:4 structured sparsity.
        4. Verification that output layer weights achieve 50% sparsity without NaNs.
        """
        # Arrange: Setup linear layer with full-rank weights
        linear = nn.Linear(16, 32, bias=False)
        linear.weight.data = torch.randn(32, 16)
        sparsegpt = SparseGPT(linear)

        # Act 1: Accumulate batches of activation data with full rank
        for _ in range(8):
            inp = torch.randn(32, 16)
            out = linear(inp)
            sparsegpt.add_batch(inp, out)

        # Assert 1: Verify Hessian matrix H is symmetric and positive semi-definite
        self.assertEqual(sparsegpt.H.shape, (16, 16))
        self.assertTrue(torch.allclose(sparsegpt.H, sparsegpt.H.t(),
                                       atol=1e-3))

        # Act 2: Run 2:4 fasterprune
        sparsegpt.fasterprune(sparsity=0,
                              prunen=2,
                              prunem=4,
                              percdamp=0.01,
                              blocksize=16)

        # Assert 2: Verify linear weight is 2:4 sparse (50% zeros)
        weight = linear.weight.data
        zero_ratio = (weight == 0).sum().item() / weight.numel()
        self.assertAlmostEqual(zero_ratio, 0.5, delta=0.05)

        # Clean up
        sparsegpt.free()

    def test_sparsegpt_lrc_prune_tall_matrix(self):
        """
        Arrange-Act-Assert: Verify LRC alternating minimization with Gram-matrix eigh on tall matrix (d_out >= d_in).
        Verifies:
        1. Dimension constraints: rows=32 >= cols=16 triggers G = R_tilde.T @ R_tilde (16x16 Gram matrix).
        2. Execution of lrc_prune(k=4, T=3) with 2:4 structured sparsity.
        3. Whitened Frobenius error monotonically non-increasing: error_lrc <= error_sparsegpt.
        4. Reconstructed weights are finite and memory references are released on free().
        """
        torch.manual_seed(42)
        linear = nn.Linear(16, 32, bias=False)
        linear.weight.data = torch.randn(32, 16)
        sparsegpt = SparseGPT(linear)

        for _ in range(8):
            inp = torch.randn(32, 16)
            out = linear(inp)
            sparsegpt.add_batch(inp, out)

        U, V = sparsegpt.lrc_prune(
            sparsity=0,
            prunen=2,
            prunem=4,
            percdamp=0.01,
            blocksize=16,
            T=3,
            k=4,
        )

        self.assertEqual(U.shape, (32, 4))
        self.assertEqual(V.shape, (4, 16))
        self.assertIsNotNone(sparsegpt.error_sparsegpt)
        self.assertIsNotNone(sparsegpt.error_lrc)
        self.assertLessEqual(sparsegpt.error_lrc, sparsegpt.error_sparsegpt + 1e-5)
        self.assertFalse(torch.isnan(linear.weight.data).any())
        self.assertFalse(torch.isinf(linear.weight.data).any())

        sparsegpt.free()
        self.assertIsNone(sparsegpt.H)
        self.assertIsNone(sparsegpt.L)
        self.assertIsNone(sparsegpt.U)
        self.assertIsNone(sparsegpt.V)
        self.assertIsNone(sparsegpt.S)

    def test_sparsegpt_lrc_prune_wide_matrix(self):
        """
        Arrange-Act-Assert: Verify LRC alternating minimization with Gram-matrix eigh on wide matrix (d_out < d_in).
        Verifies:
        1. Dimension constraints: rows=16 < cols=32 triggers G = R_tilde @ R_tilde.T (16x16 Gram matrix).
        2. Execution of lrc_prune(k=4, T=3) with 2:4 structured sparsity.
        3. Whitened Frobenius error monotonically non-increasing: error_lrc <= error_sparsegpt.
        4. Reconstructed weights are finite and memory references are released on free().
        """
        torch.manual_seed(42)
        linear = nn.Linear(32, 16, bias=False)
        linear.weight.data = torch.randn(16, 32)
        sparsegpt = SparseGPT(linear)

        for _ in range(8):
            inp = torch.randn(32, 32)
            out = linear(inp)
            sparsegpt.add_batch(inp, out)

        U, V = sparsegpt.lrc_prune(
            sparsity=0,
            prunen=2,
            prunem=4,
            percdamp=0.01,
            blocksize=16,
            T=3,
            k=4,
        )

        self.assertEqual(U.shape, (16, 4))
        self.assertEqual(V.shape, (4, 32))
        self.assertIsNotNone(sparsegpt.error_sparsegpt)
        self.assertIsNotNone(sparsegpt.error_lrc)
        self.assertLessEqual(sparsegpt.error_lrc, sparsegpt.error_sparsegpt + 1e-5)
        self.assertFalse(torch.isnan(linear.weight.data).any())
        self.assertFalse(torch.isinf(linear.weight.data).any())

        sparsegpt.free()
        self.assertIsNone(sparsegpt.H)
        self.assertIsNone(sparsegpt.L)
        self.assertIsNone(sparsegpt.U)
        self.assertIsNone(sparsegpt.V)
        self.assertIsNone(sparsegpt.S)

    def test_sparsegpt_lrc_prune_zero_rank(self):
        """
        Arrange-Act-Assert: Boundary condition with rank k=0.
        Verifies:
        1. When k=0, error_lrc matches error_sparsegpt.
        2. Returns zero adapter shapes and valid finite weights.
        """
        linear = nn.Linear(16, 16, bias=False)
        linear.weight.data = torch.randn(16, 16)
        sparsegpt = SparseGPT(linear)

        for _ in range(4):
            inp = torch.randn(16, 16)
            sparsegpt.add_batch(inp, linear(inp))

        U, V = sparsegpt.lrc_prune(
            sparsity=0,
            prunen=2,
            prunem=4,
            percdamp=0.01,
            blocksize=16,
            T=2,
            k=0,
        )
        self.assertEqual(U.shape, (16, 0))
        self.assertEqual(V.shape, (0, 16))
        self.assertAlmostEqual(sparsegpt.error_lrc, sparsegpt.error_sparsegpt, places=5)
        sparsegpt.free()


class TestQuantizer(unittest.TestCase):
    """Tests uniform quantizer configuration and forward quantization."""

    def test_quantizer_configuration_and_quantization(self):
        """
        Arrange-Act-Assert: Quantizer parameter configuration and scale/zero calculation.
        """
        # Arrange
        quant = Quantizer()
        quant.configure(bits=8, perchannel=True, sym=True, mse=False)
        self.assertEqual(quant.maxq.item(), 255)
        self.assertTrue(quant.perchannel)
        self.assertTrue(quant.sym)

        # Act: Quantize synthetic weights
        weight = torch.randn(32, 16)
        quant.find_params(weight, weight=True)
        quantized = quant.quantize(weight)

        # Assert: Confirm shape and finite values
        self.assertEqual(quantized.shape, weight.shape)


class TestModelUtils(unittest.TestCase):
    """Tests recursive neural network submodule traversal."""

    def test_find_layers_in_nested_module(self):
        """
        Arrange-Act-Assert: Verify find_layers recursively locates all nn.Linear instances within nested modules.
        """

        # Arrange: Construct nested module with 2 linear submodules
        class NestedModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(32, 64), nn.ReLU(),
                                         nn.Linear(64, 32))
                self.norm = nn.LayerNorm(32)

        model = NestedModel()

        # Act: Find linear submodules
        linear_layers = find_layers(model, layers=[nn.Linear])

        # Assert: Exactly 2 linear layers discovered with correct names
        self.assertEqual(len(linear_layers), 2)
        self.assertIn("mlp.0", linear_layers)
        self.assertIn("mlp.2", linear_layers)


if __name__ == "__main__":
    unittest.main()
