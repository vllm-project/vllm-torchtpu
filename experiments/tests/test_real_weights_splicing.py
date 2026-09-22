"""
test_real_weights_splicing.py
-----------------------------
Integration and verification test suite operating directly on real Gemma 4 31B-IT model weights:
- Extracts actual linear projection weights (Layer 0 and Layer 1) from real safetensors shards.
- Tests real-weight 2:4 structured magnitude pruning and verifies 50% sparsity without shape distortion.
- Tests real-weight layer splicing between real dense and real sparse Gemma 4 weights.
- Tests real-weight linear scaling (W <- c * W) on actual Gemma 4 projection matrices.
"""

import json
import os
import tempfile
import unittest

import torch
from resolvers.model_splicer import ModelSplicer
from safetensors import safe_open
from safetensors.torch import save_file
from sparsity.naive_magnitude_pruner import prune_tensor_magnitude_nm

try:
    from tests.fixtures import REAL_GEMMA4_PATH, has_real_gemma4_weights
except ImportError:
    from fixtures import REAL_GEMMA4_PATH, has_real_gemma4_weights


class TestRealGemma4WeightsSplicing(unittest.TestCase):
    """Executes tests on real Gemma 4 31B-IT weight shards when present on disk."""

    def setUp(self):
        # Gracefully skip tests if real Gemma 4 weights are not available in the current environment
        if not has_real_gemma4_weights():
            self.skipTest("Real Gemma 4 31B-IT weights not found on disk")
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_real_gemma4_layer0_magnitude_pruning(self):
        """
        Arrange-Act-Assert: Prune real Gemma 4 Layer 0 projection weights with 2:4 structured sparsity.
        Verifies:
        1. Loads an actual multi-megabyte Gemma 4 projection matrix (e.g., q_proj / down_proj) from disk.
        2. Applies vectorized 2:4 structured pruning.
        3. Validates that exactly 50% of weight elements are zeroed out while shape and dtype are preserved.
        """
        # Arrange: Locate first safetensors shard containing Layer 0
        shard_file = os.path.join(REAL_GEMMA4_PATH,
                                  "model-00001-of-00002.safetensors")
        self.assertTrue(os.path.exists(shard_file),
                        f"Shard {shard_file} should exist")

        with safe_open(shard_file, framework="pt", device="cpu") as f:
            # Find a real linear projection weight in the language model
            target_key = None
            for k in f.keys():
                if "language_model.layers.0" in k and "weight" in k and "norm" not in k:
                    target_key = k
                    break
            self.assertIsNotNone(
                target_key,
                "Language model Layer 0 linear weight not found in shard 1")
            real_tensor = f.get_tensor(target_key)

        # Act: Apply 2:4 structured pruning
        pruned_tensor = prune_tensor_magnitude_nm(real_tensor,
                                                  prunen=2,
                                                  prunem=4)

        # Assert: Shape and exact 50% zero-fraction
        self.assertEqual(pruned_tensor.shape, real_tensor.shape,
                         "Tensor shape must remain unchanged")
        zero_fraction = (pruned_tensor
                         == 0).sum().item() / pruned_tensor.numel()
        self.assertAlmostEqual(
            zero_fraction,
            0.5,
            delta=0.01,
            msg="Must achieve exactly 50% structured sparsity")

    def test_real_gemma4_layer0_weight_scaling(self):
        """
        Arrange-Act-Assert: Scale real Gemma 4 Layer 0 weights by constant multiplier c = 1.05.
        Verifies:
        1. Multiplies real projection matrix by 1.05 in bfloat16.
        2. Validates that the tensor mean scales proportionally by 1.05x.
        """
        # Arrange: Load real Layer 0 weight
        shard_file = os.path.join(REAL_GEMMA4_PATH,
                                  "model-00001-of-00002.safetensors")
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "language_model.layers.0" in k and "weight" in k and "norm" not in k:
                    real_tensor = f.get_tensor(k)
                    break

        # Act: Scale weights by c = 1.05
        scale_factor = 1.05
        scaled_tensor = (real_tensor.to(torch.float32) * scale_factor).to(
            torch.bfloat16)

        # Assert: Check shape and statistical scaling
        self.assertEqual(scaled_tensor.shape, real_tensor.shape)
        self.assertAlmostEqual(
            scaled_tensor.float().mean().item(),
            real_tensor.float().mean().item() * scale_factor,
            delta=0.05,
            msg="Mean tensor magnitude should scale linearly by 1.05x")

    def test_real_gemma4_layer_splicing_mock_shard(self):
        """
        Arrange-Act-Assert: Test ModelSplicer on a real 2-layer slice of Gemma 4 weights.
        Verifies:
        1. Constructs mini dense and 2:4 sparse checkpoints extracted from real Gemma 4 weights.
        2. Splices only Layer 0 into the destination directory.
        3. Confirms Layer 0 projection weights are 50% sparse while Layer 1 remains 100% dense.
        """
        # Arrange: Setup directories and extract real layer 0 & 1 tensors
        dense_mini = os.path.join(self.test_dir.name, "dense_mini")
        sparse_mini = os.path.join(self.test_dir.name, "sparse_mini")
        spliced_out = os.path.join(self.test_dir.name, "spliced_mini")
        os.makedirs(dense_mini, exist_ok=True)
        os.makedirs(sparse_mini, exist_ok=True)

        shard_file = os.path.join(REAL_GEMMA4_PATH,
                                  "model-00001-of-00002.safetensors")
        dense_tensors = {}
        sparse_tensors = {}
        weight_map = {}

        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "language_model.layers.0." in k or "language_model.layers.1." in k:
                    tensor = f.get_tensor(k)
                    dense_tensors[k] = tensor
                    if "weight" in k and "norm" not in k and tensor.dim() == 2:
                        sparse_tensors[k] = prune_tensor_magnitude_nm(tensor,
                                                                      prunen=2,
                                                                      prunem=4)
                    else:
                        sparse_tensors[k] = tensor
                    weight_map[k] = "model-00001-of-00001.safetensors"

        # Save mini safetensors shards and index JSON
        save_file(dense_tensors,
                  os.path.join(dense_mini, "model-00001-of-00001.safetensors"))
        save_file(
            sparse_tensors,
            os.path.join(sparse_mini, "model-00001-of-00001.safetensors"))

        config_path = os.path.join(REAL_GEMMA4_PATH, "config.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)
        config_data["num_hidden_layers"] = 2
        with open(os.path.join(dense_mini, "config.json"), "w") as f:
            json.dump(config_data, f)
        with open(os.path.join(sparse_mini, "config.json"), "w") as f:
            json.dump(config_data, f)

        with open(os.path.join(dense_mini, "model.safetensors.index.json"),
                  "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
        with open(os.path.join(sparse_mini, "model.safetensors.index.json"),
                  "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)

        # Act: Splice Layer 0 from sparse checkpoint into dense base
        splicer = ModelSplicer()
        splicer.splice_checkpoint(dense_model_dir=dense_mini,
                                  fully_sparse_model_dir=sparse_mini,
                                  sparse_layer_indices={0},
                                  output_dir=spliced_out)

        # Assert: Verify Layer 0 is 2:4 sparse and Layer 1 is dense
        with safe_open(os.path.join(spliced_out,
                                    "model-00001-of-00001.safetensors"),
                       framework="pt",
                       device="cpu") as f:
            for k in f.keys():
                if "weight" in k and "norm" not in k and "embed" not in k:
                    tensor = f.get_tensor(k)
                    zero_fraction = (tensor == 0).sum().item() / tensor.numel()
                    if "language_model.layers.0." in k:
                        self.assertAlmostEqual(
                            zero_fraction,
                            0.5,
                            delta=0.05,
                            msg=f"Real layer 0 tensor {k} should be sparse")
                    elif "language_model.layers.1." in k:
                        self.assertEqual(
                            zero_fraction,
                            0.0,
                            msg=f"Real layer 1 tensor {k} should be dense")


if __name__ == "__main__":
    unittest.main()
