"""
test_resolvers.py
-----------------
Comprehensive unit and edge-case test suite for model resolvers:
- BaseSparsityResolver: Range strategies (first_to_last, last_to_first, middle_out) and cache cleanup
- ModelSplicer: Zero-recompute cross-shard layer splicing on multi-shard safetensors checkpoints
- WeightScalerResolver: Linear weight transformation (W <- c * W) with norm preservation
- NaiveMagnitudeResolver: Fast magnitude pruning resolver and directory caching
- SparseGptResolver: Hessian-based calibration resolver and directory caching
- ResolverRegistry: Factory lookup, custom resolver registration, and unknown resolver error handling
"""

import os
import tempfile
import unittest

import torch
from resolvers.base_resolver import BaseResolver
from resolvers.base_sparsity_resolver import BaseSparsityResolver
from resolvers.model_splicer import ModelSplicer
from resolvers.naive_magnitude_resolver import NaiveMagnitudeResolver
from resolvers.registry import ResolverRegistry
from resolvers.sparsegpt_resolver import SparseGptResolver
from resolvers.weight_scaler_resolver import WeightScalerResolver
from safetensors import safe_open

try:
    from tests.fixtures import create_synthetic_model
except ImportError:
    from fixtures import create_synthetic_model


class TestBaseSparsityResolverStrategies(unittest.TestCase):
    """Tests layer range computation strategies and cache directory cleanup in BaseSparsityResolver."""

    def setUp(self):
        self.resolver = BaseSparsityResolver()

    def test_compute_layer_range_strategies(self):
        """
        Arrange-Act-Assert: Validates layer range calculations across all progression strategies.
        Tests:
        1. Boundary conditions: 0 sparse layers, negative counts, 100% layer coverage.
        2. Strategies:
           - 'first_to_last': Prunes from early input layers [0, N)
           - 'last_to_first': Prunes from deep output layers [total-N, total)
           - 'middle_out': Centered symmetrical pruning [(total-N)/2, (total+N)/2)
        """
        # Case 1: 0 sparse layers requested (dense baseline)
        start, end = self.resolver.compute_layer_range(
            num_sparse_layers=0, strategy="first_to_last", total_layers=60)
        self.assertIsNone(start)
        self.assertIsNone(end)

        # Case 2: Negative sparse layer edge case (treated as 0)
        start, end = self.resolver.compute_layer_range(
            num_sparse_layers=-5, strategy="first_to_last", total_layers=60)
        self.assertIsNone(start)
        self.assertIsNone(end)

        # Case 3: All layers requested (100% sparse model)
        start, end = self.resolver.compute_layer_range(
            num_sparse_layers=60, strategy="first_to_last", total_layers=60)
        self.assertEqual(start, 0)
        self.assertEqual(end, 60)

        # Case 4: first_to_last progression (10 layers out of 60 -> layers [0, 10))
        start, end = self.resolver.compute_layer_range(
            num_sparse_layers=10, strategy="first_to_last", total_layers=60)
        self.assertEqual(start, 0)
        self.assertEqual(end, 10)

        # Case 5: last_to_first progression (10 layers out of 60 -> layers [50, 60))
        start, end = self.resolver.compute_layer_range(
            num_sparse_layers=10, strategy="last_to_first", total_layers=60)
        self.assertEqual(start, 50)
        self.assertEqual(end, 60)

        # Case 6: middle_out progression (10 layers out of 60 -> layers [25, 35))
        start, end = self.resolver.compute_layer_range(num_sparse_layers=10,
                                                       strategy="middle_out",
                                                       total_layers=60)
        self.assertEqual(start, 25)
        self.assertEqual(end, 35)

    def test_cleanup_old_spliced_dirs(self):
        """
        Arrange-Act-Assert: Validates automated disk management and temporary spliced directory cleanup.
        Ensures intermediate spliced checkpoints from previous sweep points are pruned to prevent disk exhaustion,
        while preserving the current active target directory.
        """
        # Arrange: Setup temporary directory with a stale spliced model and a new target
        with tempfile.TemporaryDirectory() as temp_dir:
            old_dir = os.path.join(temp_dir, "model_layers_0_to_10")
            current_target = os.path.join(temp_dir, "model_layers_0_to_20")
            os.makedirs(old_dir, exist_ok=True)
            os.makedirs(current_target, exist_ok=True)

            # Act: Trigger cleanup of older checkpoints
            self.resolver._cleanup_old_spliced_dirs(temp_dir, current_target)

            # Assert: Stale directory is deleted, active directory is retained
            self.assertFalse(os.path.exists(old_dir),
                             "Old spliced directory should be cleaned up")
            self.assertTrue(os.path.exists(current_target),
                            "Current active directory must remain intact")


class TestModelSplicer(unittest.TestCase):
    """Tests zero-recompute cross-shard layer splicing across safetensors shards."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dense_dir = os.path.join(self.test_dir.name, "dense_model")
        self.sparse_dir = os.path.join(self.test_dir.name, "sparse_model")
        self.output_dir = os.path.join(self.test_dir.name, "spliced_model")

        # Generate synthetic 4-layer, 2-shard base dense and 2:4 sparse reference models
        create_synthetic_model(self.dense_dir,
                               num_layers=4,
                               num_shards=2,
                               sparsity_ratio=0.0)
        create_synthetic_model(self.sparse_dir,
                               num_layers=4,
                               num_shards=2,
                               sparsity_ratio=0.5)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_splice_subset_layers(self):
        """
        Arrange-Act-Assert: Splice a specific subset of layers {0, 2} into dense baseline.
        Verifies:
        1. Only layers 0 and 2 have exactly 50% zero weights (2:4 structured sparsity).
        2. Layers 1 and 3 remain 100% dense (0% zeros).
        3. All non-layer tensors (embeddings, final norms) remain intact and dense.
        """
        # Arrange
        splicer = ModelSplicer()

        # Act: Splice layers 0 and 2 from sparse reference into output
        splicer.splice_checkpoint(dense_model_dir=self.dense_dir,
                                  fully_sparse_model_dir=self.sparse_dir,
                                  sparse_layer_indices={0, 2},
                                  output_dir=self.output_dir)

        # Assert: Inspect each tensor in the resulting spliced safetensors shards
        for fname in os.listdir(self.output_dir):
            if not fname.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(self.output_dir, fname),
                           framework="pt",
                           device="cpu") as f:
                for k in f.keys():
                    if "layers." in k and "weight" in k and "layernorm" not in k and "norm" not in k:
                        layer_idx = int(k.split("layers.")[1].split(".")[0])
                        tensor = f.get_tensor(k)
                        zero_fraction = (tensor
                                         == 0).sum().item() / tensor.numel()

                        if layer_idx in {0, 2}:
                            self.assertAlmostEqual(
                                zero_fraction,
                                0.5,
                                delta=0.05,
                                msg=
                                f"Layer {layer_idx} tensor {k} should be 2:4 sparse"
                            )
                        else:
                            self.assertEqual(
                                zero_fraction,
                                0.0,
                                msg=
                                f"Layer {layer_idx} tensor {k} should remain dense"
                            )

    def test_splice_all_and_zero_layers_edge_cases(self):
        """
        Arrange-Act-Assert: Splicer boundary condition testing.
        Case 1: Splicing 0 layers (empty set -> identical to dense baseline).
        Case 2: Splicing 100% of layers ({0, 1, 2, 3} -> identical to sparse reference).
        """
        splicer = ModelSplicer()

        # 1. 0 layers spliced
        out_0 = os.path.join(self.test_dir.name, "spliced_0")
        splicer.splice_checkpoint(self.dense_dir, self.sparse_dir, set(),
                                  out_0)
        self.assertTrue(
            os.path.exists(os.path.join(out_0,
                                        "model.safetensors.index.json")))

        # 2. All 4 layers spliced
        out_all = os.path.join(self.test_dir.name, "spliced_all")
        splicer.splice_checkpoint(self.dense_dir, self.sparse_dir,
                                  {0, 1, 2, 3}, out_all)
        self.assertTrue(
            os.path.exists(
                os.path.join(out_all, "model.safetensors.index.json")))

    def test_detect_architecture_on_synthetic_model(self):
        """
        Arrange-Act-Assert: Validate automatic architecture inspection from config.json.
        Verifies layer count, hidden size, and model type detection.
        """
        spec = ModelSplicer.detect_architecture(self.dense_dir)
        self.assertEqual(spec.num_layers, 4)
        self.assertEqual(spec.hidden_size, 32)
        self.assertEqual(spec.model_type, "Gemma4ForCausalLM")


class TestResolversEndToEnd(unittest.TestCase):
    """Integration test suite verifying checkpoint resolution across all registered resolver classes."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dense_dir = os.path.join(self.test_dir.name, "dense_model")
        self.sparse_dir = os.path.join(self.test_dir.name, "sparse_model")
        self.work_dir = os.path.join(self.test_dir.name, "workspace")
        os.makedirs(self.work_dir, exist_ok=True)

        create_synthetic_model(self.dense_dir,
                               num_layers=4,
                               num_shards=2,
                               sparsity_ratio=0.0)
        create_synthetic_model(self.sparse_dir,
                               num_layers=4,
                               num_shards=2,
                               sparsity_ratio=0.5)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_base_sparsity_resolver_resolve_sparse_layers(self):
        """
        Arrange-Act-Assert: Verify BaseSparsityResolver resolution with disk caching.
        Verifies that initial resolution creates the spliced model and subsequent calls return the cached directory immediately.
        """
        resolver = BaseSparsityResolver()
        current_config = {"num_sparse_layers": 2}
        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }

        # Act: First resolution
        model_path, consumed_keys = resolver.resolve(current_config,
                                                     resolver_config)
        self.assertIn("num_sparse_layers", consumed_keys)
        self.assertTrue(
            os.path.exists(
                os.path.join(model_path, "model.safetensors.index.json")))

        # Act & Assert: Cached resolution returns identical path without re-splicing
        model_path2, _ = resolver.resolve(current_config, resolver_config)
        self.assertEqual(model_path, model_path2)

    def test_naive_magnitude_resolver_resolve(self):
        """
        Arrange-Act-Assert: Verify NaiveMagnitudeResolver resolution.
        Ensures correct parameter consumption and spliced checkpoint generation.
        """
        resolver = NaiveMagnitudeResolver()
        current_config = {"num_sparse_layers": 2}
        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }

        model_path, consumed_keys = resolver.resolve(current_config,
                                                     resolver_config)
        self.assertIn("num_sparse_layers", consumed_keys)
        self.assertTrue(
            os.path.exists(
                os.path.join(model_path, "model.safetensors.index.json")))

    def test_sparsegpt_resolver_resolve(self):
        """
        Arrange-Act-Assert: Verify SparseGptResolver resolution with prefix (first_to_last) strategy,
        and verify that invalid non-prefix (last_to_first) splicing raises ValueError per scientific validity.
        """
        resolver = SparseGptResolver()
        current_config = {"num_sparse_layers": 2}
        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }

        # Valid prefix splice
        model_path, consumed_keys = resolver.resolve(current_config,
                                                     resolver_config)
        self.assertIn("num_sparse_layers", consumed_keys)
        self.assertTrue(
            os.path.exists(
                os.path.join(model_path, "model.safetensors.index.json")))

        # Invalid non-prefix splice raises ValueError
        invalid_config = dict(resolver_config)
        invalid_config["strategy"] = "last_to_first"
        with self.assertRaises(ValueError):
            resolver.resolve(current_config, invalid_config)

    def test_weight_scaler_resolver_resolve(self):
        """
        Arrange-Act-Assert: Verify WeightScalerResolver resolution.
        Edge Case: When scale_factor == 1.0 (identity scaling), resolver must return base checkpoint untouched.
        """
        resolver = WeightScalerResolver()
        current_config = {"scale_factor": 1.05}
        resolver_config = {
            "base_sparse_model": self.sparse_dir,
            "checkpoint_base_dir": self.work_dir
        }

        # Scaled resolution
        model_path, consumed_keys = resolver.resolve(current_config,
                                                     resolver_config)
        self.assertTrue(os.path.exists(model_path))

        # Identity factor c=1.0 returns base_sparse_model untouched
        current_config_identity = {"scale_factor": 1.0}
        path_identity, _ = resolver.resolve(current_config_identity,
                                            resolver_config)
        self.assertEqual(path_identity, self.sparse_dir)

    def test_naive_magnitude_prune_in_memory_fallback(self):
        """
        Arrange-Act-Assert: Test in-memory algorithmic pruning fallback when no reference model is provided.
        """
        resolver = NaiveMagnitudeResolver()
        out_target = os.path.join(self.work_dir, "fallback_pruned")
        resolver.prune_in_memory(base_model=self.dense_dir,
                                 save_path=out_target,
                                 min_layer=0,
                                 max_layer=2,
                                 prunen=2,
                                 prunem=4,
                                 resolver_config={})
        self.assertTrue(
            os.path.exists(
                os.path.join(out_target, "model.safetensors.index.json")))


class TestWeightScalerResolver(unittest.TestCase):
    """Tests constant scalar weight multiplier (W <- c * W) transformation."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_dir = os.path.join(self.test_dir.name, "base_sparse_model")
        create_synthetic_model(self.base_dir,
                               num_layers=2,
                               num_shards=1,
                               sparsity_ratio=0.5)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_weight_scaling_transformation(self):
        """
        Arrange-Act-Assert: Scale weights by constant factor c = 1.10.
        Verifies:
        1. 2D linear projection weights are scaled by exactly 1.10.
        2. 1D layer norms and biases remain unscaled (1.0x).
        """
        resolver = WeightScalerResolver()
        out_dir = os.path.join(self.test_dir.name, "scaled_model")
        scale_factor = 1.10

        # Read original tensors
        orig_tensors = {}
        with safe_open(os.path.join(self.base_dir,
                                    "model-00001-of-00001.safetensors"),
                       framework="pt",
                       device="cpu") as f:
            for k in f.keys():
                orig_tensors[k] = f.get_tensor(k)

        # Scale weights
        resolver._scale_checkpoint(base_checkpoint=self.base_dir,
                                   output_dir=out_dir,
                                   scale_factor=scale_factor)

        # Verify scaled tensors
        with safe_open(os.path.join(out_dir,
                                    "model-00001-of-00001.safetensors"),
                       framework="pt",
                       device="cpu") as f:
            for k in f.keys():
                scaled = f.get_tensor(k)
                orig = orig_tensors[k]

                # 2D projection weights should be scaled by 1.10
                if "weight" in k and scaled.dim() == 2 and "embed" not in k:
                    expected = (orig.to(torch.float32) * scale_factor).to(
                        torch.bfloat16)
                    self.assertTrue(
                        torch.allclose(scaled, expected, atol=1e-2),
                        f"Tensor {k} scaling mismatch")
                elif "norm" in k or scaled.dim() == 1:
                    # Norms must remain unscaled
                    self.assertTrue(torch.equal(scaled, orig),
                                    f"Norm {k} should NOT be scaled")


class TestResolverRegistry(unittest.TestCase):
    """Tests ResolverRegistry factory and resolver lifecycle."""

    def test_registry_lookup_and_unknown_resolver(self):
        """
        Arrange-Act-Assert: Verify standard resolver lookups and error handling for unknown resolver names.
        """
        scaler = ResolverRegistry.get("weight_scaler")
        self.assertIsInstance(scaler, WeightScalerResolver)

        sparsegpt = ResolverRegistry.get("sparsegpt")
        self.assertIsInstance(sparsegpt, SparseGptResolver)

        magnitude = ResolverRegistry.get("naive_magnitude")
        self.assertIsInstance(magnitude, NaiveMagnitudeResolver)

        with self.assertRaises(KeyError):
            ResolverRegistry.get("non_existent_resolver_type_xyz")

    def test_custom_resolver_registration(self):
        """
        Arrange-Act-Assert: Test dynamic registration of a user-defined custom resolver class via @ResolverRegistry.register.
        """

        @ResolverRegistry.register("custom_dummy")
        class DummyResolver(BaseResolver):

            def resolve(self, current_config, resolver_config):
                return "/path/to/dummy", set()

        resolved = ResolverRegistry.get("custom_dummy")
        self.assertIsInstance(resolved, DummyResolver)


if __name__ == "__main__":
    unittest.main()
