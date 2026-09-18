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
import shutil
import tempfile
import unittest
from unittest.mock import patch

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

    def test_assert_spliceable_table(self):
        """
        Table test for the prefix-only splice guard. Prefix sets (contiguous
        and starting at layer 0), the empty set and the full stack are
        accepted; any non-zero-start or gapped set raises unless
        allow_nonprefix_splice is set.
        """
        ok = [
            ("prefix", {0, 1, 2}, {}),
            ("empty", set(), {}),
            ("full_stack", set(range(60)), {}),
            ("bypass", {2, 3}, {
                "allow_nonprefix_splice": True
            }),
        ]
        for name, indices, cfg in ok:
            with self.subTest(name=name):
                self.resolver._assert_spliceable(indices, name, cfg)

        bad = [
            ("last_to_first", {1, 2, 3}, r"\[1, 2, 3\]"),
            ("gapped", {0, 1, 3}, r"\[0, 1, 3\]"),
            ("middle_out", set(range(25, 35)), r"\[25, 26, .*34\]"),
        ]
        for name, indices, pattern in bad:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, pattern):
                    self.resolver._assert_spliceable(indices, name, {})

    def test_cleanup_old_spliced_dirs(self):
        """
        Arrange-Act-Assert: Validates automated disk management and temporary spliced directory cleanup.
        Ensures intermediate spliced checkpoints from previous sweep points (both prefix and non-prefix)
        are pruned to prevent disk exhaustion, while preserving the current active target directory.
        """
        # Arrange: Setup temporary directory with a stale spliced model, stale non-prefix model, and a new target
        with tempfile.TemporaryDirectory() as temp_dir:
            old_dir = os.path.join(temp_dir, "model_layers_0_to_10")
            old_nonprefix_dir = os.path.join(
                temp_dir, "gemma4_sparsegpt_2of4_256c4_layers_14l_abcdef12")
            current_target = os.path.join(temp_dir, "model_layers_0_to_20")
            os.makedirs(old_dir, exist_ok=True)
            os.makedirs(old_nonprefix_dir, exist_ok=True)
            os.makedirs(current_target, exist_ok=True)

            # Act: Trigger cleanup of older checkpoints
            self.resolver._cleanup_old_spliced_dirs(temp_dir, current_target)

            # Assert: Stale directories are deleted, active directory is retained
            self.assertFalse(os.path.exists(old_dir),
                             "Old spliced directory should be cleaned up")
            self.assertFalse(
                os.path.exists(old_nonprefix_dir),
                "Old non-prefix calibrated directory should be cleaned up")
            self.assertTrue(os.path.exists(current_target),
                            "Current active directory must remain intact")

    def test_cleanup_old_spliced_dirs_real_listing(self):
        """
        Validates that _cleanup_old_spliced_dirs correctly purges all temporary checkpoint shapes:
          1. <n>layers_<hash> (e.g. gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_881bb184)
          2. <n>layers [no hash] (e.g. gemma4_unified_sparsegpt_2of4_256c4_custom_order_16layers)
          3. layers_<n>l_<hash> (e.g. gemma4_unified_sparsegpt_2of4_64c4_layers_3l_786ba7d1)
          4. Range/prefix spliced: *_layers_*_to_* (e.g. gemma4_sparsegpt_2of4_256c4_first_to_last_layers_0_to_14)
        while strictly PRESERVING:
          - Canonical reference checkpoints (*_canonical_*, *_full)
          - The current active target directory (even if matching one of the temporary patterns)
        """
        real_listing = [
            # Canonical reference checkpoints (MUST BE PRESERVED)
            "gemma4_12b_2of4_canonical_256c4_full",
            "gemma4_12b_2of4_canonical_256c4_prefix24",
            "gemma4_12b_2of4_spectral_entropy_16l",
            "gemma4_12b_lrc_k128_T3_256c4_full",
            "gemma4_12b_lrc_k128_T3_256c4_prefix24",
            "gemma4_26b_a4b_2of4_canonical_256c4_full",
            "gemma4_2of4_scaled_c_1p050",
            "gemma4_31b_2of4_canonical_256c4_full",
            "gemma4_31b_lrc_k128_T3_256c4_full",
            "test_l0",
            # Temporary checkpoints to be reaped:
            # Shape 1: Range / prefix spliced
            "gemma4_sparsegpt_2of4_256c4_first_to_last_layers_0_to_14",
            # Shape 2: <n>layers (no hash)
            "gemma4_unified_sparsegpt_2of4_256c4_custom_order_16layers",
            # Shape 3: <n>layers_<hash>
            "gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_881bb184",
            "gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_e5b47bdd",
            "gemma4_unified_sparsegpt_2of4_256c4_custom_order_8layers_2523ac16",
            "gemma4_unified_sparsegpt_2of4_256c4_custom_order_8layers_40e40e3c",
            "gemma4_unified_sparsegpt_2of4_64c4_custom_order_16layers_6783bc5d",
            "gemma4_unified_sparsegpt_2of4_64c4_custom_order_16layers_9331b82c",
            # Shape 4: layers_<n>l_<hash>
            "gemma4_unified_sparsegpt_2of4_64c4_layers_3l_786ba7d1",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for item in real_listing:
                os.makedirs(os.path.join(temp_dir, item), exist_ok=True)

            # Designate one of the temporary custom_order checkpoints as current_target
            current_target = os.path.join(
                temp_dir,
                "gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_e5b47bdd"
            )

            self.resolver._cleanup_old_spliced_dirs(temp_dir, current_target)

            remaining = set(os.listdir(temp_dir))

            # Verify current target is preserved
            self.assertIn(
                "gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_e5b47bdd",
                remaining)

            # Verify all 10 reference / canonical models are preserved
            canonical_refs = [
                "gemma4_12b_2of4_canonical_256c4_full",
                "gemma4_12b_2of4_canonical_256c4_prefix24",
                "gemma4_12b_2of4_spectral_entropy_16l",
                "gemma4_12b_lrc_k128_T3_256c4_full",
                "gemma4_12b_lrc_k128_T3_256c4_prefix24",
                "gemma4_26b_a4b_2of4_canonical_256c4_full",
                "gemma4_2of4_scaled_c_1p050",
                "gemma4_31b_2of4_canonical_256c4_full",
                "gemma4_31b_lrc_k128_T3_256c4_full",
                "test_l0",
            ]
            for ref in canonical_refs:
                self.assertIn(
                    ref, remaining,
                    f"Canonical reference {ref} must never be reaped")

            # Verify all other temporary checkpoints are reaped
            reaped_expected = [
                "gemma4_sparsegpt_2of4_256c4_first_to_last_layers_0_to_14",
                "gemma4_unified_sparsegpt_2of4_256c4_custom_order_16layers",
                "gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_881bb184",
                "gemma4_unified_sparsegpt_2of4_256c4_custom_order_8layers_2523ac16",
                "gemma4_unified_sparsegpt_2of4_256c4_custom_order_8layers_40e40e3c",
                "gemma4_unified_sparsegpt_2of4_64c4_custom_order_16layers_6783bc5d",
                "gemma4_unified_sparsegpt_2of4_64c4_custom_order_16layers_9331b82c",
                "gemma4_unified_sparsegpt_2of4_64c4_layers_3l_786ba7d1",
            ]
            for r in reaped_expected:
                self.assertNotIn(
                    r, remaining,
                    f"Temporary checkpoint {r} should have been cleaned up")


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

        # Verify module-level resolve_model entrypoint
        from resolvers.sparsegpt_resolver import resolve_model
        fn_path, fn_keys = resolve_model(current_config, resolver_config)
        self.assertEqual(model_path, fn_path)
        self.assertEqual(consumed_keys, fn_keys)

        # Invalid non-prefix splice raises ValueError ...
        invalid_config = dict(resolver_config)
        invalid_config["strategy"] = "last_to_first"
        with self.assertRaisesRegex(ValueError, r"\[2, 3\]"):
            resolver.resolve(current_config, invalid_config)

        # ... and must NOT have deleted the valid cached splice on its way
        # to raising (guard runs before _cleanup_old_spliced_dirs).
        self.assertTrue(
            os.path.exists(
                os.path.join(model_path, "model.safetensors.index.json")),
            "invalid config must not delete valid cached splices")

    def test_splice_guard_end_to_end(self):
        """
        Table test of the splice guard through resolve() on the 4-layer
        safetensors fixture (no mocks): prefix / empty / full-stack resolve,
        last_to_first and middle_out raise, allow_nonprefix_splice bypasses.
        """
        resolver = BaseSparsityResolver()
        base_cfg = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "checkpoint_base_dir": self.work_dir,
        }
        ok = [
            ("prefix_{0,1,2}", "first_to_last", 3, {}),
            ("empty", "first_to_last", 0, {}),
            ("full_stack", "first_to_last", 4, {}),
            ("middle_out_full", "middle_out", 4, {}),
            ("bypass_{2,3}", "last_to_first", 2, {
                "allow_nonprefix_splice": True
            }),
        ]
        for name, strategy, n, extra in ok:
            with self.subTest(name=name):
                cfg = dict(base_cfg, strategy=strategy, **extra)
                path, _ = resolver.resolve({"num_sparse_layers": n}, cfg)
                self.assertTrue(os.path.isdir(path), name)

        bad = [
            ("last_to_first_{1,2,3}", "last_to_first", 3, r"\[1, 2, 3\]"),
            ("middle_out_{1,2}", "middle_out", 2, r"\[1, 2\]"),
        ]
        for name, strategy, n, pattern in bad:
            with self.subTest(name=name):
                cfg = dict(base_cfg, strategy=strategy)
                with self.assertRaisesRegex(ValueError, pattern):
                    resolver.resolve({"num_sparse_layers": n}, cfg)

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

    def test_sparsegpt_resolver_outbound_signature_guard(self):
        """
        Contract Guard Test: Validates that all outbound arguments dispatched by
        SparseGptResolver.prune_in_memory bind cleanly against prune_gemma4's real
        inspect.signature without parameter divergence or unexpected kwargs.
        """
        import inspect

        from sparsity import prune_gemma4

        sig = inspect.signature(prune_gemma4)
        intercepted_calls = []

        def signature_verifier_spy(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            intercepted_calls.append(bound)
            return "/tmp/stub_output_dir"

        with patch("resolvers.sparsegpt_resolver.prune_gemma4",
                   side_effect=signature_verifier_spy):
            resolver = SparseGptResolver()
            resolver.prune_in_memory(
                base_model="/mock/base",
                save_path="/mock/save",
                min_layer=0,
                max_layer=4,
                prunen=2,
                prunem=4,
                resolver_config={
                    "dataset": "c4",
                    "nsamples": 32,
                    "seed": 42,
                    "percdamp": 0.02,
                    "blocksize": 64,
                    "use_lrc": True,
                    "lrc_rank": 64,
                    "lrc_iters": 5,
                    "save_prefix24_path": "/mock/prefix24",
                },
                sparse_layer_indices={1, 3},
            )

        self.assertEqual(len(intercepted_calls), 1)
        bound = intercepted_calls[0]
        self.assertEqual(bound.arguments["model_path"], "/mock/base")
        self.assertEqual(bound.arguments["dataset"], "c4")
        self.assertEqual(bound.arguments["minlayer"], 0)
        self.assertEqual(bound.arguments["maxlayer"], 4)
        self.assertEqual(bound.arguments["prunen"], 2)
        self.assertEqual(bound.arguments["prunem"], 4)
        self.assertEqual(bound.arguments["nsamples"], 32)
        self.assertEqual(bound.arguments["seed"], 42)
        self.assertEqual(bound.arguments["percdamp"], 0.02)
        self.assertEqual(bound.arguments["blocksize"], 64)
        self.assertEqual(bound.arguments["save_path"], "/mock/save")
        self.assertEqual(bound.arguments["sparse_layer_indices"], {1, 3})
        self.assertTrue(bound.arguments["use_lrc"])
        self.assertEqual(bound.arguments["lrc_rank"], 64)
        self.assertEqual(bound.arguments["lrc_iters"], 5)
        self.assertEqual(bound.arguments["save_prefix24_path"],
                         "/mock/prefix24")

    def test_sparsegpt_resolver_prune_in_memory_nonprefix_unmocked_boundary(
            self):
        """
        Arrange-Act-Assert: Exercises SparseGptResolver.prune_in_memory for a non-prefix layer set
        reaching prune_gemma4 with NO mock on the resolver->prune_gemma4 boundary.
        Executes real sequential SparseGPT calibration on a synthetic module, asserting that
        layer 1 is 2:4 pruned while layer 0 remains dense.
        """
        import types

        import torch.nn as nn

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

        mock_model = MockGemmaModel()
        mock_dataloader = [[torch.arange(0, 32).unsqueeze(0)]]

        # Verify initial state: both layers have 0% zeros
        w0_init = mock_model.model.language_model.layers[
            0].self_attn_q.weight.data
        w1_init = mock_model.model.language_model.layers[
            1].self_attn_q.weight.data
        self.assertEqual((w0_init == 0).sum().item(), 0)
        self.assertEqual((w1_init == 0).sum().item(), 0)

        resolver = SparseGptResolver()
        resolver_config = {
            "dataset": "c4",
            "nsamples": 1,
            "seed": 42,
            "percdamp": 0.01,
            "blocksize": 32,
            "use_lrc": False,
        }

        # IMPORTANT: prune_gemma4 is UNMOCKED. The resolver calls the real prune_gemma4 function.
        # Only HuggingFace model loading, dataset streaming, and causal masks are patched to run fast on CPU.
        with patch("sparsity.gemma4.get_gemma4", return_value=mock_model), \
             patch("sparsity.get_loaders", return_value=(mock_dataloader, None)), \
             patch("transformers.models.gemma4.modeling_gemma4.create_causal_mask", return_value=None), \
             patch("transformers.models.gemma4.modeling_gemma4.create_sliding_window_causal_mask", return_value=None):
            out = resolver.prune_in_memory(
                base_model="mock_model_dir",
                save_path="",
                min_layer=0,
                max_layer=2,
                prunen=2,
                prunem=4,
                resolver_config=resolver_config,
                sparse_layer_indices={1},
            )
            self.assertEqual(out, "")

        # Layer 0: was not in sparse_layer_indices -> preserved 100% dense
        w0_post = mock_model.model.language_model.layers[
            0].self_attn_q.weight.data
        zero_ratio_0 = (w0_post == 0).sum().item() / w0_post.numel()
        self.assertEqual(zero_ratio_0, 0.0, "Layer 0 must remain dense")

        # Layer 1: was in sparse_layer_indices={1} -> 2:4 SparseGPT pruned
        w1_post = mock_model.model.language_model.layers[
            1].self_attn_q.weight.data
        zero_ratio_1 = (w1_post == 0).sum().item() / w1_post.numel()
        self.assertAlmostEqual(zero_ratio_1,
                               0.5,
                               delta=0.05,
                               msg="Layer 1 must be 2:4 sparse")

    def test_base_sparsity_resolver_layer_bounds_validation(self):
        """
        Arrange-Act-Assert: Verify strict layer bounds validation (0 <= layer_idx < total_layers).
        Prevents silent layer-dropping or out-of-bounds corruption.
        """
        resolver = BaseSparsityResolver()
        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }

        # Negative layer index must raise ValueError
        with self.assertRaises(ValueError):
            resolver.resolve({"layers": [-1, 1]}, resolver_config)

        # Out-of-bounds layer index (>= total_layers=4) must raise ValueError
        with self.assertRaises(ValueError):
            resolver.resolve({"layers": [0, 4]}, resolver_config)

        with self.assertRaises(ValueError):
            resolver.resolve({"layers": [48]}, resolver_config)

        # Invalid layers type (not list/set/tuple) must raise ValueError
        with self.assertRaises(ValueError):
            resolver.resolve({"layers": "invalid_string"}, resolver_config)

        # Missing base_model in resolver_config must raise ValueError
        with self.assertRaises(ValueError):
            resolver.resolve({"layers": [1, 2]}, {})

    def test_base_sparsity_resolver_nonprefix_layers_bypass_splicing(self):
        """
        Arrange-Act-Assert: Verify that an arbitrary non-prefix layer set bypasses
        reference splicing and falls through to in-memory algorithmic pruning.
        """
        resolver = NaiveMagnitudeResolver()
        current_config = {"layers": [1, 2]}
        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }

        # Should NOT raise ValueError from _assert_spliceable; falls through to prune_in_memory
        model_path, consumed_keys = resolver.resolve(current_config,
                                                     resolver_config)
        self.assertIn("layers", consumed_keys)
        self.assertTrue(
            os.path.exists(
                os.path.join(model_path, "model.safetensors.index.json")))

    def test_base_sparsity_resolver_nonprefix_layers_allow_splice(self):
        """
        Arrange-Act-Assert: Verify that allow_nonprefix_splice permits splicing non-prefix layer sets.
        """
        resolver = NaiveMagnitudeResolver()
        current_config = {"layers": [1, 2]}
        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "allow_nonprefix_splice": True,
            "checkpoint_base_dir": self.work_dir
        }

        model_path, consumed_keys = resolver.resolve(current_config,
                                                     resolver_config)
        self.assertIn("layers", consumed_keys)
        self.assertTrue(
            os.path.exists(
                os.path.join(model_path, "model.safetensors.index.json")))

    def test_base_sparsity_resolver_ratio_in_cache_key(self):
        """
        Arrange-Act-Assert: Verify that prunen and prunem from current_config are respected,
        added to CONSUMED_KEYS, and embedded into the non-prefix SHA1 cache key and checkpoint name.
        """
        resolver = NaiveMagnitudeResolver()
        resolver_config = {
            "base_model": self.dense_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }

        cur_1of4 = {"prunen": 1, "prunem": 4, "layers": [1, 2]}
        model_path_1of4, consumed_1of4 = resolver.resolve(
            cur_1of4, resolver_config)
        self.assertIn("prunen", consumed_1of4)
        self.assertIn("prunem", consumed_1of4)
        self.assertIn("_1of4_", model_path_1of4)

        cur_3of4 = {"prunen": 3, "prunem": 4, "layers": [1, 2]}
        model_path_3of4, consumed_3of4 = resolver.resolve(
            cur_3of4, resolver_config)
        self.assertIn("_3of4_", model_path_3of4)

        self.assertNotEqual(model_path_1of4, model_path_3of4)

    def test_base_sparsity_resolver_reference_ratio_assertion(self):
        """
        Arrange-Act-Assert: Verify that reference_sparse_model ratio mismatch
        (e.g., 2:4 reference for 1:4 sweep) is strictly rejected with ValueError.
        """
        resolver = BaseSparsityResolver()
        ref_2of4_dir = os.path.join(self.work_dir, "model_2of4_canonical_full")
        os.makedirs(ref_2of4_dir, exist_ok=True)
        shutil.copytree(self.sparse_dir, ref_2of4_dir, dirs_exist_ok=True)

        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": ref_2of4_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }

        with self.assertRaises(ValueError) as ctx:
            resolver.resolve({
                "prunen": 1,
                "prunem": 4,
                "num_sparse_layers": 2
            }, resolver_config)
        self.assertIn("Reference sparse model ratio mismatch",
                      str(ctx.exception))

        path, consumed = resolver.resolve(
            {
                "prunen": 2,
                "prunem": 4,
                "num_sparse_layers": 2
            }, resolver_config)
        self.assertIn("_2of4_", path)


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


class TestCheckpointIntegrityGate(unittest.TestCase):
    """Contract & Integrity Guard tests: validates checkpoint safetensors key parity and substitution."""

    def test_checkpoint_integrity_gate_pass_and_failures(self):
        """
        Contract & Integrity Guard Test: Validates that assert_checkpoint_integrity
        passes when safetensors keys match 1:1, and fails loudly naming missing
        or extra keys on any mismatch.
        """
        import tempfile

        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "base")
            valid_ckpt_dir = os.path.join(tmpdir, "valid_ckpt")
            missing_key_dir = os.path.join(tmpdir, "missing_key_ckpt")
            extra_key_dir = os.path.join(tmpdir, "extra_key_ckpt")

            os.makedirs(base_dir)
            os.makedirs(valid_ckpt_dir)
            os.makedirs(missing_key_dir)
            os.makedirs(extra_key_dir)

            base_tensors = {
                "model.language_model.layers.0.self_attn.q_proj.weight":
                torch.randn(8, 8),
                "model.language_model.layers.1.self_attn.q_proj.weight":
                torch.randn(8, 8),
                "model.vision_embedder.patch_dense.weight":
                torch.randn(4, 4),
                "model.language_model.embed_tokens.weight":
                torch.randn(16, 8),
            }
            save_file(base_tensors, os.path.join(base_dir,
                                                 "model.safetensors"))

            # 1. Valid checkpoint: exact matching keys
            valid_tensors = dict(base_tensors)
            valid_tensors[
                "model.language_model.layers.1.self_attn.q_proj.weight"] = torch.zeros(
                    8, 8)
            save_file(valid_tensors,
                      os.path.join(valid_ckpt_dir, "model.safetensors"))

            # Must pass without error
            ModelSplicer.assert_checkpoint_integrity(base_dir, valid_ckpt_dir)

            # 2. Missing key: omitted vision_embedder (reproducing the reported defect)
            missing_tensors = dict(base_tensors)
            del missing_tensors["model.vision_embedder.patch_dense.weight"]
            save_file(missing_tensors,
                      os.path.join(missing_key_dir, "model.safetensors"))

            with self.assertRaises(ValueError) as ctx_missing:
                ModelSplicer.assert_checkpoint_integrity(
                    base_dir, missing_key_dir)
            self.assertIn("Missing keys (1)", str(ctx_missing.exception))
            self.assertIn("model.vision_embedder.patch_dense.weight",
                          str(ctx_missing.exception))

            # 3. Extra key: unexpected multimodal/tower key
            extra_tensors = dict(base_tensors)
            extra_tensors["model.audio_tower.extra.weight"] = torch.randn(4, 4)
            save_file(extra_tensors,
                      os.path.join(extra_key_dir, "model.safetensors"))

            with self.assertRaises(ValueError) as ctx_extra:
                ModelSplicer.assert_checkpoint_integrity(
                    base_dir, extra_key_dir)
            self.assertIn("Extra keys (1)", str(ctx_extra.exception))
            self.assertIn("model.audio_tower.extra.weight",
                          str(ctx_extra.exception))

    def test_save_substituted_checkpoint_preserves_multimodal_keys(self):
        """
        Arrange-Act-Assert: Verifies that ModelSplicer.save_substituted_checkpoint
        correctly updates target layer weights while preserving non-layer multimodal
        weights (such as model.vision_embedder.*) and config files from base_model.
        """
        import json
        import tempfile

        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "base")
            save_dir = os.path.join(tmpdir, "saved")
            os.makedirs(base_dir)

            base_w0 = torch.ones(8, 8)
            base_w1 = torch.ones(8, 8)
            vision_w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            embed_w = torch.ones(16, 8) * 5.0

            base_tensors = {
                "model.language_model.layers.0.self_attn.q_proj.weight":
                base_w0,
                "model.language_model.layers.1.self_attn.q_proj.weight":
                base_w1,
                "model.vision_embedder.patch_dense.weight": vision_w,
                "model.language_model.embed_tokens.weight": embed_w,
            }
            save_file(base_tensors, os.path.join(base_dir,
                                                 "model.safetensors"))
            with open(os.path.join(base_dir, "config.json"), "w") as f:
                json.dump(
                    {"architectures": ["Gemma4ForConditionalGeneration"]}, f)

            # Replacement state dict modifies layer 1 only, plus a phantom key that must not leak
            calibrated_w1 = torch.zeros(8, 8)
            replacements = {
                "model.language_model.layers.1.self_attn.q_proj.weight":
                calibrated_w1,
                "model.phantom_audio_tower.ignored": torch.randn(4, 4),
            }

            ModelSplicer.save_substituted_checkpoint(
                base_model_dir=base_dir,
                output_dir=save_dir,
                replacement_state_dict=replacements,
                target_layer_indices={1},
            )

            # Assert checkpoint integrity passes
            ModelSplicer.assert_checkpoint_integrity(base_dir, save_dir)

            # Verify tensor contents
            with safe_open(os.path.join(save_dir, "model.safetensors"),
                           framework="pt",
                           device="cpu") as sf:
                self.assertEqual(set(sf.keys()), set(base_tensors.keys()))
                # Layer 0 remained base_w0
                self.assertTrue(
                    torch.equal(
                        sf.get_tensor(
                            "model.language_model.layers.0.self_attn.q_proj.weight"
                        ), base_w0))
                # Layer 1 updated to calibrated_w1
                self.assertTrue(
                    torch.equal(
                        sf.get_tensor(
                            "model.language_model.layers.1.self_attn.q_proj.weight"
                        ), calibrated_w1))
                # Vision embedder preserved exactly
                self.assertTrue(
                    torch.equal(
                        sf.get_tensor(
                            "model.vision_embedder.patch_dense.weight"),
                        vision_w))
                # Embed tokens preserved exactly
                self.assertTrue(
                    torch.equal(
                        sf.get_tensor(
                            "model.language_model.embed_tokens.weight"),
                        embed_w))

            # Verify config.json was copied
            self.assertTrue(
                os.path.exists(os.path.join(save_dir, "config.json")))


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


class TestLrcSparsityResolution(unittest.TestCase):
    """Tests Low-Rank Compensated (LRC: W = S + UV) sparsity resolution, naming, and verification."""

    def setUp(self):
        self.resolver = BaseSparsityResolver()
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

    def test_checkpoint_naming_with_lrc(self):
        """
        Arrange-Act-Assert: Verify checkpoint naming with use_lrc enabled.
        Ensures Arm B does not collide with Arm A's cached directory.
        """
        name_arm_a = self.resolver.generate_checkpoint_name(
            arch_type="gemma4",
            prunen=2,
            prunem=4,
            strategy="first_to_last",
            min_layer=0,
            max_layer=24,
            resolver_config={"use_lrc": False},
        )
        name_arm_b = self.resolver.generate_checkpoint_name(
            arch_type="gemma4",
            prunen=2,
            prunem=4,
            strategy="first_to_last",
            min_layer=0,
            max_layer=24,
            resolver_config={
                "use_lrc": True,
                "lrc_rank": 128,
                "lrc_iters": 3
            },
        )
        name_arm_b_custom = self.resolver.generate_checkpoint_name(
            arch_type="gemma4",
            prunen=2,
            prunem=4,
            strategy="first_to_last",
            min_layer=0,
            max_layer=24,
            resolver_config={
                "use_lrc": True,
                "lrc_rank": 64,
                "lrc_iters": 5
            },
        )
        name_is_lrc = self.resolver.generate_checkpoint_name(
            arch_type="gemma4",
            prunen=2,
            prunem=4,
            strategy="first_to_last",
            min_layer=0,
            max_layer=24,
            resolver_config={
                "is_lrc": True,
                "lrc_k": 32,
                "lrc_T": 2
            },
        )

        self.assertNotIn("_lrc", name_arm_a)
        self.assertIn("_lrc_k128_T3", name_arm_b)
        self.assertIn("_lrc_k64_T5", name_arm_b_custom)
        self.assertIn("_lrc_k32_T2", name_is_lrc)
        self.assertNotEqual(name_arm_a, name_arm_b)
        self.assertNotEqual(name_arm_b, name_arm_b_custom)

    def test_nonprefix_cache_payload_and_naming_with_lrc(self):
        """
        Arrange-Act-Assert: Verify non-prefix layer resolution embeds LRC parameters into
        cache_payload and checkpoint naming, preventing collision between standard SparseGPT and LRC.
        """
        resolver = NaiveMagnitudeResolver()
        resolver_config = {
            "base_model": self.dense_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir,
        }

        # Standard non-prefix (no LRC)
        path_std, consumed_std = resolver.resolve(
            {"layers": [1, 2]},
            resolver_config,
        )

        # Non-prefix with LRC
        lrc_config = dict(resolver_config)
        lrc_config["use_lrc"] = True
        lrc_config["lrc_rank"] = 128
        lrc_config["lrc_iters"] = 3
        path_lrc, consumed_lrc = resolver.resolve(
            {"layers": [1, 2]},
            lrc_config,
        )

        self.assertNotIn("_lrc", path_std)
        self.assertIn("_lrc_k128_T3", path_lrc)
        self.assertNotEqual(path_std, path_lrc)
        self.assertIn("use_lrc", consumed_lrc)

    def test_lrc_bypasses_non_lrc_reference_sparse_model(self):
        """
        Arrange-Act-Assert: Verify that when use_lrc is True, reference_sparse_model reuse
        and splicing are bypassed unless the reference model is explicitly tagged as an LRC checkpoint.
        """
        resolver = NaiveMagnitudeResolver()

        # Non-LRC reference model: must be bypassed, falling through to prune_in_memory
        resolver_config_std_ref = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "checkpoint_base_dir": self.work_dir,
            "use_lrc": True,
            "lrc_rank": 128,
            "lrc_iters": 3,
        }
        path_bypassed, _ = resolver.resolve(
            {"layers": [0]},
            resolver_config_std_ref,
        )
        self.assertIn("_lrc_k128_T3", path_bypassed)

        # LRC-tagged reference model: should NOT be bypassed for prefix splicing
        ref_lrc_dir = os.path.join(self.work_dir, "gemma4_lrc_k128_T3_full")
        shutil.copytree(self.sparse_dir, ref_lrc_dir, dirs_exist_ok=True)
        resolver_config_lrc_ref = {
            "base_model": self.dense_dir,
            "reference_sparse_model": ref_lrc_dir,
            "checkpoint_base_dir": self.work_dir,
            "use_lrc": True,
            "lrc_rank": 128,
            "lrc_iters": 3,
        }
        path_spliced, _ = resolver.resolve(
            {"layers": [0]},
            resolver_config_lrc_ref,
        )
        self.assertIn("_lrc_k128_T3", path_spliced)


if __name__ == "__main__":
    unittest.main()
