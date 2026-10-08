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
from resolvers.lrc_resolver import LrcResolver
from resolvers.model_splicer import ModelSplicer
from resolvers.naive_magnitude_resolver import NaiveMagnitudeResolver
from resolvers.registry import ResolverRegistry
from resolvers.sparsegpt_resolver import SparseGptResolver
from resolvers.weight_scaler_resolver import WeightScalerResolver
from safetensors import safe_open
from sparsity.lrc import make_lrc_pruner

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

    def test_cleanup_old_spliced_dirs_preserves_target_and_unmarked(self):
        """
        The reaper must never remove the current target, canonical
        references, or any directory it cannot prove it created itself.
        A snapshot of a shared checkpoint directory (artifacts from other
        tooling: gemma4_unified_*, custom_order_*, hand-made dirs) is used
        so that name shape alone is never a reason to delete.
        """
        real_listing = [
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
            "gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_e5b47bdd",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for item in real_listing:
                os.makedirs(os.path.join(temp_dir, item), exist_ok=True)
            current_target = os.path.join(
                temp_dir,
                "gemma4_unified_sparsegpt_2of4_256c4_custom_order_4layers_e5b47bdd"
            )

            self.resolver._cleanup_old_spliced_dirs(temp_dir, current_target)

            self.assertEqual(set(os.listdir(temp_dir)), set(real_listing))


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

    def test_partial_cached_checkpoint_is_regenerated_not_reused(self):
        """
        Regression: a cached checkpoint whose last shard was never written
        (ENOSPC mid-save) must not be returned as a cache hit; resolve()
        regenerates it in place.
        """
        resolver = BaseSparsityResolver()
        current_config = {"num_sparse_layers": 2}
        resolver_config = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir
        }
        model_path, _ = resolver.resolve(current_config, resolver_config)
        shard_names = sorted(f for f in os.listdir(model_path)
                             if f.endswith(".safetensors"))
        self.assertEqual(len(shard_names), 2)

        # The copied index survives; the last shard does not.
        os.remove(os.path.join(model_path, shard_names[-1]))
        with self.assertRaises(ValueError):
            ModelSplicer.assert_checkpoint_integrity(self.dense_dir,
                                                     model_path)

        model_path2, _ = resolver.resolve(current_config, resolver_config)
        self.assertEqual(model_path, model_path2)
        self.assertTrue(
            os.path.isfile(os.path.join(model_path2, shard_names[-1])))
        ModelSplicer.assert_checkpoint_integrity(self.dense_dir, model_path2)

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

    def test_sparsegpt_and_lrc_prune_in_memory_delegation(self):
        """Verify SparseGptResolver and LrcResolver delegate expected kwargs to prune_gemma4."""
        with patch("resolvers.sparsegpt_resolver.prune_gemma4") as mock_sgpt:
            SparseGptResolver().prune_in_memory(
                base_model="/mock/base",
                save_path="/mock/save",
                min_layer=0,
                max_layer=4,
                prunen=2,
                prunem=4,
                resolver_config={
                    "nsamples": 32,
                    "seed": 42
                },
                sparse_layer_indices={1, 3},
            )
            mock_sgpt.assert_called_once()
            self.assertEqual(
                mock_sgpt.call_args.kwargs["sparse_layer_indices"], {1, 3})
            self.assertNotIn("use_lrc", mock_sgpt.call_args.kwargs)

        with patch("resolvers.lrc_resolver.prune_gemma4") as mock_lrc, \
             patch("resolvers.lrc_resolver.make_lrc_pruner", wraps=make_lrc_pruner) as mock_pruner:
            LrcResolver().prune_in_memory(
                base_model="/mock/base",
                save_path="/mock/save",
                min_layer=0,
                max_layer=4,
                prunen=2,
                prunem=4,
                resolver_config={
                    "nsamples": 32,
                    "lrc_rank": 64,
                    "lrc_iters": 5
                },
                sparse_layer_indices={0, 2},
            )
            mock_pruner.assert_called_once_with(prunen=2,
                                                prunem=4,
                                                percdamp=0.01,
                                                blocksize=128,
                                                T=5,
                                                k=64)
            mock_lrc.assert_called_once()
            self.assertTrue(callable(mock_lrc.call_args.kwargs["prune_fn"]))

    def test_nonprefix_layers_ratio_and_bounds_resolution(self):
        """Verify layer bounds, non-prefix calibration fallback, and N:M ratio cache keys/guards."""
        resolver = NaiveMagnitudeResolver()
        cfg = {
            "base_model": self.dense_dir,
            "reference_sparse_model": self.sparse_dir,
            "strategy": "first_to_last",
            "checkpoint_base_dir": self.work_dir,
        }
        for bad_layers in ([-1, 1], [0, 4], [48], "invalid"):
            with self.assertRaises(ValueError):
                resolver.resolve({"layers": bad_layers}, cfg)
        with self.assertRaises(ValueError):
            resolver.resolve({"layers": [1, 2]}, {})

        # Non-prefix bypasses splice and calibrates in memory; ratio is embedded in cache key
        p_1of4, consumed = resolver.resolve(
            {
                "prunen": 1,
                "prunem": 4,
                "layers": [1, 2]
            },
            {
                "base_model": self.dense_dir,
                "checkpoint_base_dir": self.work_dir
            },
        )
        p_3of4, _ = resolver.resolve(
            {
                "prunen": 3,
                "prunem": 4,
                "layers": [1, 2]
            },
            {
                "base_model": self.dense_dir,
                "checkpoint_base_dir": self.work_dir
            },
        )
        self.assertTrue({"layers", "prunen", "prunem"}.issubset(consumed))
        self.assertIn("_1of4_", p_1of4)
        self.assertIn("_3of4_", p_3of4)
        self.assertNotEqual(p_1of4, p_3of4)

        # Reference ratio mismatch raises ValueError
        ref_2of4 = os.path.join(self.work_dir, "model_2of4_canonical_full")
        shutil.copytree(self.sparse_dir, ref_2of4, dirs_exist_ok=True)
        with self.assertRaisesRegex(ValueError,
                                    "Reference sparse model ratio mismatch"):
            resolver.resolve(
                {
                    "prunen": 1,
                    "prunem": 4,
                    "num_sparse_layers": 2
                },
                dict(cfg, reference_sparse_model=ref_2of4),
            )


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
        resolver = WeightScalerResolver()
        out_dir = os.path.join(self.test_dir.name, "scaled_model")
        scale_factor = 1.10
        orig_tensors = {}
        with safe_open(os.path.join(self.base_dir,
                                    "model-00001-of-00001.safetensors"),
                       framework="pt",
                       device="cpu") as f:
            for k in f.keys():
                orig_tensors[k] = f.get_tensor(k)

        resolver._scale_checkpoint(base_checkpoint=self.base_dir,
                                   output_dir=out_dir,
                                   scale_factor=scale_factor)

        with safe_open(os.path.join(out_dir,
                                    "model-00001-of-00001.safetensors"),
                       framework="pt",
                       device="cpu") as f:
            for k in f.keys():
                scaled = f.get_tensor(k)
                orig = orig_tensors[k]
                if "weight" in k and scaled.dim() == 2 and "embed" not in k:
                    expected = (orig.to(torch.float32) * scale_factor).to(
                        torch.bfloat16)
                    self.assertTrue(torch.allclose(scaled, expected,
                                                   atol=1e-2))
                elif "norm" in k or scaled.dim() == 1:
                    self.assertTrue(torch.equal(scaled, orig))


class TestCheckpointIntegrityGate(unittest.TestCase):
    """Tests shard-header checkpoint integrity gate and save_substituted_checkpoint."""

    def test_integrity_gate_and_substituted_save(self):
        import json

        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "base")
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            save_dir = os.path.join(tmpdir, "saved")
            os.makedirs(base_dir)
            os.makedirs(ckpt_dir)

            shards = {
                "model-00001-of-00002.safetensors": {
                    "model.language_model.layers.0.self_attn.q_proj.weight":
                    torch.ones(8, 8),
                    "model.vision_embedder.patch_dense.weight":
                    torch.ones(4, 4),
                },
                "model-00002-of-00002.safetensors": {
                    "model.language_model.layers.1.self_attn.q_proj.weight":
                    torch.ones(8, 8),
                },
            }
            index = {
                "metadata": {},
                "weight_map": {
                    k: s
                    for s, ts in shards.items()
                    for k in ts
                },
            }
            for d in (base_dir, ckpt_dir):
                with open(os.path.join(d, "model.safetensors.index.json"),
                          "w") as f:
                    json.dump(index, f)
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump(
                        {"architectures": ["Gemma4ForConditionalGeneration"]},
                        f)
            for s, ts in shards.items():
                save_file(ts, os.path.join(base_dir, s))

            first, last = sorted(shards)
            save_file(shards[first], os.path.join(ckpt_dir, first))

            # Missing shard fails even when listed in index.json
            with self.assertRaisesRegex(ValueError, "Missing shard files"):
                ModelSplicer.assert_checkpoint_integrity(base_dir, ckpt_dir)

            # Complete checkpoint passes; truncated shard fails
            last_path = os.path.join(ckpt_dir, last)
            save_file(shards[last], last_path)
            ModelSplicer.assert_checkpoint_integrity(base_dir, ckpt_dir)
            with open(last_path, "r+b") as f:
                f.truncate(os.path.getsize(last_path) - 16)
            with self.assertRaisesRegex(ValueError, "truncated"):
                ModelSplicer.assert_checkpoint_integrity(base_dir, ckpt_dir)

            # save_substituted_checkpoint: raises on key mismatch or disjoint target layers
            with self.assertRaisesRegex(ValueError, "replaced 0 tensors"):
                ModelSplicer.save_substituted_checkpoint(
                    base_dir, save_dir,
                    {"layers.1.q_proj.weight": torch.zeros(8, 8)}, {1})

            # Valid substitution updates layer 1 and preserves layer 0 + multimodal weights
            ModelSplicer.save_substituted_checkpoint(
                base_dir,
                save_dir,
                {
                    "model.language_model.layers.1.self_attn.q_proj.weight":
                    torch.zeros(8, 8)
                },
                {1},
            )
            ModelSplicer.assert_checkpoint_integrity(base_dir, save_dir)
            with safe_open(os.path.join(save_dir, last),
                           framework="pt",
                           device="cpu") as sf:
                self.assertTrue(
                    torch.equal(
                        sf.get_tensor(
                            "model.language_model.layers.1.self_attn.q_proj.weight"
                        ),
                        torch.zeros(8, 8),
                    ))


class TestResolverRegistry(unittest.TestCase):
    """Tests ResolverRegistry factory and LrcResolver lifecycle."""

    def test_registry_lookup_and_lrc_resolver(self):
        self.assertIsInstance(ResolverRegistry.get("weight_scaler"),
                              WeightScalerResolver)
        self.assertIsInstance(ResolverRegistry.get("sparsegpt"),
                              SparseGptResolver)
        self.assertIsInstance(ResolverRegistry.get("lrc"), LrcResolver)
        self.assertIsInstance(ResolverRegistry.get("resolvers.lrc_resolver"),
                              LrcResolver)
        self.assertIsInstance(ResolverRegistry.get("naive_magnitude"),
                              NaiveMagnitudeResolver)
        with self.assertRaises(KeyError):
            ResolverRegistry.get("non_existent_resolver_type_xyz")

        lrc = LrcResolver()
        sgpt = SparseGptResolver()
        name_lrc = lrc.generate_checkpoint_name("gemma4", 2, 4,
                                                "first_to_last", 0, 24, {
                                                    "lrc_rank": 64,
                                                    "lrc_iters": 5
                                                })
        name_sgpt = sgpt.generate_checkpoint_name("gemma4", 2, 4,
                                                  "first_to_last", 0, 24, {})
        self.assertIn("gemma4_lrc_2of4_first_to_last_layers_0_to_24_k64_T5",
                      name_lrc)
        self.assertNotEqual(name_lrc, name_sgpt)

    def test_custom_resolver_registration(self):

        @ResolverRegistry.register("custom_dummy")
        class DummyResolver(BaseResolver):

            def resolve(self, current_config, resolver_config):
                return "/path/to/dummy", set()

        self.assertIsInstance(ResolverRegistry.get("custom_dummy"),
                              DummyResolver)


if __name__ == "__main__":
    unittest.main()
