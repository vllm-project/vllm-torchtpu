# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU interpreter regressions and a native-kernel check enabled on TPU hosts.

Run without pytest's Torch/accelerator fixtures:
  JAX_PLATFORMS=cpu PYTHONPATH=src python -m unittest discover \
      -s tests/kernels -p test_stable_order_and_inverse.py -v
"""

import functools
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.inverse_permutation import inverse_permutation_128
from vllm_torchtpu.kernels.stable_order_and_inverse import \
    stable_order_and_inverse_128


def _numpy_order_and_inverse(keys):
    order = np.argsort(keys, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(keys.size, dtype=order.dtype)
    return order, inverse


def _check_order_and_inverse(test, keys, order, inverse):
    order, inverse = np.asarray(order), np.asarray(inverse)
    expected_order, expected_inverse = _numpy_order_and_inverse(keys)
    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_array_equal(inverse, expected_inverse)
    positions = np.arange(keys.size)
    np.testing.assert_array_equal(inverse[order], positions)
    np.testing.assert_array_equal(order[inverse], positions)
    tied = keys[order[:-1]] == keys[order[1:]]
    test.assertTrue(np.all(order[:-1][tied] < order[1:][tied]))


def _baseline_order_and_inverse(keys):
    # The immediate candidate2 chain, with only Pallas interpretation enabled.
    order = jnp.argsort(keys)
    return order, inverse_permutation_128(order, interpret=True)


class StableOrderAndInverseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(jax.default_device(jax.devices("cpu")[0]))
        cls.sort = staticmethod(
            jax.jit(
                functools.partial(stable_order_and_inverse_128,
                                  interpret=True)))

    def _check(self, keys):
        keys = np.asarray(keys, dtype=np.int32)
        order, inverse = self.sort(jnp.asarray(keys))
        for output in (order, inverse):
            self.assertEqual(output.shape, (128, ))
            self.assertEqual(output.dtype, jnp.int32)
        _check_order_and_inverse(self, keys, order, inverse)

    def test_equal_monotonic_duplicate_negative_and_extreme_keys(self):
        limits = np.iinfo(np.int32)
        cases = {
            "all_equal":
            np.full(128, -17),
            "all_min":
            np.full(128, limits.min),
            "all_max":
            np.full(128, limits.max),
            "ascending":
            np.arange(128),
            "reversed":
            np.arange(127, -1, -1),
            "negative":
            np.arange(-1, -129, -1),
            "duplicates_across_tokens":
            np.tile([7, 0, 7, 255, 0, 255, 7, 0], 16),
            "sentinels":
            np.tile([256, 255, 0, 256, 0, 255, 256, 0], 16),
            "all_sentinel":
            np.full(128, 256),
            "extrema":
            np.tile([
                limits.min, limits.max, -1, 0, 1, limits.min + 1,
                limits.max - 1, limits.min
            ], 16),
        }
        for name, keys in cases.items():
            with self.subTest(name=name):
                self._check(keys)

    def test_random_full_int32_and_duplicate_keys(self):
        limits = np.iinfo(np.int32)
        for seed in range(32):
            rng = np.random.default_rng(seed)
            with self.subTest(seed=seed, distribution="full_int32"):
                self._check(
                    rng.integers(limits.min,
                                 limits.max,
                                 size=128,
                                 dtype=np.int32,
                                 endpoint=True))
            with self.subTest(seed=seed, distribution="duplicates"):
                self._check(rng.integers(-8, 9, size=128, dtype=np.int32))

    def test_route_zero_at_every_destination(self):
        for destination in range(128):
            with self.subTest(destination=destination):
                self._check(np.roll(np.arange(128), -destination))

    def test_explicit_int32_outputs_with_x64_enabled(self):
        with jax.enable_x64(True):
            self._check(
                np.tile(
                    [np.iinfo(np.int32).min, 0,
                     np.iinfo(np.int32).max, -1], 32))

    def test_rejects_unsupported_shapes_and_dtypes(self):
        for shape in ((0, ), (127, ), (129, ), (16, 8), (1, 128)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                stable_order_and_inverse_128(jax.ShapeDtypeStruct(
                    shape, jnp.int32),
                                             interpret=True)
        for dtype in (jnp.int16, jnp.int64, jnp.uint32, jnp.float32):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                stable_order_and_inverse_128(jax.ShapeDtypeStruct((128, ),
                                                                  dtype),
                                             interpret=True)


class RoutedStableOrderAndInverseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(jax.default_device(jax.devices("cpu")[0]))
        cls.enterClassContext(jax.enable_x64(False))
        # Load production JAX code without registering Torch attention ops.
        path = (Path(__file__).resolve().parents[2] /
                "src/vllm_torchtpu/layers/common/fused_moe_gmm.py")
        spec = importlib.util.spec_from_file_location("_joint_test_routing",
                                                      path)
        cls.routing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.routing)
        cls.enterClassContext(
            patch.object(envs, "TPU_MOE_JOINT_ROUTING_128", True))

    def _inputs(self,
                *,
                tokens=16,
                topk=8,
                dtype=jnp.int32,
                mask="mixed",
                hidden_size=16,
                seed=23):
        rng = np.random.default_rng(seed)
        # Duplicate expert IDs span every token, including experts 0 and 255.
        ids = rng.choice(np.array([0, 1, 7, 127, 255], dtype=np.int32),
                         size=(tokens, topk))
        ids.ravel()[:2] = [0, 255]
        weights = rng.random((tokens, topk), dtype=np.float32)
        weights[1:] = 0
        if mask == "mixed":
            ids.ravel()[2::5] = -1
        elif mask == "all":
            ids.fill(-1)
        elif mask == "zero_weights":
            weights.fill(0)
        elif mask == "poison_weights":
            weights.ravel()[:4] = [np.nan, np.inf, -np.inf, -0.0]
        hidden = jnp.asarray(rng.standard_normal((tokens, hidden_size)),
                             jnp.bfloat16)
        return hidden, jnp.asarray(ids, dtype), jnp.asarray(weights)

    def _prepare(self, inputs, **kwargs):
        return self.routing.prepare_routed_gmm_inputs(*inputs,
                                                      local_num_experts=256,
                                                      topk=inputs[1].shape[1],
                                                      **kwargs)

    def _check_preparation(self, inputs, outputs, skip_padding):
        hidden, ids, weights = map(np.asarray, inputs)
        valid = ids.ravel() >= 0
        if skip_padding:
            valid &= weights.ravel() != 0
        keys = np.where(valid, ids.ravel(), 256)
        order, inverse = _numpy_order_and_inverse(keys)
        expected = (hidden[np.repeat(np.arange(hidden.shape[0]),
                                     ids.shape[1])[order]],
                    np.bincount(ids.ravel()[valid].astype(np.int64),
                                minlength=256), inverse, weights.ravel(),
                    valid, order)
        self.assertEqual(len(outputs), 6)
        for got, want in zip(outputs, expected):
            got = np.asarray(got)
            want = np.asarray(want, dtype=got.dtype)
            np.testing.assert_array_equal(got, want)
            # Also preserve NaN payloads and signed zeros in gathered values
            # and routing weights: these are unchanged array movements.
            self.assertEqual(got.tobytes(), want.tobytes())
        _check_order_and_inverse(self, keys, outputs[-1], outputs[2])

    def test_all_six_outputs_match_numpy_and_immediate_baseline(self):
        for mask in ("none", "mixed", "all", "zero_weights", "poison_weights"):
            for skip_padding in (False, True):
                for sparse_core in (False, True):
                    with self.subTest(mask=mask,
                                      skip_padding=skip_padding,
                                      sparse_core=sparse_core):
                        inputs = self._inputs(mask=mask)
                        kwargs = dict(use_ep=False,
                                      use_sparse_core=sparse_core,
                                      skip_padded_tokens=skip_padding)
                        with patch.object(
                                self.routing,
                                "stable_order_and_inverse_128",
                                wraps=functools.partial(
                                    stable_order_and_inverse_128,
                                    interpret=True)) as joint, patch.object(
                                        self.routing, "inverse_permutation_128"
                                    ) as old_inverse:
                            actual = self._prepare(inputs, **kwargs)
                        joint.assert_called_once()
                        old_inverse.assert_not_called()
                        with patch.object(self.routing,
                                          "stable_order_and_inverse_128",
                                          _baseline_order_and_inverse):
                            baseline = self._prepare(inputs, **kwargs)
                        for got, want in zip(actual, baseline):
                            self.assertEqual(got.shape, want.shape)
                            self.assertEqual(got.dtype, want.dtype)
                            self.assertEqual(
                                np.asarray(got).tobytes(),
                                np.asarray(want).tobytes())
                        self._check_preparation(inputs, actual, skip_padding)

    def test_jitted_full_shape_preparation_with_dynamic_inputs(self):
        with patch.object(
                self.routing, "stable_order_and_inverse_128",
                functools.partial(stable_order_and_inverse_128,
                                  interpret=True)):
            prepare = jax.jit(
                functools.partial(self._prepare,
                                  use_ep=False,
                                  use_sparse_core=False,
                                  skip_padded_tokens=True))
            for seed in range(4):
                inputs = self._inputs(hidden_size=2048, seed=seed)
                actual = prepare(inputs)
                self._check_preparation(inputs, actual, True)
                self.assertEqual(actual[0].shape, (128, 2048))

    def test_fallback_for_ep_route_counts_and_input_or_index_dtypes(self):
        cases = [(16, jnp.int32, True, False), (1, jnp.int32, False, False),
                 (15, jnp.int32, False, False), (17, jnp.int32, False, False),
                 (16, jnp.int16, False, False), (16, jnp.uint32, False, False),
                 (16, jnp.float32, False, False), (16, jnp.int32, False, True),
                 (16, jnp.int64, False, True)]
        for tokens, dtype, use_ep, x64 in cases:
            with self.subTest(tokens=tokens,
                              dtype=dtype,
                              use_ep=use_ep,
                              x64=x64), jax.enable_x64(x64):
                inputs = self._inputs(tokens=tokens, dtype=dtype, mask="none")
                with patch.object(
                        self.routing,
                        "stable_order_and_inverse_128") as joint, patch.object(
                            self.routing,
                            "inverse_permutation_128") as old_inverse:
                    actual = self._prepare(inputs,
                                           use_ep=use_ep,
                                           use_sparse_core=False)
                joint.assert_not_called()
                old_inverse.assert_not_called()
                self._check_preparation(inputs, actual, False)
                expected_dtype = jnp.int64 if x64 else jnp.int32
                self.assertEqual(actual[2].dtype, expected_dtype)
                self.assertEqual(actual[-1].dtype, expected_dtype)

    def test_non_int32_sort_keys_retain_older_inverse_eligibility(self):
        inputs = self._inputs()
        original_where = jnp.where

        def promoted_sort_keys(condition, x, y):
            result = original_where(condition, x, y)
            # Isolate the constructed-key dtype guard with otherwise eligible
            # IDs/indices. Mask and key values are unchanged.
            if isinstance(y, int) and y == 256:
                return result.astype(jnp.float32)
            return result

        with patch.object(self.routing.jnp, "where", promoted_sort_keys), \
                patch.object(self.routing, "stable_order_and_inverse_128"
                             ) as joint, patch.object(
                                 self.routing, "inverse_permutation_128",
                                 wraps=functools.partial(
                                     inverse_permutation_128, interpret=True)
                             ) as old_inverse:
            actual = self._prepare(inputs, use_ep=False, use_sparse_core=False)
        joint.assert_not_called()
        old_inverse.assert_called_once()
        self._check_preparation(inputs, actual, False)

    def test_disabled_flag_preserves_original_sorts_and_all_outputs(self):
        for skip_padding in (False, True):
            with self.subTest(skip_padding=skip_padding), patch.object(
                    envs, "TPU_MOE_JOINT_ROUTING_128", False), patch.object(
                        self.routing, "stable_order_and_inverse_128") as joint, \
                    patch.object(self.routing, "inverse_permutation_128") as inverse, \
                    patch.object(self.routing.jnp, "argsort", wraps=jnp.argsort) as sorts:
                inputs = self._inputs(mask="poison_weights")
                actual = self._prepare(inputs,
                                       use_ep=False,
                                       use_sparse_core=False,
                                       skip_padded_tokens=skip_padding)
                joint.assert_not_called()
                inverse.assert_not_called()
                self.assertEqual(sorts.call_count, 2)
                self._check_preparation(inputs, actual, skip_padding)

    def test_128_routes_with_different_token_and_topk_counts(self):
        inputs = self._inputs(tokens=32, topk=4)
        with patch.object(
                self.routing, "stable_order_and_inverse_128",
                functools.partial(stable_order_and_inverse_128,
                                  interpret=True)):
            actual = self._prepare(inputs, use_ep=False, use_sparse_core=False)
        self._check_preparation(inputs, actual, False)


class RoutingFlagTest(unittest.TestCase):

    def test_default_off_and_boolean_values(self):
        name = "TPU_MOE_JOINT_ROUTING_128"
        with patch.dict(os.environ):
            os.environ.pop(name, None)
            self.assertFalse(envs.environment_variables[name]())
            for value, expected in [("0", False), ("1", True),
                                    ("false", False), ("true", True)]:
                with self.subTest(value=value):
                    os.environ[name] = value
                    self.assertEqual(envs.environment_variables[name](),
                                     expected)


class NativeRoutingKernelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Keep accelerator initialization out of test collection, so CI can
        # run worker-spawning tests before in-process TPU tests.
        if jax.default_backend() != "tpu":
            raise unittest.SkipTest("requires a TPU")

    def test_native_order_and_inverse_match_numpy(self):
        """Exercise real Mosaic lowering in TPU CI, without interpretation."""
        sort = jax.jit(stable_order_and_inverse_128)
        invert = jax.jit(inverse_permutation_128)
        limits = np.iinfo(np.int32)
        rng = np.random.default_rng(1729)
        cases = [
            np.zeros(128, dtype=np.int32),
            np.arange(127, -1, -1, dtype=np.int32),
            np.tile(np.array([limits.min, limits.max, -1, 0], dtype=np.int32),
                    32),
            rng.integers(-8, 9, size=128, dtype=np.int32),
            rng.integers(limits.min,
                         limits.max,
                         size=128,
                         dtype=np.int32,
                         endpoint=True),
        ]
        with jax.default_device(jax.devices("tpu")[0]):
            for i, keys in enumerate(cases):
                with self.subTest(case=i):
                    order, inverse = sort(jnp.asarray(keys))
                    _check_order_and_inverse(self, keys, order, inverse)
                    np.testing.assert_array_equal(invert(order), inverse)


if __name__ == "__main__":
    unittest.main()
