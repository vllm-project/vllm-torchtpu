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
"""CPU interpreter checks; these do not validate Mosaic or TPU execution.

Run without Torch, pytest or accelerator initialization:
  JAX_PLATFORMS=cpu PYTHONPATH=src python -m unittest discover \
      -s tests/kernels -p test_inverse_permutation.py -v
"""

import functools
import importlib.util
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


def _load_routing_module():
    # Load the real JAX source without layers/__init__.py registering Torch
    # attention operators. No routing or floating-point source is rewritten.
    path = (Path(__file__).resolve().parents[2] /
            "src/vllm_torchtpu/layers/common/fused_moe_gmm.py")
    spec = importlib.util.spec_from_file_location("_inverse_test_routing",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InversePermutationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(jax.default_device(jax.devices("cpu")[0]))
        cls.invert = staticmethod(
            jax.jit(functools.partial(inverse_permutation_128,
                                      interpret=True)))

    def _check_permutation(self, permutation):
        permutation = np.asarray(permutation, dtype=np.int32)
        actual = np.asarray(self.invert(jnp.asarray(permutation)))
        positions = np.arange(128, dtype=np.int32)
        # An indexed assignment is independent of both the equality/sum
        # implementation and the original inverse argsort.
        expected = np.empty(128, dtype=np.int32)
        expected[permutation] = positions
        self.assertEqual(actual.shape, (128, ))
        self.assertEqual(actual.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[permutation], positions)
        np.testing.assert_array_equal(permutation[actual], positions)
        np.testing.assert_array_equal(actual, np.argsort(permutation))

    def test_identity(self):
        self._check_permutation(np.arange(128))

    def test_reverse(self):
        self._check_permutation(np.arange(127, -1, -1))

    def test_every_cyclic_shift(self):
        for shift in range(128):
            with self.subTest(shift=shift):
                self._check_permutation(np.roll(np.arange(128), shift))

    def test_random_permutations(self):
        for seed in range(64):
            with self.subTest(seed=seed):
                self._check_permutation(
                    np.random.default_rng(seed).permutation(128))

    def test_duplicate_experts_and_masked_keys(self):
        rng = np.random.default_rng(19)
        for all_masked in (False, True):
            ids = rng.integers(0, 8, size=(16, 8), dtype=np.int32)
            ids.ravel()[::3] = -1
            if all_masked:
                ids.fill(-1)
            keys = np.where(ids.ravel() >= 0, ids.ravel(), 256)
            # Preserve the same first stable expert sort as production.
            permutation = np.asarray(jnp.argsort(jnp.asarray(keys)))
            np.testing.assert_array_equal(permutation,
                                          np.argsort(keys, kind="stable"))
            self._check_permutation(permutation)

    def test_rejects_unsupported_shapes_and_dtypes(self):
        for shape in ((0, ), (127, ), (129, ), (16, 8), (1, 128)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                inverse_permutation_128(jax.ShapeDtypeStruct(shape, jnp.int32),
                                        interpret=True)
        for dtype in (jnp.int16, jnp.int64, jnp.uint32, jnp.float32):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                inverse_permutation_128(jax.ShapeDtypeStruct((128, ), dtype),
                                        interpret=True)


class RoutedInversePermutationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(jax.default_device(jax.devices("cpu")[0]))
        cls.routing = _load_routing_module()
        cls.enterClassContext(
            patch.object(envs, "TPU_MOE_JOINT_ROUTING_128", True))

    def _inputs(self, tokens=16, dtype=jnp.int32, mask="mixed"):
        rng = np.random.default_rng(23)
        ids = rng.integers(0, 8, size=(tokens, 8), dtype=np.int32)
        ids[0] = [255, 0, 128, 127, 1, 254, 64, 63]
        weights = rng.random((tokens, 8), dtype=np.float32)
        weights[1:] = 0  # one active token, keeping all padded routes
        if mask == "mixed":
            ids.ravel()[::5] = -1
        elif mask == "all":
            ids.fill(-1)
        hidden = jnp.asarray(rng.standard_normal((tokens, 16)), jnp.bfloat16)
        return hidden, jnp.asarray(ids, dtype), jnp.asarray(weights)

    def _prepare(self, inputs, **kwargs):
        return self.routing.prepare_routed_gmm_inputs(*inputs,
                                                      local_num_experts=256,
                                                      topk=8,
                                                      **kwargs)

    def test_routing_metadata_matches_stable_sort_reference(self):
        for mask in ("none", "mixed", "all"):
            for skip_padding in (False, True):
                for sparse_core in (False, True):
                    with self.subTest(mask=mask,
                                      skip_padding=skip_padding,
                                      sparse_core=sparse_core):
                        inputs = self._inputs(mask=mask)
                        with patch.object(self.routing,
                                          "stable_order_and_inverse_128",
                                          wraps=functools.partial(
                                              stable_order_and_inverse_128,
                                              interpret=True)) as kernel:
                            actual = self._prepare(
                                inputs,
                                use_ep=False,
                                use_sparse_core=sparse_core,
                                skip_padded_tokens=skip_padding)
                        kernel.assert_called_once()
                        hidden, ids, weights = map(np.asarray, inputs)
                        valid = ids.ravel() >= 0
                        if skip_padding:
                            valid &= weights.ravel() != 0
                        order = np.argsort(np.where(valid, ids.ravel(), 256),
                                           kind="stable")
                        expected = (hidden[np.repeat(np.arange(16), 8)[order]],
                                    np.bincount(ids.ravel()[valid],
                                                minlength=256),
                                    np.argsort(order), weights.ravel(), valid,
                                    order)
                        for got, want in zip(actual, expected):
                            np.testing.assert_array_equal(got, want)
                        self.assertEqual(actual[0].shape, (128, 16))

    def test_fallback_for_ep_other_sizes_and_index_dtypes(self):
        cases = [(16, jnp.int32, True), (1, jnp.int32, False),
                 (15, jnp.int32, False), (17, jnp.int32, False),
                 (16, jnp.int16, False), (16, jnp.uint32, False),
                 (16, jnp.int64, False)]
        for tokens, dtype, use_ep in cases:
            with jax.enable_x64(dtype == jnp.int64):
                with self.subTest(tokens=tokens, dtype=dtype, use_ep=use_ep):
                    inputs = self._inputs(tokens, dtype, mask="none")
                    with patch.object(self.routing,
                                      "inverse_permutation_128") as kernel:
                        actual = self._prepare(inputs,
                                               use_ep=use_ep,
                                               use_sparse_core=False)
                    kernel.assert_not_called()
                    expected = np.argsort(np.asarray(actual[-1]))
                    np.testing.assert_array_equal(actual[2], expected)

    def test_downstream_combine_matches_original_argsort_path(self):
        # Exercise the real moe_gmm gather, weighting, masking, reduction and
        # output cast. CPU substitutes fixed GMM outputs; actual TPU GMM and
        # complete model parity remain separate controller gates.
        rng = np.random.default_rng(37)
        for dtype in (jnp.bfloat16, jnp.float32):
            for skip_padding in (False, True):
                with self.subTest(dtype=dtype, skip_padding=skip_padding):
                    inputs = self._inputs()
                    kwargs = dict(use_ep=False,
                                  use_sparse_core=False,
                                  skip_padded_tokens=skip_padding)
                    with patch.object(
                            self.routing, "stable_order_and_inverse_128",
                            functools.partial(stable_order_and_inverse_128,
                                              interpret=True)):
                        optimized = self._prepare(inputs, **kwargs)
                    # Adapt only the joint helper's test wiring, keeping the
                    # existing original-argsort reference below unchanged.
                    def original_order_chain(keys):
                        order = jnp.argsort(keys)
                        return (order,
                                self.routing.inverse_permutation_128(order))

                    # Restore exactly the original second argsort for the
                    # baseline; do not change any reference implementation.
                    with patch.object(self.routing, "inverse_permutation_128",
                                      jnp.argsort), patch.object(
                                          self.routing,
                                          "stable_order_and_inverse_128",
                                          original_order_chain):
                        baseline = self._prepare(inputs, **kwargs)
                    for got, want in zip(optimized, baseline):
                        np.testing.assert_array_equal(got, want)

                    routes = jnp.asarray(rng.standard_normal((128, 16)), dtype)
                    # Invalid GMM tail rows may be undefined, including NaN.
                    routes = jnp.where(optimized[4][:, None], routes, jnp.nan)
                    sorted_routes = routes[baseline[-1]]

                    def combine(metadata):
                        x, groups, inverse, weights, valid, order = metadata
                        with patch.object(self.routing,
                                          "gmm_wrapper",
                                          side_effect=[x, sorted_routes]):
                            return self.routing.moe_gmm(x,
                                                        jnp.zeros(
                                                            (256, 16, 32),
                                                            dtype),
                                                        None,
                                                        None,
                                                        jnp.zeros(
                                                            (256, 16, 16),
                                                            dtype),
                                                        None,
                                                        None,
                                                        groups,
                                                        inverse,
                                                        weights,
                                                        valid,
                                                        order,
                                                        activation="silu",
                                                        num_tokens=16,
                                                        topk=8,
                                                        **kwargs)

                    actual = combine(optimized)
                    np.testing.assert_array_equal(actual, combine(baseline))
                    # Independently combine in original token/slot order.
                    weighted = routes.reshape(16, 8, 16) * inputs[2][:, :,
                                                                     None]
                    expected = jnp.where(optimized[4].reshape(16, 8,
                                                              1), weighted,
                                         0).sum(axis=1).astype(inputs[0].dtype)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, inputs[0].dtype)


if __name__ == "__main__":
    unittest.main()
