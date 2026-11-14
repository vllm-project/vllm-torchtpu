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
"""Joint stable order and inverse for 128 int32 keys vs a stable argsort.

``interpret=True`` runs the kernel body on the host, so the ranks, the
tie-break and the inverse are gated without a TPU. The ``tpu_test`` case
covers Mosaic lowering.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.stable_order_and_inverse import (
    stable_order_and_inverse_128,
)

_INT32 = np.iinfo(np.int32)
_EDGE_KEYS = {
    "all_equal": np.full(128, -17),
    "all_min": np.full(128, _INT32.min),
    "all_max": np.full(128, _INT32.max),
    "ascending": np.arange(128),
    "reversed": np.arange(127, -1, -1),
    "negative": np.arange(-1, -129, -1),
    "duplicates_across_tokens": np.tile([7, 0, 7, 255, 0, 255, 7, 0], 16),
    "sentinels": np.tile([256, 255, 0, 256, 0, 255, 256, 0], 16),
    "all_sentinel": np.full(128, 256),
    "extrema": np.tile(
        [_INT32.min, _INT32.max, -1, 0, 1, _INT32.min + 1, _INT32.max - 1, _INT32.min],
        16,
    ),
}

_interpreted = jax.jit(functools.partial(stable_order_and_inverse_128, interpret=True))


def _assert_stable_order_and_inverse(keys, order, inverse):
    order, inverse = np.asarray(order), np.asarray(inverse)
    assert order.shape == inverse.shape == (128,)
    assert order.dtype == inverse.dtype == np.int32
    expected_order = np.argsort(keys, kind="stable")
    expected_inverse = np.empty_like(expected_order)
    expected_inverse[expected_order] = np.arange(128)
    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_array_equal(inverse, expected_inverse)


def _check(keys):
    keys = np.asarray(keys, dtype=np.int32)
    _assert_stable_order_and_inverse(keys, *_interpreted(jnp.asarray(keys)))


@pytest.mark.parametrize("name", list(_EDGE_KEYS))
def test_matches_stable_argsort_on_edge_keys(name):
    _check(_EDGE_KEYS[name])


@pytest.mark.parametrize("distribution", ["full_int32", "duplicates"])
def test_matches_stable_argsort_on_random_keys(distribution):
    for seed in range(32):
        rng = np.random.default_rng(seed)
        if distribution == "full_int32":
            keys = rng.integers(
                _INT32.min, _INT32.max, size=128, dtype=np.int32, endpoint=True
            )
        else:
            keys = rng.integers(-8, 9, size=128, dtype=np.int32)
        _check(keys)


def test_matches_stable_argsort_with_smallest_key_at_every_position():
    for shift in range(128):
        _check(np.roll(np.arange(128), -shift))


def test_outputs_stay_int32_with_x64_enabled():
    with jax.enable_x64(True):
        _check(np.tile([_INT32.min, 0, _INT32.max, -1], 32))


@pytest.mark.parametrize("shape", [(0,), (127,), (129,), (16, 8), (1, 128)])
def test_rejects_shapes_other_than_128(shape):
    with pytest.raises(ValueError, match="shape"):
        stable_order_and_inverse_128(
            jax.ShapeDtypeStruct(shape, jnp.int32), interpret=True
        )


@pytest.mark.parametrize("dtype", [jnp.int16, jnp.int64, jnp.uint32, jnp.float32])
def test_rejects_non_int32_keys(dtype):
    with pytest.raises(TypeError, match="int32"):
        stable_order_and_inverse_128(
            jax.ShapeDtypeStruct((128,), dtype), interpret=True
        )


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


@pytest.mark.tpu_test
def test_native_kernel_matches_stable_argsort():
    _require_tpu()
    native = jax.jit(stable_order_and_inverse_128)
    rng = np.random.default_rng(1729)
    cases = [
        np.zeros(128, dtype=np.int32),
        np.arange(127, -1, -1, dtype=np.int32),
        np.tile(np.array([_INT32.min, _INT32.max, -1, 0], dtype=np.int32), 32),
        rng.integers(-8, 9, size=128, dtype=np.int32),
        rng.integers(_INT32.min, _INT32.max, size=128, dtype=np.int32, endpoint=True),
    ]
    for keys in cases:
        _assert_stable_order_and_inverse(keys, *native(jnp.asarray(keys)))
