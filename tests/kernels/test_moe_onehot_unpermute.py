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
"""CPU trace and validation tests for blockwise output one-hot combine."""

import contextlib
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.megablox import moe_onehot_unpermute


def _shape(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _tpu7_info():
    return tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7, 1)


def _tpu6e_info():
    return tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V6E, 1)


@contextlib.contextmanager
def _cpu_tpu7_registry():
    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with patch.dict(tpu_info.registry, {"cpu": _tpu7_info}):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def test_qwen35_target_shape_traces_on_cpu():
    with _cpu_tpu7_registry():
        output = jax.eval_shape(
            lambda routes, tokens, weights, count: moe_onehot_unpermute.
            blockwise_onehot_unpermute(
                routes,
                tokens,
                weights,
                count,
                num_tokens=256,
            ),
            _shape((2560, 4096), jnp.bfloat16),
            _shape((2560, ), jnp.int32),
            _shape((2560, ), jnp.bfloat16),
            _shape((1, ), jnp.int32),
        )

    assert output.shape == (256, 4096)
    assert output.dtype == jnp.bfloat16


def test_qwen35_target_shape_fits_explicit_vmem_budget():
    with _cpu_tpu7_registry():
        info = _tpu7_info()
        explicit = moe_onehot_unpermute._explicit_vmem_bytes(
            route_capacity=2560,
            hidden_size=4096,
            num_tokens=256,
        )
        estimated = (explicit +
                     moe_onehot_unpermute._VMEM_WORKSPACE_MARGIN_BYTES)

    assert explicit == 8409088
    assert estimated == 12603392
    assert estimated < int(info.vmem_capacity_bytes * 0.9)


def test_qwen35_target_shape_is_not_enabled_on_unmeasured_tpu_generation():
    with patch.object(
            moe_onehot_unpermute.pltpu,
            "get_tpu_info",
            return_value=_tpu6e_info(),
    ):
        assert not moe_onehot_unpermute.can_use_blockwise_onehot_unpermute(
            _shape((2560, 4096), jnp.bfloat16),
            _shape((2560, ), jnp.int32),
            _shape((2560, ), jnp.bfloat16),
            _shape((1, ), jnp.int32),
            num_tokens=256,
        )


@pytest.mark.parametrize(
    ("routes", "tokens", "weights", "count", "num_tokens"),
    [
        (
            _shape((2432, 4096), jnp.bfloat16),
            _shape((2432, ), jnp.int32),
            _shape((2432, ), jnp.bfloat16),
            _shape((1, ), jnp.int32),
            256,
        ),
        (
            _shape((2560, 4096), jnp.float32),
            _shape((2560, ), jnp.int32),
            _shape((2560, ), jnp.bfloat16),
            _shape((1, ), jnp.int32),
            256,
        ),
        (
            _shape((2560, 4096), jnp.bfloat16),
            _shape((2560, ), jnp.int32),
            _shape((2560, ), jnp.float32),
            _shape((1, ), jnp.int32),
            256,
        ),
        (
            _shape((2560, 4096), jnp.bfloat16),
            _shape((2560, ), jnp.int32),
            _shape((2560, ), jnp.bfloat16),
            _shape((1, ), jnp.int32),
            128,
        ),
    ],
)
def test_unmeasured_or_incompatible_shapes_use_dense_fallback(
    routes,
    tokens,
    weights,
    count,
    num_tokens,
):
    with _cpu_tpu7_registry():
        assert not moe_onehot_unpermute.can_use_blockwise_onehot_unpermute(
            routes,
            tokens,
            weights,
            count,
            num_tokens=num_tokens,
        )
