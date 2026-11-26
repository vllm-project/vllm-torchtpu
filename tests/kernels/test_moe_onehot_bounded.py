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
"""CPU trace and admission tests for the bounded EP one-hot permute/combine."""

import contextlib
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.megablox import moe_onehot_bounded


@contextlib.contextmanager
def _cpu_tpu7_registry():
    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with patch.dict(
            tpu_info.registry,
            {
                "cpu": lambda: tpu_info.get_tpu_info_for_chip(
                    tpu_info.ChipVersion.TPU_7, 1
                )
            },
        ):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def _shape(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def test_minimax_decode_shapes_trace_on_cpu():
    # MiniMax-M3.1 c64 decode: 512 tokens x top-4 slots, hidden 6144.
    with _cpu_tpu7_registry():
        routed = jax.eval_shape(
            moe_onehot_bounded.bounded_onehot_permute,
            _shape((512, 6144), jnp.bfloat16),
            _shape((2048,), jnp.int32),
            _shape((1,), jnp.int32),
        )
        out = jax.eval_shape(
            moe_onehot_bounded.bounded_onehot_combine,
            _shape((2048, 6144), jnp.bfloat16),
            _shape((2048,), jnp.int32),
            _shape((512, 4), jnp.bfloat16),
            _shape((1,), jnp.int32),
        )
        fused = jax.eval_shape(
            moe_onehot_bounded.bounded_onehot_combine,
            _shape((2048, 6144), jnp.bfloat16),
            _shape((2048,), jnp.int32),
            _shape((512, 4), jnp.bfloat16),
            _shape((1,), jnp.int32),
            (
                _shape((512, 384), jnp.bfloat16),
                _shape((384, 6144), jnp.float8_e4m3fn),
                _shape((1, 6144), jnp.float32),
            ),
        )
    assert (routed.shape, routed.dtype) == ((2048, 6144), jnp.bfloat16)
    assert (out.shape, out.dtype) == ((512, 6144), jnp.bfloat16)
    assert (fused.shape, fused.dtype) == ((512, 6144), jnp.bfloat16)


def test_admission():
    with _cpu_tpu7_registry():
        assert moe_onehot_bounded.can_use_bounded_onehot(512, 2048, 6144)
        assert moe_onehot_bounded.can_use_bounded_onehot(256, 1024, 6144)
        # c64's 416-token decode bucket: tokens off the 128 lane width.
        assert moe_onehot_bounded.can_use_bounded_onehot(416, 1664, 6144)
        # Below 256 tokens off the lane width (c48's 192), partial sublane
        # tiles of tokens or 128-slot blocks stay dense.
        assert not moe_onehot_bounded.can_use_bounded_onehot(192, 768, 6144)
        assert not moe_onehot_bounded.can_use_bounded_onehot(24, 96, 6144)
        # The decode buckets with MiniMax-M3's shared expert (384 per rank).
        assert moe_onehot_bounded.can_use_bounded_onehot(384, 1536, 6144, 384)
        assert moe_onehot_bounded.can_use_bounded_onehot(256, 1024, 6144, 384)
        assert not moe_onehot_bounded.can_use_bounded_onehot(256, 1024, 6144, 200)
        # An f32 owner the size of a prefill chunk does not fit VMEM.
        assert not moe_onehot_bounded.can_use_bounded_onehot(4096, 16384, 6144)
