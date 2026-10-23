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
"""Independent operation-order oracle for index RoPE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from qwen4_exp_exact_bf16 import (
    fma_bits,
    fp32_multiply_add_bits,
    fp32_rope_bits,
    rope_bits,
    rope_cases,
    round_bits,
)

from vllm_torchtpu.kernels.qwen4_exp.qsa import index_rope


def test_exact_oracle_ties_and_cancellation():
    assert round_bits(257, -8) == 0x3F80
    assert round_bits(259, -8) == 0x3F82
    assert round_bits(-257, -8) == 0xBF80
    assert fma_bits(0x3F80, 0x3F80, 0xBF80) == 0
    assert fma_bits(0, 0, 1) == 1
    assert fma_bits(0x8000, 0x3F80, 0x8000) == 0x8000


def test_simple_fp32_operations_are_not_a_fused_single_rounding_contract():
    cases = [(51784, 48944, 46164), (51898, 15136, 45408)]
    for row in cases:
        assert fp32_multiply_add_bits(*row) != fma_bits(*row)


def _check_rope():
    rows = rope_cases()
    bits = np.asarray(rows, np.uint16)
    x = jax.lax.bitcast_convert_type(jnp.asarray(bits[:, :2]), jnp.bfloat16)
    trig = jax.lax.bitcast_convert_type(jnp.asarray(bits[:, 2:]), jnp.bfloat16)
    trig = jnp.pad(trig, ((0, 0), (0, 126)))
    got = jax.jit(lambda x, t: index_rope(x, jnp.arange(len(x)), t, 2))(x, trig)
    expected = np.asarray([fp32_rope_bits(r) for r in rows], np.uint16)
    historical = np.asarray(
        [rope_bits(r, target="inner_b_cos") for r in rows], np.uint16
    )
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), expected)
    assert np.any(expected != historical), "This is not the historical fused target"


def test_cpu_jit_index_rope_matches_independent_fp32_operation_order():
    _check_rope()


@pytest.mark.tpu_test
def test_tpu_jit_index_rope_matches_independent_fp32_operation_order():
    assert jax.default_backend() == "tpu"
    _check_rope()


def test_index_rope_real_trig_positions_and_passthrough():
    rng = np.random.default_rng(1964)
    x = jnp.asarray(rng.normal(size=(17, 3, 128)), jnp.bfloat16)
    angles = np.arange(48)[:, None] / np.power(10000, np.arange(16)[None] / 16)
    trig = jnp.asarray(
        np.pad(np.concatenate([np.cos(angles), np.sin(angles)], -1), ((0, 0), (0, 96))),
        jnp.bfloat16,
    )
    positions = jnp.asarray([11, 0, 31, 2, 17, 3, 5, 9, 1, 12, 7, 8, 4, 6, 13, 15, 16])
    got = jax.jit(lambda a, p, t: index_rope(a, p, t, 32))(x, positions, trig)
    xb, tb = np.asarray(x).view(np.uint16), np.asarray(trig).view(np.uint16)
    expected = xb.copy()
    for i, pos in enumerate(np.asarray(positions)):
        for h in range(3):
            for d in range(16):
                expected[i, h, d], expected[i, h, d + 16] = fp32_rope_bits(
                    (
                        int(xb[i, h, d]),
                        int(xb[i, h, d + 16]),
                        int(tb[pos, d]),
                        int(tb[pos, d + 16]),
                    )
                )
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), expected)
