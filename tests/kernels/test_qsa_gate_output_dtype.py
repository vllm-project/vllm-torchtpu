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
"""Independent gating math, dtype, masking, and readonly input contracts."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import _gate_output


def check_math(raw, gate, lengths):
    before_raw = np.asarray(
        jax.lax.bitcast_convert_type(
            raw, jnp.uint16 if raw.dtype == jnp.bfloat16 else jnp.uint32
        )
    ).copy()
    before_gate = np.asarray(gate).view(np.uint32).copy()
    values = np.asarray(raw, np.float64)
    logistic = 1 / (1 + np.exp(-np.asarray(gate, np.float64)))
    # CPU FP32 multiplication flushes subnormal inputs and products.
    # Apply FP32 flushing separately from the normal-number error bound.
    tiny = np.finfo(np.float32).tiny
    values = np.where(np.abs(values) < tiny, 0, values)
    with np.errstate(invalid="ignore"):
        expected = values * logistic
    expected = np.where(np.abs(expected) < tiny, 0, expected)
    expected = np.where((np.asarray(lengths) > 0)[:, None, None], expected, 0)
    actual = _gate_output(raw, gate, jnp.asarray(lengths))
    assert actual.dtype == jnp.bfloat16 and actual.shape == raw.shape
    observed = np.asarray(actual, np.float64)
    np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(observed), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(observed), np.isneginf(expected))
    finite = np.isfinite(expected)
    info = ml_dtypes.finfo(ml_dtypes.bfloat16)
    # One final BF16 conversion bounds FP32 gating by one representable step.
    np.testing.assert_allclose(
        observed[finite],
        expected[finite],
        rtol=float(info.eps),
        atol=float(info.smallest_subnormal),
    )
    assert np.all(observed[np.asarray(lengths) <= 0] == 0)
    np.testing.assert_array_equal(
        before_raw,
        np.asarray(
            jax.lax.bitcast_convert_type(
                raw, jnp.uint16 if raw.dtype == jnp.bfloat16 else jnp.uint32
            )
        ),
    )
    np.testing.assert_array_equal(before_gate, np.asarray(gate).view(np.uint32))


def test_all_bf16_payload_math_and_readonly_input():
    bits = jnp.arange(65536, dtype=jnp.uint16).reshape(-1, 1, 1)
    raw = jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
    check_math(raw, jnp.zeros(raw.shape, jnp.float32), np.ones(len(raw), np.int32))


def test_float32_nonfinite_and_inactive_rows():
    words = np.array(
        [
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0x7F812345,
            0x7F800001,
            0xFFC12345,
        ],
        np.uint32,
    )
    raw = jnp.asarray(words.view(np.float32).reshape(-1, 1, 1))
    check_math(
        raw,
        jnp.zeros(raw.shape, jnp.float32),
        np.array([1, 1, 1, 1, 1, 1, 1, 0], np.int32),
    )


@pytest.mark.parametrize("tokens", [0, 1, 5, 17])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_finite_boundaries_tails_and_empty(tokens, dtype):
    words = np.array(
        [
            0x3F807FFF,
            0x3F808000,
            0x3F808001,
            0x3F817FFF,
            0x3F818000,
            0xBF808000,
            0xBF818000,
            0,
            0x80000000,
        ],
        np.uint32,
    )
    raw = jnp.asarray(np.resize(words.view(np.float32), (tokens, 1, 9)), dtype)
    gate = jnp.asarray(
        np.resize(np.array([-8.0, -1.0, 0.0, 1.0, 8.0], np.float32), raw.shape)
    )
    lengths = np.ones(tokens, np.int32)
    lengths[::3] = 0
    check_math(raw, gate, lengths)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_input_and_output_underflow_boundaries(dtype):
    words = np.array(
        [0x007F0000, 0x00800000, 0x01000000, 0x807F0000, 0x80800000, 0x81000000],
        np.uint32,
    )
    raw = jnp.asarray(words.view(np.float32).reshape(-1, 1, 1), dtype)
    gate = jnp.zeros(raw.shape, jnp.float32)
    check_math(raw, gate, np.ones(len(raw), np.int32))
    observed = np.asarray(_gate_output(raw, gate, jnp.ones(len(raw), jnp.int32)))
    assert np.all(observed[[0, 1, 3, 4]] == 0)
    assert observed[2, 0, 0] == np.finfo(np.float32).tiny
    assert observed[5, 0, 0] == -np.finfo(np.float32).tiny
