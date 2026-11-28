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
"""Tests for gmm_v2.

Most helper logic in this file is copied from
../tpu-inference/tests/kernels/gmm_test.py and adapted to this repo's pytest
test runner.
"""

import collections
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    Dimensions,
    InputConfigs,
    TileSizes,
    apply_act_fn,
    calculate_tiling,
    gmm_v2,
    interleave_lane,
    silu_and_mul_bf16,
    situ_and_mul,
)

_GroupConfig = collections.namedtuple(
    "_GroupConfig", ["num_groups", "group_offset", "num_local_groups"]
)


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _assert_allclose(actual, expected, atol=1e-5, rtol=1e-5) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul(linear_beta):
    gate = jnp.asarray([-3.0, 0.5, 7.0])
    up = jnp.asarray([-30.0, 2.0, 40.0])
    expected_gate = 4.0 * jnp.tanh(gate / 4.0) * jax.nn.sigmoid(gate)
    expected_up = (
        up if linear_beta is None else linear_beta * jnp.tanh(up / linear_beta)
    )

    _assert_allclose(
        situ_and_mul(gate, up, 4.0, linear_beta), expected_gate * expected_up
    )


@pytest.mark.parametrize("linear_beta", ["none", "25.0"])
def test_apply_act_fn_dispatches_situ(monkeypatch, linear_beta):
    monkeypatch.setattr(
        "vllm_torchtpu.kernels.megablox.gmm_v2.pltpu.get_tpu_info",
        lambda: SimpleNamespace(num_lanes=1),
    )
    gate = jnp.asarray([-3.0, 0.5, 7.0])
    up = jnp.asarray([-30.0, 2.0, 40.0])
    expected = situ_and_mul(
        gate,
        up,
        beta=4.0,
        linear_beta=None if linear_beta == "none" else float(linear_beta),
    )

    actual = apply_act_fn(interleave_lane(gate, up), f"situ:4.0:{linear_beta}")
    _assert_allclose(actual, expected)


def test_apply_act_fn_silu_bf16_rounds_intermediates_to_bfloat16(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.kernels.megablox.gmm_v2.pltpu.get_tpu_info",
        lambda: SimpleNamespace(num_lanes=1),
    )
    gate = jnp.asarray([-3.86777902], dtype=jnp.float32)
    up = jnp.asarray([2.9283731], dtype=jnp.float32)
    interleaved = interleave_lane(gate, up)

    actual = apply_act_fn(interleaved, "silu_bf16")

    # bf16(bf16(silu(bf16(gate))) * bf16(up)). Dropping any one of the three
    # intermediate casts changes the value (checked below), so the test fails
    # if the kernel skips one.
    expected = np.asarray([-0.2294921875], dtype=np.float32)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(actual), expected)
    np.testing.assert_array_equal(
        np.asarray(silu_and_mul_bf16(gate, up).astype(jnp.float32)), expected
    )
    bf16, f32 = jnp.bfloat16, jnp.float32
    silu_of_bf16_gate = jax.nn.silu(gate.astype(bf16).astype(f32))
    without_one_cast = {
        "gate": jax.nn.silu(gate).astype(bf16) * up.astype(bf16),
        "up": silu_of_bf16_gate.astype(bf16).astype(f32) * up,
        "silu": silu_of_bf16_gate * up.astype(bf16).astype(f32),
        "all": jax.nn.silu(gate) * up,
    }
    for name, value in without_one_cast.items():
        value = np.asarray(value.astype(bf16).astype(f32))
        assert not np.array_equal(value, expected), name


def get_group_sizes(batch_size: int, num_groups: int) -> jax.Array:
    distribution = jax.random.uniform(
        jax.random.key(0), (num_groups - 1,), dtype=jnp.float32
    )
    distribution = distribution / jnp.sum(distribution)
    group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
    return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def quantize_tensor(
    x: jax.Array, dtype: jnp.dtype, axis: int = -1, block_size: int = 256
):
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = int(dtype_info.max)
        min_val = int(dtype_info.min)
    else:
        dtype_info = jnp.finfo(dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)

    orig_shape = x.shape
    blocked_shape = orig_shape[:axis] + (-1, block_size) + orig_shape[axis + 1 :]
    x_blocked = x.reshape(blocked_shape)

    x_blocked_abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
    scale = x_blocked_abs_max / max_val
    x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)

    x_q = x_blocked_q.reshape(orig_shape)
    x_q = jnp.nan_to_num(x_q)
    scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
    return x_q, scale


def reference_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
):
    num_tokens = lhs.shape[0]
    num_groups, in_size, out_size = rhs.shape
    assert lhs.shape[1] == in_size

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]

    num_blocks = rhs_scale.shape[1] if rhs_scale is not None else 1
    block_size = in_size // num_blocks

    start = 0
    gmm_out = []
    for global_group in range(group_sizes.size):
        group_size = group_sizes[global_group]

        group = global_group - group_offset[0]
        end = min(start + group_size, num_tokens)
        group_size = end - start
        if 0 <= group < num_groups:
            lhs_slice = lhs[start:end]
            rhs_slice = rhs[group]

            out = 0
            for block in range(num_blocks):
                block_start = block * block_size
                block_end = block_start + block_size
                lhs_block = lhs_slice[:, block_start:block_end].astype(jnp.float32)
                rhs_block = rhs_slice[block_start:block_end, :].astype(jnp.float32)

                acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
                if rhs_scale is not None:
                    acc *= rhs_scale[group][block]
                out += acc
            if rhs_bias is not None:
                out = out + rhs_bias[group]
        else:
            out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

        gmm_out.append(out.astype(lhs.dtype))
        start = end

    return jnp.concat(gmm_out, axis=0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("group_offset", [0, 2])
def test_gmm_matches_reference(has_bias, group_offset):
    _require_tpu()

    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 16
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
        rhs_bias = jax.random.normal(
            key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
        )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs, rhs, group_sizes, rhs_bias=rhs_bias, group_offset=group_offset
    )

    actual = gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    _assert_allclose(actual, expected)


@pytest.mark.tpu_test
@pytest.mark.parametrize("weight_dtype", [jnp.int8, jnp.float8_e4m3fn])
@pytest.mark.parametrize("block_size", [256, 512])
def test_gmm_weight_quantized_matches_reference(weight_dtype, block_size):
    _require_tpu()

    batch_size = 128
    in_size = 1024
    out_size = 512
    num_groups = 16
    group_offset = jnp.array(0, dtype=jnp.int32)
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    expected = reference_gmm(
        lhs,
        rhs_q,
        get_group_sizes(batch_size, num_groups),
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_v2(
        lhs,
        rhs_q,
        get_group_sizes(batch_size, num_groups),
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    _assert_allclose(actual, expected, atol=3e-1, rtol=3e-1)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fuse_act", ["silu", "silu_bf16", "swigluoai"])
def test_gmm_fused_activation_matches_reference(fuse_act):
    _require_tpu()

    batch_size = 128
    in_size = 512
    out_size = 512
    final_out_size = out_size // 2
    num_groups = 16
    block_size = 512
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, jnp.int8, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    rhs_bias = jax.random.normal(key, (num_groups, 1, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array([0], dtype=jnp.int32)

    lhs_block_size = min(512, in_size)
    lhs_q, lhs_scale_factor = quantize_tensor(
        lhs, jnp.int8, axis=1, block_size=lhs_block_size
    )
    lhs_q_blocked = lhs_q.reshape(batch_size, -1, lhs_block_size).astype(jnp.float32)
    lhs_scale_expanded = jnp.expand_dims(lhs_scale_factor, axis=2)
    lhs_simulated = (
        (lhs_q_blocked * lhs_scale_expanded).reshape(lhs.shape).astype(lhs.dtype)
    )

    raw_expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )
    raw_gate, raw_up = jnp.split(raw_expected, 2, axis=-1)
    raw_expected = interleave_lane(raw_gate, raw_up)
    expected = apply_act_fn(raw_expected.astype(jnp.float32), fuse_act).astype(
        lhs.dtype
    )

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    assert actual.shape == (batch_size, final_out_size)
    _assert_allclose(actual, expected, atol=4.0, rtol=2.0)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "weight_dtype,maybe_quantize_lhs",
    [(None, False), (jnp.int8, False), (jnp.int8, True)],
)
def test_gmm_fused_silu_bf16_matches_unfused_silu_and_mul(
    weight_dtype, maybe_quantize_lhs
):
    _require_tpu()

    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 16
    key = jax.random.key(0)
    lhs_key, rhs_key, bias_key = jax.random.split(key, 3)

    lhs = jax.random.uniform(lhs_key, (batch_size, in_size), jnp.bfloat16, -1.0, 1.0)
    rhs = jax.random.uniform(
        rhs_key, (num_groups, in_size, out_size), jnp.bfloat16, -0.1, 0.1
    )
    rhs_bias = jax.random.uniform(
        bias_key, (num_groups, 1, out_size), jnp.bfloat16, -0.1, 0.1
    )
    rhs_scale = None
    if weight_dtype is not None:
        rhs, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=512)
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)

    def run(fuse_act):
        return gmm_v2(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            maybe_quantize_lhs=maybe_quantize_lhs,
            fuse_act=fuse_act,
        )

    # Unfused reference: the same GEMM writes bfloat16 gate/up, then a
    # standalone bfloat16 SiluAndMul.
    gate, up = jnp.split(run(None).astype(jnp.bfloat16), 2, axis=-1)
    expected = np.asarray(
        jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * up
    )
    actual_silu_bf16 = run("silu_bf16")
    actual_silu = np.asarray(run("silu").astype(jnp.bfloat16))

    assert actual_silu_bf16.dtype == jnp.bfloat16
    actual = np.asarray(actual_silu_bf16).astype(np.float32)
    # Every element within 2 bfloat16 ULPs; most are bit-equal.
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual, expected.astype(np.float32), rtol=2**-6, atol=2**-10
    )
    bf16_equal = np.mean(np.asarray(actual_silu_bf16) == expected)
    fp32_equal = np.mean(actual_silu == expected)
    assert bf16_equal > 0.99, (bf16_equal, fp32_equal)
    # Reverting to float32 intermediates must fail the check above.
    assert fp32_equal < 0.9, (bf16_equal, fp32_equal)


@pytest.mark.tpu_test
def test_gmm_weight_quantized_block_larger_than_tile_k():
    _require_tpu()

    batch_size = 128
    in_size = 1024
    out_size = 512
    num_groups = 16
    block_size = 1024
    tile_info = TileSizes(tile_m=128, bucket_base=128, tile_k=256, tile_n=out_size)
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(0, dtype=jnp.int32)

    expected = reference_gmm(
        lhs, rhs_q, group_sizes, rhs_scale=rhs_scale, group_offset=group_offset
    )

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    _assert_allclose(actual, expected, atol=3e-1, rtol=3e-1)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "group_config",
    [
        _GroupConfig(num_groups=16, group_offset=2, num_local_groups=4),
        _GroupConfig(num_groups=16, group_offset=0, num_local_groups=8),
    ],
)
def test_gmm_nonlocal_groups_produce_zeros(group_config):
    _require_tpu()

    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups, group_offset, num_local_groups = group_config
    lhs_key, rhs_key, bias_key = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.normal(lhs_key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        rhs_key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = jax.random.normal(
        bias_key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs, rhs, group_sizes, rhs_bias=rhs_bias, group_offset=group_offset
    )

    actual = gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    assert actual.shape == (batch_size, out_size)
    _assert_allclose(actual, expected)


@pytest.mark.tpu_test
def test_gmm_v2_int4_packed_matches_reference():
    _require_tpu()

    batch_size = 128
    logical_k = 512
    storage_k = logical_k // 8
    out_size = 256
    num_groups = 4
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, logical_k), jnp.bfloat16, -1, 1)

    weight_key, scale_key = jax.random.split(key)
    rhs_int4 = jax.random.randint(
        weight_key,
        (num_groups, logical_k, out_size),
        minval=-8,
        maxval=8,
        dtype=jnp.int32,
    )

    group_size = 64
    num_blocks = logical_k // group_size
    rhs_scale = jax.random.uniform(
        scale_key,
        (num_groups, num_blocks, 1, out_size),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.bfloat16,
    )

    scale_tiled = jnp.repeat(rhs_scale, group_size, axis=1)
    scale_tiled = jnp.squeeze(scale_tiled, axis=2)

    rhs_dequantized = rhs_int4.astype(jnp.bfloat16) * scale_tiled

    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_gmm(
        lhs,
        rhs_dequantized,
        group_sizes,
    )

    rhs_uint4 = (rhs_int4 + 8) & 0x0F

    rhs_reshaped = rhs_uint4.reshape(num_groups, storage_k, 8, out_size)

    shifts = jnp.arange(8, dtype=jnp.int32) * 4
    shifted = rhs_reshaped << shifts[None, None, :, None]
    rhs_packed_int32 = jnp.sum(shifted, axis=2).astype(jnp.int32)

    INT4_SIGN_XOR = -2004318072
    rhs_packed_int32_xor = rhs_packed_int32 ^ INT4_SIGN_XOR

    actual = gmm_v2(
        lhs,
        rhs_packed_int32_xor,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_quant_dtype=jnp.int4,
    )

    assert actual.shape == (batch_size, out_size)
    _assert_allclose(actual, expected, atol=3e-1, rtol=3e-1)


_E2M1_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def make_nvfp4_rhs(key, num_groups: int, size_k: int, size_n: int):
    """K-major uint8 [G, K/2, N] E2M1 pairs and E4M3 [G, K/16, N] block scales.

    Block scales are raw codes over the whole finite E4M3 range (0x7F is NaN),
    as real checkpoints put most block scales near the 448 maximum.
    """
    weight_key, scale_key = jax.random.split(key)
    packed = jax.random.randint(
        weight_key, (num_groups, size_k // 2, size_n), 0, 256, dtype=jnp.int32
    ).astype(jnp.uint8)
    codes = jax.random.randint(
        scale_key, (num_groups, size_k // 16, size_n), 0, 0x7F, dtype=jnp.int32
    ).astype(jnp.uint8)
    return packed, jax.lax.bitcast_convert_type(codes, jnp.float8_e4m3fn)


def make_global_scale(key, num_groups: int, size_n: int) -> jax.Array:
    """FP32 [G, 1, N] with a different multiplier on every output lane."""
    return jax.random.uniform(key, (num_groups, 1, size_n), minval=1e-3, maxval=5e-2)


def dequantize_nvfp4_kmajor(
    packed: jax.Array,
    block_scale: jax.Array,
    global_scale: jax.Array,
    dtype: jnp.dtype,
) -> jax.Array:
    """Row 2j is the low nibble of byte row j. The block-scaled weights are exact
    in FP32, and the global scale multiplies them before the one rounding."""
    lut = jnp.asarray(_E2M1_VALUES, dtype=jnp.float32)
    values = jnp.stack((lut[packed & 15], lut[packed >> 4]), axis=2)
    values = values.reshape(packed.shape[0], -1, packed.shape[-1])
    weights = values * jnp.repeat(block_scale.astype(jnp.float32), 16, axis=1)
    return (weights * global_scale).astype(dtype)


def read_back_nvfp4_rhs(packed, rhs_scale, **kwargs) -> jax.Array:
    """One-hot lhs rows make the output each group's dequantized rhs, exactly."""
    num_groups, size_k = packed.shape[0], packed.shape[1] * 2
    lhs = jnp.tile(jnp.eye(size_k, dtype=jnp.bfloat16), (num_groups, 1))
    out = gmm_v2(
        lhs,
        packed,
        jnp.full((num_groups,), size_k, dtype=jnp.int32),
        rhs_scale=rhs_scale,
        rhs_quant_dtype=jnp.float4_e2m1fn,
        maybe_quantize_lhs=False,
        **kwargs,
    )
    return out.reshape(num_groups, size_k, -1)


def _assert_bitwise_equal(actual, expected) -> None:
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tile_info",
    [
        calculate_tiling,
        TileSizes(tile_m=128, tile_k=512, tile_n=128, bucket_base=128),
    ],
    ids=["auto_tiles", "split_k_and_n"],
)
def test_gmm_v2_nvfp4_global_scale_decodes_every_weight(tile_info):
    _require_tpu()

    num_groups, size_k, size_n = 4, 1024, 512
    packed, scale = make_nvfp4_rhs(jax.random.key(0), num_groups, size_k, size_n)
    global_scale = make_global_scale(jax.random.key(1), num_groups, size_n)
    expected = dequantize_nvfp4_kmajor(packed, scale, global_scale, jnp.bfloat16)
    rhs_scale = jax.lax.bitcast_convert_type(scale, jnp.uint8)

    actual = read_back_nvfp4_rhs(
        packed,
        rhs_scale,
        rhs_global_scale=global_scale,
        tile_info=tile_info,
    )

    _assert_bitwise_equal(actual, expected)


@pytest.mark.tpu_test
def test_gmm_v2_nvfp4_global_scale_matches_dequantized_reference():
    _require_tpu()

    batch_size, num_groups, size_k, size_n = 512, 8, 1024, 512
    rhs_key, lhs_key = jax.random.split(jax.random.key(2))
    packed, scale = make_nvfp4_rhs(rhs_key, num_groups, size_k, size_n)
    global_scale = make_global_scale(jax.random.key(3), num_groups, size_n)
    rhs = dequantize_nvfp4_kmajor(packed, scale, global_scale, jnp.float32)
    lhs = jax.random.uniform(lhs_key, (batch_size, size_k), jnp.bfloat16, -1, 1)
    group_sizes = get_group_sizes(batch_size, num_groups)
    with jax.default_matmul_precision("highest"):
        expected = reference_gmm(lhs, rhs, group_sizes)
    rhs_scale = jax.lax.bitcast_convert_type(scale, jnp.uint8)

    actual = gmm_v2(
        lhs,
        packed,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_global_scale=global_scale,
        rhs_quant_dtype=jnp.float4_e2m1fn,
        maybe_quantize_lhs=False,
        preferred_element_type=jnp.bfloat16,
        acc_dtype=jnp.float32,
    )

    _assert_allclose(actual, expected, atol=2e-2, rtol=1e-2)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("size_k", "tile_info"),
    [
        (640, TileSizes(tile_m=128, tile_k=640, tile_n=256, bucket_base=128)),
        (2880, calculate_tiling),
    ],
    ids=["tile_k_640_all_blocks", "size_k_2880_auto_tiles"],
)
def test_gmm_v2_compact_scales_with_k_off_512(size_k, tile_info):
    """Compact scale tiles that hold all of K's blocks, or a K that auto tiling
    must split on a 512-row boundary, decode every weight exactly."""
    _require_tpu()

    num_groups, size_n = 2, 256
    packed, scale = make_nvfp4_rhs(jax.random.key(11), num_groups, size_k, size_n)
    global_scale = make_global_scale(jax.random.key(12), num_groups, size_n)
    expected = dequantize_nvfp4_kmajor(packed, scale, global_scale, jnp.bfloat16)

    actual = read_back_nvfp4_rhs(
        packed,
        jax.lax.bitcast_convert_type(scale, jnp.uint8),
        rhs_global_scale=global_scale,
        tile_info=tile_info,
    )

    _assert_bitwise_equal(actual, expected)


@pytest.mark.tpu_test
def test_gmm_v2_nvfp4_global_scale_fused_silu_matches_reference():
    """The fused gate/up path slices a per-lane global scale for each half."""
    _require_tpu()

    batch_size, num_groups, size_k, size_n = 512, 8, 1024, 512
    rhs_key, lhs_key = jax.random.split(jax.random.key(13))
    packed, scale = make_nvfp4_rhs(rhs_key, num_groups, size_k, size_n)
    global_scale = make_global_scale(jax.random.key(14), num_groups, size_n)
    rhs = dequantize_nvfp4_kmajor(packed, scale, global_scale, jnp.float32)
    lhs = jax.random.uniform(lhs_key, (batch_size, size_k), jnp.bfloat16, -1, 1)
    group_sizes = get_group_sizes(batch_size, num_groups)
    with jax.default_matmul_precision("highest"):
        reference = reference_gmm(lhs.astype(jnp.float32), rhs, group_sizes)
    gate, up = jnp.split(reference, 2, axis=-1)
    expected = apply_act_fn(interleave_lane(gate, up), "silu").astype(jnp.bfloat16)
    rhs_scale = jax.lax.bitcast_convert_type(scale, jnp.uint8)

    actual = gmm_v2(
        lhs,
        packed,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_global_scale=global_scale,
        rhs_quant_dtype=jnp.float4_e2m1fn,
        maybe_quantize_lhs=False,
        fuse_act="silu",
    )

    assert actual.shape == (batch_size, size_n // 2)
    scale_of_outputs = float(jnp.max(jnp.abs(expected.astype(jnp.float32))))
    _assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        atol=1e-3 * scale_of_outputs,
        rtol=2e-2,
    )


def _validation_tpu_info(monkeypatch, mxu_column_size: int = 256) -> None:
    monkeypatch.setattr(
        "vllm_torchtpu.kernels.megablox.gmm_v2.pltpu.get_tpu_info",
        lambda: SimpleNamespace(
            generation=7,
            mxu_column_size=mxu_column_size,
            num_lanes=128,
            vmem_capacity_bytes=64 * 1024 * 1024,
            fp8_ops_per_second=0,
            int8_ops_per_second=0,
            get_sublane_tiling=lambda dtype: 8
            * (32 // jax.dtypes.itemsize_bits(dtype)),
        ),
    )


def _nvfp4_validation_call(rhs_scale, rhs_global_scale, size_k=512, tile_k=512):
    num_groups, size_n = 2, 256
    packed, _ = make_nvfp4_rhs(jax.random.key(6), num_groups, size_k, size_n)
    return gmm_v2(
        jnp.zeros((num_groups * 8, size_k), jnp.bfloat16),
        packed,
        jnp.full((num_groups,), 8, dtype=jnp.int32),
        rhs_scale=rhs_scale,
        rhs_global_scale=rhs_global_scale,
        rhs_quant_dtype=jnp.float4_e2m1fn,
        maybe_quantize_lhs=False,
        tile_info=TileSizes(tile_m=16, tile_k=tile_k, tile_n=256, bucket_base=16),
    )


def test_gmm_v2_rejects_compact_scale_tiles_off_the_sublane_tiling(monkeypatch):
    _validation_tpu_info(monkeypatch)
    _, scale = make_nvfp4_rhs(jax.random.key(15), 2, 1024, 256)
    with pytest.raises(ValueError, match="got tile_k=384, quant block size 16"):
        _nvfp4_validation_call(
            jax.lax.bitcast_convert_type(scale, jnp.uint8),
            make_global_scale(jax.random.key(16), 2, 256),
            size_k=1024,
            tile_k=384,
        )


def _compact_tiling(
    monkeypatch, size_m, size_k, size_n, quant_block_size=16, vmem_mib=48
):
    """calculate_tiling for packed NVFP4 weights with compact E4M3 scales."""
    _validation_tpu_info(monkeypatch)
    dims = Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=8,
        size_lhs_group=8,
        size_lhs_sublane=16,
    )
    rhs_cfgs = InputConfigs(
        quant_dtype=jnp.float4_e2m1fn,
        quant_block_size=quant_block_size,
        dtype=jnp.uint8,
        has_scale=True,
        has_global_scale=True,
        has_compact_scale=True,
    )
    lhs_cfgs = InputConfigs(quant_dtype=None, quant_block_size=512, dtype=jnp.bfloat16)
    return calculate_tiling(dims, lhs_cfgs, rhs_cfgs, vmem_mib * 1024 * 1024)


@pytest.mark.parametrize(
    ("quant_block_size", "size_k", "tile_k"),
    [(16, 640, 640), (16, 2880, 1536), (64, 2880, 2048)],
    ids=["one_tile_holds_all_blocks", "block_16", "block_64"],
)
def test_calculate_tiling_aligns_compact_scale_tiles(
    monkeypatch, quant_block_size, size_k, tile_k
):
    """A K tile holds all of K's blocks or a multiple of 32 blocks (the uint8
    sublane tiling), with K split evenly."""
    tiles = _compact_tiling(monkeypatch, 1024, size_k, 4096, quant_block_size)
    assert tiles.tile_k == tile_k, tiles


def test_calculate_tiling_splits_k_when_tile_n_stops_at_its_limit(monkeypatch):
    """With N=512, tile_n starts at its limit (twice the MXU columns), so only
    a smaller tile_k can bring the estimate under the VMEM limit."""
    tiles = _compact_tiling(monkeypatch, 512, 65536, 512, vmem_mib=58)
    assert tiles.tile_n == 512 and tiles.tile_k < 65536, tiles


def test_gmm_v2_rejects_malformed_global_scale(monkeypatch):
    _validation_tpu_info(monkeypatch)
    _, scale = make_nvfp4_rhs(jax.random.key(8), 2, 512, 256)
    with pytest.raises(ValueError, match="rhs_global_scale must be FP32"):
        _nvfp4_validation_call(
            jax.lax.bitcast_convert_type(scale, jnp.uint8),
            jnp.ones((2, 1, 256), jnp.bfloat16),
        )


def test_gmm_v2_rejects_compact_scales_on_dequant_after_matmul(monkeypatch):
    _validation_tpu_info(monkeypatch, mxu_column_size=16)
    _, scale = make_nvfp4_rhs(jax.random.key(9), 2, 512, 256)
    with pytest.raises(NotImplementedError, match="not below the MXU column size 16"):
        _nvfp4_validation_call(jax.lax.bitcast_convert_type(scale, jnp.uint8), None)


def test_gmm_v2_rejects_partial_m_buckets_with_split_k(monkeypatch):
    _validation_tpu_info(monkeypatch)
    with pytest.raises(NotImplementedError, match="need a single K step"):
        gmm_v2(
            jnp.zeros((512, 2048), jnp.bfloat16),
            jnp.zeros((2, 2048, 256), jnp.bfloat16),
            jnp.full((2,), 256, dtype=jnp.int32),
            maybe_quantize_lhs=False,
            tile_info=TileSizes(tile_m=256, tile_k=1024, tile_n=256, bucket_base=64),
        )


def test_calculate_tiling_keeps_one_bucket_when_k_is_split(monkeypatch):
    """Partial-M buckets finalize after the first K step, so a K that compact
    scale tiles split (2880 rows) keeps one bucket."""
    tiles = _compact_tiling(monkeypatch, 5760, 2880, 256)
    assert tiles.tile_k < 2880 and tiles.tile_m == tiles.bucket_base, tiles


@pytest.mark.tpu_test
def test_gmm_v2_mxfp4_sparse_decode_fused_situ_matches_reference():
    _require_tpu()

    batch_size = 128
    in_size = 1536
    out_size = 1024
    num_groups = 16
    block_size = 32
    num_blocks = in_size // block_size
    fuse_act = "situ:4.0:25.0"
    key = jax.random.key(7)
    k_lhs, k_rhs, k_exp = jax.random.split(key, 3)

    lhs = jax.random.uniform(k_lhs, (batch_size, in_size), jnp.bfloat16, -1.0, 1.0)
    rhs_fp4 = jax.random.uniform(
        k_rhs, (num_groups, in_size, out_size), jnp.float32, -3.0, 3.0
    ).astype(jnp.float4_e2m1fn)
    exp_int = jax.random.randint(
        k_exp,
        (num_groups, num_blocks, 1, out_size),
        minval=-3,
        maxval=2,
        dtype=jnp.int32,
    )
    rhs_scale_e8m0 = jnp.exp2(exp_int.astype(jnp.float32)).astype(jnp.float8_e8m0fnu)
    group_sizes = get_group_sizes(batch_size, num_groups)

    scale_expanded = jnp.squeeze(
        jnp.repeat(rhs_scale_e8m0.astype(jnp.bfloat16), block_size, axis=1), axis=2
    )
    rhs_dequant = rhs_fp4.astype(jnp.bfloat16) * scale_expanded
    raw_expected = reference_gmm(lhs, rhs_dequant, group_sizes)
    raw_gate, raw_up = jnp.split(raw_expected, 2, axis=-1)
    expected = apply_act_fn(
        interleave_lane(raw_gate, raw_up).astype(jnp.float32), fuse_act
    ).astype(lhs.dtype)

    actual = gmm_v2(
        lhs,
        rhs_fp4,
        group_sizes,
        rhs_scale=rhs_scale_e8m0,
        maybe_quantize_lhs=False,
        fuse_act=fuse_act,
    )

    assert actual.shape == (batch_size, out_size // 2)
    _assert_allclose(actual, expected, atol=6e-2, rtol=6e-2)


def test_calculate_tiling_scales_bucket_base_with_group_density(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.kernels.megablox.gmm_v2.pltpu.get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    lhs_bf16 = InputConfigs(quant_dtype=None, quant_block_size=512, dtype=jnp.bfloat16)
    lhs_fp8 = InputConfigs(
        quant_dtype=jnp.float8_e4m3fn, quant_block_size=512, dtype=jnp.bfloat16
    )
    rhs_mxfp4 = InputConfigs(
        quant_dtype=jnp.float4_e2m1fn,
        quant_block_size=32,
        dtype=jnp.float4_e2m1fn,
        has_scale=True,
    )
    rhs_fp8 = InputConfigs(
        quant_dtype=jnp.float8_e4m3fn,
        quant_block_size=512,
        dtype=jnp.float8_e4m3fn,
        has_scale=True,
    )
    vmem_limit = 64 * 1024 * 1024

    def _tiles_for(
        size_m: int,
        *,
        size_group: int = 56,
        size_lhs_group: int | None = None,
        size_k: int = 3584,
        size_n: int = 3072,
        lhs_cfgs: InputConfigs = lhs_bf16,
        rhs_cfgs: InputConfigs = rhs_mxfp4,
        limit: int = vmem_limit,
        fuse_act: str | None = "silu",
    ) -> TileSizes:
        dims = Dimensions(
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            size_group=size_group,
            size_lhs_group=size_lhs_group or size_group,
            size_lhs_sublane=16,
        )
        return calculate_tiling(dims, lhs_cfgs, rhs_cfgs, limit, fuse_act=fuse_act)

    # TPU v7x sparse decode batches (< 24 rows/group, bf16 LHS) use {16, 32, 48, 64}.
    t_m64 = _tiles_for(64)
    assert (t_m64.bucket_base, t_m64.tile_m) == (16, 64)
    t_m1024 = _tiles_for(1024)
    assert (t_m1024.bucket_base, t_m1024.tile_m) == (16, 64)
    t_g96_m2048 = _tiles_for(2048, size_group=96)
    assert (t_g96_m2048.bucket_base, t_g96_m2048.tile_m) == (16, 64)
    # EP-sharded group_sizes (size_lhs_group=128 > size_group=16) uses size_lhs_group.
    t_ep = _tiles_for(1024, size_group=16, size_lhs_group=128)
    assert (t_ep.bucket_base, t_ep.tile_m) == (16, 64)

    # Dense / prefill batches (>= 24 rows/group) keep the full 64-row bucket_base.
    t_g56_m2048 = _tiles_for(2048, size_group=56)
    assert (t_g56_m2048.bucket_base, t_g56_m2048.tile_m) == (64, 256)
    t_prefill = _tiles_for(4096, size_group=56)
    assert (t_prefill.bucket_base, t_prefill.tile_m) == (64, 256)

    # W8A8 calls (lhs_cfgs.should_quantize=True) keep their 128-row bucket_base.
    t_w8a8 = _tiles_for(512, size_group=64, lhs_cfgs=lhs_fp8, rhs_cfgs=rhs_fp8)
    assert (t_w8a8.bucket_base, t_w8a8.tile_m) == (128, 512)

    # Single-group calls (size_group == 1) keep the full unscaled bucket_base.
    t_dense = _tiles_for(64, size_group=1)
    assert (t_dense.bucket_base, t_dense.tile_m) == (64, 64)

    # Unvalidated generations (e.g. v6e) keep the 64-row bucket_base.
    monkeypatch.setattr(
        "vllm_torchtpu.kernels.megablox.gmm_v2.pltpu.get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1),
    )
    t_v6e = _tiles_for(1024)
    assert (t_v6e.bucket_base, t_v6e.tile_m) == (64, 256)
