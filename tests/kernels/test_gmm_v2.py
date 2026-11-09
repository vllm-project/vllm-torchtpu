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

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    TileSizes,
    apply_act_fn,
    gmm_v2,
    gmm_v2_mlp,
    interleave_lane,
    mlp_supported,
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


@pytest.mark.parametrize("fuse_act", ["silu", "swigluoai"])
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
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
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

    assert actual.shape == (batch_size, out_size)
    _assert_allclose(actual, expected)


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


def _mlp_inputs(num_groups, size_k, size_inter, size_n, seed=0, block_size=256):
    keys = jax.random.split(jax.random.key(seed), 4)
    w1 = jax.random.uniform(
        keys[0], (num_groups, size_k, 2 * size_inter), jnp.float32, -6, 6
    ).astype(jnp.float4_e2m1fn)
    w2 = jax.random.uniform(
        keys[1], (num_groups, size_inter, size_n), jnp.float32, -6, 6
    ).astype(jnp.float4_e2m1fn)
    s1 = jax.random.uniform(
        keys[2],
        (num_groups, size_k // block_size, 1, 2 * size_inter),
        jnp.float32,
        0.005,
        0.02,
    )
    s2 = jax.random.uniform(
        keys[3],
        (num_groups, size_inter // block_size, 1, size_n),
        jnp.float32,
        0.005,
        0.02,
    )
    return w1, s1, w2, s2


def _two_call_mlp(lhs, w1, s1, w2, s2, group_sizes, group_offset):
    inter = gmm_v2(
        lhs,
        w1,
        group_sizes,
        rhs_scale=s1,
        group_offset=group_offset,
        fuse_act="silu",
        maybe_quantize_lhs=False,
        zero_initialize=False,
    )
    return gmm_v2(
        inter[:, : w2.shape[1]],
        w2,
        group_sizes,
        rhs_scale=s2,
        group_offset=group_offset,
        maybe_quantize_lhs=False,
        zero_initialize=False,
        preferred_element_type=lhs.dtype,
    )


def _assert_mlp_bitwise(
    lhs,
    w1,
    s1,
    w2,
    s2,
    sizes,
    group_offset,
    num_parts,
    down_parts=1,
    buckets=None,
    tile_rows=None,
):
    group_sizes = jnp.asarray(sizes, jnp.int32)
    offset = jnp.asarray([group_offset], jnp.int32)
    kwargs = dict(
        num_parts=num_parts,
        down_parts=down_parts,
        buckets=buckets,
        tile_rows=tile_rows,
    )
    assert mlp_supported(lhs, w1, s1, w2, s2, group_sizes, fuse_act="silu", **kwargs)
    expected = _two_call_mlp(lhs, w1, s1, w2, s2, group_sizes, offset)
    actual = gmm_v2_mlp(
        lhs,
        w1,
        w2,
        group_sizes,
        s1,
        s2,
        offset,
        fuse_act="silu",
        maybe_quantize_lhs=False,
        **kwargs,
    )
    assert actual.shape == expected.shape
    # Rows past the routed rows are not written by either path.
    start = sum(sizes[:group_offset])
    stop = start + sum(sizes[group_offset:])
    expected_bits = jax.lax.bitcast_convert_type(expected[start:stop], jnp.uint16)
    actual_bits = jax.lax.bitcast_convert_type(actual[start:stop], jnp.uint16)
    np.testing.assert_array_equal(np.asarray(actual_bits), np.asarray(expected_bits))


@pytest.mark.parametrize(
    "num_parts, down_parts, buckets, tile_rows",
    [
        (1, 1, None, None),
        (2, 1, None, None),
        (4, 1, None, None),
        (2, 2, None, None),
        # Row tiles as large as the gate/up kernel's.
        (2, 1, (64, 128, 192, 256), 256),
        (4, 2, (16, 32, 256), 256),
        (2, 1, (16, 256), 256),
        (2, 1, (16, 32), 32),
    ],
)
@pytest.mark.parametrize(
    "size_m, sizes, group_offset",
    [
        (128, [9, 0, 21, 3, 0, 14, 1, 30], 0),  # empty and uneven groups
        (128, [1] * 8, 0),  # one row per group
        (120, [0, 0, 0, 37, 0, 0, 0, 5], 0),  # empty leading groups, padded rows
        (128, [0] * 8, 0),  # no rows
        (640, [15, 1, 300, 17, 0, 129, 31, 2], 0),  # groups spanning tiles
        (128, [4, 6, 9, 3, 0, 14, 1, 30, 2, 5], 2),  # group offset
    ],
)
def test_gmm_v2_mlp_matches_two_calls_bitwise(
    size_m, sizes, group_offset, num_parts, down_parts, buckets, tile_rows
):
    _require_tpu()

    w1, s1, w2, s2 = _mlp_inputs(len(sizes) - group_offset, 1024, 512, 1024)
    lhs = jax.random.uniform(jax.random.key(1), (size_m, 1024), jnp.bfloat16, -1, 1)
    _assert_mlp_bitwise(
        lhs,
        w1,
        s1,
        w2,
        s2,
        sizes,
        group_offset,
        num_parts,
        down_parts,
        buckets,
        tile_rows,
    )


@pytest.mark.parametrize(
    "buckets, tile_rows", [(None, None), (None, 128), (16, 128), (64, 128)]
)
@pytest.mark.parametrize(
    "size_m, num_groups, num_rows",
    [(256, 16, 120), (960, 64, 120), (240, 64, 30)],
)
def test_gmm_v2_mlp_matches_two_calls_bitwise_served_shapes(
    size_m, num_groups, num_rows, buckets, tile_rows
):
    _require_tpu()

    w1, s1, w2, s2 = _mlp_inputs(num_groups, 8192, 2048, 8192)
    rng = np.random.RandomState(num_groups + num_rows)
    sizes = [
        int(v) for v in rng.multinomial(num_rows, np.ones(num_groups) / num_groups)
    ]
    # Large activations drive some gate pre-activations below the sigmoid's
    # underflow, so the activation holds signed zeros.
    lhs = jax.random.uniform(jax.random.key(2), (size_m, 8192), jnp.bfloat16, -50, 50)
    buckets = (buckets,) if buckets else None
    _assert_mlp_bitwise(
        lhs, w1, s1, w2, s2, sizes, 0, 2, buckets=buckets, tile_rows=tile_rows
    )


def test_gmm_v2_mlp_supported_shapes():
    _require_tpu()

    lhs = jax.ShapeDtypeStruct((128, 1024), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    w1, s1, w2, s2 = _mlp_inputs(8, 1024, 512, 1024)
    assert mlp_supported(lhs, w1, s1, w2, s2, sizes, fuse_act="silu")
    # No fused activation, or the block-16 scales applied before the matmul.
    assert not mlp_supported(lhs, w1, s1, w2, s2, sizes, fuse_act=None)
    w1, s1, w2, s2 = _mlp_inputs(8, 1024, 512, 1024, block_size=16)
    assert not mlp_supported(lhs, w1, s1, w2, s2, sizes, fuse_act="silu")
    # Activations the gate/up kernel would quantize.
    w1, s1, w2, s2 = _mlp_inputs(8, 1024, 512, 1024)
    w1 = w1.astype(jnp.float8_e4m3fn)
    w2 = w2.astype(jnp.float8_e4m3fn)
    assert not mlp_supported(
        lhs, w1, s1, w2, s2, sizes, fuse_act="silu", maybe_quantize_lhs=True
    )


def test_gmm_v2_mlp_row_results_do_not_depend_on_their_neighbours():
    """A replica of an expert computes a route's row in another group and beside
    other rows. The same 20 rows give the same bits alone in a group (a 32-row
    bucket), among 40 more rows (one 64-row tile), across a tile boundary in a
    300-row group, and in a ninth group holding a replica of the expert's weights
    (served sizes)."""
    _require_tpu()

    w1, s1, w2, s2 = _mlp_inputs(8, 8192, 2048, 8192)
    # Group 8 holds a replica of expert 3.
    experts = jnp.asarray(list(range(8)) + [3])
    w1, s1, w2, s2 = (a[experts] for a in (w1, s1, w2, s2))
    rows = jax.random.uniform(jax.random.key(4), (20, 8192), jnp.bfloat16, -1, 1)
    other = jax.random.uniform(jax.random.key(5), (320, 8192), jnp.bfloat16, -1, 1)

    def layout(group, before, after):
        """lhs, group sizes and the first of the 20 rows, which sit in
        ``group`` after ``before`` and before ``after`` of its other rows."""
        sizes = [3, 4, 0, 10, 5, 0, 2, 6, 0]
        sizes[group] = before + 20 + after
        parts, used = [], 0
        for g, n in enumerate(sizes):
            if g == group:
                parts += [other[used : used + before], rows]
                parts.append(other[used + before : used + before + after])
                used += before + after
            else:
                parts.append(other[used : used + n])
                used += n
        return jnp.concatenate(parts), sizes, sum(sizes[:group]) + before

    results = []
    for group, before, after in ((3, 0, 0), (3, 20, 20), (3, 120, 160), (8, 0, 0)):
        lhs, sizes, start = layout(group, before, after)
        group_sizes = jnp.asarray(sizes, jnp.int32)
        assert mlp_supported(lhs, w1, s1, w2, s2, group_sizes, fuse_act="silu")
        out = gmm_v2_mlp(
            lhs,
            w1,
            w2,
            group_sizes,
            s1,
            s2,
            jnp.zeros((1,), jnp.int32),
            fuse_act="silu",
            maybe_quantize_lhs=False,
        )
        bits = jax.lax.bitcast_convert_type(out[start : start + 20], jnp.uint16)
        results.append(np.asarray(bits))
    assert (results[0] != 0).any()
    for got in results[1:]:
        np.testing.assert_array_equal(got, results[0])
