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

from vllm_torchtpu.kernels.megablox import gmm_v2 as gmm_v2_module
from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    TileSizes,
    apply_act_fn,
    gmm_v2,
    interleave_lane,
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


# The Qwen3.5 FP8 MoE paths behind gmm_v2.RHS_STREAM (hand-streamed
# whole-expert rhs slabs) and gmm_v2.TUNED_FP8_MOE (tuned tiles and the
# rhs-scale hoist). Both are off by default; the Qwen3.5 kernel-numerics rule
# turns them on for Qwen3.5-397B-A17B on TPU7x. gmm_v2 is jitted and reads
# both flags at trace time, so every run below clears JAX's caches around it.
_FP8_MOE_K, _FP8_MOE_N = 1024, 4096  # GMM2: the down projection, one k tile
# (lhs groups, weight slots, group offset, group sizes). "spread": local groups
# 4..11 of 16, four of them non-empty with empty groups between and after,
# and rows of non-local groups on both sides. "single": one non-empty local
# group, so the stream has one run. "slots": the weights hold two more slots
# than group_sizes lists (a copy-slot layer); the kernel must not read them.
# "tail" and "tail_empty": at offset 4 the 8 weight slots outnumber the 4 lhs
# groups left, and the trailing local groups (all of them in "tail_empty") are
# empty, so the stream's search for its first group must stop at the lhs
# groups.
_FP8_MOE_LAYOUTS = {
    "spread": (16, 8, 4, [5, 7, 0, 9, 12, 0, 20, 0, 0, 16, 8, 0, 30, 0, 21, 0]),
    "single": (16, 8, 4, [10, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 54, 0, 0, 0]),
    "slots": (8, 10, 0, [12, 0, 20, 0, 16, 8, 0, 72]),
    "tail": (8, 8, 4, [5, 7, 9, 12, 0, 6, 0, 0]),
    "tail_empty": (8, 8, 4, [5, 7, 9, 12, 0, 0, 0, 0]),
}
_WHOLE_EXPERT_TILES = TileSizes(
    tile_m=128, tile_k=_FP8_MOE_K, tile_n=_FP8_MOE_N, bucket_base=128
)
# An 80-row GMM2 over 64 local experts: an entry of the tuned tile table.
_TUNED_TABLE_SIZES = [3, 0, 2, 0] * 16
_FP8_MOE_TOLERANCE = 0.05


def _channel_magnitudes(key, slots, n):
    """A factor of 2**U(-2, 2) per (weight slot, output channel). Uniform(-1, 1)
    bf16 weights reach the same amax in every column, so their per-channel
    scales would all be equal, and a scale applied to the wrong channel or
    expert would change nothing; with these factors it changes the output by
    far more than the tests' tolerances."""
    return jnp.exp2(jax.random.uniform(key, (slots, 1, n), jnp.float32, -2, 2))


def _fp8_moe_inputs(slots, group_sizes, as_arrays=True):
    """bf16 rows and a GMM2-shaped FP8 weight with one scale per output channel
    (a single quant block over k), as FP8 MoE checkpoints load."""
    rows = int(sum(group_sizes))
    if not as_arrays:
        return (
            jax.ShapeDtypeStruct((rows, _FP8_MOE_K), jnp.bfloat16),
            jax.ShapeDtypeStruct((slots, _FP8_MOE_K, _FP8_MOE_N), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((slots, 1, 1, _FP8_MOE_N), jnp.float32),
        )
    lhs_key, rhs_key = jax.random.split(jax.random.key(0))
    lhs = jax.random.uniform(lhs_key, (rows, _FP8_MOE_K), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        rhs_key, (slots, _FP8_MOE_K, _FP8_MOE_N), jnp.float32, -1, 1
    )
    magnitudes = _channel_magnitudes(jax.random.fold_in(rhs_key, 1), slots, _FP8_MOE_N)
    rhs = (rhs * magnitudes).astype(jnp.bfloat16)
    rhs_q, rhs_scale = quantize_tensor(
        rhs, jnp.float8_e4m3fn, axis=1, block_size=_FP8_MOE_K
    )
    return lhs, rhs_q, jnp.expand_dims(rhs_scale, axis=2)


def _fp8_moe_configs(slots, group_sizes, tile_info):
    """The configs gmm_v2 builds for these inputs on the current device."""
    lhs, rhs, rhs_scale = _fp8_moe_inputs(slots, group_sizes, as_arrays=False)
    vmem = int(gmm_v2_module.pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    return gmm_v2_module.make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        None,
        jax.ShapeDtypeStruct((len(group_sizes),), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        tile_info=tile_info,
        vmem_limit_bytes=vmem,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=True,
    )


def _relative_l2(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


@pytest.mark.parametrize("layout", sorted(_FP8_MOE_LAYOUTS))
@pytest.mark.parametrize("tuned", [False, True], ids=["main_order", "tuned"])
@pytest.mark.parametrize("rhs_stream", [False, True], ids=["pipeline", "stream"])
def test_fp8_moe_cases_take_the_paths_they_test(monkeypatch, rhs_stream, tuned, layout):
    """On TPU7x (chip description only), the device cases below stream the rhs
    slabs exactly when RHS_STREAM is set and hoist the rhs scale exactly when
    TUNED_FP8_MOE is set, and the tuned case takes the tuned table's tiles."""
    info = gmm_v2_module.pltpu.get_tpu_info_for_chip(
        gmm_v2_module.pltpu.ChipVersion.TPU_7X, 1
    )
    monkeypatch.setattr(gmm_v2_module.pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(gmm_v2_module, "RHS_STREAM", rhs_stream)
    monkeypatch.setattr(gmm_v2_module, "TUNED_FP8_MOE", tuned)
    _, slots, _, sizes = _FP8_MOE_LAYOUTS[layout]
    cfgs = _fp8_moe_configs(slots, sizes, _WHOLE_EXPERT_TILES)
    assert cfgs.lhs_cfgs.should_quantize
    assert cfgs.dims.size_group == min(slots, len(sizes))
    assert bool(gmm_v2_module.use_rhs_stream(cfgs)) is rhs_stream
    assert gmm_v2_module.hoist_rhs_scale(cfgs) is tuned
    table = _fp8_moe_configs(64, _TUNED_TABLE_SIZES, gmm_v2_module.calculate_tiling)
    tiles = (table.tiles.tile_m, table.tiles.bucket_base)
    assert (tiles == gmm_v2_module._FP8_MOE_GMM2_TILES[80]) is tuned


def _run_fp8_moe(
    monkeypatch, rhs_stream, tuned, lhs, rhs, rhs_scale, sizes, offset, tile_info
):
    monkeypatch.setattr(gmm_v2_module, "RHS_STREAM", rhs_stream)
    monkeypatch.setattr(gmm_v2_module, "TUNED_FP8_MOE", tuned)
    jax.clear_caches()
    try:
        return gmm_v2(
            lhs,
            rhs,
            jnp.asarray(sizes, dtype=jnp.int32),
            rhs_scale=rhs_scale,
            group_offset=jnp.array(offset, dtype=jnp.int32),
            tile_info=tile_info,
        )
    finally:
        jax.clear_caches()


@pytest.mark.tpu_test
@pytest.mark.parametrize("layout", sorted(_FP8_MOE_LAYOUTS))
@pytest.mark.parametrize("tuned", [False, True], ids=["main_order", "tuned"])
@pytest.mark.parametrize("rhs_stream", [False, True], ids=["pipeline", "stream"])
def test_gmm_fp8_moe_whole_expert_paths_match_reference(
    monkeypatch, rhs_stream, tuned, layout
):
    _require_tpu()
    _, slots, offset, sizes = _FP8_MOE_LAYOUTS[layout]
    lhs, rhs, rhs_scale = _fp8_moe_inputs(slots, sizes)
    expected = reference_gmm(
        lhs,
        rhs,
        jnp.asarray(sizes, dtype=jnp.int32),
        rhs_scale=rhs_scale,
        group_offset=jnp.array(offset, dtype=jnp.int32),
    )
    actual = _run_fp8_moe(
        monkeypatch,
        rhs_stream,
        tuned,
        lhs,
        rhs,
        rhs_scale,
        sizes,
        offset,
        _WHOLE_EXPERT_TILES,
    )
    assert actual.shape == expected.shape
    if not np.any(np.asarray(expected)):
        # No local rows: nothing is computed and the output stays zero.
        assert not np.any(np.asarray(actual))
        return
    # The kernel quantizes the lhs to fp8 per 512-wide block, which alone puts
    # it about 2.6% (relative L2) from this unquantized reference; a slab from
    # the wrong expert or group is off by more than 100%.
    assert _relative_l2(actual, expected) < _FP8_MOE_TOLERANCE


@pytest.mark.tpu_test
@pytest.mark.parametrize("tuned", [False, True], ids=["main_tiles", "tuned_tiles"])
@pytest.mark.parametrize("rhs_stream", [False, True], ids=["pipeline", "stream"])
def test_gmm_fp8_moe_tuned_table_shape_matches_reference(
    monkeypatch, rhs_stream, tuned
):
    _require_tpu()
    lhs, rhs, rhs_scale = _fp8_moe_inputs(64, _TUNED_TABLE_SIZES)
    expected = reference_gmm(
        lhs,
        rhs,
        jnp.asarray(_TUNED_TABLE_SIZES, dtype=jnp.int32),
        rhs_scale=rhs_scale,
    )
    actual = _run_fp8_moe(
        monkeypatch,
        rhs_stream,
        tuned,
        lhs,
        rhs,
        rhs_scale,
        _TUNED_TABLE_SIZES,
        0,
        gmm_v2_module.calculate_tiling,
    )
    assert actual.shape == expected.shape
    assert _relative_l2(actual, expected) < _FP8_MOE_TOLERANCE


# GMM1, the fused gate/up projection the GMM1 table covers: k=4096, n=2048 (gate
# then up, 1024 columns each), fuse_act="silu", 80 rows over 64 local experts.
_FP8_MOE_GMM1_K, _FP8_MOE_GMM1_N = 4096, 2048


def _fp8_moe_gmm1_inputs(as_arrays=True, k=_FP8_MOE_GMM1_K):
    """bf16 rows and an FP8 gate/up weight with one scale per output channel.
    ``k`` is the contracting width; the device case uses the GMM1 table's."""
    rows, slots = int(sum(_TUNED_TABLE_SIZES)), len(_TUNED_TABLE_SIZES)
    n = _FP8_MOE_GMM1_N
    if not as_arrays:
        return (
            jax.ShapeDtypeStruct((rows, k), jnp.bfloat16),
            jax.ShapeDtypeStruct((slots, k, n), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((slots, 1, 1, n), jnp.float32),
        )
    lhs_key, rhs_key = jax.random.split(jax.random.key(1))
    lhs = jax.random.uniform(lhs_key, (rows, k), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(rhs_key, (slots, k, n), jnp.float32, -1, 1)
    rhs = rhs * _channel_magnitudes(jax.random.fold_in(rhs_key, 1), slots, n)
    rhs_q, rhs_scale = quantize_tensor(
        rhs.astype(jnp.bfloat16), jnp.float8_e4m3fn, axis=1, block_size=k
    )
    return lhs, rhs_q, jnp.expand_dims(rhs_scale, axis=2)


@pytest.mark.parametrize("tuned", [False, True], ids=["main_tiles", "tuned_tiles"])
def test_the_fused_gate_up_case_takes_the_gmm1_table_and_hoist(monkeypatch, tuned):
    """On TPU7x (chip description only), the GMM1 device case below takes the
    GMM1 table's tiles and hoists the rhs scale exactly when TUNED_FP8_MOE is
    set, and never streams (only non-fused projections stream)."""
    info = gmm_v2_module.pltpu.get_tpu_info_for_chip(
        gmm_v2_module.pltpu.ChipVersion.TPU_7X, 1
    )
    monkeypatch.setattr(gmm_v2_module.pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(gmm_v2_module, "RHS_STREAM", True)
    monkeypatch.setattr(gmm_v2_module, "TUNED_FP8_MOE", tuned)
    lhs, rhs, rhs_scale = _fp8_moe_gmm1_inputs(as_arrays=False)
    cfgs = gmm_v2_module.make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        None,
        jax.ShapeDtypeStruct((len(_TUNED_TABLE_SIZES),), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        tile_info=gmm_v2_module.calculate_tiling,
        vmem_limit_bytes=int(info.vmem_capacity_bytes * 0.9),
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        fuse_act="silu",
    )
    tiles = (cfgs.tiles.tile_m, cfgs.tiles.bucket_base)
    assert (tiles == gmm_v2_module._FP8_MOE_GMM1_TILES[80]) is tuned
    assert gmm_v2_module.hoist_rhs_scale(cfgs) is tuned
    assert not gmm_v2_module.use_rhs_stream(cfgs)


@pytest.mark.parametrize("which", ["gmm2", "gmm1"])
def test_the_fp8_moe_inputs_give_each_channel_and_expert_its_own_scale(which):
    """The device cases can only catch a scale applied to the wrong output
    channel (the fused gate/up halves) or expert (a streamed slab's scale) if
    those scales differ."""
    if which == "gmm1":
        # A short contracting width: each scale is one (slot, channel)'s amax
        # over k, so k does not change how the scales vary, and the full
        # 64 x 4096 x 2048 float32 weight would take 2 GiB on the CPU lane.
        _, _, scale = _fp8_moe_gmm1_inputs(k=128)
    else:
        _, _, scale = _fp8_moe_inputs(8, _FP8_MOE_LAYOUTS["spread"][3])
    s = np.asarray(scale, dtype=np.float64)[:, 0, 0, :]
    half = s.shape[1] // 2
    # Halves of one expert, and two experts: typical ratios of 2 to 4.
    assert np.median(np.abs(np.log2(s[:, :half] / s[:, half:]))) > 0.5
    assert np.median(np.abs(np.log2(s[0] / s[1]))) > 0.5


@pytest.mark.tpu_test
@pytest.mark.parametrize("tuned", [False, True], ids=["main_tiles", "tuned_tiles"])
def test_gmm_fp8_moe_fused_gate_up_matches_reference(monkeypatch, tuned):
    """GMM1 with the fused SiLU-and-multiply, on the default tiles and order and
    on the tuned GMM1 tiles with the hoisted rhs scale, against the unquantized
    reference followed by the same activation."""
    _require_tpu()
    lhs, rhs, rhs_scale = _fp8_moe_gmm1_inputs()
    sizes = jnp.asarray(_TUNED_TABLE_SIZES, dtype=jnp.int32)
    raw = reference_gmm(lhs, rhs, sizes, rhs_scale=rhs_scale)
    raw_gate, raw_up = jnp.split(raw, 2, axis=-1)
    expected = apply_act_fn(
        interleave_lane(raw_gate, raw_up).astype(jnp.float32), "silu"
    ).astype(lhs.dtype)
    monkeypatch.setattr(gmm_v2_module, "TUNED_FP8_MOE", tuned)
    jax.clear_caches()
    try:
        actual = gmm_v2(lhs, rhs, sizes, rhs_scale=rhs_scale, fuse_act="silu")
    finally:
        jax.clear_caches()
    assert actual.shape == expected.shape == (lhs.shape[0], _FP8_MOE_GMM1_N // 2)
    # The lhs is quantized to fp8 per 512-wide block in the kernel and not in
    # the reference, and the product of the two projections compounds that
    # error; a wrong expert's weights or scales are off by more than 100%.
    assert _relative_l2(actual, expected) < 2 * _FP8_MOE_TOLERANCE
