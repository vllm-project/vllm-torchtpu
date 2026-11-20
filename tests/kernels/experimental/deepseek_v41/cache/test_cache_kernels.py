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
"""Lean CPU interpret and TPU7x hardware tests for DeepSeek-V4.1 cache kernels."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
    DEFAULT_PAGE_TOKENS,
    GATHER_PAGE_BURST_ROWS,
    GATHER_SUB_BURST_ROWS,
    INDEXER_HEAD_DIM,
    INDEXER_KEY_WORDS,
    LATENT_DIM,
    NOPE_DIM,
    ROPE_DIM,
    ROPE_WORDS,
    ROW_BYTES,
    ROW_WORDS,
    VALID_SLICE_WORDS,
    WORD_BYTES,
    benchmark_cache,
    gather_cache_rows,
    pack_fp8_kv_rows,
    write_sub_rows,
)

SCALE_GROUP_DIM = 64
FP8_MIN_AMAX = 1e-4
FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
MIN_INT32 = -(1 << 31)
MAX_INT32 = (1 << 31) - 1

_JIT_WRITE_SUB_ROWS = jax.jit(
    write_sub_rows,
    static_argnames=("page_tokens", "interpret", "enable_named_scope"),
)
_JIT_PACK_FP8 = jax.jit(
    pack_fp8_kv_rows,
    static_argnames=("interpret", "enable_named_scope"),
)
_JIT_GATHER_ROWS = jax.jit(
    gather_cache_rows,
    static_argnames=("interpret", "enable_named_scope"),
)


@pytest.fixture(autouse=True)
def _tpu7x_abstract_mesh(request: pytest.FixtureRequest) -> Iterator[None]:
    if jax.default_backend() == "tpu":
        yield
        return
    if request.node.get_closest_marker("tpu_test") is not None:
        pytest.skip("Requires TPU hardware backend")
    device = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
        yield


def _oracle_write_sub_rows(
    cache_words: np.ndarray,
    updates: np.ndarray,
    slot_mapping: np.ndarray,
) -> np.ndarray:
    expected = cache_words.copy()
    num_rows = expected.shape[0]
    slice_words = updates.shape[1]
    sub_slots_per_row = ROW_WORDS // slice_words
    total_slots = num_rows * sub_slots_per_row
    for token_idx, raw_slot in enumerate(slot_mapping.tolist()):
        if 0 <= raw_slot < total_slots:
            row_idx, lane_group = divmod(raw_slot, sub_slots_per_row)
            lane_start = lane_group * slice_words
            expected[row_idx, lane_start : lane_start + slice_words] = updates[
                token_idx
            ]
    return expected


def _oracle_pack_fp8_kv_rows(latent_bf16: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    num_tokens = latent_bf16.shape[0]
    if num_tokens == 0:
        return (
            np.zeros((0, ROW_WORDS), dtype=np.int32),
            np.zeros((0, ROPE_WORDS), dtype=np.int32),
        )
    latent_np = np.asarray(latent_bf16)
    nope_f32 = latent_np[:, :NOPE_DIM].astype(np.float32)
    nope_groups = nope_f32.reshape(
        num_tokens, NOPE_DIM // SCALE_GROUP_DIM, SCALE_GROUP_DIM
    )
    amax = np.clip(np.max(np.abs(nope_groups), axis=1), FP8_MIN_AMAX, None)
    scale_ue8m0 = np.exp2(np.ceil(np.log2(amax / FP8_MAX))).astype(jnp.float8_e8m0fnu)
    exact_scale_f32 = scale_ue8m0.astype(np.float32)
    quantized_nope = (
        (nope_groups / exact_scale_f32[:, None, :])
        .astype(jnp.float8_e4m3fn)
        .reshape(num_tokens, NOPE_DIM)
    )
    quantized_u8 = quantized_nope.view(np.uint8)
    scale_u8 = scale_ue8m0.view(np.uint8)
    slabs_u32 = (
        np.concatenate([quantized_u8, scale_u8], axis=1)
        .reshape(num_tokens, 4, ROW_WORDS)
        .astype(np.uint32)
    )
    nope_u32 = (
        slabs_u32[:, 0, :]
        | (slabs_u32[:, 1, :] << np.uint32(8))
        | (slabs_u32[:, 2, :] << np.uint32(16))
        | (slabs_u32[:, 3, :] << np.uint32(24))
    )
    nope_words = nope_u32.view(np.int32)
    rope_u32 = latent_np[:, NOPE_DIM:].view(np.uint16).astype(np.uint32)
    rope_words = (
        rope_u32[:, :ROPE_WORDS] | (rope_u32[:, ROPE_WORDS:] << np.uint32(16))
    ).view(np.int32)
    return nope_words, rope_words


def _dequantize_packed_kv(nope_words: np.ndarray, rope_words: np.ndarray) -> np.ndarray:
    num_tokens = nope_words.shape[0]
    nope_u32 = nope_words.view(np.uint32)
    out_f32 = np.zeros((num_tokens, LATENT_DIM), dtype=np.float32)
    scale_bytes = ((nope_u32[:, 64:] >> np.uint32(24)) & np.uint32(0xFF)).astype(
        np.uint32
    )
    scales_f32 = (scale_bytes << np.uint32(23)).view(np.float32)
    slabs_u8 = np.stack(
        [
            ((nope_u32 >> np.uint32(8 * plane)) & np.uint32(0xFF)).astype(np.uint8)
            for plane in range(4)
        ],
        axis=1,
    ).reshape(num_tokens, 4 * ROW_WORDS)
    fp8_vals = slabs_u8[:, :NOPE_DIM].view(jnp.float8_e4m3fn).astype(np.float32)
    scale_full = np.tile(scales_f32, (1, NOPE_DIM // SCALE_GROUP_DIM))
    out_f32[:, :NOPE_DIM] = fp8_vals * scale_full

    rope_u32 = rope_words.view(np.uint32)
    rope_u16 = np.empty((num_tokens, ROPE_DIM), dtype=np.uint16)
    rope_u16[:, :ROPE_WORDS] = (rope_u32 & np.uint32(0xFFFF)).astype(np.uint16)
    rope_u16[:, ROPE_WORDS:] = ((rope_u32 >> np.uint32(16)) & np.uint32(0xFFFF)).astype(
        np.uint16
    )
    out_f32[:, NOPE_DIM:] = rope_u16.view(jnp.bfloat16).astype(np.float32)
    return out_f32


def _oracle_gather_cache_rows(pool: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    num_rows = pool.shape[0]
    expected = np.zeros((row_indices.shape[0], ROW_WORDS), dtype=np.int32)
    for q_idx, row_idx in enumerate(row_indices.tolist()):
        if 0 <= row_idx < num_rows:
            expected[q_idx] = pool[row_idx]
    return expected


def test_should_export_expected_layout_constants() -> None:
    assert ROW_WORDS == 128
    assert WORD_BYTES == 4
    assert ROW_BYTES == 512
    assert NOPE_DIM == 448
    assert ROPE_DIM == 64
    assert LATENT_DIM == 512
    assert ROPE_WORDS == 32
    assert INDEXER_HEAD_DIM == 128
    assert INDEXER_KEY_WORDS == 64
    assert DEFAULT_PAGE_TOKENS == 16
    assert GATHER_PAGE_BURST_ROWS == 32
    assert GATHER_SUB_BURST_ROWS == 8
    assert VALID_SLICE_WORDS == (32, 64, 128)


def test_should_reject_non_tpu7x_device_unconditionally() -> None:
    if jax.default_backend() == "tpu":
        pytest.skip("Abstract non-TPU7x check runs under CPU backend")
    v6e = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=v6e)
    with jax.sharding.use_abstract_mesh(mesh):
        pool = jnp.zeros((8, ROW_WORDS), dtype=jnp.int32)
        upd = jnp.ones((4, ROPE_WORDS), dtype=jnp.int32)
        slots = jnp.arange(4, dtype=jnp.int32)
        lat = jnp.zeros((8, LATENT_DIM), dtype=jnp.bfloat16)
        with pytest.raises(RuntimeError, match="TPU7x"):
            write_sub_rows(pool, upd, slots, interpret=True)
        with pytest.raises(RuntimeError, match="TPU7x"):
            pack_fp8_kv_rows(lat, interpret=True)
        with pytest.raises(RuntimeError, match="TPU7x"):
            gather_cache_rows(pool, slots, interpret=True)


def test_should_execute_under_multirank_dp_abstract_mesh() -> None:
    if jax.default_backend() == "tpu":
        pytest.skip("Abstract multirank DP mesh check runs under CPU backend")
    device = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    mesh = jax.sharding.AbstractMesh(
        (2, 4),
        ("data", "model"),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        abstract_device=device,
    )
    with jax.sharding.use_abstract_mesh(mesh):
        pool = jax.ShapeDtypeStruct((16, ROW_WORDS), jnp.int32)
        upd = jax.ShapeDtypeStruct((8, ROPE_WORDS), jnp.int32)
        slots = jax.ShapeDtypeStruct((8,), jnp.int32)
        exported = jax.export.export(
            jax.jit(
                lambda c, u, s: gather_cache_rows(
                    write_sub_rows(c, u, s, interpret=False), s[:2], interpret=False
                )
            ),
            platforms=["tpu"],
        )(pool, upd, slots)
        assert "tpu_custom_call" in exported.mlir_module()


@pytest.mark.parametrize(
    ("slice_words", "num_tokens", "page_tokens", "enable_named_scope"),
    [
        (ROPE_WORDS, 3, 8, True),
        (INDEXER_KEY_WORDS, 2, 8, False),
        (ROW_WORDS, 3, DEFAULT_PAGE_TOKENS, False),
    ],
)
def test_should_handle_small_sub_row_counts_and_duplicates(
    slice_words: int,
    num_tokens: int,
    page_tokens: int,
    enable_named_scope: bool,
) -> None:
    rng = np.random.default_rng(100 + slice_words + num_tokens)
    num_rows = 6
    sub_slots_per_row = ROW_WORDS // slice_words
    total_slots = num_rows * sub_slots_per_row
    pool_warm = rng.integers(100, 999, size=(num_rows, ROW_WORDS), dtype=np.int32)
    pool_cold = np.zeros_like(pool_warm)
    updates = rng.integers(-5000, 5000, size=(num_tokens, slice_words), dtype=np.int32)
    if slice_words == ROPE_WORDS:
        slots = np.array([5, 5, -1], dtype=np.int32)
        after_tok: jax.Array | None = jnp.ones((1,), jnp.bfloat16)
    elif slice_words == INDEXER_KEY_WORDS:
        slots = np.array([total_slots - 1, total_slots], dtype=np.int32)
        after_tok = jnp.zeros((0,), jnp.int32)
    else:
        slots = np.array([0, 4, 5], dtype=np.int32)
        after_tok = None

    for pool_np in (pool_cold, pool_warm):
        got = np.asarray(
            _JIT_WRITE_SUB_ROWS(
                jnp.asarray(pool_np),
                jnp.asarray(updates),
                jnp.asarray(slots),
                after=after_tok,
                page_tokens=page_tokens,
                interpret=True,
                enable_named_scope=enable_named_scope,
            )
        )
        exp = _oracle_write_sub_rows(pool_np, updates, slots)
        np.testing.assert_array_equal(got, exp)


def test_should_handle_multiblock_bursts_partial_merges_all_invalid_and_after() -> None:
    slice_words = ROPE_WORDS
    page_tokens = 8
    rng = np.random.default_rng(303 + slice_words)
    num_tokens = 72
    num_rows = 48
    pool_np = rng.integers(10, 500, size=(num_rows, ROW_WORDS), dtype=np.int32)
    updates_np = rng.integers(-400, 400, size=(num_tokens, slice_words), dtype=np.int32)
    total_slots = num_rows * (ROW_WORDS // slice_words)

    write_fn = jax.jit(partial(write_sub_rows, page_tokens=page_tokens, interpret=True))
    prior_read = jnp.asarray(pool_np[:3])

    # All-invalid slots leave warm pool bitwise unchanged (including int32 extrema).
    invalid_slots = np.tile(
        np.array([-1, MIN_INT32, total_slots, MAX_INT32], dtype=np.int32),
        num_tokens // 4,
    )
    unchanged = np.asarray(
        write_fn(
            jnp.asarray(pool_np),
            jnp.asarray(updates_np),
            jnp.asarray(invalid_slots),
            after=prior_read[0, 0],
        )
    )
    np.testing.assert_array_equal(unchanged, pool_np)

    # Contiguous page bursts, partial sub-row merges, duplicates, and OOB slots.
    slots_np = np.concatenate(
        [
            np.arange(0, 32, dtype=np.int32),
            np.arange(48, 72, dtype=np.int32),
            np.array(
                [
                    1,
                    1,
                    38,
                    41,
                    43,
                    -1,
                    MIN_INT32,
                    total_slots - 1,
                    total_slots,
                    MAX_INT32,
                ],
                dtype=np.int32,
            ),
            np.full((num_tokens - 66,), -1, dtype=np.int32),
        ]
    )
    got = np.asarray(
        write_fn(
            jnp.asarray(pool_np),
            jnp.asarray(updates_np),
            jnp.asarray(slots_np),
            after=prior_read[0, 0],
        )
    )
    exp = _oracle_write_sub_rows(pool_np, updates_np, slots_np)
    np.testing.assert_array_equal(got, exp)

    # Lock structural data dependency from `after` into pallas_call SMEM operand.
    for sample_after in (
        jnp.array([17, -9], dtype=jnp.int32),
        jnp.array([-2.0, 1.5], dtype=jnp.float8_e4m3fn),
    ):
        closed = jax.make_jaxpr(
            lambda c, u, s, a: write_sub_rows(
                c, u, s, after=a, page_tokens=page_tokens, interpret=True
            )
        )(
            jnp.asarray(pool_np[:6]),
            jnp.asarray(updates_np[:3]),
            jnp.asarray(slots_np[:3]),
            sample_after,
        )
        var_cls = type(closed.jaxpr.invars[3])
        tainted = {closed.jaxpr.invars[3]}
        pallas_eqns = []
        for eqn in closed.jaxpr.eqns:
            if any(isinstance(v, var_cls) and v in tainted for v in eqn.invars):
                tainted.update(eqn.outvars)
            if eqn.primitive.name == "pallas_call":
                pallas_eqns.append(eqn)
        assert len(pallas_eqns) == 1
        assert pallas_eqns[0].invars[0] in tainted


@pytest.mark.parametrize(("num_rows", "num_tokens"), [(0, 4), (4, 0), (0, 0)])
def test_should_return_unchanged_cache_when_rows_or_tokens_zero(
    num_rows: int, num_tokens: int
) -> None:
    pool = jnp.asarray(np.ones((num_rows, ROW_WORDS), dtype=np.int32))
    updates = jnp.asarray(np.ones((num_tokens, ROPE_WORDS), dtype=np.int32))
    slots = jnp.asarray(np.zeros((num_tokens,), dtype=np.int32))
    out = write_sub_rows(pool, updates, slots, interpret=True)
    assert out.shape == (num_rows, ROW_WORDS)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(pool))


@pytest.mark.parametrize(
    (
        "cache_shape",
        "cache_dtype",
        "upd_shape",
        "upd_dtype",
        "slot_shape",
        "slot_dtype",
        "page_tokens",
    ),
    [
        ((ROW_WORDS,), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, 16),
        ((2, 4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, 16),
        ((4, 64), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, 16),
        ((4, ROW_WORDS), jnp.float32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, 16),
        ((4, ROW_WORDS), jnp.int32, (ROPE_WORDS,), jnp.int32, (2,), jnp.int32, 16),
        ((4, ROW_WORDS), jnp.int32, (2, 16), jnp.int32, (2,), jnp.int32, 16),
        ((4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.float32, (2,), jnp.int32, 16),
        ((4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.int32, (3,), jnp.int32, 16),
        ((4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.float32, 16),
        ((4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, 0),
        ((4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, -8),
        ((4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, 6),
        ((4, ROW_WORDS), jnp.int32, (2, ROPE_WORDS), jnp.int32, (2,), jnp.int32, 12),
    ],
)
def test_should_raise_value_error_on_invalid_write_sub_rows_inputs(
    cache_shape: tuple[int, ...],
    cache_dtype: jnp.dtype,
    upd_shape: tuple[int, ...],
    upd_dtype: jnp.dtype,
    slot_shape: tuple[int, ...],
    slot_dtype: jnp.dtype,
    page_tokens: int,
) -> None:
    cache_words = jnp.zeros(cache_shape, dtype=cache_dtype)
    updates = jnp.zeros(upd_shape, dtype=upd_dtype)
    slots = jnp.zeros(slot_shape, dtype=slot_dtype)
    with pytest.raises(ValueError):
        write_sub_rows(
            cache_words, updates, slots, page_tokens=page_tokens, interpret=True
        )


@pytest.mark.parametrize(
    ("num_tokens", "enable_named_scope"),
    [(0, False), (3, True)],
)
def test_should_match_fp8_and_rope_oracle_when_packing_latents(
    num_tokens: int, enable_named_scope: bool
) -> None:
    rng = np.random.default_rng(404 + num_tokens)
    latent_np = rng.normal(0.0, 2.5, size=(num_tokens, LATENT_DIM)).astype(np.float32)
    if num_tokens >= 3:
        # Inject lane-strided outliers, exact zeros, subnormals, and finite extrema.
        latent_np[0, :] = 0.01
        latent_np[0, 0] = 320.0
        latent_np[0, 131] = -18.0
        latent_np[1, :] = 0.0
        latent_np[2, :NOPE_DIM] = 1e-6
        latent_np[2, 63] = -4096.0
        latent_np[2, NOPE_DIM:] = 1024.0
    latent_bf16 = jnp.asarray(latent_np, dtype=jnp.bfloat16)
    got_nope, got_rope = _JIT_PACK_FP8(
        latent_bf16, interpret=True, enable_named_scope=enable_named_scope
    )
    exp_nope, exp_rope = _oracle_pack_fp8_kv_rows(latent_bf16)
    np.testing.assert_array_equal(np.asarray(got_nope), exp_nope)
    np.testing.assert_array_equal(np.asarray(got_rope), exp_rope)
    if num_tokens >= 3:
        # Scalar per-channel byte-level check of lane-strided UE8M0 scales and FP8 data.
        latent_f32 = np.asarray(latent_bf16).astype(np.float32)
        got_nope_u32 = np.asarray(got_nope).view(np.uint32)
        for tok in range(2):
            expected_scales_f32 = np.zeros(SCALE_GROUP_DIM, dtype=np.float32)
            for j in range(SCALE_GROUP_DIM):
                strided_vals = [
                    abs(float(latent_f32[tok, j + 64 * k])) for k in range(7)
                ]
                amax = max(max(strided_vals), FP8_MIN_AMAX)
                exponent = math.ceil(math.log2(amax / FP8_MAX))
                expected_scale_u8 = exponent + 127
                got_scale_u8 = int((got_nope_u32[tok, 64 + j] >> 24) & 0xFF)
                assert got_scale_u8 == expected_scale_u8
                expected_scales_f32[j] = math.ldexp(1.0, exponent)
            for chan in range(NOPE_DIM):
                plane = chan // ROW_WORDS
                lane = chan % ROW_WORDS
                got_fp8_u8 = int((got_nope_u32[tok, lane] >> (8 * plane)) & 0xFF)
                exp_fp8_u8 = int(
                    np.float32(
                        latent_f32[tok, chan]
                        / expected_scales_f32[chan % SCALE_GROUP_DIM]
                    )
                    .astype(jnp.float8_e4m3fn)
                    .view(np.uint8)
                )
                assert got_fp8_u8 == exp_fp8_u8
        # Discriminate 64 lane-strided scales from 7 contiguous blocks on token 0.
        assert int((got_nope_u32[0, 64] >> 24) & 0xFF) > int(
            (got_nope_u32[0, 65] >> 24) & 0xFF
        )


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((LATENT_DIM,), jnp.bfloat16),
        ((2, 4, LATENT_DIM), jnp.bfloat16),
        ((4, 256), jnp.bfloat16),
        ((4, LATENT_DIM), jnp.float32),
    ],
)
def test_should_raise_value_error_on_invalid_pack_fp8_kv_rows_inputs(
    shape: tuple[int, ...], dtype: jnp.dtype
) -> None:
    latent = jnp.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError):
        pack_fp8_kv_rows(latent, interpret=True)


def test_should_handle_empty_and_small_pools_in_gather_cache_rows() -> None:
    empty_pool = jnp.zeros((0, ROW_WORDS), dtype=jnp.int32)
    idx = jnp.array([0, 1, -1], dtype=jnp.int32)
    out_empty_pool = gather_cache_rows(empty_pool, idx, interpret=True)
    np.testing.assert_array_equal(
        np.asarray(out_empty_pool), np.zeros((3, ROW_WORDS), dtype=np.int32)
    )

    rng = np.random.default_rng(505)
    pool_tiny = rng.integers(10, 999, size=(6, ROW_WORDS), dtype=np.int32)
    out_empty_q = gather_cache_rows(
        jnp.asarray(pool_tiny), jnp.zeros((0,), dtype=jnp.int32), interpret=True
    )
    assert out_empty_q.shape == (0, ROW_WORDS)

    # num_rows=6 < 8 exercises early return in DMA helper for 32- and 8-row bursts.
    idx_tiny = np.array([-1, 0, 5, 6, 2, 3, MIN_INT32, MAX_INT32], dtype=np.int32)
    got_tiny = np.asarray(
        _JIT_GATHER_ROWS(
            jnp.asarray(pool_tiny),
            jnp.asarray(idx_tiny),
            interpret=True,
            enable_named_scope=True,
        )
    )
    np.testing.assert_array_equal(
        got_tiny, _oracle_gather_cache_rows(pool_tiny, idx_tiny)
    )


def test_should_gather_bursts_scattered_and_straddling_oob_runs() -> None:
    rng = np.random.default_rng(606)
    num_rows = 60
    pool = rng.integers(10, 9999, size=(num_rows, ROW_WORDS), dtype=np.int32)
    # 128 queries = four 32-row page chunks covering:
    # - chunk 0: aligned 32-row page burst in [8, 40)
    # - chunk 1: 8-row sub-burst [40..47], tight +1 straddling end [53..60],
    #            tight -1 straddling start [-1..6], 8-row scattered/OOB
    # - chunk 2: 32-row contiguous run tight +1 straddling pool end [29..60]
    # - chunk 3: 32-row contiguous run tight -1 straddling pool start [-1..30]
    indices = np.concatenate(
        [
            np.arange(8, 40, dtype=np.int32),
            np.arange(40, 48, dtype=np.int32),
            np.arange(53, 61, dtype=np.int32),
            np.arange(-1, 7, dtype=np.int32),
            np.array([59, -1, 0, 60, 27, MIN_INT32, MAX_INT32, 12], dtype=np.int32),
            np.arange(29, 61, dtype=np.int32),
            np.arange(-1, 31, dtype=np.int32),
        ]
    )
    got = np.asarray(
        _JIT_GATHER_ROWS(jnp.asarray(pool), jnp.asarray(indices), interpret=True)
    )
    exp = _oracle_gather_cache_rows(pool, indices)
    np.testing.assert_array_equal(got, exp)


def test_should_roundtrip_pack_write_and_gather_composed_pipeline() -> None:
    rng = np.random.default_rng(707)
    num_tokens = 3
    latent_bf16 = jnp.asarray(
        rng.normal(0.0, 1.5, size=(num_tokens, LATENT_DIM)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    nope_words, rope_words = _JIT_PACK_FP8(
        latent_bf16, interpret=True, enable_named_scope=True
    )
    nope_pool = jnp.zeros((6, ROW_WORDS), dtype=jnp.int32)
    rope_pool = jnp.zeros((6, ROW_WORDS), dtype=jnp.int32)
    slots = jnp.array([1, 4, 2], dtype=jnp.int32)
    rope_slots = jnp.array([2, 13, 21], dtype=jnp.int32)

    nope_pool = _JIT_WRITE_SUB_ROWS(
        nope_pool,
        nope_words,
        slots,
        after=None,
        page_tokens=DEFAULT_PAGE_TOKENS,
        interpret=True,
        enable_named_scope=False,
    )
    rope_pool = _JIT_WRITE_SUB_ROWS(
        rope_pool,
        rope_words,
        rope_slots,
        after=jnp.ones((1,), jnp.bfloat16),
        page_tokens=8,
        interpret=True,
        enable_named_scope=True,
    )
    nope_idx_8 = jnp.pad(slots, (0, 8 - num_tokens), constant_values=-1)
    rope_idx_8 = jnp.pad(rope_slots // 4, (0, 8 - num_tokens), constant_values=-1)
    gathered_nope = np.asarray(
        _JIT_GATHER_ROWS(
            nope_pool, nope_idx_8, interpret=True, enable_named_scope=True
        )[:num_tokens]
    )
    gathered_rope_rows = np.asarray(
        _JIT_GATHER_ROWS(
            rope_pool, rope_idx_8, interpret=True, enable_named_scope=True
        )[:num_tokens]
    )
    gathered_rope = np.stack(
        [
            gathered_rope_rows[
                i, (int(s) % 4) * ROPE_WORDS : ((int(s) % 4) + 1) * ROPE_WORDS
            ]
            for i, s in enumerate(np.asarray(rope_slots))
        ],
        axis=0,
    )
    reconstructed = _dequantize_packed_kv(gathered_nope, gathered_rope)
    reference_f32 = np.asarray(latent_bf16.astype(jnp.float32))
    np.testing.assert_allclose(reconstructed[:, NOPE_DIM:], reference_f32[:, NOPE_DIM:])
    np.testing.assert_allclose(
        reconstructed[:, :NOPE_DIM], reference_f32[:, :NOPE_DIM], rtol=0.15, atol=0.25
    )


def test_should_accept_exact_vmem_capacity_and_reject_one_byte_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exact VMEM bytes for pack_fp8_kv_rows with T=3 (padded tile_tokens=8):
    # DOUBLE_BUFFER * 8 * (512 * 2 + (128 + 32) * 4) = 26,624 B.
    exact_pack_bytes = 2 * 8 * (LATENT_DIM * 2 + (ROW_WORDS + ROPE_WORDS) * 4)
    latent = jnp.zeros((3, LATENT_DIM), dtype=jnp.bfloat16)

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            generation=7, num_lanes=ROW_WORDS, vmem_capacity_bytes=exact_pack_bytes
        ),
    )
    # Avoid _JIT_PACK_FP8 cache hits bypassing Python-side _check_vmem_budget.
    closed = jax.make_jaxpr(
        lambda x: pack_fp8_kv_rows(x, interpret=True, enable_named_scope=True)
    )(latent)
    assert tuple(v.aval.shape for v in closed.jaxpr.outvars) == (
        (3, ROW_WORDS),
        (3, ROPE_WORDS),
    )

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            generation=7, num_lanes=ROW_WORDS, vmem_capacity_bytes=exact_pack_bytes - 1
        ),
    )
    with pytest.raises(ValueError, match="VMEM"):
        pack_fp8_kv_rows(latent, interpret=True, enable_named_scope=True)


def test_should_aot_export_cache_kernels_for_tpu7x() -> None:
    if jax.default_backend() == "tpu":
        pytest.skip("Offline AOT export check runs under CPU AbstractDevice mesh")
    pool = jax.ShapeDtypeStruct((2048, ROW_WORDS), jnp.int32)
    for sw in VALID_SLICE_WORDS:
        upd = jax.ShapeDtypeStruct((64, sw), jnp.int32)
        slots = jax.ShapeDtypeStruct((64,), jnp.int32)
        exported_w = jax.export.export(
            jax.jit(
                lambda c, u, s: write_sub_rows(c, u, s, interpret=False),
                donate_argnums=(0,),
            ),
            platforms=["tpu"],
        )(pool, upd, slots)
        assert "tpu_custom_call" in exported_w.mlir_module()

    lat = jax.ShapeDtypeStruct((64, LATENT_DIM), jnp.bfloat16)
    exported_p = jax.export.export(
        jax.jit(lambda x: pack_fp8_kv_rows(x, interpret=False)),
        platforms=["tpu"],
    )(lat)
    assert "tpu_custom_call" in exported_p.mlir_module()

    idx = jax.ShapeDtypeStruct((64,), jnp.int32)
    exported_g = jax.export.export(
        jax.jit(lambda p, i: gather_cache_rows(p, i, interpret=False)),
        platforms=["tpu"],
    )(pool, idx)
    assert "tpu_custom_call" in exported_g.mlir_module()


@pytest.mark.parametrize(
    ("pool_shape", "pool_dtype", "idx_shape", "idx_dtype"),
    [
        ((ROW_WORDS,), jnp.int32, (2,), jnp.int32),
        ((2, 4, ROW_WORDS), jnp.int32, (2,), jnp.int32),
        ((4, 64), jnp.int32, (2,), jnp.int32),
        ((4, ROW_WORDS), jnp.float32, (2,), jnp.int32),
        ((4, ROW_WORDS), jnp.int32, (2, 1), jnp.int32),
        ((4, ROW_WORDS), jnp.int32, (2,), jnp.float32),
    ],
)
def test_should_raise_value_error_on_invalid_gather_cache_rows_inputs(
    pool_shape: tuple[int, ...],
    pool_dtype: jnp.dtype,
    idx_shape: tuple[int, ...],
    idx_dtype: jnp.dtype,
) -> None:
    pool = jnp.zeros(pool_shape, dtype=pool_dtype)
    idx = jnp.zeros(idx_shape, dtype=idx_dtype)
    with pytest.raises(ValueError):
        gather_cache_rows(pool, idx, interpret=True)


def test_should_fail_fast_in_benchmark_cache_outside_physical_tpu7x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if jax.default_backend() != "tpu":
        with pytest.raises(RuntimeError, match="TPU7x"):
            benchmark_cache.run_benchmarks(warmup=1, repeat=1)

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(generation=6, num_lanes=ROW_WORDS),
    )
    with pytest.raises(RuntimeError, match="TPU7x"):
        benchmark_cache.main(["--warmup", "1", "--repeat", "1"])


def test_should_extract_xprof_device_latency_and_fail_fast_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_profile = SimpleNamespace(
        planes=[
            SimpleNamespace(
                name="/device:TPU:0",
                lines=[
                    SimpleNamespace(
                        name="XLA Modules",
                        events=[
                            SimpleNamespace(duration_ns=12_000),
                            SimpleNamespace(duration_ns=14_000),
                            SimpleNamespace(duration_ns=18_000),
                        ],
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(jax.profiler, "trace", lambda _d: nullcontext())
    monkeypatch.setattr(
        benchmark_cache.glob,
        "glob",
        lambda _p, recursive=True: ["/tmp/fake.xplane.pb"],
    )
    monkeypatch.setattr(jax.profiler.ProfileData, "from_file", lambda _f: fake_profile)
    dev_us, host_us = benchmark_cache.measure_device_and_host_us(
        lambda: jnp.zeros((1,), dtype=jnp.int32),
        warmup=1,
        repeat=2,
    )
    assert dev_us == pytest.approx(14.0)
    assert host_us >= 0.0

    # Missing .xplane.pb file raises RuntimeError.
    monkeypatch.setattr(benchmark_cache.glob, "glob", lambda _p, recursive=True: [])
    with pytest.raises(RuntimeError, match=r"No \.xplane\.pb"):
        benchmark_cache.measure_device_and_host_us(
            lambda: jnp.zeros((1,), dtype=jnp.int32), warmup=0, repeat=1
        )

    # Missing plane, missing line, and empty events raise RuntimeError.
    monkeypatch.setattr(
        benchmark_cache.glob,
        "glob",
        lambda _p, recursive=True: ["/tmp/fake.xplane.pb"],
    )
    for bad_planes, match in [
        ([SimpleNamespace(name="/host:CPU", lines=[])], "Missing profiler plane"),
        (
            [SimpleNamespace(name="/device:TPU:0", lines=[])],
            "Missing profiler line",
        ),
        (
            [
                SimpleNamespace(
                    name="/device:TPU:0",
                    lines=[SimpleNamespace(name="XLA Modules", events=[])],
                )
            ],
            "Empty device event samples",
        ),
    ]:
        monkeypatch.setattr(
            jax.profiler.ProfileData,
            "from_file",
            lambda _f, p=bad_planes: SimpleNamespace(planes=p),
        )
        with pytest.raises(RuntimeError, match=match):
            benchmark_cache.measure_device_and_host_us(
                lambda: jnp.zeros((1,), dtype=jnp.int32), warmup=0, repeat=1
            )


@pytest.mark.tpu_test
@pytest.mark.parametrize("slice_words", VALID_SLICE_WORDS)
@pytest.mark.parametrize("page_tokens", [8, DEFAULT_PAGE_TOKENS])
def test_tpu_should_execute_write_sub_rows_and_donated_hazard_cleanly(
    slice_words: int, page_tokens: int
) -> None:
    rng = np.random.default_rng(1100 + slice_words + page_tokens)
    num_rows = 256
    num_tokens = 300
    sub_slots = ROW_WORDS // slice_words
    total_slots = num_rows * sub_slots
    pool_np = rng.integers(10, 500, size=(num_rows, ROW_WORDS), dtype=np.int32)
    upd_np = rng.integers(-5000, 5000, size=(num_tokens, slice_words), dtype=np.int32)
    slots_np = rng.integers(-4, total_slots + 4, size=(num_tokens,), dtype=np.int32)
    slots_np[:16] = np.arange(0, 16, dtype=np.int32)
    slots_np[16:20] = slots_np[0]
    read_rows = jnp.arange(0, 32, dtype=jnp.int32)

    @jax.jit(donate_argnums=(0,))
    def _read_then_write(
        c: jax.Array, u: jax.Array, s: jax.Array, r: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        before = gather_cache_rows(c, r)
        after_pool = write_sub_rows(
            c, u, s, after=before[0, 0], page_tokens=page_tokens
        )
        return before, after_pool

    got_before, got_pool = _read_then_write(
        jnp.asarray(pool_np), jnp.asarray(upd_np), jnp.asarray(slots_np), read_rows
    )
    np.testing.assert_array_equal(np.asarray(got_before), pool_np[:32])
    np.testing.assert_array_equal(
        np.asarray(got_pool), _oracle_write_sub_rows(pool_np, upd_np, slots_np)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_tokens", [2048, 4096])
def test_tpu_should_emit_zero_sparsecore_offloads_in_write_sub_rows_hlo(
    num_tokens: int,
) -> None:
    pool = jnp.zeros((2048, ROW_WORDS), dtype=jnp.int32)
    updates = jnp.ones((num_tokens, ROPE_WORDS), dtype=jnp.int32)
    slots = jnp.arange(num_tokens, dtype=jnp.int32)

    def _argsort_gather_baseline(
        upd: jax.Array, idx: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        order = jnp.argsort(idx, stable=True)
        return idx[order], upd[order]

    baseline_hlo = (
        jax.jit(_argsort_gather_baseline).lower(updates, slots).compile().as_text()
    )
    assert baseline_hlo is not None
    assert "gather_offload" in baseline_hlo
    assert 'async_execution_thread="sparsecore"' in baseline_hlo

    hlo_text = (
        jax.jit(write_sub_rows, donate_argnums=(0,))
        .lower(pool, updates, slots)
        .compile()
        .as_text()
    )
    assert hlo_text is not None
    assert "gather_offload" not in hlo_text
    assert 'async_execution_thread="sparsecore"' not in hlo_text


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_tokens", [1, 40, 520, 4096])
def test_tpu_should_execute_pack_fp8_kv_rows_cleanly(num_tokens: int) -> None:
    rng = np.random.default_rng(1200 + num_tokens)
    latent = jnp.asarray(
        rng.normal(0.0, 2.0, size=(num_tokens, LATENT_DIM)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    got_nope, got_rope = jax.jit(pack_fp8_kv_rows)(latent)
    exp_nope, exp_rope = _oracle_pack_fp8_kv_rows(latent)
    np.testing.assert_array_equal(np.asarray(got_nope), exp_nope)
    np.testing.assert_array_equal(np.asarray(got_rope), exp_rope)


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_queries", [48, 2304, 12288])
def test_tpu_should_execute_gather_cache_rows_cleanly(num_queries: int) -> None:
    rng = np.random.default_rng(1300 + num_queries)
    num_rows = 2048
    pool = rng.integers(1, 10000, size=(num_rows, ROW_WORDS), dtype=np.int32)
    indices = rng.integers(-8, num_rows + 8, size=(num_queries,), dtype=np.int32)
    indices[:32] = np.arange(0, 32, dtype=np.int32)
    if num_queries >= 96:
        indices[32:64] = np.arange(num_rows - 16, num_rows + 16, dtype=np.int32)
        indices[64:72] = np.arange(num_rows - 4, num_rows + 4, dtype=np.int32)
    got = np.asarray(
        jax.jit(gather_cache_rows)(jnp.asarray(pool), jnp.asarray(indices))
    )
    np.testing.assert_array_equal(got, _oracle_gather_cache_rows(pool, indices))
