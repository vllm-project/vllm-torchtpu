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
"""Unit and TPU7x hardware tests for DeepSeek-V4.1 word-row cache kernels."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compress_store_ref,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
    COMPRESSED_PAGE_TOKENS,
    DEFAULT_PAGE_TOKENS,
    INDEXER_KEY_WORDS,
    LATENT_DIM,
    NOPE_DIM,
    ROPE_WORDS,
    ROW_WORDS,
    VALID_SLICE_WORDS,
    benchmark_cache,
    gather_cache_rows,
    pack_fp8_kv_rows,
    write_sub_rows,
)

SCALE_GROUP_DIM = 64


@pytest.fixture(autouse=True)
def _setup_tpu7x_abstract_mesh() -> Iterator[None]:
    """Provide a TPU7x abstract mesh for CPU interpret-mode tests."""
    if jax.default_backend() == "tpu":
        yield
        return
    abstract_device = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    abstract_mesh = jax.sharding.AbstractMesh((), (), abstract_device=abstract_device)
    with jax.sharding.use_abstract_mesh(abstract_mesh):
        yield


def _ref_write_sub_rows(
    cache_words: np.ndarray,
    updates: np.ndarray,
    slot_mapping: np.ndarray,
    *,
    slot_offset: int = 0,
) -> np.ndarray:
    """NumPy reference implementation of sub-row cache mutation."""
    out = cache_words.copy()
    num_rows = out.shape[0]
    slice_words = updates.shape[1]
    sub_slots_per_row = ROW_WORDS // slice_words
    bank_slots = num_rows * sub_slots_per_row
    for tok_idx, raw_slot in enumerate(slot_mapping.tolist()):
        local_slot = int(raw_slot) - slot_offset
        if 0 <= local_slot < bank_slots:
            row_idx = local_slot // sub_slots_per_row
            sub_slot = local_slot % sub_slots_per_row
            start_word = sub_slot * slice_words
            out[row_idx, start_word : start_word + slice_words] = updates[tok_idx]
    return out


def _ref_pack_fp8_kv_rows(latent: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """Reference using base ``compress_store_ref`` and ``csa_cache_layout``."""
    num_tokens = latent.shape[0]
    if num_tokens == 0:
        return (
            np.zeros((0, ROW_WORDS), dtype=np.int32),
            np.zeros((0, ROPE_WORDS), dtype=np.int32),
        )
    quantized_full, scale_full = compress_store_ref.quantize_fp8_lane_periodic(
        latent.astype(jnp.float32), SCALE_GROUP_DIM, NOPE_DIM
    )
    quantized_u8 = jax.lax.bitcast_convert_type(quantized_full, jnp.uint8).reshape(
        num_tokens, LATENT_DIM
    )
    scale_u8 = jax.lax.bitcast_convert_type(scale_full, jnp.uint8).reshape(
        num_tokens, SCALE_GROUP_DIM
    )
    slabs_u8 = jnp.concatenate([quantized_u8[:, :NOPE_DIM], scale_u8], axis=1).reshape(
        num_tokens, 4, ROW_WORDS
    )
    nope_words = csa_cache_layout.slabs_to_words(slabs_u8)
    rope_words = csa_cache_layout.rope_words(latent[:, NOPE_DIM:])
    return np.asarray(nope_words), np.asarray(rope_words)


def _ref_gather_cache_rows(
    pool_lo: np.ndarray,
    row_indices: np.ndarray,
    *,
    pool_hi: np.ndarray | None = None,
) -> np.ndarray:
    """NumPy reference implementation of 1-bank / 2-bank row gather."""
    rows_lo = pool_lo.shape[0]
    rows_hi = 0 if pool_hi is None else pool_hi.shape[0]
    out = np.zeros((row_indices.shape[0], ROW_WORDS), dtype=np.int32)
    for query_idx, raw_idx in enumerate(row_indices.tolist()):
        row_idx = int(raw_idx)
        if 0 <= row_idx < rows_lo:
            out[query_idx] = pool_lo[row_idx]
        elif pool_hi is not None and rows_lo <= row_idx < rows_lo + rows_hi:
            out[query_idx] = pool_hi[row_idx - rows_lo]
    return out


def test_check_tpu7x_rejects_non_tpu7x_device() -> None:
    """Verify all three kernels fail fast on non-TPU7x abstract devices."""
    non_tpu7x_device = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    non_tpu7x_mesh = jax.sharding.AbstractMesh((), (), abstract_device=non_tpu7x_device)
    cache_words = jnp.zeros((4, ROW_WORDS), dtype=jnp.int32)
    updates = jnp.zeros((2, INDEXER_KEY_WORDS), dtype=jnp.int32)
    slots = jnp.zeros((2,), dtype=jnp.int32)
    latent = jnp.zeros((2, LATENT_DIM), dtype=jnp.bfloat16)
    with jax.sharding.use_abstract_mesh(non_tpu7x_mesh):
        with pytest.raises(RuntimeError, match="Requires TPU7x"):
            write_sub_rows(cache_words, updates, slots, interpret=True)
        with pytest.raises(RuntimeError, match="Requires TPU7x"):
            pack_fp8_kv_rows(latent, interpret=True)
        with pytest.raises(RuntimeError, match="Requires TPU7x"):
            gather_cache_rows(cache_words, slots, interpret=True)


@pytest.mark.parametrize("slice_words", VALID_SLICE_WORDS)
@pytest.mark.parametrize("slot_offset", [0, 8])
@pytest.mark.parametrize(
    "page_tokens", [4, COMPRESSED_PAGE_TOKENS, DEFAULT_PAGE_TOKENS, 512]
)
@pytest.mark.parametrize("enable_named_scope", [False, True])
def test_write_sub_rows_cpu_interpret(
    slice_words: int,
    slot_offset: int,
    page_tokens: int,
    enable_named_scope: bool,
) -> None:
    rng = np.random.default_rng(100 + slice_words + slot_offset + page_tokens)
    num_rows = 6
    cache_np = rng.integers(-1000, 1000, size=(num_rows, ROW_WORDS), dtype=np.int32)
    raw_slots = np.array([-1, 0, 1, 1, 2, 3, 4, 5, 6, 7, 2, 99, -5, 0], dtype=np.int32)
    slots_np = np.where(raw_slots >= 0, raw_slots + slot_offset, raw_slots).astype(
        np.int32
    )
    updates_np = rng.integers(
        1, 5000, size=(slots_np.shape[0], slice_words), dtype=np.int32
    )

    got = np.asarray(
        write_sub_rows(
            jnp.asarray(cache_np),
            jnp.asarray(updates_np),
            jnp.asarray(slots_np),
            slot_offset=slot_offset,
            page_tokens=page_tokens,
            interpret=True,
            enable_named_scope=enable_named_scope,
        )
    )
    expected = _ref_write_sub_rows(
        cache_np, updates_np, slots_np, slot_offset=slot_offset
    )
    np.testing.assert_array_equal(got, expected)


def test_write_sub_rows_two_bank_window_split() -> None:
    """Verify 2-bank windowing where 0 <= slot_mapping < slot_offset."""
    bank0 = np.zeros((2, ROW_WORDS), dtype=np.int32)
    bank1 = np.zeros((2, ROW_WORDS), dtype=np.int32)
    updates = np.arange(1, 6 * INDEXER_KEY_WORDS + 1, dtype=np.int32).reshape(
        6, INDEXER_KEY_WORDS
    )
    slots = np.array([0, 1, 3, 4, 5, 7], dtype=np.int32)

    got0 = np.asarray(
        write_sub_rows(
            jnp.asarray(bank0),
            jnp.asarray(updates),
            jnp.asarray(slots),
            slot_offset=0,
            interpret=True,
        )
    )
    got1 = np.asarray(
        write_sub_rows(
            jnp.asarray(bank1),
            jnp.asarray(updates),
            jnp.asarray(slots),
            slot_offset=4,
            interpret=True,
        )
    )
    np.testing.assert_array_equal(
        got0, _ref_write_sub_rows(bank0, updates, slots, slot_offset=0)
    )
    np.testing.assert_array_equal(
        got1, _ref_write_sub_rows(bank1, updates, slots, slot_offset=4)
    )


@pytest.mark.parametrize("slice_words", VALID_SLICE_WORDS)
@pytest.mark.parametrize("page_tokens", [COMPRESSED_PAGE_TOKENS, DEFAULT_PAGE_TOKENS])
def test_write_sub_rows_multiblock_burst_partial_and_min_int32(
    slice_words: int, page_tokens: int
) -> None:
    """Exercise multi-block contiguous bursts, partial merges, empty tails, & -2**31."""
    rng = np.random.default_rng(150 + slice_words + page_tokens)
    sub_slots_per_row = ROW_WORDS // slice_words
    num_rows = 200
    cache_np = rng.integers(-1000, 1000, size=(num_rows, ROW_WORDS), dtype=np.int32)
    slot_offset = 16
    contig = np.arange(
        slot_offset, slot_offset + 128 * sub_slots_per_row, dtype=np.int32
    )
    partial_and_oob = np.array(
        [
            -(2**31),
            -1,
            slot_offset + 140 * sub_slots_per_row,
            slot_offset + 145 * sub_slots_per_row + (sub_slots_per_row - 1),
            slot_offset + 140 * sub_slots_per_row,
            slot_offset + num_rows * sub_slots_per_row + 5,
        ],
        dtype=np.int32,
    )
    trailing_oob = np.full((140,), -1, dtype=np.int32)
    slots_np = np.concatenate([contig, partial_and_oob, trailing_oob])
    updates_np = rng.integers(
        1, 9000, size=(slots_np.shape[0], slice_words), dtype=np.int32
    )

    got = np.asarray(
        write_sub_rows(
            jnp.asarray(cache_np),
            jnp.asarray(updates_np),
            jnp.asarray(slots_np),
            slot_offset=slot_offset,
            page_tokens=page_tokens,
            interpret=True,
            enable_named_scope=True,
        )
    )
    expected = _ref_write_sub_rows(
        cache_np, updates_np, slots_np, slot_offset=slot_offset
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    ("num_rows", "num_tokens"),
    [(0, 4), (4, 0), (0, 0)],
)
def test_write_sub_rows_zero_length(num_rows: int, num_tokens: int) -> None:
    cache = jnp.ones((num_rows, ROW_WORDS), dtype=jnp.int32)
    updates = jnp.ones((num_tokens, INDEXER_KEY_WORDS), dtype=jnp.int32)
    slots = jnp.zeros((num_tokens,), dtype=jnp.int32)
    out = write_sub_rows(cache, updates, slots, interpret=True)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cache))


@pytest.mark.parametrize("slice_words", VALID_SLICE_WORDS)
def test_write_sub_rows_after_dependency(slice_words: int) -> None:
    """Verify after= vs None bit-equality (int32/bf16/uint8/fp8/empty) & SMEM edge."""
    rng = np.random.default_rng(180 + slice_words)
    cache_np = rng.integers(-500, 500, size=(8, ROW_WORDS), dtype=np.int32)
    slots_np = np.array([0, 1, 3, 5, -1], dtype=np.int32)
    updates_np = rng.integers(
        1, 2000, size=(slots_np.shape[0], slice_words), dtype=np.int32
    )
    expected = _ref_write_sub_rows(cache_np, updates_np, slots_np)
    got_none = np.asarray(
        write_sub_rows(
            jnp.asarray(cache_np),
            jnp.asarray(updates_np),
            jnp.asarray(slots_np),
            after=None,
            interpret=True,
        )
    )
    np.testing.assert_array_equal(got_none, expected)

    def _chained_read_then_write(
        pool: jax.Array, upd: jax.Array, slots: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        pre_read = gather_cache_rows(
            pool, jnp.array([0, 1, 2], dtype=jnp.int32), interpret=True
        )
        updated = write_sub_rows(pool, upd, slots, after=pre_read, interpret=True)
        return pre_read, updated

    jit_chained = jax.jit(_chained_read_then_write, donate_argnums=(0,))
    got_read, got_i32 = jit_chained(
        jnp.asarray(cache_np), jnp.asarray(updates_np), jnp.asarray(slots_np)
    )
    np.testing.assert_array_equal(np.asarray(got_read), cache_np[:3])
    np.testing.assert_array_equal(np.asarray(got_i32), got_none)

    after_samples = (
        jnp.array([17, -9], dtype=jnp.int32),
        jnp.array([[-3.5, 0.0], [7.25, -1.25]], dtype=jnp.bfloat16),
        jnp.array([255, 0, 42], dtype=jnp.uint8),
        jnp.array([-2.0, 1.5], dtype=jnp.float8_e4m3fn),
        jnp.empty((0,), dtype=jnp.bfloat16),
    )
    for sample_after in after_samples:
        got = np.asarray(
            write_sub_rows(
                jnp.asarray(cache_np),
                jnp.asarray(updates_np),
                jnp.asarray(slots_np),
                after=sample_after,
                interpret=True,
            )
        )
        np.testing.assert_array_equal(got, got_none)
        if sample_after.size > 0:
            closed = jax.make_jaxpr(
                lambda c, u, s, a: write_sub_rows(c, u, s, after=a, interpret=True)
            )(
                jnp.asarray(cache_np),
                jnp.asarray(updates_np),
                jnp.asarray(slots_np),
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


@pytest.mark.parametrize(
    (
        "cache_shape",
        "cache_dtype",
        "upd_shape",
        "upd_dtype",
        "slot_shape",
        "slot_dtype",
        "offset",
        "page_tokens",
    ),
    [
        ((128,), jnp.int32, (2, 64), jnp.int32, (2,), jnp.int32, 0, 16),
        ((4, 64), jnp.int32, (2, 64), jnp.int32, (2,), jnp.int32, 0, 16),
        ((4, 128), jnp.float32, (2, 64), jnp.int32, (2,), jnp.int32, 0, 16),
        ((4, 128), jnp.int32, (64,), jnp.int32, (64,), jnp.int32, 0, 16),
        ((4, 128), jnp.int32, (2, 48), jnp.int32, (2,), jnp.int32, 0, 16),
        ((4, 128), jnp.int32, (2, 64), jnp.float32, (2,), jnp.int32, 0, 16),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (3,), jnp.int32, 0, 16),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (2, 1), jnp.int32, 0, 16),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (2,), jnp.float32, 0, 16),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (2,), jnp.int32, -1, 16),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (2,), jnp.int32, 2**31, 16),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (2,), jnp.int32, 0, 0),
        ((4, 128), jnp.int32, (2, 32), jnp.int32, (2,), jnp.int32, 0, -4),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (2,), jnp.int32, 0, 6),
        ((4, 128), jnp.int32, (2, 64), jnp.int32, (2,), jnp.int32, 0, 12),
    ],
)
def test_write_sub_rows_validation_errors(
    cache_shape: tuple[int, ...],
    cache_dtype: jnp.dtype,
    upd_shape: tuple[int, ...],
    upd_dtype: jnp.dtype,
    slot_shape: tuple[int, ...],
    slot_dtype: jnp.dtype,
    offset: int,
    page_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        write_sub_rows(
            jnp.zeros(cache_shape, dtype=cache_dtype),
            jnp.zeros(upd_shape, dtype=upd_dtype),
            jnp.zeros(slot_shape, dtype=slot_dtype),
            slot_offset=offset,
            page_tokens=page_tokens,
            interpret=True,
        )


@pytest.mark.parametrize("num_tokens", [0, 1, 7, 32, 45, 520, 1024])
@pytest.mark.parametrize("enable_named_scope", [False, True])
def test_pack_fp8_kv_rows_cpu_interpret(
    num_tokens: int, enable_named_scope: bool
) -> None:
    rng = np.random.default_rng(200 + num_tokens)
    latent_f32 = rng.standard_normal((num_tokens, LATENT_DIM), dtype=np.float32) * 3.5
    latent = jnp.asarray(latent_f32, dtype=jnp.bfloat16)

    got_nope, got_rope = pack_fp8_kv_rows(
        latent, interpret=True, enable_named_scope=enable_named_scope
    )
    exp_nope, exp_rope = _ref_pack_fp8_kv_rows(latent)
    np.testing.assert_array_equal(np.asarray(got_nope), exp_nope)
    np.testing.assert_array_equal(np.asarray(got_rope), exp_rope)


def test_pack_fp8_kv_rows_zero_and_subnormal_blocks() -> None:
    """Verify exact scale-byte parity with base reference when amax < 1e-4."""
    zeros = jnp.zeros((2, LATENT_DIM), dtype=jnp.bfloat16)
    tiny = jnp.full((2, LATENT_DIM), 1e-6, dtype=jnp.bfloat16)
    latent = jnp.concatenate([zeros, tiny], axis=0)

    got_nope, got_rope = pack_fp8_kv_rows(latent, interpret=True)
    exp_nope, exp_rope = _ref_pack_fp8_kv_rows(latent)
    np.testing.assert_array_equal(np.asarray(got_nope), exp_nope)
    np.testing.assert_array_equal(np.asarray(got_rope), exp_rope)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((512,), jnp.bfloat16),
        ((4, 256), jnp.bfloat16),
        ((4, 512, 1), jnp.bfloat16),
        ((4, 512), jnp.float32),
    ],
)
def test_pack_fp8_kv_rows_validation_errors(
    shape: tuple[int, ...], dtype: jnp.dtype
) -> None:
    with pytest.raises(ValueError, match="latent"):
        pack_fp8_kv_rows(jnp.zeros(shape, dtype=dtype), interpret=True)


@pytest.mark.parametrize("num_queries", [0, 1, 5, 8, 19])
@pytest.mark.parametrize("use_pool_hi", [False, True])
@pytest.mark.parametrize("enable_named_scope", [False, True])
def test_gather_cache_rows_cpu_interpret(
    num_queries: int, use_pool_hi: bool, enable_named_scope: bool
) -> None:
    rng = np.random.default_rng(300 + num_queries + int(use_pool_hi))
    pool_lo_np = rng.integers(-2000, 2000, size=(6, ROW_WORDS), dtype=np.int32)
    pool_hi_np = (
        rng.integers(-2000, 2000, size=(5, ROW_WORDS), dtype=np.int32)
        if use_pool_hi
        else None
    )
    raw_indices = np.array(
        [-3, 0, 2, 5, 6, 8, 10, 11, 25, -1, 4, 7, 1, 9, 3, 6, 0, 12, -9],
        dtype=np.int32,
    )[:num_queries]

    got = np.asarray(
        gather_cache_rows(
            jnp.asarray(pool_lo_np),
            jnp.asarray(raw_indices),
            pool_hi=None if pool_hi_np is None else jnp.asarray(pool_hi_np),
            interpret=True,
            enable_named_scope=enable_named_scope,
        )
    )
    expected = _ref_gather_cache_rows(pool_lo_np, raw_indices, pool_hi=pool_hi_np)
    np.testing.assert_array_equal(got, expected)


def test_gather_cache_rows_empty_pools() -> None:
    empty = jnp.zeros((0, ROW_WORDS), dtype=jnp.int32)
    non_empty = jnp.arange(2 * ROW_WORDS, dtype=jnp.int32).reshape(2, ROW_WORDS)
    idx = jnp.array([-1, 0, 1, 2], dtype=jnp.int32)

    out_both_empty = gather_cache_rows(empty, idx, pool_hi=empty, interpret=True)
    np.testing.assert_array_equal(
        np.asarray(out_both_empty), np.zeros((4, ROW_WORDS), dtype=np.int32)
    )

    out_lo_empty = gather_cache_rows(empty, idx, pool_hi=non_empty, interpret=True)
    expected = _ref_gather_cache_rows(np.asarray(non_empty), np.asarray(idx))
    np.testing.assert_array_equal(np.asarray(out_lo_empty), expected)


@pytest.mark.parametrize("use_pool_hi", [False, True])
def test_gather_cache_rows_page_burst_and_scattered(use_pool_hi: bool) -> None:
    """Exercise 32-row bursts, 8-row sub-bursts, pure-burst blocks, and OOB rows."""
    rng = np.random.default_rng(350 + int(use_pool_hi))
    pool_lo_np = rng.integers(-2000, 2000, size=(48, ROW_WORDS), dtype=np.int32)
    pool_hi_np = (
        rng.integers(-2000, 2000, size=(48, ROW_WORDS), dtype=np.int32)
        if use_pool_hi
        else None
    )
    hi_jnp = None if pool_hi_np is None else jnp.asarray(pool_hi_np)

    pure_burst = np.concatenate(
        [
            np.arange(8, 40, dtype=np.int32),
            np.arange(52, 84, dtype=np.int32)
            if use_pool_hi
            else np.arange(12, 44, dtype=np.int32),
        ]
    )
    got_pure = np.asarray(
        gather_cache_rows(
            jnp.asarray(pool_lo_np),
            jnp.asarray(pure_burst),
            pool_hi=hi_jnp,
            interpret=True,
        )
    )
    np.testing.assert_array_equal(
        got_pure,
        _ref_gather_cache_rows(pool_lo_np, pure_burst, pool_hi=pool_hi_np),
    )

    burst_lo = np.arange(8, 40, dtype=np.int32)
    burst_hi = np.arange(52, 84, dtype=np.int32)
    sub8_lo = np.arange(2, 10, dtype=np.int32)
    sub8_hi = np.arange(56, 64, dtype=np.int32)
    scattered = np.array(
        [-5, 0, 47, 48, 95, 96, -1, 15, 60, 31, 999, -(2**31), 3, 50, 12],
        dtype=np.int32,
    )
    idx_np = np.concatenate([burst_lo, burst_hi, sub8_lo, sub8_hi, scattered])

    got = np.asarray(
        gather_cache_rows(
            jnp.asarray(pool_lo_np),
            jnp.asarray(idx_np),
            pool_hi=hi_jnp,
            interpret=True,
            enable_named_scope=True,
        )
    )
    expected = _ref_gather_cache_rows(pool_lo_np, idx_np, pool_hi=pool_hi_np)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    ("lo_shape", "lo_dtype", "hi_shape", "hi_dtype", "idx_shape", "idx_dtype"),
    [
        ((128,), jnp.int32, None, jnp.int32, (2,), jnp.int32),
        ((4, 64), jnp.int32, None, jnp.int32, (2,), jnp.int32),
        ((4, 128), jnp.float32, None, jnp.int32, (2,), jnp.int32),
        ((4, 128), jnp.int32, (128,), jnp.int32, (2,), jnp.int32),
        ((4, 128), jnp.int32, (4, 64), jnp.int32, (2,), jnp.int32),
        ((4, 128), jnp.int32, (4, 128), jnp.float32, (2,), jnp.int32),
        ((4, 128), jnp.int32, None, jnp.int32, (2, 1), jnp.int32),
        ((4, 128), jnp.int32, None, jnp.int32, (2,), jnp.float32),
    ],
)
def test_gather_cache_rows_validation_errors(
    lo_shape: tuple[int, ...],
    lo_dtype: jnp.dtype,
    hi_shape: tuple[int, ...] | None,
    hi_dtype: jnp.dtype,
    idx_shape: tuple[int, ...],
    idx_dtype: jnp.dtype,
) -> None:
    with pytest.raises(ValueError):
        gather_cache_rows(
            jnp.zeros(lo_shape, dtype=lo_dtype),
            jnp.zeros(idx_shape, dtype=idx_dtype),
            pool_hi=None if hi_shape is None else jnp.zeros(hi_shape, dtype=hi_dtype),
            interpret=True,
        )


def test_benchmark_cache_interpret_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_cache CLI and XProf ProfileData parsing execute cleanly."""
    exit_code = benchmark_cache.main(["--interpret", "--warmup", "1", "--repeat", "1"])
    assert exit_code == 0

    fake_event = SimpleNamespace(duration_ns=2500.0)
    fake_line = SimpleNamespace(
        name=benchmark_cache.XLA_MODULES_LINE, events=[fake_event]
    )
    fake_plane = SimpleNamespace(
        name=benchmark_cache.TPU_DEVICE_PLANE, lines=[fake_line]
    )
    fake_profile = SimpleNamespace(planes=[fake_plane])
    monkeypatch.setattr(benchmark_cache.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        benchmark_cache.jax.profiler, "trace", lambda _dir: nullcontext()
    )
    monkeypatch.setattr(benchmark_cache.glob, "glob", lambda *_a, **_k: ["fake.pb"])
    monkeypatch.setattr(
        benchmark_cache.jax.profiler.ProfileData,
        "from_file",
        lambda _path: fake_profile,
    )
    dev_us, host_us = benchmark_cache._measure_device_and_host_us(
        lambda: jnp.zeros((1,), dtype=jnp.int32),
        warmup=1,
        repeat=1,
        interpret=False,
    )
    assert dev_us == pytest.approx(2.5)
    assert host_us > 0.0


@pytest.mark.tpu_test
@pytest.mark.parametrize("slice_words", VALID_SLICE_WORDS)
@pytest.mark.parametrize("page_tokens", [COMPRESSED_PAGE_TOKENS, DEFAULT_PAGE_TOKENS])
def test_write_sub_rows_tpu(slice_words: int, page_tokens: int) -> None:
    rng = np.random.default_rng(400 + slice_words + page_tokens)
    sub_slots_per_row = ROW_WORDS // slice_words
    cache_np = rng.integers(-500, 500, size=(64, ROW_WORDS), dtype=np.int32)
    burst = np.arange(2, 2 + 32 * sub_slots_per_row, dtype=np.int32)
    tail = np.array(
        [-(2**31), -1, 2, 3, 3, 40 * sub_slots_per_row + 1, 9999, 0, 1],
        np.int32,
    )
    slots_np = np.concatenate([burst, tail])
    updates_np = rng.integers(
        1, 1000, size=(slots_np.shape[0], slice_words), dtype=np.int32
    )
    after = (
        None
        if slice_words == ROPE_WORDS
        else (
            jnp.array([-2.5, 1.0], dtype=jnp.bfloat16)
            if slice_words == INDEXER_KEY_WORDS
            else jnp.array([99], dtype=jnp.int32)
        )
    )
    got = np.asarray(
        write_sub_rows(
            jnp.asarray(cache_np),
            jnp.asarray(updates_np),
            jnp.asarray(slots_np),
            slot_offset=2,
            after=after,
            page_tokens=page_tokens,
            interpret=False,
            enable_named_scope=True,
        )
    )
    expected = _ref_write_sub_rows(cache_np, updates_np, slots_np, slot_offset=2)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_tokens", [40, 520])
def test_pack_fp8_kv_rows_tpu(num_tokens: int) -> None:
    rng = np.random.default_rng(500 + num_tokens)
    latent = jnp.asarray(
        rng.standard_normal((num_tokens, LATENT_DIM), dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    got_nope, got_rope = pack_fp8_kv_rows(
        latent, interpret=False, enable_named_scope=True
    )
    exp_nope, exp_rope = _ref_pack_fp8_kv_rows(latent)
    np.testing.assert_array_equal(np.asarray(got_nope), exp_nope)
    np.testing.assert_array_equal(np.asarray(got_rope), exp_rope)


@pytest.mark.tpu_test
@pytest.mark.parametrize("use_pool_hi", [False, True])
def test_gather_cache_rows_tpu(use_pool_hi: bool) -> None:
    rng = np.random.default_rng(600 + int(use_pool_hi))
    pool_lo_np = rng.integers(-1000, 1000, size=(48, ROW_WORDS), dtype=np.int32)
    pool_hi_np = (
        rng.integers(-1000, 1000, size=(48, ROW_WORDS), dtype=np.int32)
        if use_pool_hi
        else None
    )
    idx_np = np.concatenate(
        [
            np.arange(4, 36, dtype=np.int32),
            np.arange(52, 84, dtype=np.int32),
            np.arange(6, 14, dtype=np.int32),
            np.arange(56, 64, dtype=np.int32),
            np.array([-1, 0, 3, 47, 48, 95, 96, 999, -(2**31), 2], dtype=np.int32),
        ]
    )
    got = np.asarray(
        gather_cache_rows(
            jnp.asarray(pool_lo_np),
            jnp.asarray(idx_np),
            pool_hi=None if pool_hi_np is None else jnp.asarray(pool_hi_np),
            interpret=False,
            enable_named_scope=True,
        )
    )
    expected = _ref_gather_cache_rows(pool_lo_np, idx_np, pool_hi=pool_hi_np)
    np.testing.assert_array_equal(got, expected)
