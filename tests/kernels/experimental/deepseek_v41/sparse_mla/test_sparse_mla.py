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
"""Unit and TPU7x hardware tests for Sparse MLA + SWA gather and attention."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import glob
import inspect
import types
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.profiler
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    benchmark_sparse_mla,
    csa_gather,
    gather_sparse_fp8_kv,
    sparse_mla_swa_attention,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    kernel as mla_k,
)

_FP8 = ml_dtypes.float8_e4m3fn
_BF16 = ml_dtypes.bfloat16


@pytest.fixture(autouse=True)
def _tpu7x_abstract_mesh() -> Any:
    if jax.default_backend() == "tpu":
        yield
        return
    dev = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=dev)
    with jax.sharding.use_abstract_mesh(mesh):
        yield


def _patch_vmem_token_head_budget(monkeypatch: pytest.MonkeyPatch, budget: int) -> int:
    base_info = pltpu.get_tpu_info()
    vmem_bytes = (
        budget
        * mla_k.VMEM_BUDGET_DIVISOR
        * mla_k.QUERY_OUT_VMEM_DIVISOR
        * mla_k.QUERY_OUT_BYTES_PER_TOKEN_HEAD
    )
    patched_info = dataclasses.replace(base_info, vmem_capacity_bytes=vmem_bytes)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: patched_info)
    active_vmem_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    return (
        active_vmem_bytes
        // mla_k.VMEM_BUDGET_DIVISOR
        // mla_k.QUERY_OUT_VMEM_DIVISOR
        // mla_k.QUERY_OUT_BYTES_PER_TOKEN_HEAD
    )


def _make_raw_fp8_pool(
    rng: np.random.Generator, n_rows: int, *, include_zero_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sign = rng.integers(0, 2, size=(n_rows, 448), dtype=np.uint8) << 7
    expo = rng.integers(4, 10, size=(n_rows, 448), dtype=np.uint8) << 3
    fp8_bytes = sign | expo | rng.integers(0, 8, size=(n_rows, 448), dtype=np.uint8)
    scale_bytes = rng.integers(123, 129, size=(n_rows, 64), dtype=np.uint8)
    if include_zero_scale and n_rows >= 2:
        scale_bytes[0, :8] = 0
        scale_bytes[1, :] = 255
    record = np.concatenate([fp8_bytes, scale_bytes], axis=1)
    slab = record.reshape(n_rows, 4, 128).astype(np.uint32)
    words = slab[:, 0] | (slab[:, 1] << 8) | (slab[:, 2] << 16) | (slab[:, 3] << 24)
    nope_words = words.view(np.int32)
    vals = fp8_bytes.view(_FP8).astype(np.float64)
    dims = np.arange(448)
    sc_exp = scale_bytes[:, dims % 64].astype(np.float64)
    scales_f64 = np.where(sc_exp == 255.0, np.nan, np.exp2(sc_exp - 127.0))
    nope_f64 = vals * scales_f64
    rope_bf16 = (rng.standard_normal((n_rows, 64)) * 0.5).astype(_BF16)
    bits = rope_bf16.view(np.uint16).astype(np.uint32)
    rw = (bits[:, :32] | (bits[:, 32:] << 16)).view(np.int32)
    rope_words = rw.reshape(n_rows // 4, 128)
    kv_f64 = np.concatenate([nope_f64, rope_bf16.astype(np.float64)], axis=1)
    return nope_words, rope_words, kv_f64


def _independent_csa_gather_reference(
    nope_cache: np.ndarray,
    rope_cache: np.ndarray,
    indices: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_nope = nope_cache.reshape(-1, 128)
    flat_rope = rope_cache.reshape(-1, 32).astype(np.uint32)
    nope_out = flat_nope[indices]
    half = top_k // 2
    num_periods = indices.size // top_k
    rope_u16 = np.empty((num_periods, half, 128), dtype=np.uint16)
    for period_idx in range(num_periods):
        period_indices = indices[period_idx * top_k : (period_idx + 1) * top_k]
        for half_offset, lane_start in ((0, 0), (half, 64)):
            words = flat_rope[period_indices[half_offset : half_offset + half]]
            rope_u16[period_idx, :, lane_start : lane_start + 32] = (
                words & 0xFFFF
            ).astype(np.uint16)
            rope_u16[period_idx, :, lane_start + 32 : lane_start + 64] = (
                words >> 16
            ).astype(np.uint16)
    rope_paired = rope_u16.reshape(-1, 2, 128).astype(np.uint32)
    rope_out = (rope_paired[:, 0, :] | (rope_paired[:, 1, :] << 16)).view(np.int32)
    return nope_out, rope_out


def _independent_float64_reference(
    q: np.ndarray,
    kv_rows_f64: np.ndarray,
    nope_words: np.ndarray,
    rope_words: np.ndarray,
    topk_indices: np.ndarray,
    block_tables: np.ndarray,
    attention_sinks: np.ndarray,
    seq_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    *,
    page_size: int,
    sm_scale: float,
    swa_window: int,
    swa_kv: np.ndarray | None,
    kv_compress_ratio: int,
    swa_min_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens, topk = topk_indices.shape
    heads = q.shape[1]
    n_pages = nope_words.shape[0] // page_size
    gn = np.zeros((tokens, topk, 128), np.int32)
    gr = np.zeros((tokens, topk // 4, 128), np.int32)
    out = np.zeros((tokens, heads, 512), np.float64)

    for s in range(len(seq_lens)):
        q_start, q_end = int(cu_q_lens[s]), int(cu_q_lens[s + 1])
        min_p = 0 if swa_min_pos is None else max(0, int(swa_min_pos[s]))
        for t in range(max(0, q_start), min(tokens, q_end)):
            pos = int(seq_lens[s]) - (q_end - t)
            lower = max(min_p, pos - swa_window + 1)
            vals: list[np.ndarray] = []
            if swa_kv is not None and swa_window > 0:
                if swa_kv.ndim == 2:
                    w_len = (swa_kv.shape[0] - tokens) // len(seq_lens)
                    p0 = int(seq_lens[s]) - (q_end - q_start)
                    for w in range(w_len):
                        p_pos = (pos - w_len + 1) + w
                        if lower <= p_pos <= pos:
                            row_2d = (
                                swa_kv[s * w_len + (p_pos - (p0 - w_len))]
                                if p_pos < p0
                                else swa_kv[
                                    len(seq_lens) * w_len + q_start + (p_pos - p0)
                                ]
                            )
                            vals.append(row_2d.astype(np.float64)[None])
                elif swa_kv.shape[1] > 0:
                    w_len = swa_kv.shape[1]
                    base = max(0, pos - w_len + 1)
                    for w in range(w_len):
                        p_pos = base + w
                        if lower <= p_pos <= pos:
                            vals.append(swa_kv[t, w].astype(np.float64)[None])
            for k in range(topk):
                idx = int(topk_indices[t, k])
                in_attn = 0 <= idx < (pos + 1) // kv_compress_ratio
                ph = (
                    int(block_tables[s, idx // page_size])
                    if idx >= 0 and idx // page_size < block_tables.shape[1]
                    else -1
                )
                if not (0 <= ph < n_pages):
                    if in_attn:
                        vals.append(np.zeros((1, 512), np.float64))
                    continue
                row = ph * page_size + idx % page_size
                gn[t, k] = nope_words[row]
                src = rope_words[row // 4, 32 * (row % 4) : 32 * (row % 4) + 32]
                gr[t, k // 4, 32 * (k % 4) : 32 * (k % 4) + 32] = src
                if in_attn:
                    vals.append(kv_rows_f64[row][None])
            v_mat = (
                np.concatenate(vals, axis=0) if vals else np.zeros((0, 512), np.float64)
            )
            qf = q[t].astype(np.float64)
            lg = np.concatenate(
                [attention_sinks.astype(np.float64)[:, None], qf @ v_mat.T * sm_scale],
                axis=1,
            )
            m_val = lg.max(axis=1, keepdims=True)
            p_val = np.exp(lg - m_val)
            out[t] = (p_val[:, 1:] @ v_mat) / p_val.sum(axis=1, keepdims=True)
    return gn, gr, out


@pytest.mark.parametrize(
    ("topk", "heads", "swa_w", "swa_win", "ratio", "scope"),
    [
        (32, 8, 4, 0, 1, False),
        (36, 12, 18, 5, 2, True),
        (0, 3, 7, 128, 1, False),
    ],
)
def test_gather_and_sparse_mla_swa_independent_ref(
    topk: int,
    heads: int,
    swa_w: int,
    swa_win: int,
    ratio: int,
    scope: bool,
) -> None:
    rng = np.random.default_rng(100 + topk + heads + swa_w)
    page_size, n_pages = 16, 6
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(
        rng, n_pages * page_size, include_zero_scale=(topk == 32)
    )
    kv_pool = jnp.asarray(nope_w.reshape(n_pages, page_size, 128))
    rope_pool = jnp.asarray(rope_w.reshape(n_pages, page_size // 4, 128))

    tokens = 2 if topk == 0 else 4
    cu = np.array([0, 1, 1] if topk == 0 else [0, 1, 3], dtype=np.int32)
    sl = np.array([3 if swa_win > 0 else 28, 65], dtype=np.int32)
    bt = np.array([[0, 2, 4, 3], [1, -1, 99, 5]], dtype=np.int32)
    if topk == 32:
        bt = np.pad(bt, ((0, 0), (0, 128)), constant_values=-1)
    tk = rng.integers(-2, 52, size=(tokens, topk), dtype=np.int32)
    if topk > 0:
        tk[0, :] = -1
        tk[1, 0] = 0
        tk[1, 1] = -1
        tk[1, 2::2] = 19
        # Logical page 4 == max_pages (unpadded bt) with valid last col bt[1, 3] = 5.
        tk[1, 3] = 4 * page_size + 5
        tk[1, 4] = 1 * page_size + 2
        tk[1, 5] = 2 * page_size + 3

    q = (rng.standard_normal((tokens, heads, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-1.0, 1.0, size=(heads,)).astype(np.float32)
    if swa_win == 0:
        swa = np.full((tokens, swa_w, 512), np.nan, dtype=_BF16)
    else:
        swa = (rng.standard_normal((tokens, swa_w, 512)) * 0.2).astype(_BF16)
        swa[0, 3:, :] = _BF16(np.nan)
    sm = 0.0625

    def _run_gather_and_attn(
        kv_p: jax.Array,
        rp_p: jax.Array,
        q_in: jax.Array,
        tk_in: jax.Array,
        bt_in: jax.Array,
        sinks_in: jax.Array,
        sl_in: jax.Array,
        cu_in: jax.Array,
        swa_in: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        gn_, gr_ = gather_sparse_fp8_kv(
            kv_p,
            rp_p,
            tk_in,
            bt_in,
            cu_in,
            page_size=page_size,
            interpret=True,
            enable_named_scope=scope,
        )
        gn_in_ = gn_.at[1, 1, :].set(jnp.int32(-1)) if topk == 32 else gn_
        out_ = sparse_mla_swa_attention(
            q_in,
            gn_in_,
            gr_,
            tk_in,
            sinks_in,
            sl_in,
            cu_in,
            sm_scale=sm,
            swa_window=swa_win,
            swa_kv=swa_in,
            kv_compress_ratio=ratio,
            interpret=True,
            enable_named_scope=scope,
        )
        return gn_, gr_, out_

    gn, gr, out = jax.jit(_run_gather_and_attn)(
        kv_pool,
        rope_pool,
        jnp.asarray(q),
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        jnp.asarray(swa),
    )
    rgn, rgr, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=page_size,
        sm_scale=sm,
        swa_window=swa_win,
        swa_kv=swa,
        kv_compress_ratio=ratio,
    )
    assert not np.any(np.isnan(np.asarray(out, dtype=np.float32)))
    if topk == 36:
        assert not np.any(np.asarray(gn)[1, 3])
    np.testing.assert_array_equal(np.asarray(gn), rgn)
    np.testing.assert_array_equal(np.asarray(gr), rgr)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(out[int(cu[-1]) :], dtype=np.float32), 0.0)


def test_should_isolate_layers_when_gathering_homogeneous_page_scale2_shared_pool() -> (
    None
):
    rng = np.random.default_rng(42)
    page_size, logical_blocks = 32, 4
    total_pages = logical_blocks * 2
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, total_pages * page_size)
    shared_nope = jnp.asarray(nope_w.reshape(total_pages, page_size, 128))
    shared_rope = jnp.asarray(rope_w.reshape(total_pages, page_size // 4, 128))

    base_bt = np.array([[1, 2], [0, 3]], dtype=np.int32)
    l2_bt = base_bt * 2
    l8_bt = base_bt * 2 + 1
    cu = np.array([0, 1, 2], dtype=np.int32)
    sl = np.array([128, 128], dtype=np.int32)
    tk = np.array([[0, 5, 33, -1], [2, 31, 63, 7]], dtype=np.int32)
    q = (rng.standard_normal((2, 8, 512)) * 0.1).astype(_BF16)
    sinks = np.zeros((8,), dtype=np.float32)

    gather_jit = jax.jit(
        functools.partial(gather_sparse_fp8_kv, page_size=page_size, interpret=True)
    )
    gn_l2, gr_l2 = gather_jit(
        shared_nope, shared_rope, jnp.asarray(tk), jnp.asarray(l2_bt), jnp.asarray(cu)
    )
    gn_l8, gr_l8 = gather_jit(
        shared_nope, shared_rope, jnp.asarray(tk), jnp.asarray(l8_bt), jnp.asarray(cu)
    )
    rgn_l2, rgr_l2, _ = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        l2_bt,
        sinks,
        sl,
        cu,
        page_size=page_size,
        sm_scale=0.0625,
        swa_window=0,
        swa_kv=None,
        kv_compress_ratio=2,
    )
    rgn_l8, rgr_l8, _ = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        l8_bt,
        sinks,
        sl,
        cu,
        page_size=page_size,
        sm_scale=0.0625,
        swa_window=0,
        swa_kv=None,
        kv_compress_ratio=2,
    )
    np.testing.assert_array_equal(np.asarray(gn_l2), rgn_l2)
    np.testing.assert_array_equal(np.asarray(gr_l2), rgr_l2)
    np.testing.assert_array_equal(np.asarray(gn_l8), rgn_l8)
    np.testing.assert_array_equal(np.asarray(gr_l8), rgr_l8)
    assert not np.array_equal(np.asarray(gn_l2), np.asarray(gn_l8))


def test_zero_dimension_short_circuits() -> None:
    kv_pool = jnp.zeros((2, 16, 128), dtype=jnp.int32)
    rope_pool = jnp.zeros((2, 4, 128), dtype=jnp.int32)
    bt = jnp.zeros((1, 2), dtype=jnp.int32)
    cu = jnp.array([0, 0], dtype=jnp.int32)
    nope_empty, rope_empty = csa_gather(
        kv_pool, rope_pool, jnp.zeros((0,), dtype=jnp.int32), top_k=256
    )
    assert nope_empty.shape == (0, 128) and rope_empty.shape == (0, 128)

    gn0, gr0 = gather_sparse_fp8_kv(
        kv_pool,
        rope_pool,
        jnp.zeros((0, 32), dtype=jnp.int32),
        bt,
        cu,
        page_size=16,
        interpret=True,
    )
    assert gn0.shape == (0, 32, 128) and gr0.shape == (0, 8, 128)

    out0 = sparse_mla_swa_attention(
        jnp.zeros((0, 8, 512), dtype=jnp.bfloat16),
        gn0,
        gr0,
        jnp.zeros((0, 32), dtype=jnp.int32),
        jnp.zeros((8,), dtype=jnp.float32),
        jnp.array([16], dtype=jnp.int32),
        cu,
        sm_scale=0.1,
        swa_window=0,
        interpret=True,
    )
    assert out0.shape == (0, 8, 512)


@functools.cache
def _valid_csa_inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.zeros((2, 16, 128), dtype=jnp.int32),
        jnp.zeros((2, 4, 128), dtype=jnp.int32),
        jnp.zeros((256,), dtype=jnp.int32),
    )


@functools.cache
def _valid_gather_inputs() -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    return (
        jnp.zeros((2, 16, 128), dtype=jnp.int32),
        jnp.zeros((2, 4, 128), dtype=jnp.int32),
        jnp.zeros((2, 32), dtype=jnp.int32),
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.array([0, 2], dtype=jnp.int32),
    )


@functools.cache
def _valid_attn_inputs() -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    return (
        jnp.zeros((4, 8, 512), dtype=jnp.bfloat16),
        jnp.zeros((4, 32, 128), dtype=jnp.int32),
        jnp.zeros((4, 8, 128), dtype=jnp.int32),
        jnp.zeros((4, 32), dtype=jnp.int32),
        jnp.zeros((8,), dtype=jnp.float32),
        jnp.array([16, 16], dtype=jnp.int32),
        jnp.array([0, 2, 4], dtype=jnp.int32),
    )


@pytest.mark.parametrize(
    ("bad_field", "exc_type", "match"),
    [
        ("nope_ndim", ValueError, "Expected int32 nope_cache"),
        ("nope_dtype", ValueError, "Expected int32 nope_cache"),
        ("nope_pages", ValueError, "Expected int32 nope_cache"),
        ("nope_page_size", ValueError, "Expected int32 nope_cache"),
        ("nope_cols", ValueError, "Expected int32 nope_cache"),
        ("rope_shape", ValueError, "Expected int32 rope_cache"),
        ("rope_dtype", ValueError, "Expected int32 rope_cache"),
        ("indices_ndim", ValueError, "Expected 1D int32 indices"),
        ("indices_dtype", ValueError, "Expected 1D int32 indices"),
        ("top_k_zero", ValueError, "Expected top_k positive multiple of 128"),
        ("top_k_unaligned", ValueError, "Expected top_k positive multiple of 128"),
        ("out_size_unaligned", ValueError, "divisible by top_k"),
        ("missing_sc", RuntimeError, "TPU7x SparseCore"),
    ],
)
def test_should_reject_invalid_csa_gather_inputs(
    monkeypatch: pytest.MonkeyPatch,
    bad_field: str,
    exc_type: type[Exception],
    match: str,
) -> None:
    nope_cache, rope_cache, indices = _valid_csa_inputs()
    top_k = 256

    if bad_field == "nope_ndim":
        nope_cache = jnp.zeros((32, 128), dtype=jnp.int32)
    elif bad_field == "nope_dtype":
        nope_cache = jnp.zeros((2, 16, 128), dtype=jnp.float32)
    elif bad_field == "nope_pages":
        nope_cache = jnp.zeros((0, 16, 128), dtype=jnp.int32)
    elif bad_field == "nope_page_size":
        nope_cache = jnp.zeros((2, 14, 128), dtype=jnp.int32)
    elif bad_field == "nope_cols":
        nope_cache = jnp.zeros((2, 16, 64), dtype=jnp.int32)
    elif bad_field == "rope_shape":
        rope_cache = jnp.zeros((2, 3, 128), dtype=jnp.int32)
    elif bad_field == "rope_dtype":
        rope_cache = jnp.zeros((2, 4, 128), dtype=jnp.float32)
    elif bad_field == "indices_ndim":
        indices = jnp.zeros((1, 256), dtype=jnp.int32)
    elif bad_field == "indices_dtype":
        indices = jnp.zeros((256,), dtype=jnp.float32)
    elif bad_field == "top_k_zero":
        top_k = 0
    elif bad_field == "top_k_unaligned":
        top_k = 192
    elif bad_field == "out_size_unaligned":
        indices = jnp.zeros((300,), dtype=jnp.int32)
    elif bad_field == "missing_sc":
        cast(Any, csa_gather).clear_cache()
        no_sc_info = dataclasses.replace(pltpu.get_tpu_info(), sparse_core=None)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: no_sc_info)

    with pytest.raises(exc_type, match=match):
        csa_gather(nope_cache, rope_cache, indices, top_k=top_k)


@pytest.mark.parametrize(
    ("bad_field", "match"),
    [
        ("page_size", "Expected page_size > 0"),
        ("kv_ndim", "Expected kv_pool"),
        ("kv_shape", "Expected kv_pool"),
        ("rp", "Expected rope_pool"),
        ("topk", "Expected topk_indices"),
        ("bt", "Expected block_tables"),
        ("cu", "Expected cu_q_lens"),
    ],
)
def test_gather_validation_errors(bad_field: str, match: str) -> None:
    kv_pool, rope_pool, tk, bt, cu = _valid_gather_inputs()
    page_size = 16

    if bad_field == "page_size":
        page_size = 14
    elif bad_field == "kv_ndim":
        kv_pool = jnp.zeros((16, 128), dtype=jnp.int32)
    elif bad_field == "kv_shape":
        kv_pool = jnp.zeros((2, 16, 64), dtype=jnp.int32)
    elif bad_field == "rp":
        rope_pool = jnp.zeros((2, 3, 128), dtype=jnp.int32)
    elif bad_field == "topk":
        tk = jnp.zeros((2, 30), dtype=jnp.int32)
    elif bad_field == "bt":
        bt = jnp.zeros((1, 0), dtype=jnp.int32)
    elif bad_field == "cu":
        cu = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=match):
        gather_sparse_fp8_kv(
            kv_pool,
            rope_pool,
            tk,
            bt,
            cu,
            page_size=page_size,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("bad_field", "match"),
    [
        ("q", "Expected q"),
        ("topk", "Expected topk_indices"),
        ("topk_0d", "Expected topk_indices"),
        ("nope", "Expected gathered_nope"),
        ("rope", "Expected gathered_rope"),
        ("sinks", "Expected attention_sinks"),
        ("seq_lens", "Expected seq_lens"),
        ("seq_lens_0d", "Expected seq_lens"),
        ("cu", "Expected cu_q_lens"),
        ("window", "Expected swa_window >= 0"),
        ("ratio", "Expected swa_window >= 0"),
        ("swa_kv_shape", "Expected swa_kv"),
        ("swa_kv_tokens", "Expected swa_kv"),
        ("swa_kv_ndim", "Expected swa_kv"),
        ("swa_kv_0d", "Expected swa_kv"),
        ("swa_kv_2d_cols", "Expected swa_kv"),
        ("swa_kv_2d_lt_tokens", "Expected swa_kv"),
        ("swa_kv_2d_zero_width", "Expected swa_kv"),
        ("swa_kv_2d_unaligned_rem", "Expected swa_kv"),
        ("swa_kv_2d_dtype", "Expected swa_kv"),
        ("swa_min_pos_shape", "Expected swa_min_pos"),
        ("swa_min_pos_ndim", "Expected swa_min_pos"),
        ("swa_min_pos_dtype", "Expected swa_min_pos"),
    ],
)
def test_attention_validation_errors(bad_field: str, match: str) -> None:
    q, gn, gr, tk, sinks, sl, cu = _valid_attn_inputs()
    swa_w, ratio = 4, 1
    swa_kv: jax.Array | None = None
    swa_min_pos: jax.Array | None = None

    if bad_field == "q":
        q = jnp.zeros((4, 8, 256), dtype=jnp.bfloat16)
    elif bad_field == "topk":
        tk = jnp.zeros((4, 30), dtype=jnp.int32)
    elif bad_field == "topk_0d":
        tk = jnp.zeros((), dtype=jnp.int32)
    elif bad_field == "nope":
        gn = jnp.zeros((4, 16, 128), dtype=jnp.int32)
    elif bad_field == "rope":
        gr = jnp.zeros((4, 4, 128), dtype=jnp.int32)
    elif bad_field == "sinks":
        sinks = jnp.zeros((4,), dtype=jnp.float32)
    elif bad_field == "seq_lens":
        sl = jnp.zeros((0,), dtype=jnp.int32)
    elif bad_field == "seq_lens_0d":
        sl = jnp.zeros((), dtype=jnp.int32)
    elif bad_field == "cu":
        cu = jnp.array([0, 4], dtype=jnp.int32)
    elif bad_field == "window":
        swa_w = -1
    elif bad_field == "ratio":
        ratio = 3
    elif bad_field == "swa_kv_shape":
        swa_kv = jnp.zeros((4, 4, 256), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_tokens":
        swa_kv = jnp.zeros((2, 4, 512), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_ndim":
        swa_kv = jnp.zeros((512,), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_0d":
        swa_kv = jnp.zeros((), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_2d_cols":
        swa_kv = jnp.zeros((12, 256), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_2d_lt_tokens":
        swa_kv = jnp.zeros((2, 512), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_2d_zero_width":
        swa_kv = jnp.zeros((4, 512), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_2d_unaligned_rem":
        swa_kv = jnp.zeros((9, 512), dtype=jnp.bfloat16)
    elif bad_field == "swa_kv_2d_dtype":
        swa_kv = jnp.zeros((12, 512), dtype=jnp.float32)
    elif bad_field == "swa_min_pos_shape":
        swa_min_pos = jnp.zeros((3,), dtype=jnp.int32)
    elif bad_field == "swa_min_pos_ndim":
        swa_min_pos = jnp.zeros((2, 1), dtype=jnp.int32)
    elif bad_field == "swa_min_pos_dtype":
        swa_min_pos = jnp.zeros((2,), dtype=jnp.float32)

    with pytest.raises(ValueError, match=match):
        sparse_mla_swa_attention(
            q,
            gn,
            gr,
            tk,
            sinks,
            sl,
            cu,
            sm_scale=0.1,
            swa_window=swa_w,
            swa_kv=swa_kv,
            kv_compress_ratio=ratio,
            swa_min_pos=swa_min_pos,
            interpret=True,
        )


def test_should_compile_csa_gather_and_sparse_mla_for_tpu7x_aot() -> None:
    nope_spec = jax.ShapeDtypeStruct((4, 16, 128), jnp.int32)
    rope_spec = jax.ShapeDtypeStruct((4, 4, 128), jnp.int32)

    idx_spec = jax.ShapeDtypeStruct((8192,), jnp.int32)
    hlo_sc = (
        csa_gather.trace(nope_spec, rope_spec, idx_spec, top_k=256)
        .lower(lowering_platforms=("tpu",))
        .as_text()
    )
    assert "sc_csa_gather" in hlo_sc
    assert "tensor<8192xi1>" not in hlo_sc

    kv_pool = jax.ShapeDtypeStruct((4, 16, 128), jnp.int32)
    rope_pool = jax.ShapeDtypeStruct((4, 4, 128), jnp.int32)
    tk = jax.ShapeDtypeStruct((4, 32), jnp.int32)
    bt = jax.ShapeDtypeStruct((2, 4), jnp.int32)
    sl = jax.ShapeDtypeStruct((2,), jnp.int32)
    cu = jax.ShapeDtypeStruct((3,), jnp.int32)
    lowered_gather = (
        jax.jit(
            lambda kv_, rp_, tk_, bt_, cu_: gather_sparse_fp8_kv(
                kv_,
                rp_,
                tk_,
                bt_,
                cu_,
                page_size=16,
                interpret=False,
            )
        )
        .trace(kv_pool, rope_pool, tk, bt, cu)
        .lower(lowering_platforms=("tpu",))
    )
    gather_hlo = lowered_gather.as_text()
    assert "dsv41_gather_page_ids" in gather_hlo
    assert "sc_csa_gather" in gather_hlo

    # Avoid materializing bf16[640,128,512] across multi-chunk scan: T=640, H=64
    # (step=128 -> 5 chunks) must emit 1 extract + 1 attention call inside while.
    q_long = jax.ShapeDtypeStruct((640, 64, 512), jnp.bfloat16)
    gn_long = jax.ShapeDtypeStruct((640, 512, 128), jnp.int32)
    gr_long = jax.ShapeDtypeStruct((640, 128, 128), jnp.int32)
    tk_long = jax.ShapeDtypeStruct((640, 512), jnp.int32)
    sinks_long = jax.ShapeDtypeStruct((64,), jnp.float32)
    swa_2d_long = jax.ShapeDtypeStruct((2 * 128 + 640, 512), jnp.bfloat16)
    hlo_long = (
        jax.jit(
            lambda q_, gn_, gr_, tk_, s_, sl_, cu_, swa_: sparse_mla_swa_attention(
                q_,
                gn_,
                gr_,
                tk_,
                s_,
                sl_,
                cu_,
                sm_scale=0.0625,
                swa_window=128,
                swa_kv=swa_,
                interpret=False,
            )
        )
        .trace(q_long, gn_long, gr_long, tk_long, sinks_long, sl, cu, swa_2d_long)
        .lower(lowering_platforms=("tpu",))
        .as_text()
    )
    assert hlo_long.count("dsv41_extract_swa_windows") == 1
    assert hlo_long.count("dsv41_sparse_mla_swa") == 1
    assert "bf16[640,128,512]" not in hlo_long


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("num_indices", "top_k"),
    [
        (256, 256),
        (1536, 256),
        (384, 128),
        (4096, 128),
        (8192, 256),
        (12288, 256),
        (16384, 512),
    ],
)
def test_should_match_reference_for_csa_gather_on_tpu7x(
    num_indices: int, top_k: int
) -> None:
    _require_physical_tpu()
    rng = np.random.default_rng(300 + num_indices + top_k)
    nope_w, rope_w, _ = _make_raw_fp8_pool(rng, 1024)
    nope_cache = nope_w.reshape(64, 16, 128)
    rope_cache = rope_w.reshape(64, 4, 128)
    indices = rng.integers(0, 1024, size=(num_indices,), dtype=np.int32)

    nope_out, rope_out = csa_gather(
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        jnp.asarray(indices),
        top_k=top_k,
    )
    ref_nope, ref_rope = _independent_csa_gather_reference(
        nope_cache, rope_cache, indices, top_k=top_k
    )
    np.testing.assert_array_equal(np.asarray(nope_out), ref_nope)
    np.testing.assert_array_equal(np.asarray(rope_out), ref_rope)


def _require_physical_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("TPU7x hardware test requires physical TPU backend.")


def _make_tpu7x_kv_pool(
    rng: np.random.Generator, num_pages: int = 64, page_size: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, num_pages * page_size)
    return (
        nope_w,
        rope_w,
        kv_f64,
        nope_w.reshape(num_pages, page_size, 128),
        rope_w.reshape(num_pages, page_size // 4, 128),
    )


@pytest.mark.tpu_test
def test_should_match_reference_small_compressed_3d_swa_on_tpu7x() -> None:
    _require_physical_tpu()
    rng = np.random.default_rng(17)
    nope_w, rope_w, kv_f64, npw, rpw = _make_tpu7x_kv_pool(rng, num_pages=4)
    tokens, topk, heads = 6, 36, 12
    tk = rng.integers(-1, 28, size=(tokens, topk), dtype=np.int32)
    tk[2, 0] = 2 * 16 + 3
    bt = np.array([[0, 2], [1, 3]], dtype=np.int32)
    cu = np.array([0, 2, 5], dtype=np.int32)
    sl = np.array([24, 30], dtype=np.int32)
    q = (rng.standard_normal((tokens, heads, 512)) * 0.15).astype(_BF16)
    sinks = rng.uniform(-0.8, 0.8, size=(heads,)).astype(np.float32)
    swa = (rng.standard_normal((tokens, 18, 512)) * 0.2).astype(_BF16)

    gn_tpu, gr_tpu = gather_sparse_fp8_kv(
        jnp.asarray(npw),
        jnp.asarray(rpw),
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=False,
    )
    out_tpu = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn_tpu,
        gr_tpu,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=jnp.asarray(swa),
        kv_compress_ratio=2,
        interpret=False,
    )
    rgn, rgr, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=swa,
        kv_compress_ratio=2,
    )
    np.testing.assert_array_equal(np.asarray(gn_tpu), rgn)
    np.testing.assert_array_equal(np.asarray(gr_tpu), rgr)
    np.testing.assert_allclose(
        np.asarray(out_tpu, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    assert np.all(np.asarray(out_tpu[5:], dtype=np.float32) == 0.0)


@pytest.mark.tpu_test
def test_should_match_reference_k512_h64_compressed_3d_swa_on_tpu7x() -> None:
    _require_physical_tpu()
    rng = np.random.default_rng(18)
    nope_w, rope_w, kv_f64, npw, rpw = _make_tpu7x_kv_pool(rng, num_pages=64)
    tk = rng.integers(-1, 500, size=(4, 512), dtype=np.int32)
    bt = np.arange(64, dtype=np.int32).reshape(2, 32)
    cu = np.array([0, 2, 4], dtype=np.int32)
    sl = np.array([512, 1024], dtype=np.int32)
    q = (rng.standard_normal((4, 64, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(64,)).astype(np.float32)
    swa = (rng.standard_normal((4, 128, 512)) * 0.1).astype(_BF16)

    gn, gr = gather_sparse_fp8_kv(
        jnp.asarray(npw),
        jnp.asarray(rpw),
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=False,
    )
    out = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=jnp.asarray(swa),
        kv_compress_ratio=2,
        interpret=False,
    )
    rgn, rgr, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=swa,
        kv_compress_ratio=2,
    )
    np.testing.assert_array_equal(np.asarray(gn), rgn)
    np.testing.assert_array_equal(np.asarray(gr), rgr)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )


@pytest.mark.tpu_test
def test_should_mask_nan_replay_bounds_compact_2d_swa_on_tpu7x() -> None:
    _require_physical_tpu()
    rng = np.random.default_rng(19)
    nope_w, rope_w, kv_f64, npw, rpw = _make_tpu7x_kv_pool(rng, num_pages=64)
    bt = rng.integers(0, 64, size=(4, 300), dtype=np.int32)
    cu = np.array([0, 0, 5, 27, 27], dtype=np.int32)
    sl = np.array([0, 12, 60, 0], dtype=np.int32)
    tokens, swa_w = 29, 18
    tk = rng.integers(-1, 60, size=(tokens, 36), dtype=np.int32)
    q = (rng.standard_normal((tokens, 12, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(12,)).astype(np.float32)
    seq_swa = (rng.standard_normal((4, swa_w, 512)) * 0.15).astype(_BF16)
    # Avoid finite sentinels in out-of-window slots: NaNs prove causal/replay masks
    # zero excluded SWA rows before q_dot rather than relying on exp(-1e30) == 0.
    min_pos = np.array([0, -4, 30, 0], dtype=np.int32)
    seq_swa[1, :11, :] = _BF16(np.nan)
    seq_swa[2, :10, :] = _BF16(np.nan)
    seq_swa[0, :, :] = _BF16(np.nan)
    seq_swa[3, :, :] = _BF16(np.nan)
    kv_rot = (rng.standard_normal((tokens, 512)) * 0.15).astype(_BF16)
    kv_rot[27:, :] = _BF16(np.nan)
    swa_2d = np.concatenate([seq_swa.reshape(4 * swa_w, 512), kv_rot], axis=0)

    gn, gr = gather_sparse_fp8_kv(
        jnp.asarray(npw),
        jnp.asarray(rpw),
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=False,
    )
    out = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=16,
        swa_kv=jnp.asarray(swa_2d),
        kv_compress_ratio=1,
        swa_min_pos=jnp.asarray(min_pos),
        interpret=False,
    )
    rgn, rgr, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=16,
        swa_kv=swa_2d,
        kv_compress_ratio=1,
        swa_min_pos=min_pos,
    )
    assert not np.any(np.isnan(np.asarray(out, dtype=np.float32)))
    np.testing.assert_array_equal(np.asarray(gn), rgn)
    np.testing.assert_array_equal(np.asarray(gr), rgr)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(out[27:], dtype=np.float32), 0.0)


@pytest.mark.tpu_test
def test_should_match_reference_multi_chunk_2d_swa_on_tpu7x() -> None:
    _require_physical_tpu()
    rng = np.random.default_rng(20)
    nope_w, rope_w, kv_f64, npw, rpw = _make_tpu7x_kv_pool(rng, num_pages=64)
    tokens = 300
    bt = np.arange(64, dtype=np.int32).reshape(2, 32)
    tk = rng.integers(0, 500, size=(tokens, 256), dtype=np.int32)
    cu = np.array([0, 150, 300], dtype=np.int32)
    sl = np.array([600, 800], dtype=np.int32)
    q = (rng.standard_normal((tokens, 64, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(64,)).astype(np.float32)
    swa_2d = (rng.standard_normal((2 * 128 + tokens, 512)) * 0.1).astype(_BF16)

    gn, gr = gather_sparse_fp8_kv(
        jnp.asarray(npw),
        jnp.asarray(rpw),
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=False,
    )
    out = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=jnp.asarray(swa_2d),
        kv_compress_ratio=1,
        interpret=False,
    )
    _, _, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=swa_2d,
        kv_compress_ratio=1,
    )
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )


@pytest.mark.tpu_test
def test_should_match_reference_k2048_h64_3d_swa_on_tpu7x() -> None:
    _require_physical_tpu()
    rng = np.random.default_rng(21)
    nope_w, rope_w, kv_f64, npw, rpw = _make_tpu7x_kv_pool(rng, num_pages=64)
    bt = np.arange(64, dtype=np.int32).reshape(2, 32)
    tk = rng.integers(-1, 1024, size=(2, 2048), dtype=np.int32)
    cu = np.array([0, 1, 2], dtype=np.int32)
    sl = np.array([900, 1024], dtype=np.int32)
    q = (rng.standard_normal((2, 64, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(64,)).astype(np.float32)
    swa = (rng.standard_normal((2, 128, 512)) * 0.1).astype(_BF16)

    gn, gr = gather_sparse_fp8_kv(
        jnp.asarray(npw),
        jnp.asarray(rpw),
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=False,
    )
    out = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=jnp.asarray(swa),
        kv_compress_ratio=1,
        interpret=False,
    )
    rgn, rgr, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=swa,
        kv_compress_ratio=1,
    )
    np.testing.assert_array_equal(np.asarray(gn), rgn)
    np.testing.assert_array_equal(np.asarray(gr), rgr)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    ("tokens", "active_tokens", "topk", "swa_w", "swa_win", "budget"),
    [
        (2, 1, 8, 0, 0, 8),
        (3, 3, 8, 0, 0, 16),
        (3, 3, 8, 4, 4, 16),
    ],
)
def test_sparse_mla_swa_multi_chunk_tiling(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    active_tokens: int,
    topk: int,
    swa_w: int,
    swa_win: int,
    budget: int,
) -> None:
    token_head_budget = _patch_vmem_token_head_budget(monkeypatch, budget)
    rng = np.random.default_rng(23 + tokens + active_tokens + swa_w)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 256)
    tk = rng.integers(-1, 200, size=(tokens, topk), dtype=np.int32)
    bt = np.arange(32, dtype=np.int32).reshape(2, 16) % 16
    cu = np.array([0, 1, active_tokens], dtype=np.int32)
    sl = np.array([180, 220], dtype=np.int32)
    heads = 8
    padded_heads = ((heads + mla_k.SUBLANES - 1) // mla_k.SUBLANES) * mla_k.SUBLANES
    assert token_head_budget == budget
    assert token_head_budget < tokens * padded_heads

    q = (rng.standard_normal((tokens, heads, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(heads,)).astype(np.float32)
    swa = (
        (rng.standard_normal((len(sl) * swa_w + tokens, 512)) * 0.1).astype(_BF16)
        if swa_w > 0
        else None
    )
    rgn, rgr, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=swa_win,
        swa_kv=swa,
        kv_compress_ratio=1,
    )
    out = jax.jit(
        lambda q_, gn_, gr_, tk_, s_, sl_, cu_, swa_: sparse_mla_swa_attention(
            q_,
            gn_,
            gr_,
            tk_,
            s_,
            sl_,
            cu_,
            sm_scale=0.0625,
            swa_window=swa_win,
            swa_kv=swa_,
            kv_compress_ratio=1,
            interpret=True,
        )
    )(
        jnp.asarray(q),
        jnp.asarray(rgn),
        jnp.asarray(rgr),
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        None if swa is None else jnp.asarray(swa),
    )
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    if active_tokens < tokens:
        np.testing.assert_array_equal(
            np.asarray(out[active_tokens:], dtype=np.float32), 0.0
        )


def test_should_unpack_csa_gather_interleaved_rope_matching_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _emulate_csa_gather(
        nope_cache: jax.Array,
        rope_cache: jax.Array,
        indices: jax.Array,
        *,
        top_k: int = 1024,
    ) -> tuple[jax.Array, jax.Array]:
        n = indices.size
        nope_out = nope_cache.reshape(-1, 128)[indices]
        gathered_rope = rope_cache.reshape(-1, 32)[indices]
        periods = n // top_k
        g_u32 = jax.lax.bitcast_convert_type(gathered_rope, jnp.uint32).reshape(
            periods, 2, top_k // 4, 2, 32
        )
        even = g_u32[:, :, :, 0, :]
        odd = g_u32[:, :, :, 1, :]
        col_lo = (even & jnp.uint32(0xFFFF)) | (odd << jnp.uint32(16))
        col_hi = (even >> jnp.uint32(16)) | (odd & jnp.uint32(0xFFFF0000))
        half_words = jnp.concatenate([col_lo, col_hi], axis=-1)
        rope_out_u32 = jnp.concatenate(
            [half_words[:, 0], half_words[:, 1]], axis=-1
        ).reshape(n // 4, 128)
        return nope_out, jax.lax.bitcast_convert_type(rope_out_u32, jnp.int32)

    orig_pallas_call: Any = pl.pallas_call

    def _interpret_pallas_call(*args: Any, **kwargs: Any) -> Any:
        return orig_pallas_call(*args, **{**kwargs, "interpret": True})

    monkeypatch.setattr(pl, "pallas_call", _interpret_pallas_call)
    monkeypatch.setattr(mla_k, "csa_gather", _emulate_csa_gather)

    tokens, topk = 1, 8
    rng = np.random.default_rng(701 + tokens * 17 + topk)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 24 * 16)
    kv_pool = jnp.asarray(nope_w.reshape(24, 16, 128))
    rope_pool = jnp.asarray(rope_w.reshape(24, 4, 128))
    bt = np.array(
        [[0, 18, -1, 5, 22, 30, 1, 17]],
        dtype=np.int32,
    )
    cu = np.array([0, 1], dtype=np.int32)
    tk = rng.integers(-2, 8 * 16, size=(tokens, topk), dtype=np.int32)

    gn_sc, gr_sc = jax.jit(
        functools.partial(gather_sparse_fp8_kv, page_size=16, interpret=False)
    )(
        kv_pool,
        rope_pool,
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
    )
    rgn, rgr, _ = _independent_float64_reference(
        np.zeros((tokens, 8, 512), dtype=_BF16),
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        np.zeros((8,), dtype=np.float32),
        np.array([128], dtype=np.int32),
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=0,
        swa_kv=None,
        kv_compress_ratio=1,
    )
    np.testing.assert_array_equal(np.asarray(gn_sc), rgn)
    np.testing.assert_array_equal(np.asarray(gr_sc), rgr)


def test_should_fail_fast_in_benchmark_and_xprof_extraction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    empty_mesh = jax.sharding.AbstractMesh((), ())
    with (
        jax.sharding.use_abstract_mesh(empty_mesh),
        pytest.raises(RuntimeError, match="physical TPU backend"),
    ):
        benchmark_sparse_mla.main(["--warmup", "1", "--repeat", "1"])

    tpu7x_dev = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    tpu7x_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=tpu7x_dev)
    with (
        jax.sharding.use_abstract_mesh(tpu7x_mesh),
        pytest.raises(RuntimeError, match="physical TPU backend"),
    ):
        benchmark_sparse_mla.main(["--warmup", "1", "--repeat", "1"])

    v6e_dev = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_dev)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    with (
        jax.sharding.use_abstract_mesh(v6e_mesh),
        pytest.raises(RuntimeError, match=r"run_benchmarks requires TPU7x \(8x128"),
    ):
        benchmark_sparse_mla.main(["--warmup", "1", "--repeat", "1"])

    with jax.sharding.use_abstract_mesh(tpu7x_mesh):
        base_tpu7x = pltpu.get_tpu_info()
    for field_name, bad_val in (("num_lanes", 64), ("num_sublanes", 16)):
        bad_info = dataclasses.replace(base_tpu7x, **{field_name: bad_val})
        with (
            monkeypatch.context() as mp,
            jax.sharding.use_abstract_mesh(tpu7x_mesh),
            pytest.raises(RuntimeError, match=r"run_benchmarks requires TPU7x \(8x128"),
        ):
            mp.setattr(pltpu, "get_tpu_info", lambda info=bad_info: info)
            benchmark_sparse_mla.main(["--warmup", "1", "--repeat", "1"])

    one = jnp.int32(1)
    monkeypatch.setattr(jax.profiler, "trace", contextlib.nullcontext)
    monkeypatch.setattr(glob, "glob", lambda *_a, **_kw: [])
    with pytest.raises(RuntimeError, match=r"No \.xplane\.pb file produced"):
        benchmark_sparse_mla.measure_device_and_host_us(lambda: one, warmup=1, repeat=1)

    monkeypatch.setattr(glob, "glob", lambda *_a, **_kw: ["/tmp/fake.xplane.pb"])
    no_tpu_plane = types.SimpleNamespace(
        planes=[types.SimpleNamespace(name="/host:CPU", lines=[])]
    )
    monkeypatch.setattr(
        jax.profiler.ProfileData,
        "from_file",
        staticmethod(lambda _path: no_tpu_plane),
    )
    with pytest.raises(RuntimeError, match="Missing profiler plane"):
        benchmark_sparse_mla.measure_device_and_host_us(lambda: one, warmup=1, repeat=1)

    no_xla_line = types.SimpleNamespace(
        planes=[
            types.SimpleNamespace(
                name=benchmark_sparse_mla.TPU_DEVICE_PLANE_NAME,
                lines=[types.SimpleNamespace(name="Steps", events=[])],
            )
        ]
    )
    monkeypatch.setattr(
        jax.profiler.ProfileData,
        "from_file",
        staticmethod(lambda _path: no_xla_line),
    )
    with pytest.raises(RuntimeError, match="Missing profiler line"):
        benchmark_sparse_mla.measure_device_and_host_us(lambda: one, warmup=1, repeat=1)

    empty_events = types.SimpleNamespace(
        planes=[
            types.SimpleNamespace(
                name=benchmark_sparse_mla.TPU_DEVICE_PLANE_NAME,
                lines=[
                    types.SimpleNamespace(
                        name=benchmark_sparse_mla.XLA_MODULES_LINE_NAME, events=[]
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr(
        jax.profiler.ProfileData,
        "from_file",
        staticmethod(lambda _path: empty_events),
    )
    with pytest.raises(RuntimeError, match="Empty device event samples"):
        benchmark_sparse_mla.measure_device_and_host_us(lambda: one, warmup=1, repeat=1)

    valid_profile = types.SimpleNamespace(
        planes=[
            types.SimpleNamespace(name="/host:CPU", lines=[]),
            types.SimpleNamespace(
                name=benchmark_sparse_mla.TPU_DEVICE_PLANE_NAME,
                lines=[
                    types.SimpleNamespace(name="Steps", events=[]),
                    types.SimpleNamespace(
                        name=benchmark_sparse_mla.XLA_MODULES_LINE_NAME,
                        events=[
                            types.SimpleNamespace(duration_ns=84_540.0),
                            types.SimpleNamespace(duration_ns=85_460.0),
                        ],
                    ),
                ],
            ),
        ]
    )
    monkeypatch.setattr(
        jax.profiler.ProfileData,
        "from_file",
        staticmethod(lambda _path: valid_profile),
    )
    trace_active = [False]
    call_trace_states: list[bool] = []

    @contextlib.contextmanager
    def _track_trace(_tmpdir: str) -> Any:
        trace_active[0] = True
        try:
            yield
        finally:
            trace_active[0] = False

    def _tracked_step() -> jax.Array:
        call_trace_states.append(trace_active[0])
        return one

    monkeypatch.setattr(jax.profiler, "trace", _track_trace)
    dev_us, host_us = benchmark_sparse_mla.measure_device_and_host_us(
        _tracked_step, warmup=1, repeat=2
    )
    assert call_trace_states == [False, False, False, True, True]
    assert dev_us == pytest.approx(85.0)
    assert host_us > 0.0
    monkeypatch.setattr(jax.profiler, "trace", contextlib.nullcontext)

    tiny_tpu7x = dataclasses.replace(base_tpu7x, vmem_capacity_bytes=4096)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: tiny_tpu7x)
    monkeypatch.setattr(benchmark_sparse_mla, "TPU_UNCOMPRESSED_MAX_PAGES_PER_SEQ", 4)
    monkeypatch.setattr(benchmark_sparse_mla, "TPU_SWA_WINDOW", 4)
    monkeypatch.setattr(
        benchmark_sparse_mla,
        "WORKLOADS",
        (("decode_tiny", 1, 1, 8, 32, 1), ("mtp_tiny", 1, 2, 8, 32, 2)),
    )
    monkeypatch.setattr(jax, "jit", lambda fn, *_a, **_kw: fn)
    monkeypatch.setattr(
        jax.random,
        "randint",
        lambda _k, shape, *_a, dtype=jnp.int32, **_kw: jnp.zeros(shape, dtype=dtype),
    )
    monkeypatch.setattr(
        jax.random,
        "normal",
        lambda _k, shape, dtype=jnp.bfloat16: jnp.zeros(shape, dtype=dtype),
    )
    monkeypatch.setattr(
        mla_k,
        "gather_sparse_fp8_kv",
        lambda _kv, _rp, tk, _bt, _cu, **_kw: (
            jnp.zeros((tk.shape[0], tk.shape[1], mla_k.ROW_WORDS), jnp.int32),
            jnp.zeros(
                (
                    tk.shape[0],
                    tk.shape[1] // mla_k.ROPE_TOKENS_PER_ROW,
                    mla_k.ROW_WORDS,
                ),
                jnp.int32,
            ),
        ),
    )
    monkeypatch.setattr(
        mla_k,
        "sparse_mla_swa_attention",
        lambda q, *_a, **_kw: jnp.zeros_like(q),
    )
    with jax.sharding.use_abstract_mesh(tpu7x_mesh):
        rc = benchmark_sparse_mla.main(["--warmup", "1", "--repeat", "1"])
    stdout = capsys.readouterr().out
    assert rc == 0
    assert "gather_sparse_fp8_kv(decode_tiny)" in stdout
    assert "sparse_mla_swa_attention(decode_tiny)" in stdout
    assert "sparse_mla_swa_attention_3d(decode_tiny)" not in stdout
    assert "gather_sparse_fp8_kv(mtp_tiny)" in stdout
    assert "sparse_mla_swa_attention(mtp_tiny)" in stdout
    assert "sparse_mla_swa_attention_3d(mtp_tiny)" in stdout


def test_should_enforce_tpu7x_guard_and_import_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert mla_k.ROW_WORDS == 128
    assert mla_k.ROPE_TOKENS_PER_ROW == 4
    csa_mod = inspect.getmodule(csa_gather)
    assert csa_mod is not None
    expected_imports = (
        (csa_mod, []),
        (
            mla_k,
            ["vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla.csa_gather"],
        ),
        (
            benchmark_sparse_mla,
            ["vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla"],
        ),
    )
    for mod, expected in expected_imports:
        tree = ast.parse(inspect.getsource(mod))
        repo_imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("vllm_torchtpu.")
        ]
        assert repo_imports == expected

    tpu_vmem_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    assert (
        tpu_vmem_bytes
        // mla_k.VMEM_BUDGET_DIVISOR
        // mla_k.QUERY_OUT_VMEM_DIVISOR
        // mla_k.QUERY_OUT_BYTES_PER_TOKEN_HEAD
    ) == mla_k.MAX_VMEM_TOKEN_HEADS

    kv_pool = jnp.zeros((1, 16, 128), jnp.int32)
    rope_pool = jnp.zeros((1, 4, 128), jnp.int32)
    tk = jnp.zeros((1, 32), jnp.int32)
    bt = jnp.zeros((1, 1), jnp.int32)
    cu = jnp.array([0, 1], jnp.int32)
    sl = jnp.array([16], jnp.int32)
    q = jnp.zeros((1, 8, 512), jnp.bfloat16)
    gn = jnp.zeros((1, 32, 128), jnp.int32)
    gr = jnp.zeros((1, 8, 128), jnp.int32)
    sinks = jnp.zeros((8,), jnp.float32)

    v6e_dev = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_dev)
    cast(Any, csa_gather).clear_cache()
    with jax.sharding.use_abstract_mesh(v6e_mesh):
        with pytest.raises(RuntimeError, match="csa_gather requires TPU7x SparseCore"):
            csa_gather(kv_pool, rope_pool, jnp.zeros((256,), jnp.int32), top_k=256)
        with pytest.raises(RuntimeError, match=r"requires TPU7x \(8x128 lanes\)"):
            gather_sparse_fp8_kv(
                kv_pool, rope_pool, tk, bt, cu, page_size=16, interpret=True
            )
        with pytest.raises(RuntimeError, match=r"requires TPU7x \(8x128 lanes\)"):
            sparse_mla_swa_attention(
                q, gn, gr, tk, sinks, sl, cu, sm_scale=0.1, swa_window=0, interpret=True
            )

    base_info = pltpu.get_tpu_info()
    assert base_info.sparse_core is not None
    for sc_field, bad_sc_val in (("num_cores", 1), ("num_subcores", 8)):
        bad_sc = dataclasses.replace(base_info.sparse_core, **{sc_field: bad_sc_val})
        bad_sc_info = dataclasses.replace(base_info, sparse_core=bad_sc)
        cast(Any, csa_gather).clear_cache()
        with (
            monkeypatch.context() as mp,
            pytest.raises(RuntimeError, match="csa_gather requires TPU7x SparseCore"),
        ):
            mp.setattr(pltpu, "get_tpu_info", lambda info=bad_sc_info: info)
            csa_gather(kv_pool, rope_pool, jnp.zeros((256,), jnp.int32), top_k=256)

    for field_name, bad_val in (("num_lanes", 64), ("num_sublanes", 16)):
        bad_info = dataclasses.replace(base_info, **{field_name: bad_val})
        with monkeypatch.context() as mp:
            mp.setattr(pltpu, "get_tpu_info", lambda info=bad_info: info)
            with pytest.raises(RuntimeError, match=r"requires TPU7x \(8x128 lanes\)"):
                gather_sparse_fp8_kv(
                    kv_pool, rope_pool, tk, bt, cu, page_size=16, interpret=True
                )
            with pytest.raises(RuntimeError, match=r"requires TPU7x \(8x128 lanes\)"):
                sparse_mla_swa_attention(
                    q,
                    gn,
                    gr,
                    tk,
                    sinks,
                    sl,
                    cu,
                    sm_scale=0.1,
                    swa_window=0,
                    interpret=True,
                )

    empty_mesh = jax.sharding.AbstractMesh((), ())
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    cast(Any, csa_gather).clear_cache()
    with jax.sharding.use_abstract_mesh(empty_mesh):
        with pytest.raises(RuntimeError, match="csa_gather requires TPU7x"):
            csa_gather(kv_pool, rope_pool, jnp.zeros((256,), jnp.int32), top_k=256)
        with pytest.raises(RuntimeError, match="gather_sparse_fp8_kv requires TPU7x"):
            gather_sparse_fp8_kv(
                kv_pool, rope_pool, tk, bt, cu, page_size=16, interpret=True
            )
        with pytest.raises(
            RuntimeError, match="sparse_mla_swa_attention requires TPU7x"
        ):
            sparse_mla_swa_attention(
                q, gn, gr, tk, sinks, sl, cu, sm_scale=0.1, swa_window=0, interpret=True
            )


def test_compact_2d_swa_kv_and_swa_min_pos_replay_lower_bound() -> None:
    rng = np.random.default_rng(137)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 128)
    bt = np.arange(32, dtype=np.int32).reshape(4, 8) % 8
    cu = np.array([0, 0, 1, 3, 4], dtype=np.int32)
    sl = np.array([0, 4, 40, 50], dtype=np.int32)
    min_pos = np.array([0, -3, 30, 500], dtype=np.int32)
    tokens, swa_w, swa_win, heads = 5, 12, 10, 8
    tk = rng.integers(-1, 64, size=(tokens, 8), dtype=np.int32)
    q = (rng.standard_normal((tokens, heads, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(heads,)).astype(np.float32)

    seq_swa = (rng.standard_normal((4, swa_w, 512)) * 0.15).astype(_BF16)
    kv_rot = (rng.standard_normal((tokens, 512)) * 0.15).astype(_BF16)
    # Avoid benign zeros in pre-replay / negative prefix slots: NaNs prove causal
    # and replay lower-bound masking exclude out-of-range rows in 2D layout.
    seq_swa[0, :, :] = _BF16(np.nan)
    seq_swa[1, :9, :] = _BF16(np.nan)
    seq_swa[2, :4, :] = _BF16(np.nan)
    seq_swa[3, :, :] = _BF16(np.nan)
    kv_rot[3:, :] = _BF16(np.nan)
    swa_2d = np.concatenate([seq_swa.reshape(4 * swa_w, 512), kv_rot], axis=0)

    rgn, rgr, rout = _independent_float64_reference(
        q,
        kv_f64,
        nope_w,
        rope_w,
        tk,
        bt,
        sinks,
        sl,
        cu,
        page_size=16,
        sm_scale=0.0625,
        swa_window=swa_win,
        swa_kv=swa_2d,
        kv_compress_ratio=1,
        swa_min_pos=min_pos,
    )
    gn, gr = jnp.asarray(rgn), jnp.asarray(rgr)
    attn_2d_jit = jax.jit(
        lambda swa_in: sparse_mla_swa_attention(
            jnp.asarray(q),
            gn,
            gr,
            jnp.asarray(tk),
            jnp.asarray(sinks),
            jnp.asarray(sl),
            jnp.asarray(cu),
            sm_scale=0.0625,
            swa_window=swa_win,
            swa_kv=swa_in,
            swa_min_pos=jnp.asarray(min_pos),
            interpret=True,
        )
    )
    out_2d = attn_2d_jit(jnp.asarray(swa_2d))
    assert not np.any(np.isnan(np.asarray(out_2d, dtype=np.float32)))
    np.testing.assert_allclose(
        np.asarray(out_2d, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(out_2d[4:], dtype=np.float32), 0.0)

    # Positive control: corrupting an active slot at pos == swa_min_pos yields NaN.
    swa_2d_bad = swa_2d.copy()
    swa_2d_bad[2 * swa_w + 4, 0] = _BF16(np.nan)
    out_bad = attn_2d_jit(jnp.asarray(swa_2d_bad))
    assert np.any(np.isnan(np.asarray(out_bad[1], dtype=np.float32)))
