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
"""Unit and TPU7x hardware tests for DeepSeek-V4.1 Sparse MLA + SWA kernels."""

from __future__ import annotations

import contextlib
import glob
import types
from typing import Any

import jax
import jax.numpy as jnp
import jax.profiler
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    gather_sparse_fp8_kv,
    sparse_mla_swa_attention,
)

_FP8 = ml_dtypes.float8_e4m3fn
_BF16 = ml_dtypes.bfloat16


@pytest.fixture(autouse=True)
def _tpu7x_abstract_mesh() -> Any:
    """Provide an abstract TPU7x mesh on CPU so ``pltpu.get_tpu_info()`` succeeds."""
    if jax.default_backend() == "tpu":
        yield
        return
    dev = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=dev)
    with jax.sharding.use_abstract_mesh(mesh):
        yield


def _make_raw_fp8_pool(
    rng: np.random.Generator, n_rows: int, *, include_zero_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode raw FP8+E8M0 NoPE and BF16 RoPE word-row cache pools."""
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
    """Independent float64 reference without kernel helper reuse."""
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


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    ("has_hi", "topk", "heads", "swa_w", "swa_win", "ratio", "idx_mode", "scope"),
    [
        (False, 32, 8, 0, 0, 1, "rand", False),
        (True, 36, 12, 18, 5, 1, "rand", True),
        (True, 32, 3, 7, 128, 2, "rand", False),
        (False, 40, 8, 6, 4, 1, "neg", False),
        (True, 40, 8, 6, 6, 1, "dup", False),
    ],
)
def test_gather_and_sparse_mla_swa_independent_ref(
    seed: int,
    has_hi: bool,
    topk: int,
    heads: int,
    swa_w: int,
    swa_win: int,
    ratio: int,
    idx_mode: str,
    scope: bool,
) -> None:
    rng = np.random.default_rng(seed + 100)
    page_size, n_pages = 16, 6
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, n_pages * page_size)
    npw = nope_w.reshape(n_pages, page_size, 128)
    rpw = rope_w.reshape(n_pages, page_size // 4, 128)
    if has_hi:
        kl, kh = jnp.asarray(npw[:3]), jnp.asarray(npw[3:])
        rl, rh = jnp.asarray(rpw[:3]), jnp.asarray(rpw[3:])
    else:
        kl, kh, rl, rh = jnp.asarray(npw), None, jnp.asarray(rpw), None

    tokens = 6
    cu = np.array([0, 2, 5], dtype=np.int32)
    sl = np.array([28, 65], dtype=np.int32)
    bt = np.array([[0, 2, 4, -1], [1, 3, 5, 99]], dtype=np.int32)
    if idx_mode == "neg":
        tk = -np.ones((tokens, topk), dtype=np.int32)
    elif idx_mode == "dup":
        tk = np.full((tokens, topk), 3, dtype=np.int32)
        tk[:, ::2] = 19
    else:
        tk = rng.integers(-2, 52, size=(tokens, topk), dtype=np.int32)

    q = (rng.standard_normal((tokens, heads, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-1.0, 1.0, size=(heads,)).astype(np.float32)
    swa = (
        (rng.standard_normal((tokens, swa_w, 512)) * 0.2).astype(_BF16)
        if swa_w > 0
        else None
    )
    sm = 0.0625

    gn, gr = gather_sparse_fp8_kv(
        kl,
        rl,
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=page_size,
        kv_pool_hi=kh,
        rope_pool_hi=rh,
        interpret=True,
        enable_named_scope=scope,
    )
    out = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=sm,
        swa_window=swa_win,
        swa_kv=None if swa is None else jnp.asarray(swa),
        kv_compress_ratio=ratio,
        interpret=True,
        enable_named_scope=scope,
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
    np.testing.assert_array_equal(np.asarray(gn), rgn)
    np.testing.assert_array_equal(np.asarray(gr), rgr)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(out[5:], dtype=np.float32), 0.0)


def test_e8m0_zero_scale_and_masked_nan_row_safety() -> None:
    """Verify e8m0 byte 0 decodes to 2^-127 and masked 0xFF rows do not NaN."""
    rng = np.random.default_rng(7)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 16, include_zero_scale=True)
    kl = jnp.asarray(nope_w.reshape(1, 16, 128))
    rl = jnp.asarray(rope_w.reshape(1, 4, 128))
    tk = np.full((1, 32), -1, dtype=np.int32)
    tk[0, 0] = 0
    bt = np.array([[0]], dtype=np.int32)
    cu = np.array([0, 1], dtype=np.int32)
    sl = np.array([1], dtype=np.int32)
    q = (rng.standard_normal((1, 8, 512)) * 0.2).astype(_BF16)
    sinks = np.zeros((8,), dtype=np.float32)

    gn, gr = gather_sparse_fp8_kv(
        kl,
        rl,
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=True,
    )
    gn_corrupt = gn.at[0, 1, :].set(jnp.int32(-1))
    out = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn_corrupt,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=0,
        interpret=True,
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
        swa_window=0,
        swa_kv=None,
        kv_compress_ratio=1,
    )
    assert not np.any(np.isnan(np.asarray(out, dtype=np.float32)))
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )


def test_zero_dimension_short_circuits() -> None:
    kv_lo = jnp.zeros((2, 16, 128), dtype=jnp.int32)
    rp_lo = jnp.zeros((2, 4, 128), dtype=jnp.int32)
    bt = jnp.zeros((1, 2), dtype=jnp.int32)
    cu = jnp.array([0, 0], dtype=jnp.int32)
    gn0, gr0 = gather_sparse_fp8_kv(
        kv_lo,
        rp_lo,
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


@pytest.mark.parametrize(
    ("bad_field", "match"),
    [
        ("page_size", "Expected page_size > 0"),
        ("kv_lo_ndim", "Expected kv_pool_lo"),
        ("kv_lo_shape", "Expected kv_pool_lo"),
        ("rp_lo", "Expected rope_pool_lo"),
        ("hi_mismatch", "kv_pool_hi and rope_pool_hi must both"),
        ("kv_hi_ndim", "Expected kv_pool_hi"),
        ("kv_hi_shape", "Expected kv_pool_hi"),
        ("rp_hi", "Expected rope_pool_hi"),
        ("topk", "Expected topk_indices"),
        ("bt", "Expected block_tables"),
        ("cu", "Expected cu_q_lens"),
    ],
)
def test_gather_validation_errors(bad_field: str, match: str) -> None:
    kv_lo = jnp.zeros((2, 16, 128), dtype=jnp.int32)
    rp_lo = jnp.zeros((2, 4, 128), dtype=jnp.int32)
    kv_hi = jnp.zeros((2, 16, 128), dtype=jnp.int32)
    rp_hi = jnp.zeros((2, 4, 128), dtype=jnp.int32)
    tk = jnp.zeros((2, 32), dtype=jnp.int32)
    bt = jnp.zeros((1, 2), dtype=jnp.int32)
    cu = jnp.array([0, 2], dtype=jnp.int32)
    page_size = 16

    if bad_field == "page_size":
        page_size = 14
    elif bad_field == "kv_lo_ndim":
        kv_lo = jnp.zeros((16, 128), dtype=jnp.int32)
    elif bad_field == "kv_lo_shape":
        kv_lo = jnp.zeros((2, 16, 64), dtype=jnp.int32)
    elif bad_field == "rp_lo":
        rp_lo = jnp.zeros((2, 3, 128), dtype=jnp.int32)
    elif bad_field == "hi_mismatch":
        rp_hi = None
    elif bad_field == "kv_hi_ndim":
        kv_hi = jnp.zeros((16, 128), dtype=jnp.int32)
    elif bad_field == "kv_hi_shape":
        kv_hi = jnp.zeros((2, 16, 64), dtype=jnp.int32)
    elif bad_field == "rp_hi":
        rp_hi = jnp.zeros((2, 3, 128), dtype=jnp.int32)
    elif bad_field == "topk":
        tk = jnp.zeros((2, 30), dtype=jnp.int32)
    elif bad_field == "bt":
        bt = jnp.zeros((1, 0), dtype=jnp.int32)
    elif bad_field == "cu":
        cu = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match=match):
        gather_sparse_fp8_kv(
            kv_lo,
            rp_lo,
            tk,
            bt,
            cu,
            page_size=page_size,
            kv_pool_hi=kv_hi,
            rope_pool_hi=rp_hi,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("bad_field", "match"),
    [
        ("q", "Expected q"),
        ("topk", "Expected topk_indices"),
        ("nope", "Expected gathered_nope"),
        ("rope", "Expected gathered_rope"),
        ("sinks", "Expected attention_sinks"),
        ("seq_lens", "Expected seq_lens"),
        ("cu", "Expected cu_q_lens"),
        ("window", "Expected swa_window >= 0"),
        ("ratio", "Expected swa_window >= 0"),
        ("swa_kv_shape", "Expected swa_kv"),
        ("swa_kv_tokens", "Expected swa_kv"),
        ("swa_kv_ndim", "Expected swa_kv"),
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
    q = jnp.zeros((4, 8, 512), dtype=jnp.bfloat16)
    gn = jnp.zeros((4, 32, 128), dtype=jnp.int32)
    gr = jnp.zeros((4, 8, 128), dtype=jnp.int32)
    tk = jnp.zeros((4, 32), dtype=jnp.int32)
    sinks = jnp.zeros((8,), dtype=jnp.float32)
    sl = jnp.array([16, 16], dtype=jnp.int32)
    cu = jnp.array([0, 2, 4], dtype=jnp.int32)
    swa_w, ratio = 4, 1
    swa_kv: jax.Array | None = None
    swa_min_pos: jax.Array | None = None

    if bad_field == "q":
        q = jnp.zeros((4, 8, 256), dtype=jnp.bfloat16)
    elif bad_field == "topk":
        tk = jnp.zeros((4, 30), dtype=jnp.int32)
    elif bad_field == "nope":
        gn = jnp.zeros((4, 16, 128), dtype=jnp.int32)
    elif bad_field == "rope":
        gr = jnp.zeros((4, 4, 128), dtype=jnp.int32)
    elif bad_field == "sinks":
        sinks = jnp.zeros((4,), dtype=jnp.float32)
    elif bad_field == "seq_lens":
        sl = jnp.zeros((0,), dtype=jnp.int32)
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


def test_sparse_mla_tpu7x_lowering_trace() -> None:
    """Verify `interpret=False` lowers cleanly under abstract TPU7x mesh."""
    q = jax.ShapeDtypeStruct((4, 8, 512), jnp.bfloat16)
    kv_lo = jax.ShapeDtypeStruct((4, 16, 128), jnp.int32)
    rp_lo = jax.ShapeDtypeStruct((4, 4, 128), jnp.int32)
    gn = jax.ShapeDtypeStruct((4, 32, 128), jnp.int32)
    gr = jax.ShapeDtypeStruct((4, 8, 128), jnp.int32)
    tk = jax.ShapeDtypeStruct((4, 32), jnp.int32)
    bt = jax.ShapeDtypeStruct((2, 4), jnp.int32)
    sinks = jax.ShapeDtypeStruct((8,), jnp.float32)
    sl = jax.ShapeDtypeStruct((2,), jnp.int32)
    cu = jax.ShapeDtypeStruct((3,), jnp.int32)
    swa = jax.ShapeDtypeStruct((4, 16, 512), jnp.bfloat16)
    lowered_gather = (
        jax.jit(
            lambda kl, rl, tk_, bt_, cu_: gather_sparse_fp8_kv(
                kl,
                rl,
                tk_,
                bt_,
                cu_,
                page_size=16,
                kv_pool_hi=kl,
                rope_pool_hi=rl,
                interpret=False,
            )
        )
        .trace(kv_lo, rp_lo, tk, bt, cu)
        .lower(lowering_platforms=("tpu",))
    )
    lowered = (
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
                swa_window=16,
                swa_kv=swa_,
                interpret=False,
            )
        )
        .trace(q, gn, gr, tk, sinks, sl, cu, swa)
        .lower(lowering_platforms=("tpu",))
    )
    assert "dsv41_gather_page_ids" in lowered_gather.as_text()
    assert "sc_csa_gather" in lowered_gather.as_text()
    assert "dsv41_sparse_mla_swa" in lowered.as_text()

    # Multi-chunk compact 2D swa_kv regression guard: T=640, H=64 (step=128 -> 5 chunks
    # via scan) with compact 2D swa_kv must emit 1 extract + 1 attention custom call
    # inside while without materializing bf16[640,128,512].
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
def test_sparse_mla_swa_tpu7x_hardware() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("TPU7x hardware test requires physical TPU backend.")
    rng = np.random.default_rng(17)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 64)
    npw = nope_w.reshape(4, 16, 128)
    rpw = rope_w.reshape(4, 4, 128)
    tokens, topk, heads = 6, 36, 12
    tk = rng.integers(-1, 28, size=(tokens, topk), dtype=np.int32)
    bt = np.array([[0, 2], [1, 3]], dtype=np.int32)
    cu = np.array([0, 2, 5], dtype=np.int32)
    sl = np.array([24, 30], dtype=np.int32)
    q = (rng.standard_normal((tokens, heads, 512)) * 0.15).astype(_BF16)
    sinks = rng.uniform(-0.8, 0.8, size=(heads,)).astype(np.float32)
    swa = (rng.standard_normal((tokens, 18, 512)) * 0.2).astype(_BF16)

    gn_tpu, gr_tpu = gather_sparse_fp8_kv(
        jnp.asarray(npw[:2]),
        jnp.asarray(rpw[:2]),
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        kv_pool_hi=jnp.asarray(npw[2:]),
        rope_pool_hi=jnp.asarray(rpw[2:]),
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

    # Also verify V4.1 DP=8 TP=1 production head/tile geometry (H=64, K=512, W=128).
    nope_w2, rope_w2, kv_f64_2 = _make_raw_fp8_pool(rng, 1024)
    npw2 = nope_w2.reshape(64, 16, 128)
    rpw2 = rope_w2.reshape(64, 4, 128)
    tk2 = rng.integers(-1, 500, size=(4, 512), dtype=np.int32)
    bt2 = np.arange(64, dtype=np.int32).reshape(2, 32)
    cu2 = np.array([0, 2, 4], dtype=np.int32)
    sl2 = np.array([512, 1024], dtype=np.int32)
    q2 = (rng.standard_normal((4, 64, 512)) * 0.1).astype(_BF16)
    sinks2 = rng.uniform(-0.5, 0.5, size=(64,)).astype(np.float32)
    swa2 = (rng.standard_normal((4, 128, 512)) * 0.1).astype(_BF16)
    gn2, gr2 = gather_sparse_fp8_kv(
        jnp.asarray(npw2[:32]),
        jnp.asarray(rpw2[:32]),
        jnp.asarray(tk2),
        jnp.asarray(bt2),
        jnp.asarray(cu2),
        page_size=16,
        kv_pool_hi=jnp.asarray(npw2[32:]),
        rope_pool_hi=jnp.asarray(rpw2[32:]),
        interpret=False,
    )
    out2 = sparse_mla_swa_attention(
        jnp.asarray(q2),
        gn2,
        gr2,
        jnp.asarray(tk2),
        jnp.asarray(sinks2),
        jnp.asarray(sl2),
        jnp.asarray(cu2),
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=jnp.asarray(swa2),
        kv_compress_ratio=2,
        interpret=False,
    )
    rgn2, rgr2, rout2 = _independent_float64_reference(
        q2,
        kv_f64_2,
        nope_w2,
        rope_w2,
        tk2,
        bt2,
        sinks2,
        sl2,
        cu2,
        page_size=16,
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=swa2,
        kv_compress_ratio=2,
    )
    np.testing.assert_array_equal(np.asarray(gn2), rgn2)
    np.testing.assert_array_equal(np.asarray(gr2), rgr2)
    np.testing.assert_allclose(
        np.asarray(out2, dtype=np.float32), rout2, rtol=2e-2, atol=2e-2
    )

    # Compact 2D swa_kv on TPU7x: multi-chunk block_tables (max_pages=300),
    # short sequence negative prefix positions filled with NaN, q_len > swa_w,
    # padded empty sequences, and multi-chunk scan (T=300 > step=128 at H=64).
    bt3 = rng.integers(0, 64, size=(4, 300), dtype=np.int32)
    cu3 = np.array([0, 0, 5, 27, 27], dtype=np.int32)
    sl3 = np.array([0, 12, 60, 0], dtype=np.int32)
    tok3, w3 = 29, 18
    tk3 = rng.integers(-1, 60, size=(tok3, 36), dtype=np.int32)
    q3 = (rng.standard_normal((tok3, 12, 512)) * 0.1).astype(_BF16)
    sinks3 = rng.uniform(-0.5, 0.5, size=(12,)).astype(np.float32)
    seq_swa3 = (rng.standard_normal((4, w3, 512)) * 0.15).astype(_BF16)
    # Sequence 1 has p0 = 12 - 5 = 7 < w3 = 18: slots 0..10 are at negative pos < 0.
    # Seq 2 has p0 = 60 - 22 = 38: swa_min_pos[2] = 30 masks slots 0..9 (pos 20..29).
    min_pos3 = np.array([0, -4, 30, 0], dtype=np.int32)
    seq_swa3[1, :11, :] = _BF16(np.nan)
    seq_swa3[2, :10, :] = _BF16(np.nan)
    seq_swa3[0, :, :] = _BF16(np.nan)
    seq_swa3[3, :, :] = _BF16(np.nan)
    kv_rot3 = (rng.standard_normal((tok3, 512)) * 0.15).astype(_BF16)
    kv_rot3[27:, :] = _BF16(np.nan)
    swa3_2d = np.concatenate([seq_swa3.reshape(4 * w3, 512), kv_rot3], axis=0)
    gn3, gr3 = gather_sparse_fp8_kv(
        jnp.asarray(npw2[:32]),
        jnp.asarray(rpw2[:32]),
        jnp.asarray(tk3),
        jnp.asarray(bt3),
        jnp.asarray(cu3),
        page_size=16,
        kv_pool_hi=jnp.asarray(npw2[32:]),
        rope_pool_hi=jnp.asarray(rpw2[32:]),
        interpret=False,
    )
    out3 = sparse_mla_swa_attention(
        jnp.asarray(q3),
        gn3,
        gr3,
        jnp.asarray(tk3),
        jnp.asarray(sinks3),
        jnp.asarray(sl3),
        jnp.asarray(cu3),
        sm_scale=0.0625,
        swa_window=16,
        swa_kv=jnp.asarray(swa3_2d),
        kv_compress_ratio=1,
        swa_min_pos=jnp.asarray(min_pos3),
        interpret=False,
    )
    rgn3, rgr3, rout3 = _independent_float64_reference(
        q3,
        kv_f64_2,
        nope_w2,
        rope_w2,
        tk3,
        bt3,
        sinks3,
        sl3,
        cu3,
        page_size=16,
        sm_scale=0.0625,
        swa_window=16,
        swa_kv=swa3_2d,
        kv_compress_ratio=1,
        swa_min_pos=min_pos3,
    )
    assert not np.any(np.isnan(np.asarray(out3, dtype=np.float32)))
    np.testing.assert_array_equal(np.asarray(gn3), rgn3)
    np.testing.assert_array_equal(np.asarray(gr3), rgr3)
    np.testing.assert_allclose(
        np.asarray(out3, dtype=np.float32), rout3, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(out3[27:], dtype=np.float32), 0.0)

    # Multi-chunk scan on TPU7x (T=300 > step=128 at H=64) with compact 2D swa_kv.
    tok4 = 300
    tk4 = rng.integers(0, 500, size=(tok4, 256), dtype=np.int32)
    cu4 = np.array([0, 150, 300], dtype=np.int32)
    sl4 = np.array([600, 800], dtype=np.int32)
    q4 = (rng.standard_normal((tok4, 64, 512)) * 0.1).astype(_BF16)
    swa4_2d = (rng.standard_normal((2 * 128 + tok4, 512)) * 0.1).astype(_BF16)
    gn4, gr4 = gather_sparse_fp8_kv(
        jnp.asarray(npw2),
        jnp.asarray(rpw2),
        jnp.asarray(tk4),
        jnp.asarray(bt2),
        jnp.asarray(cu4),
        page_size=16,
        interpret=False,
    )
    out4 = sparse_mla_swa_attention(
        jnp.asarray(q4),
        gn4,
        gr4,
        jnp.asarray(tk4),
        jnp.asarray(sinks2),
        jnp.asarray(sl4),
        jnp.asarray(cu4),
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=jnp.asarray(swa4_2d),
        kv_compress_ratio=1,
        interpret=False,
    )
    _, _, rout4 = _independent_float64_reference(
        q4,
        kv_f64_2,
        nope_w2,
        rope_w2,
        tk4,
        bt2,
        sinks2,
        sl4,
        cu4,
        page_size=16,
        sm_scale=0.0625,
        swa_window=128,
        swa_kv=swa4_2d,
        kv_compress_ratio=1,
    )
    np.testing.assert_allclose(
        np.asarray(out4, dtype=np.float32), rout4, rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("tokens", [6, 7])
def test_sparse_mla_swa_multi_chunk_tiling(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    """Verify per-chunk `_pack_chunk_tiles` + `jax.lax.scan` (exact & remainder)."""
    from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
        benchmark_sparse_mla,
    )
    from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
        kernel as mla_k,
    )

    monkeypatch.setattr(mla_k, "_max_vmem_token_heads", lambda: 16)
    rng = np.random.default_rng(23 + tokens)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 256)
    kl = jnp.asarray(nope_w.reshape(16, 16, 128))
    rl = jnp.asarray(rope_w.reshape(16, 4, 128))
    tk = rng.integers(-1, 200, size=(tokens, 260), dtype=np.int32)
    bt = np.arange(32, dtype=np.int32).reshape(2, 16) % 16
    cu = np.array([0, 3, tokens - 1], dtype=np.int32)
    sl = np.array([180, 220], dtype=np.int32)
    q = (rng.standard_normal((tokens, 8, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(8,)).astype(np.float32)
    swa = (rng.standard_normal((tokens, 134, 512)) * 0.1).astype(_BF16)
    gn, gr = gather_sparse_fp8_kv(
        kl,
        rl,
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=True,
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
        swa_window=130,
        swa_kv=jnp.asarray(swa),
        kv_compress_ratio=1,
        interpret=True,
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
        swa_window=130,
        swa_kv=swa,
        kv_compress_ratio=1,
    )
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(out[tokens - 1 :], dtype=np.float32), 0.0)
    if tokens == 6:
        assert (
            benchmark_sparse_mla.main(["--interpret", "--warmup", "1", "--repeat", "1"])
            == 0
        )
        fake_pd = types.SimpleNamespace(
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
        monkeypatch.setattr(glob, "glob", lambda *_a, **_kw: ["/tmp/fake.xplane.pb"])
        monkeypatch.setattr(jax.profiler, "trace", contextlib.nullcontext)
        monkeypatch.setattr(
            jax.profiler.ProfileData,
            "from_file",
            staticmethod(lambda _path: fake_pd),
        )
        dev_us, host_us = benchmark_sparse_mla._time_fn(
            lambda: jnp.int32(1), interpret=False, warmup=1, repeat=2
        )
        assert dev_us == pytest.approx(85.0)
        assert host_us > 0.0


def test_check_tpu7x_guard() -> None:
    """Verify `_check_tpu7x` passes on TPU7x and raises RuntimeError on non-TPU7x."""
    from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
        kernel as mla_k,
    )

    mla_k._check_tpu7x()
    assert mla_k._max_vmem_token_heads() == mla_k.MAX_VMEM_TOKEN_HEADS

    v6e_dev = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    v6e_mesh = jax.sharding.AbstractMesh((), (), (), abstract_device=v6e_dev)
    with (
        jax.sharding.use_abstract_mesh(v6e_mesh),
        pytest.raises(RuntimeError, match="require TPU7x"),
    ):
        mla_k._check_tpu7x()

    valid_sc = jnp.ones((256,), dtype=jnp.bool_)
    unpacked = mla_k._unpack_csa_rope_to_quarters(
        jnp.zeros((64, 128), dtype=jnp.int32), valid_sc, 1, 256
    )
    assert unpacked.shape == (1, 64, 128)


def test_compact_2d_swa_kv_cpu_parity_and_short_seq_nan_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify compact 2D `swa_kv`, NaN causal prefix mask, and >128-page gather."""
    import inspect

    from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
        kernel as mla_k,
    )

    assert "core_attention.sparse_mla" not in inspect.getsource(mla_k)
    monkeypatch.setattr(mla_k, "_max_vmem_token_heads", lambda: 32)
    rng = np.random.default_rng(91)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 256)
    kl = jnp.asarray(nope_w.reshape(16, 16, 128))
    rl = jnp.asarray(rope_w.reshape(16, 4, 128))
    # Multi-chunk block_tables (max_pages=300 -> 3 chunks of 128) + padded empty seqs.
    bt = rng.integers(0, 16, size=(4, 300), dtype=np.int32)
    cu = np.array([0, 0, 3, 12, 12], dtype=np.int32)
    sl = np.array([0, 5, 40, 0], dtype=np.int32)
    tokens, swa_w, heads = 14, 18, 8
    tk = rng.integers(-1, 300 * 16, size=(tokens, 32), dtype=np.int32)
    q = (rng.standard_normal((tokens, heads, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(heads,)).astype(np.float32)

    seq_swa = (rng.standard_normal((4, swa_w, 512)) * 0.15).astype(_BF16)
    kv_rot = (rng.standard_normal((tokens, 512)) * 0.15).astype(_BF16)
    # Build matching 3D swa_kv BEFORE injecting NaNs into masked negative/empty slots.
    swa_3d = np.zeros((tokens, swa_w, 512), dtype=_BF16)
    for s in range(4):
        q0, q1 = int(cu[s]), int(cu[s + 1])
        p0 = int(sl[s]) - (q1 - q0)
        for t in range(q0, q1):
            pos = int(sl[s]) - (q1 - t)
            base_3d = max(0, pos - swa_w + 1)
            for w in range(swa_w):
                p_pos = base_3d + w
                if p_pos <= pos:
                    swa_3d[t, w] = (
                        seq_swa[s, p_pos - (p0 - swa_w)]
                        if p_pos < p0
                        else kv_rot[q0 + (p_pos - p0)]
                    )
    # Seq 1 has p0 = 5 - 3 = 2 < swa_w = 18: slots 0..15 correspond to negative pos < 0.
    seq_swa[1, :16, :] = _BF16(np.nan)
    seq_swa[0, :, :] = _BF16(np.nan)
    seq_swa[3, :, :] = _BF16(np.nan)
    kv_rot[12:, :] = _BF16(np.nan)
    swa_2d = np.concatenate([seq_swa.reshape(4 * swa_w, 512), kv_rot], axis=0)

    gn, gr = gather_sparse_fp8_kv(
        kl,
        rl,
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=True,
    )
    out_2d = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=14,
        swa_kv=jnp.asarray(swa_2d),
        kv_compress_ratio=1,
        interpret=True,
    )
    out_3d = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=14,
        swa_kv=jnp.asarray(swa_3d),
        kv_compress_ratio=1,
        interpret=True,
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
        swa_window=14,
        swa_kv=swa_2d,
        kv_compress_ratio=1,
    )
    assert not np.any(np.isnan(np.asarray(out_2d, dtype=np.float32)))
    np.testing.assert_array_equal(np.asarray(gn), rgn)
    np.testing.assert_array_equal(np.asarray(gr), rgr)
    np.testing.assert_array_equal(np.asarray(out_2d), np.asarray(out_3d))
    np.testing.assert_allclose(
        np.asarray(out_2d, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )
    # Also verify swa_window == 0 ignores 2D swa_kv cleanly after shape validation.
    out_w0 = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=0,
        swa_kv=jnp.asarray(swa_2d),
        interpret=True,
    )
    assert not np.any(np.isnan(np.asarray(out_w0, dtype=np.float32)))


def test_swa_min_pos_replay_lower_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `swa_min_pos` masks `kv_pos < swa_min_pos[s]` in 2D & 3D `swa_kv`."""
    from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
        kernel as mla_k,
    )

    monkeypatch.setattr(mla_k, "_max_vmem_token_heads", lambda: 24)
    rng = np.random.default_rng(137)
    nope_w, rope_w, kv_f64 = _make_raw_fp8_pool(rng, 128)
    kl = jnp.asarray(nope_w.reshape(8, 16, 128))
    rl = jnp.asarray(rope_w.reshape(8, 4, 128))
    bt = np.arange(32, dtype=np.int32).reshape(4, 8) % 8
    cu = np.array([0, 3, 8, 11, 11], dtype=np.int32)
    sl = np.array([10, 40, 50, 0], dtype=np.int32)
    min_pos = np.array([-3, 30, 500, 0], dtype=np.int32)
    tokens, swa_w, swa_win, heads = 13, 18, 16, 8
    tk = rng.integers(-1, 64, size=(tokens, 32), dtype=np.int32)
    q = (rng.standard_normal((tokens, heads, 512)) * 0.1).astype(_BF16)
    sinks = rng.uniform(-0.5, 0.5, size=(heads,)).astype(np.float32)

    seq_swa = (rng.standard_normal((4, swa_w, 512)) * 0.15).astype(_BF16)
    kv_rot = (rng.standard_normal((tokens, 512)) * 0.15).astype(_BF16)
    swa_3d = np.zeros((tokens, swa_w, 512), dtype=_BF16)
    for s in range(4):
        q0, q1 = int(cu[s]), int(cu[s + 1])
        p0 = int(sl[s]) - (q1 - q0)
        mp = max(0, int(min_pos[s]))
        for t in range(q0, q1):
            pos = int(sl[s]) - (q1 - t)
            base_3d = max(0, pos - swa_w + 1)
            for w in range(swa_w):
                p_pos = base_3d + w
                if p_pos < mp:
                    swa_3d[t, w] = _BF16(np.nan)
                elif p_pos <= pos:
                    swa_3d[t, w] = (
                        seq_swa[s, p_pos - (p0 - swa_w)]
                        if p_pos < p0
                        else kv_rot[q0 + (p_pos - p0)]
                    )
    # Poison pre-replay positions (< swa_min_pos[s]) in compact 2D swa_kv as well.
    seq_swa[0, :11, :] = _BF16(np.nan)  # p0=7 -> slots 0..10 have pos -11..-1 < 0
    seq_swa[1, :13, :] = _BF16(np.nan)  # p0=35 -> slots 0..12 have pos 17..29 < 30
    seq_swa[2, :, :] = _BF16(np.nan)  # swa_min_pos[2]=500 masks all SWA rows
    kv_rot[8:11, :] = _BF16(np.nan)  # Seq 2 current-step pos 47..49 < 500
    seq_swa[3, :, :] = _BF16(np.nan)
    kv_rot[11:, :] = _BF16(np.nan)
    swa_2d = np.concatenate([seq_swa.reshape(4 * swa_w, 512), kv_rot], axis=0)

    gn, gr = gather_sparse_fp8_kv(
        kl,
        rl,
        jnp.asarray(tk),
        jnp.asarray(bt),
        jnp.asarray(cu),
        page_size=16,
        interpret=True,
    )
    out_2d = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=swa_win,
        swa_kv=jnp.asarray(swa_2d),
        swa_min_pos=jnp.asarray(min_pos),
        interpret=True,
    )
    out_3d = sparse_mla_swa_attention(
        jnp.asarray(q),
        gn,
        gr,
        jnp.asarray(tk),
        jnp.asarray(sinks),
        jnp.asarray(sl),
        jnp.asarray(cu),
        sm_scale=0.0625,
        swa_window=swa_win,
        swa_kv=jnp.asarray(swa_3d),
        swa_min_pos=jnp.asarray(min_pos),
        interpret=True,
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
        swa_window=swa_win,
        swa_kv=swa_2d,
        kv_compress_ratio=1,
        swa_min_pos=min_pos,
    )
    assert not np.any(np.isnan(np.asarray(out_2d, dtype=np.float32)))
    np.testing.assert_array_equal(np.asarray(out_2d), np.asarray(out_3d))
    np.testing.assert_allclose(
        np.asarray(out_2d, dtype=np.float32), rout, rtol=2e-2, atol=2e-2
    )

    # Prove masked rows (< swa_min_pos) get swa_v=False (zero weight, no dilution)
    # and unmasked NaN rows (>= swa_min_pos) still surface NaN.
    q1 = np.full((1, 8, 512), _BF16(0.05), dtype=_BF16)
    gn1 = jnp.zeros((1, 32, 128), dtype=jnp.int32)
    gr1 = jnp.zeros((1, 8, 128), dtype=jnp.int32)
    tk1 = -jnp.ones((1, 32), dtype=jnp.int32)
    sinks1 = jnp.full((8,), -80.0, dtype=jnp.float32)
    sl1, cu1 = jnp.array([11], dtype=jnp.int32), jnp.array([0, 1], dtype=jnp.int32)
    v_only = (rng.standard_normal((512,)) * 0.25).astype(_BF16)
    swa1 = np.full((1, 16, 512), _BF16(25.0), dtype=_BF16)
    swa1[0, 10, :] = v_only  # base=0, w=10 -> pos=10 == swa_min_pos[0]
    out_single = sparse_mla_swa_attention(
        jnp.asarray(q1),
        gn1,
        gr1,
        tk1,
        sinks1,
        sl1,
        cu1,
        sm_scale=0.0625,
        swa_window=16,
        swa_kv=jnp.asarray(swa1),
        swa_min_pos=jnp.array([10], dtype=jnp.int32),
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(out_single[0], dtype=np.float32),
        np.broadcast_to(v_only.astype(np.float32), (8, 512)),
        rtol=1e-2,
        atol=1e-2,
    )
    swa1_bad = swa1.copy()
    swa1_bad[0, 10, 0] = _BF16(np.nan)
    out_bad = sparse_mla_swa_attention(
        jnp.asarray(q1),
        gn1,
        gr1,
        tk1,
        sinks1,
        sl1,
        cu1,
        sm_scale=0.0625,
        swa_window=16,
        swa_kv=jnp.asarray(swa1_bad),
        swa_min_pos=jnp.array([10], dtype=jnp.int32),
        interpret=True,
    )
    assert np.any(np.isnan(np.asarray(out_bad, dtype=np.float32)))
