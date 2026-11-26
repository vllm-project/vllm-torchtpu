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
"""CPU JAX safety tests for Sparse MLA `-1` padding and prefix-cache isolation."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import indexer_qprep, lse_rescale
from vllm_torchtpu.kernels.mla import absorb_q, dcp_token_parallel, kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel


def _murmur_scatter_ref(
    flat_idx: np.ndarray, total_num_pages: int, page_size: int
) -> np.ndarray:
    h = flat_idx.astype(np.uint32)
    h ^= h >> np.uint32(16)
    h = np.uint32(h * np.uint32(0x85EBCA6B))
    h ^= h >> np.uint32(13)
    h = np.uint32(h * np.uint32(0xC2B2AE35))
    h ^= h >> np.uint32(16)
    hi = h.view(np.int32)
    page = np.mod((h >> np.uint32(9)).view(np.int32), total_num_pages)
    off = (
        (hi & (page_size - 1))
        if (page_size & (page_size - 1)) == 0
        else np.mod(hi & ((1 << 23) - 1), page_size)
    )
    return page * page_size + off


def _cpu_interpret_gather_page_ids(
    page_indices: jax.Array,
    seq_page_ids: jax.Array,
    seq_ids_segment: jax.Array,
    max_num_seqs: int,
    *,
    block_tokens: int = 8,
) -> jax.Array:
    num_tokens, topk = seq_page_ids.shape
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_chunks = sparse_kernel.cdiv(pages_per_seq, sparse_kernel._GATHER_PAGE_CHUNK)
    padded_pps = num_chunks * sparse_kernel._GATHER_PAGE_CHUNK
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    windows = page_table[seq_ids_segment]
    logical = jnp.clip(seq_page_ids, 0, pages_per_seq - 1)
    padded_tokens = sparse_kernel.align_to(num_tokens, block_tokens)
    if padded_tokens != num_tokens:
        pad = padded_tokens - num_tokens
        windows, logical = (
            jnp.pad(windows, ((0, pad), (0, 0))),
            jnp.pad(logical, ((0, pad), (0, 0))),
        )
    return pl.pallas_call(
        functools.partial(sparse_kernel._gather_page_ids_kernel, num_chunks=num_chunks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, padded_pps), lambda t: (t, 0)),
                pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            ],
            out_specs=pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            grid=(padded_tokens // block_tokens,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
        interpret=True,
    )(windows, logical)[:num_tokens]


def test_scatter_and_prepare_gather_indices_safety(monkeypatch):
    pages, offs = sparse_kernel.scatter_padding_slots(
        jnp.arange(64, dtype=jnp.int32), 64, 16
    )
    p_np, o_np = np.asarray(pages), np.asarray(offs)
    assert np.all((p_np >= 0) & (p_np < 64)) and np.all((o_np >= 0) & (o_np < 16))
    assert len(set(p_np.tolist())) > 16

    monkeypatch.setattr(
        sparse_kernel, "gather_page_ids", _cpu_interpret_gather_page_ids
    )
    page_size, total_num_pages, topk = 16, 64, 8
    block_table = jnp.array(
        [[17, 29, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], jnp.int32
    )
    topk_indices = jnp.array(
        [[0, 1, 18, *([-1] * 5)], [16, *([-1] * 7)], [-1] * topk, [-1] * topk],
        jnp.int32,
    )
    kv_lens, phys = sparse_kernel.prepare_gather_indices(
        topk_indices=topk_indices,
        page_indices=block_table.reshape(-1),
        cu_q_lens=jnp.array([0, 2, 2, 2, 2], jnp.int32),
        page_size=page_size,
        total_num_pages=total_num_pages,
        is_dcp_mode=True,
        num_valid_tokens=jnp.int32(2),
    )
    np.testing.assert_array_equal(np.asarray(kv_lens), [3, 1, 0, 0])
    flat_idx = np.asarray(phys).reshape(-1)
    exp_scattered = _murmur_scatter_ref(
        np.arange(topk_indices.size, dtype=np.int32), total_num_pages, page_size
    )
    assert [int(flat_idx[i]) for i in (0, 1, 2, topk)] == [
        17 * page_size,
        17 * page_size + 1,
        29 * page_size + 2,
        29 * page_size,
    ]
    np.testing.assert_array_equal(flat_idx[3:topk], exp_scattered[3:topk])
    np.testing.assert_array_equal(flat_idx[topk + 1 :], exp_scattered[topk + 1 :])

    kv_u, phys_u = sparse_kernel.prepare_gather_indices(
        topk_indices=jnp.array([[50, 75, -1, -1]], jnp.int32),
        page_indices=None,
        cu_q_lens=jnp.array([0, 1], jnp.int32),
        page_size=page_size,
        total_num_pages=total_num_pages,
    )
    np.testing.assert_array_equal(np.asarray(kv_u), [2])
    np.testing.assert_array_equal(np.asarray(phys_u)[0, :2], [50, 75])
    np.testing.assert_array_equal(np.asarray(phys_u)[0, 2:], exp_scattered[2:4])


def test_window_physical_rows_and_absorb_q():
    raw = jnp.array([[31, 42, 0, 0]], jnp.int32)
    windows = jnp.pad(raw, ((0, 0), (0, sparse_kernel._GATHER_PAGE_CHUNK - 4)))
    idx = jnp.array([[0, 257, -1, -1]], jnp.int32)
    flat_elem = jnp.arange(idx.size, dtype=jnp.int32).reshape(idx.shape)
    rows = np.asarray(
        sparse_kernel._window_physical_rows(
            idx, windows, flat_elem, page_size=256, pages_per_seq=4, total_num_pages=128
        )
    )[0]
    exp_rows = _murmur_scatter_ref(np.asarray(flat_elem), 128, 256)[0]
    assert (rows[0], rows[1], rows[2], rows[3]) == (
        31 * 256,
        42 * 256 + 1,
        exp_rows[2],
        exp_rows[3],
    )

    kq, kw = jax.random.split(jax.random.PRNGKey(7))
    q_nope = jax.random.normal(kq, (5, 4, 192), jnp.float32)
    w_uk_t = jax.random.normal(kw, (4, 192, 512), jnp.float32)
    got = absorb_q.absorb_q(q_nope, w_uk_t)
    assert got.shape == (5, 4, 512) and got.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(jnp.einsum("thd,hdk->thk", q_nope, w_uk_t)),
        rtol=1e-5,
        atol=1e-5,
    )
    spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        num_pages=16,
        page_size=256,
        head_dim=64,
    )
    assert (spec.head_dim, spec.layout) == (64, kv_cache_utils.KVCacheLayout.SPARSECORE)


@pytest.mark.parametrize("num_tokens", [4, 8, 16])
def test_pack_send_lse_words_under_send_world_cpu_pallas(num_tokens: int):
    num_q_heads, send_world = 64, 16
    send_per, send_rows = (
        num_q_heads // send_world,
        num_tokens * (num_q_heads // send_world),
    )
    lanes = sparse_kernel.send_layout_lanes(num_tokens, num_q_heads, send_world)
    lse_np = (
        np.random.default_rng(100 + num_tokens)
        .standard_normal((num_tokens, num_q_heads))
        .astype(np.float32)
    )
    lse_np[0, 0] = lse_np[num_tokens // 2, -1] = -np.inf
    lse_np[-1, 3] = np.float32(-17.125)
    src_np = np.zeros((num_tokens, lanes), dtype=np.float32)
    src_np[:, :num_q_heads] = lse_np.reshape(num_tokens, send_world, send_per)[
        :, ::-1, :
    ].reshape(num_tokens, num_q_heads)

    pk_kw = dict(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        send_world=send_world,
        send_lanes=lanes,
    )
    words = pl.pallas_call(
        lambda x_ref, out_ref: out_ref.__setitem__(
            Ellipsis,
            sparse_kernel.pack_send_lse_words(x_ref[...], **pk_kw).astype(jnp.int16),
        ),
        out_shape=jax.ShapeDtypeStruct((send_world, lanes), jnp.int16),
        interpret=True,
    )(jnp.asarray(src_np))
    w_np = np.asarray(words, dtype=np.int16)
    hi = w_np[:, :send_rows].astype(np.int32) << 16
    lo = w_np[:, send_rows : 2 * send_rows].astype(np.int32) & 0xFFFF
    rec = (
        (hi | lo)
        .astype(np.uint32)
        .view(np.float32)
        .reshape(send_world, num_tokens, send_per)
        .transpose(1, 0, 2)
        .reshape(num_tokens, num_q_heads)
    )
    np.testing.assert_array_equal(rec.view(np.uint32), lse_np.view(np.uint32))


def test_lse_rescale_all_neg_inf_indexer_qprep_odd_rows_and_flat_rows_bounds():
    out = jnp.ones((8, 4, 128), dtype=jnp.bfloat16)
    lse = jnp.full((8, 4), -jnp.inf, dtype=jnp.float32)
    rescaled = np.asarray(lse_rescale.lse_rescale_head_major_f32(out, lse, lse))
    assert np.all(np.isfinite(rescaled))
    np.testing.assert_array_equal(rescaled, np.zeros((4, 8, 128), np.float32))

    k1, k2, k3 = jax.random.split(jax.random.key(19), 3)
    q = jax.random.normal(k1, (12, 4 * 128), dtype=jnp.bfloat16)
    c, s = (jax.random.normal(k, (12, 128), dtype=jnp.bfloat16) for k in (k2, k3))
    q8, sc = indexer_qprep.indexer_q_prep(
        q, c, s, n_head=4, rope_dim=64, interpret=True
    )
    rq8, rsc = indexer_qprep.reference(q, c, s, n_head=4, rope_dim=64)
    np.testing.assert_array_equal(
        np.asarray(q8, np.float32), np.asarray(rq8, np.float32)
    )
    np.testing.assert_allclose(np.asarray(sc), np.asarray(rsc), rtol=1e-5)

    layout = dcp_token_parallel.TokenParallelLayout(4, 64, 256, 8, 4)
    mapped = np.asarray(
        dcp_token_parallel.flat_rows(
            layout, jnp.array([[0, 4096, -1]], jnp.int32), jnp.zeros(1, jnp.int32)
        )
    )[0]
    assert tuple(mapped.tolist()) == (0, -1, -1)


@pytest.mark.tpu_test
def test_lse_rescale_and_indexer_qprep_on_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    k1, k2 = jax.random.split(jax.random.key(29))
    out = jax.random.normal(k1, (16, 4, 128), dtype=jnp.bfloat16)
    lse = jax.random.normal(k2, (16, 4), dtype=jnp.float32).at[0, :].set(-jnp.inf)
    merged = jnp.maximum(lse, lse + 0.5).at[0, :].set(-jnp.inf)
    got = lse_rescale.lse_rescale_head_major_f32(out, lse, merged)
    ref = lse_rescale._lse_rescale_fallback(out, lse, merged, jnp.float32)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)

    q = jax.random.normal(k1, (16, 4 * 128), dtype=jnp.bfloat16)
    c = s = jnp.full((16, 128), 0.5, dtype=jnp.bfloat16)
    q8, sc = indexer_qprep.indexer_q_prep(q, c, s, n_head=4, rope_dim=64)
    rq8, rsc = indexer_qprep.reference(q, c, s, n_head=4, rope_dim=64)

    def _deq(x, scale):
        return np.asarray(x.astype(jnp.float32) * scale[..., None])

    np.testing.assert_allclose(_deq(q8, sc), _deq(rq8, rsc), rtol=3e-2, atol=3e-2)
