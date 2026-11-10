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
"""MSA under DCP wider than a KV head's replicas (e.g. DCP 8 at TP 8 with 4
KV heads): every rank holds all of the group's KV heads for its blocks.

The composition (writes, per-head scores and candidates, the cross-rank
merge, per-head attention and the LSE merge) is checked against the
single-rank forward of each KV head with jnp stand-ins for the block-score
and attention kernels, which other tests cover in interpret mode; the
attention kernel's KV-head selection is checked in interpret mode here.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.models.minimax_m3.packing import CACHE_DTYPES, PACKING, pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_select,
    msa_write,
)

PAGE = 128
D = 16
PAGES = 64  # physical pages per rank
SCALE = 0.3


def _xla_write_rows(cache, rows, pages, slots, *, max_runs, bits=None, interpret=False):
    del max_runs, bits, interpret
    packing = cache.shape[2]
    return cache.at[pages, slots // packing, slots % packing].set(rows, mode="drop")


def _rows(cache):
    return cache.reshape(cache.shape[0], -1, cache.shape[-1]).astype(jnp.float32)


def _jnp_block_scores(
    index_q,
    index_cache,
    seq_lens,
    block_tables,
    cu_q_lens,
    num_seqs,
    interpret,
    scale_cache=None,
    scale_row=None,
):
    """`msa_indexer.block_scores` in jnp; pages past a request's end hold NaN
    (the kernel leaves them unwritten). bf16 / fp8 caches only."""
    del num_seqs, interpret, scale_cache, scale_row
    num_tokens = index_q.shape[0]
    max_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_seqs
    seq_id, _, _ = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = block_tables.reshape(max_seqs, pages_per_seq)[seq_id]
    scores = jnp.einsum(
        "td,tpsd->tps", index_q[:, 0].astype(jnp.float32), _rows(index_cache)[pages]
    ).max(-1)
    live = jnp.arange(pages_per_seq)[None] < (-(-seq_lens // PAGE))[seq_id][:, None]
    return jnp.where(live, scores, jnp.nan)[:, None, :]


def _jnp_attention_lse(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    *,
    sm_scale,
    head=0,
    **kwargs,
):
    rows = _rows(kv_cache)[..., head * 2 * D : (head + 1) * 2 * D]
    pages = rows[page_table[jnp.maximum(selected, 0)]]  # [T, K, S, 2D]
    limit = jnp.where(selected == local_page[:, None], local_limit[:, None], PAGE)
    mask = (selected >= 0)[..., None] & (jnp.arange(PAGE) < limit[..., None])
    s = jnp.einsum("thd,tksd->thks", q.astype(jnp.float32), pages[..., :D])
    s = jnp.where(mask[:, None], s * sm_scale, -jnp.inf)
    m = s.max((-1, -2), keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(s - m)
    denom = p.sum((-1, -2))
    safe = jnp.where(denom > 0, denom, 1.0)
    out = jnp.einsum("thks,tksd->thd", p, pages[..., D:]) / safe[..., None]
    lse = jnp.where(denom > 0, m[..., 0, 0] + jnp.log(safe), -jnp.inf)
    return out.astype(q.dtype), lse


def _jnp_attention_lse_heads(q, kv_cache, selected, *args, sm_scale, **kwargs):
    num_heads = selected.shape[1]
    q_per_head = q.shape[1] // num_heads
    outs, lses = zip(
        *(
            _jnp_attention_lse(
                q[:, h * q_per_head : (h + 1) * q_per_head],
                kv_cache,
                selected[:, h],
                *args,
                sm_scale=sm_scale,
                head=h,
            )
            for h in range(num_heads)
        )
    )
    return jnp.concatenate(outs, axis=1), jnp.concatenate(lses, axis=1)


@pytest.fixture
def _fast_kernels(monkeypatch):
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)
    monkeypatch.setattr(msa, "_block_scores", _jnp_block_scores)
    monkeypatch.setattr(
        msa_attention_lse, "msa_sparse_attention_lse", _jnp_attention_lse
    )
    monkeypatch.setattr(
        msa_attention_lse, "msa_sparse_attention_lse_heads", _jnp_attention_lse_heads
    )

    def attention(q, kv_cache, selected, local_page, local_limit, **kwargs):
        table = kwargs.pop("page_table")
        return _jnp_attention_lse(
            q, kv_cache, selected, local_page, local_limit, table, **kwargs
        )[0]

    monkeypatch.setattr(msa_attention, "msa_sparse_attention", attention)


def _merge_lse(outs, lses):
    lses = np.asarray(lses, np.float64)
    m = lses.max(axis=0)
    m = np.where(np.isfinite(m), m, 0.0)
    w = np.where(np.isfinite(lses), np.exp(lses - m), 0.0)
    w = w / np.maximum(w.sum(axis=0), 1e-30)
    return (np.asarray(outs, np.float64) * w[..., None]).sum(axis=0)


@pytest.mark.skip(
    reason="_fast_kernels' jnp stand-ins mirror the pre-r5 kernel API (token-major "
    "msa_sparse_attention_lse(_heads), _block_scores without index_heads); the "
    "real kernels' DCP forward is checked by test_msa_dcp"
)
@pytest.mark.usefixtures("_fast_kernels")
@pytest.mark.parametrize("dcp,kv_heads", [(8, 4), (4, 2), (2, 2), (2, 1)])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,topk,init_blocks",
    [
        ([1, 200, 8, 8], [1500, 300, 2100, 0], 4, 0),
        ([8, 8, 8], [1023, 129, 2047], 3, 1),
    ],
)
def test_multi_head_dcp_matches_each_heads_single_rank(
    q_lens, ctx_lens, topk, init_blocks, dcp, kv_heads
):
    rng = np.random.default_rng(dcp * 10 + kv_heads + len(q_lens))
    dtype = jnp.bfloat16
    q_per_head, max_seqs = 2, 4
    num_seqs, num_real = len(q_lens), sum(q_lens)
    num_tokens = num_real + 3
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = np.add(q_lens, ctx_lens)
    cu_q = np.full(max_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    logical = -(-int(seq_lens.max()) // (PAGE * dcp))
    tables = np.zeros((max_seqs, logical), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, PAGES))[
        : num_seqs * logical
    ].reshape(num_seqs, logical)
    blocks = np.arange(logical * dcp)
    ref_tables = (blocks % dcp) * PAGES + tables[:, blocks // dcp]

    kv_rows = rounded(rng.standard_normal((dcp, PAGES, PAGE, kv_heads * 2 * D)), dtype)
    index_rows = rounded(rng.standard_normal((dcp, PAGES, PAGE, D)), dtype)

    def bf16(*shape):
        return jnp.asarray(rng.standard_normal(shape)).astype(dtype)

    q = bf16(num_tokens, kv_heads * q_per_head, D)
    k, v = bf16(num_tokens, kv_heads, D), bf16(num_tokens, kv_heads, D)
    index_q, index_k = bf16(num_tokens, kv_heads, D), bf16(num_tokens, D)
    distribution = jnp.asarray([0, 0, num_seqs], jnp.int32)
    config = dict(topk=topk, init_blocks=init_blocks, local_blocks=1)

    # Per KV head: the single-rank (DCP 1) forward over all the context.
    ref_meta = (
        jnp.asarray(seq_lens),
        jnp.asarray(ref_tables.reshape(-1)),
        jnp.asarray(cu_q),
    )
    _, pos, _ = msa.token_positions(num_tokens, ref_meta[0], ref_meta[2])
    ref_out, ref_selected = [], []
    for h in range(kv_heads):
        head_rows = kv_rows[..., h * 2 * D : (h + 1) * 2 * D]
        _, ref_index, out = msa.msa_forward(
            q[:, h * q_per_head : (h + 1) * q_per_head],
            k[:, h : h + 1],
            v[:, h : h + 1],
            index_q[:, h : h + 1],
            index_k,
            pack(head_rows.reshape(dcp * PAGES, PAGE, -1), dtype),
            pack(index_rows.reshape(dcp * PAGES, PAGE, -1), dtype),
            *ref_meta,
            distribution,
            sm_scale=SCALE,
            interpret=True,
            **config,
        )
        ref_out.append(np.asarray(out.astype(jnp.float32)))
        scores = _jnp_block_scores(
            index_q[:, h : h + 1], ref_index, *ref_meta, None, True
        )
        ref_selected.append(
            np.asarray(
                msa_select.select_blocks(
                    scores, pos, block_size=PAGE, interpret=True, **config
                )
            )[:, 0]
        )
    ref_out = np.concatenate(ref_out, axis=1)

    meta = (jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q))
    caches, candidates = [], []
    for rank in range(dcp):
        kv, _, cand = msa_dcp.msa_write_and_candidates_dcp(
            k,
            v,
            index_q,
            index_k,
            pack(kv_rows[rank], dtype),
            pack(index_rows[rank], dtype),
            *meta,
            distribution,
            rank=rank,
            dcp=dcp,
            interpret=True,
            **config,
        )
        assert cand.shape == (num_tokens, kv_heads * 2 * topk)
        caches.append(kv)
        candidates.append(cand)
    candidates = jnp.stack(candidates)

    selected = np.asarray(msa_dcp.merge_head_candidates(candidates, topk))
    for h in range(kv_heads):
        np.testing.assert_array_equal(
            selected[:num_real, h], ref_selected[h][:num_real]
        )

    outs, lses = [], []
    for rank in range(dcp):
        out, lse = msa_dcp.msa_attend_dcp(
            q,
            caches[rank],
            candidates,
            *meta,
            sm_scale=SCALE,
            topk=topk,
            rank=rank,
            dcp=dcp,
            interpret=True,
        )
        outs.append(np.asarray(out.astype(jnp.float32)))
        lses.append(np.asarray(lse))
    merged = _merge_lse(outs, lses)
    np.testing.assert_allclose(merged[:num_real], ref_out[:num_real], atol=2e-2)


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lse_kernel_attends_the_chosen_kv_head(dtype):
    """With rows of several KV heads, `head` reads exactly that head's
    ``[K | V]`` slice: equal to the kernel on a one-head cache of it."""
    rng = np.random.default_rng(11)
    num_tokens, pages, topk, heads, hd = 24, 6, 3, 2, 128
    q = jnp.asarray(rng.standard_normal((num_tokens, 4, hd))).astype(jnp.bfloat16)
    rows = rounded(rng.standard_normal((pages, PAGE, heads * 2 * hd)), dtype)
    selected = np.full((num_tokens, topk), -1, np.int32)
    for t in range(num_tokens):
        chosen = rng.choice(pages, rng.integers(1, topk + 1), replace=False)
        selected[t, : len(chosen)] = chosen
    args = (
        jnp.asarray(selected),
        jnp.asarray(selected[:, 0]),
        jnp.asarray(rng.integers(1, PAGE + 1, num_tokens), jnp.int32),
        jnp.arange(pages, dtype=jnp.int32),
    )
    for head in range(heads):
        one = pack(rows[..., head * 2 * hd : (head + 1) * 2 * hd], dtype)
        expected = msa_attention_lse.msa_sparse_attention_lse(
            q, one, *args, sm_scale=hd**-0.5, interpret=True
        )
        got = msa_attention_lse.msa_sparse_attention_lse(
            q, pack(rows, dtype), *args, sm_scale=hd**-0.5, head=head, interpret=True
        )
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


@pytest.mark.parametrize("chunk_tokens", [2048, 32])
def test_lse_heads_kernel_matches_each_head(monkeypatch, chunk_tokens):
    """One `lax.map`-ed kernel over the KV heads (and, with a small
    `MAX_TOKENS_PER_CALL`, over token chunks) equals the per-head kernel."""
    dtype = jnp.float8_e4m3fn
    rng = np.random.default_rng(13)
    num_tokens, pages, topk, heads, hd, q_per_head = 72, 6, 3, 2, 128, 2
    q = jnp.asarray(rng.standard_normal((num_tokens, heads * q_per_head, hd)))
    q = q.astype(jnp.bfloat16)
    rows = pack(
        rounded(rng.standard_normal((pages, PAGE, heads * 2 * hd)), dtype), dtype
    )
    selected = np.full((num_tokens, heads, topk), -1, np.int32)
    for t in range(num_tokens):
        for h in range(heads):
            chosen = rng.choice(pages, rng.integers(0, topk + 1), replace=False)
            selected[t, h, : len(chosen)] = chosen
    rest = (
        jnp.asarray(selected[:, 0, 0]),
        jnp.asarray(rng.integers(1, PAGE + 1, num_tokens), jnp.int32),
        jnp.arange(pages, dtype=jnp.int32),
    )
    expected = [
        msa_attention_lse.msa_sparse_attention_lse(
            q[:, h * q_per_head : (h + 1) * q_per_head],
            rows,
            jnp.asarray(selected[:, h]),
            *rest,
            sm_scale=hd**-0.5,
            head=h,
            interpret=True,
        )
        for h in range(heads)
    ]
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", chunk_tokens)
    jax.clear_caches()  # the chunk size is read at trace time
    got_out, got_lse = msa_attention_lse.msa_sparse_attention_lse_heads(
        q, rows, jnp.asarray(selected), *rest, sm_scale=hd**-0.5, interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(got_out), np.asarray(jnp.concatenate([e[0] for e in expected], 1))
    )
    np.testing.assert_array_equal(
        np.asarray(got_lse), np.asarray(jnp.concatenate([e[1] for e in expected], 1))
    )


def test_gather_group_kv_picks_one_rank_per_kv_head():
    from vllm_torchtpu.models.minimax_m3.ops.msa import gather_group_kv

    dcp, replicas, num_tokens, d = 8, 2, 5, 4

    # Rank i's k/v/index_q hold 10 * (i // replicas) + {1, 2, 3}.
    def full(value):
        return torch.full((num_tokens, 1, d), value)

    per_rank = [
        torch.cat([full(10.0 * (i // replicas) + c) for c in (1, 2, 3)], dim=1)
        for i in range(dcp)
    ]

    class _Group:  # what group.all_gather(dim=1) returns, for every rank
        def all_gather(self, x, dim):
            assert dim == 1
            return torch.cat(per_rank, dim=1)

    k, v, index_q = gather_group_kv(
        full(0.0), full(0.0), full(0.0), _Group(), dcp, replicas
    )
    for tensor, c in ((k, 1), (v, 2), (index_q, 3)):
        assert tensor.shape == (num_tokens, dcp // replicas, d)
        assert [float(x) for x in tensor[0, :, 0]] == [
            10.0 * h + c for h in range(dcp // replicas)
        ]


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_multi_head_dcp_lowers_for_tpu(dtype):
    """Both halves with 4 KV heads per rank (DCP 8, TP 8) lower to Mosaic."""
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    tokens, max_seqs, pages_per_seq, dcp, heads, hd = 256, 16, 256, 8, 4, 128
    packing = PACKING[dtype]
    i32 = jnp.int32
    meta = (
        jax.ShapeDtypeStruct((max_seqs,), i32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), i32),
        jax.ShapeDtypeStruct((max_seqs + 1,), i32),
    )
    kv = jax.ShapeDtypeStruct((4096, PAGE // packing, packing, heads * 2 * hd), dtype)
    bf16 = jnp.bfloat16
    score_args = (
        jax.ShapeDtypeStruct((tokens, heads, hd), bf16),
        jax.ShapeDtypeStruct((tokens, heads, hd), bf16),
        jax.ShapeDtypeStruct((tokens, heads, hd), bf16),
        jax.ShapeDtypeStruct((tokens, hd), bf16),
        kv,
        jax.ShapeDtypeStruct((4096, PAGE // packing, packing, hd), dtype),
        *meta,
        jax.ShapeDtypeStruct((3,), i32),
    )
    attend_args = (
        jax.ShapeDtypeStruct((tokens, heads * 16, hd), bf16),
        kv,
        jax.ShapeDtypeStruct((dcp, tokens, heads * 32), jnp.float32),
        *meta,
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    common = dict(topk=16, rank=3, dcp=dcp)
    with use_abstract_mesh(tpu7x):
        for fn, args in (
            (
                functools.partial(
                    msa_dcp.msa_write_and_candidates_dcp,
                    init_blocks=0,
                    local_blocks=1,
                    **common,
                ),
                score_args,
            ),
            (
                functools.partial(msa_dcp.msa_attend_dcp, sm_scale=hd**-0.5, **common),
                attend_args,
            ),
        ):
            lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
            assert "tpu_custom_call" in lowered.as_text()
