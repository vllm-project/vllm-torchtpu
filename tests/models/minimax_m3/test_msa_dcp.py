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
"""MSA under DCP (interpret mode): per-rank writes, the merged block
selection and the log-sum-exp merge of the ranks' partial attention against
the single-rank forward over the same context."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import CACHE_DTYPES, pack, rounded, unpack
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_select,
    msa_write,
)

PAGE = 128
HEAD_DIM = 128
HEADS = 4
MAX_SEQS = 4
PAGES = 24  # physical pages per rank
SCALE = HEAD_DIM**-0.5


def _xla_write_rows(cache, rows, pages, slots, *, max_runs, bits=None, interpret=False):
    """`msa_write.write_rows` as an XLA scatter (covered by test_msa_write)."""
    del max_runs, bits, interpret
    packing = cache.shape[2]
    return cache.at[pages, slots // packing, slots % packing].set(rows, mode="drop")


@pytest.fixture(autouse=True)
def _scatter_writes(monkeypatch):
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)


def _batch(rng, q_lens, ctx_lens, dcp, logical_pages):
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + 3
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    # vLLM's logical DCP blocks; each rank backs them with its own page.
    tables = np.zeros((MAX_SEQS, logical_pages), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, PAGES))[
        : num_seqs * logical_pages
    ].reshape(num_seqs, logical_pages)
    # The single-rank view: sparse block b of a request is rank b % dcp's
    # page tables[b // dcp], numbered rank * PAGES + page in one big cache.
    blocks = np.arange(logical_pages * dcp)
    ref_tables = (blocks % dcp) * PAGES + tables[:, blocks // dcp]
    return num_tokens, seq_lens, cu_q, tables, ref_tables


def _merge_lse(outs, lses):
    """f32 log-sum-exp merge of ``[R, T, H, D]`` partials."""
    lses = np.asarray(lses, np.float64)
    m = lses.max(axis=0)
    m = np.where(np.isfinite(m), m, 0.0)
    w = np.where(np.isfinite(lses), np.exp(lses - m), 0.0)
    w = w / np.maximum(w.sum(axis=0), 1e-30)
    return (np.asarray(outs, np.float64) * w[..., None]).sum(axis=0)


# Decode rows, a long chunk (large tiles) and a short one, topk < blocks; and
# spec-decode rows across block ends with an init block forced. Each batch
# runs at both DCP sizes and both cache dtypes (all 8 pairs pass; 4 kept, as
# interpret mode takes minutes per case).
_DECODE_AND_CHUNK = ([1, 130, 3, 1], [1100, 200, 0, 40], 4, 0)
_SPEC_ROWS = ([4, 4, 4, 4], [1000, 5, 127, 128], 3, 1)


@pytest.mark.parametrize(
    "q_lens,ctx_lens,topk,init_blocks,dcp,dtype",
    [
        (*_DECODE_AND_CHUNK, 2, jnp.bfloat16),
        (*_DECODE_AND_CHUNK, 4, jnp.float8_e4m3fn),
        (*_SPEC_ROWS, 2, jnp.float8_e4m3fn),
        (*_SPEC_ROWS, 4, jnp.bfloat16),
    ],
)
def test_dcp_forward_matches_single_rank(
    q_lens, ctx_lens, topk, init_blocks, dcp, dtype
):
    rng = np.random.default_rng(dcp * 10 + len(q_lens))
    logical_pages = -(-max(np.add(q_lens, ctx_lens)) // (PAGE * dcp))
    num_tokens, seq_lens, cu_q, tables, ref_tables = _batch(
        rng, q_lens, ctx_lens, dcp, logical_pages
    )
    kv_rows = rounded(rng.standard_normal((dcp, PAGES, PAGE, 2 * HEAD_DIM)), dtype)
    index_rows = rounded(rng.standard_normal((dcp, PAGES, PAGE, HEAD_DIM)), dtype)

    def bf16(*shape):
        return jnp.asarray(rng.standard_normal(shape)).astype(jnp.bfloat16)

    q = bf16(num_tokens, HEADS, HEAD_DIM)
    k, v = bf16(num_tokens, 1, HEAD_DIM), bf16(num_tokens, 1, HEAD_DIM)
    index_q, index_k = bf16(num_tokens, 1, HEAD_DIM), bf16(num_tokens, HEAD_DIM)
    distribution = jnp.asarray([0, 0, len(q_lens)], jnp.int32)
    config = dict(topk=topk, init_blocks=init_blocks, local_blocks=1)

    # Single rank over all the context.
    ref_meta = (
        jnp.asarray(seq_lens),
        jnp.asarray(ref_tables.reshape(-1)),
        jnp.asarray(cu_q),
    )
    ref_kv, ref_index, ref_out = msa.msa_forward(
        q,
        k,
        v,
        index_q,
        index_k,
        pack(kv_rows.reshape(dcp * PAGES, PAGE, -1), dtype),
        pack(index_rows.reshape(dcp * PAGES, PAGE, -1), dtype),
        *ref_meta,
        distribution,
        sm_scale=SCALE,
        interpret=True,
        **config,
    )
    _, pos, _ = msa.token_positions(num_tokens, ref_meta[0], ref_meta[2])
    ref_selected = msa_select.select_blocks(
        msa._block_scores(index_q, ref_index, *ref_meta, distribution[2], True),
        pos,
        block_size=PAGE,
        interpret=True,
        **config,
    )[:, 0]

    meta = (jnp.asarray(seq_lens), jnp.asarray(tables.reshape(-1)), jnp.asarray(cu_q))
    caches, candidates = [], []
    for rank in range(dcp):
        kv, index, cand = msa_dcp.msa_write_and_candidates_dcp(
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
        rows = slice(rank * PAGES, (rank + 1) * PAGES)
        np.testing.assert_array_equal(unpack(kv), unpack(ref_kv)[rows])
        np.testing.assert_array_equal(unpack(index), unpack(ref_index)[rows])
        caches.append(kv)
        candidates.append(cand)
    candidates = jnp.stack(candidates)

    num_real = sum(q_lens)
    selected = np.asarray(msa_dcp.merge_candidates(candidates, topk))
    np.testing.assert_array_equal(
        selected[:num_real], np.asarray(ref_selected)[:num_real]
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
        # Head-major partials ([H, T', D], [H, T']), as the kernel writes them.
        num_q = q.shape[0]
        outs.append(np.asarray(out.astype(jnp.float32))[:, :num_q].transpose(1, 0, 2))
        lses.append(np.asarray(lse)[:, :num_q].T)
    merged = _merge_lse(outs, lses)
    np.testing.assert_allclose(
        merged[:num_real],
        np.asarray(ref_out.astype(jnp.float32))[:num_real],
        atol=2e-2,
    )
    # Padding tokens carry no weight anywhere.
    assert np.all(np.isneginf(np.stack(lses)[:, num_real:]))


@pytest.mark.parametrize("dcp,rank", [(2, 0), (2, 1), (4, 3)])
def test_local_seq_lens_count_owned_blocks(dcp, rank):
    seq_lens = np.array([0, 1, PAGE, PAGE + 1, 5 * PAGE - 3, 9 * PAGE], np.int32)
    got = np.asarray(
        msa_dcp.local_seq_lens(
            jnp.asarray(seq_lens), block_size=PAGE, rank=rank, dcp=dcp
        )
    )
    blocks = -(-seq_lens // PAGE)
    owned = [len(range(rank, n, dcp)) for n in blocks]
    np.testing.assert_array_equal(got, np.array(owned) * PAGE)


def test_attention_lse_matches_kernel_v2():
    """The LSE kernel's output equals msa_attention's; its LSE equals a
    dense log-sum-exp, -inf for rows with no page."""
    rng = np.random.default_rng(3)
    num_tokens, pages, topk = 40, 6, 3
    dtype = jnp.bfloat16
    q = jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM))).astype(dtype)
    kv_rows = rounded(rng.standard_normal((pages, PAGE, 2 * HEAD_DIM)), dtype)
    kv = pack(kv_rows, dtype)
    selected = np.full((num_tokens, topk), -1, np.int32)
    local_page = np.full(num_tokens, -1, np.int32)
    local_limit = np.ones(num_tokens, np.int32)
    for t in range(num_tokens - 4):  # the last rows select nothing
        chosen = rng.choice(pages, rng.integers(1, topk + 1), replace=False)
        selected[t, : len(chosen)] = chosen
        local_page[t] = chosen[0] if t % 3 else -1
        local_limit[t] = rng.integers(1, PAGE + 1)
    args = tuple(map(jnp.asarray, (selected, local_page, local_limit)))
    table = jnp.arange(pages, dtype=jnp.int32)
    out, lse = msa_attention_lse.msa_sparse_attention_lse(
        q, kv, *args, table, sm_scale=SCALE, interpret=True
    )
    expected = msa_attention.msa_sparse_attention(
        q, kv, *args, sm_scale=SCALE, page_table=table, interpret=True
    )
    # The LSE kernel folds sm_scale * log2(e) into the bf16 queries and runs
    # its softmax in log2 units (exp2): close to msa_attention, not bitwise.
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
        atol=1e-2,
    )
    qf = np.asarray(q.astype(jnp.float32))
    for t in range(num_tokens):
        keys = [
            kv_rows[p, : (local_limit[t] if p == local_page[t] else PAGE), :HEAD_DIM]
            for p in selected[t]
            if p >= 0
        ]
        if not keys:
            assert np.all(np.isneginf(np.asarray(lse[t])))
            continue
        s = qf[t] @ np.concatenate(keys).T * SCALE
        m = s.max(-1)
        dense = m + np.log(np.exp(s - m[:, None]).sum(-1))
        np.testing.assert_allclose(np.asarray(lse[t]), dense, rtol=1e-5, atol=2e-2)


def test_merge_candidates_breaks_ties_to_lower_block():
    inf = float("-inf")
    scores = jnp.asarray(
        [[[5.0, 1.0, inf]], [[5.0, 3.0, inf]]], jnp.float32
    )  # [dcp=2, T=1, k=3]
    blocks = jnp.asarray([[[4, 0, -1]], [[1, 3, -1]]], jnp.int32)
    packed = jax.vmap(msa_dcp.pack_candidates)(scores, blocks)
    got = np.asarray(msa_dcp.merge_candidates(packed, 3))
    np.testing.assert_array_equal(got, [[1, 4, 3]])


def _flush_like_a_tpu_collective(x: jax.Array) -> jax.Array:
    """Subnormals to zero and NaNs canonical, as a TPU collective may."""
    tiny = jnp.finfo(jnp.float32).tiny
    x = jnp.where(jnp.abs(x) < tiny, jnp.zeros_like(x), x)
    return jnp.where(jnp.isnan(x), jnp.float32(jnp.nan), x)


def test_candidates_survive_a_flushing_collective():
    """Block ids cross the DCP all-gather inside f32 candidates. Bitcast ids
    are subnormals (and -1 a NaN): a collective that flushes them decoded
    every id as block 0, so tokens attended block 0 only (j-301d606e: a few
    GSM8K requests derailed into garbage, MTP acceptance 1.01 on them)."""
    rng = np.random.default_rng(5)
    dcp, num_tokens, topk = 2, 64, 16
    scores = rng.standard_normal((dcp, num_tokens, topk)).astype(np.float32)
    blocks = np.stack(
        [
            rng.permutation(np.arange(r, 4096, dcp))[: num_tokens * topk].reshape(
                num_tokens, topk
            )
            for r in range(dcp)
        ]
    ).astype(np.int32)
    scores[:, :5, 10:] = -np.inf  # missing picks
    blocks[:, :5, 10:] = -1
    packed = jax.vmap(msa_dcp.pack_candidates)(jnp.asarray(scores), jnp.asarray(blocks))
    expected = np.asarray(msa_dcp.merge_candidates(packed, topk))
    got = np.asarray(
        msa_dcp.merge_candidates(_flush_like_a_tpu_collective(packed), topk)
    )
    np.testing.assert_array_equal(got, expected)
    assert len(np.unique(expected[5:])) > topk  # real, distinct block ids


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_dcp_forward_lowers_for_tpu(dtype):
    """Both halves of the DCP forward lower to Mosaic without a TPU."""
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    from tests.models.minimax_m3.packing import PACKING

    tokens, max_seqs, pages_per_seq, dcp, heads = 256, 16, 1024, 2, 16
    packing = PACKING[dtype]
    i32 = jnp.int32
    meta = (
        jax.ShapeDtypeStruct((max_seqs,), i32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), i32),
        jax.ShapeDtypeStruct((max_seqs + 1,), i32),
    )
    kv = jax.ShapeDtypeStruct((4096, PAGE // packing, packing, 2 * HEAD_DIM), dtype)
    score_args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, HEAD_DIM), jnp.bfloat16),
        kv,
        jax.ShapeDtypeStruct((4096, PAGE // packing, packing, HEAD_DIM), dtype),
        *meta,
        jax.ShapeDtypeStruct((3,), i32),
    )
    attend_args = (
        jax.ShapeDtypeStruct((tokens, heads, HEAD_DIM), jnp.bfloat16),
        kv,
        jax.ShapeDtypeStruct((dcp, tokens, 32), jnp.float32),
        *meta,
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        for fn, args in (
            (
                functools.partial(
                    msa_dcp.msa_write_and_candidates_dcp,
                    topk=16,
                    init_blocks=0,
                    local_blocks=1,
                    rank=1,
                    dcp=dcp,
                ),
                score_args,
            ),
            (
                functools.partial(
                    msa_dcp.msa_attend_dcp,
                    sm_scale=SCALE,
                    topk=16,
                    rank=1,
                    dcp=dcp,
                ),
                attend_args,
            ),
        ):
            lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
            assert "tpu_custom_call" in lowered.as_text()


def test_rank_as_input_matches_int_rank():
    """`local_candidates` reads the rank from SMEM; an ``int32[1]`` rank
    gives the same picks as the Python int it holds."""
    rng = np.random.default_rng(3)
    tokens, cols, dcp, topk = 16, 12, 4, 5
    scores = jnp.asarray(rng.standard_normal((tokens, 1, cols)), jnp.float32)
    pos = jnp.asarray(rng.integers(0, cols * dcp * PAGE, tokens), jnp.int32)
    common = dict(block_size=PAGE, topk=topk, init_blocks=1, local_blocks=1, dcp=dcp)
    for rank in range(dcp):
        want = msa_dcp.local_candidates(
            scores, pos, rank=rank, interpret=True, **common
        )
        got = msa_dcp.local_candidates(
            scores,
            pos,
            rank=jnp.asarray([rank], jnp.int32),
            interpret=True,
            **common,
        )
        for w, g in zip(want, got):
            np.testing.assert_array_equal(np.asarray(w), np.asarray(g))


def test_dcp_forward_lowers_once_for_every_rank():
    """With the rank as an input both halves lower to one TPU program for
    all ranks (a Python int rank would give each rank its own)."""
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    tokens, max_seqs, pages_per_seq, dcp, heads = 256, 16, 1024, 4, 16
    dtype = jnp.float8_e4m3fn
    from tests.models.minimax_m3.packing import PACKING

    packing = PACKING[dtype]
    i32 = jnp.int32
    meta = (
        jax.ShapeDtypeStruct((max_seqs,), i32),
        jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), i32),
        jax.ShapeDtypeStruct((max_seqs + 1,), i32),
    )
    kv = jax.ShapeDtypeStruct((4096, PAGE // packing, packing, 2 * HEAD_DIM), dtype)
    rank = jax.ShapeDtypeStruct((1,), i32)
    score_args = (
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 1, HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, HEAD_DIM), jnp.bfloat16),
        kv,
        jax.ShapeDtypeStruct((4096, PAGE // packing, packing, HEAD_DIM), dtype),
        *meta,
        jax.ShapeDtypeStruct((3,), i32),
        rank,
    )
    attend_args = (
        jax.ShapeDtypeStruct((tokens, heads, HEAD_DIM), jnp.bfloat16),
        kv,
        jax.ShapeDtypeStruct((dcp, tokens, 32), jnp.float32),
        *meta,
        rank,
    )

    def score(*args):
        *rest, r = args
        return msa_dcp.msa_write_and_candidates_dcp(
            *rest, topk=16, init_blocks=0, local_blocks=1, rank=r, dcp=dcp
        )

    def attend(*args):
        *rest, r = args
        return msa_dcp.msa_attend_dcp(*rest, sm_scale=SCALE, topk=16, rank=r, dcp=dcp)

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        for fn, args in ((score, score_args), (attend, attend_args)):
            lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
            assert "tpu_custom_call" in lowered.as_text()


def test_dcp_op_name_has_no_rank():
    """The torch op's name (its compile-cache identity) is rank-free."""
    import inspect

    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    params = inspect.signature(msa_ops.get_msa_dcp_op).parameters
    assert "rank" not in params
    src = inspect.getsource(msa_ops.get_msa_dcp_op)
    assert "_r{rank}" not in src and "dcp_rank" in src
