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
import inspect

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
    selected = np.asarray(msa_dcp.merge_head_candidates(candidates, topk))[:, 0]
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


def _sorted_candidates(rng, dcp, num_cols, topk, num_blocks=4096, ties=False):
    """Every rank's local top-k as its kernels emit them: distinct blocks of
    its own (``b % dcp == rank``), score descending, ties to the lower block,
    then ``(-inf, -1)`` for missing picks. ``[dcp, N, topk]`` each."""
    scores = np.full((dcp, num_cols, topk), -np.inf, np.float32)
    blocks = np.full((dcp, num_cols, topk), -1, np.int32)
    for r in range(dcp):
        for c in range(num_cols):
            n = topk if c % 5 else int(rng.integers(0, topk + 1))
            b = rng.permutation(np.arange(r, num_blocks, dcp))[:n]
            s = (
                rng.integers(-3, 4, n).astype(np.float32)
                if ties
                else rng.standard_normal(n).astype(np.float32)
            )
            if c % 7 == 0 and n:
                s[0] = 3.0e38  # a forced score
            order = np.lexsort((b, -s))
            scores[r, c, :n], blocks[r, c, :n] = s[order], b[order]
    return scores, blocks


def _slots(scores, blocks):
    """`msa_dcp.to_slots` of each rank's one-head candidates."""
    packed = jax.vmap(msa_dcp.pack_candidates)(jnp.asarray(scores), jnp.asarray(blocks))
    return jnp.stack([msa_dcp.to_slots(p, scores.shape[-1]) for p in packed])


def test_candidates_cross_as_int32():
    """Block ids cross the DCP all-gather with the scores. As f32 bit
    patterns they were subnormals (and -1 a NaN), which a TPU collective
    flushed (j-301d606e: every id decoded as block 0, a few GSM8K requests
    derailed). `to_slots` sends int32: scores as bit patterns, ids as ints,
    which no collective rewrites, and every score bit pattern survives."""
    rng = np.random.default_rng(5)
    dcp, num_tokens, topk = 2, 64, 16
    scores, blocks = _sorted_candidates(rng, dcp, num_tokens, topk)
    scores[0, 3, 2] = np.float32(1e-41)  # a subnormal score
    slots = np.asarray(_slots(scores, blocks))
    assert slots.dtype == np.int32 and slots.shape == (dcp, 2 * topk, num_tokens)
    np.testing.assert_array_equal(
        slots[:, :topk].view(np.float32).view(np.uint32),
        scores.transpose(0, 2, 1).view(np.uint32),
    )
    np.testing.assert_array_equal(slots[:, topk:], blocks.transpose(0, 2, 1))
    # The f32 rows the slots replace do not survive a flushing collective
    # when the ids are bit patterns: the reason ids never travel as bits.
    bits = jax.lax.bitcast_convert_type(jnp.asarray(blocks), jnp.float32)
    flushed = jax.lax.bitcast_convert_type(_flush_like_a_tpu_collective(bits), jnp.int32)
    assert not np.array_equal(np.asarray(flushed), blocks)


def _np_merge(scores, blocks, topk):
    """Reference merge of ``[dcp, N, topk]`` candidates: the real ones
    (score > -inf) by score descending, then block ascending; -1 padded."""
    dcp, num_cols, _ = scores.shape
    out = np.full((num_cols, topk), -1, np.int32)
    for c in range(num_cols):
        s, b = scores[:, c].reshape(-1), blocks[:, c].reshape(-1)
        real = s > -np.inf
        order = np.lexsort((b[real], -s[real]))[:topk]
        out[c, : len(order)] = b[real][order]
    return out


@pytest.mark.parametrize("dcp,heads", [(2, 1), (4, 2), (8, 4), (8, 1)])
@pytest.mark.parametrize("ties", [False, True], ids=["distinct", "ties"])
def test_slot_merge_matches_row_merge(dcp, heads, ties):
    """`merge_head_candidates` of `to_slots` candidates (the dense ranking
    at DCP 2, the kernel above) equals the r5 merge (`merge_candidates` of
    `pack_candidates` rows) and a numpy sort, per head, ties, forced scores
    and missing picks included."""
    rng = np.random.default_rng(dcp * 10 + heads)
    num_tokens, topk = 136, 16  # 136 * G columns: a padded last block
    scores, blocks = _sorted_candidates(rng, dcp, num_tokens * heads, topk, ties=ties)
    want = _np_merge(scores, blocks, topk)
    rows = jax.vmap(msa_dcp.pack_candidates)(jnp.asarray(scores), jnp.asarray(blocks))
    np.testing.assert_array_equal(np.asarray(msa_dcp.merge_candidates(rows, topk)), want)
    got = np.asarray(
        msa_dcp.merge_head_candidates(_slots(scores, blocks), topk, heads, interpret=True)
    )
    np.testing.assert_array_equal(got, want.reshape(num_tokens, heads, topk))
    assert (want >= 0).mean() > 0.5


def _faulty(fn, old, new):
    src = inspect.getsource(fn)
    assert old in src
    namespace = dict(vars(msa_dcp))
    exec(compile(src.replace(old, new), "<fault>", "exec"), namespace)
    return namespace[fn.__name__]


@pytest.mark.parametrize("network", [True, False], ids=["network", "extract"])
def test_slot_merge_paths_match(monkeypatch, network):
    """The network merge (per-rank sorted lists) and the max-extraction
    kernel and dense ranking it replaces all give the r5 merge."""
    monkeypatch.setattr(msa_dcp, "_network_merge", lambda topk: network)
    for dcp, heads in ((2, 1), (4, 2), (8, 4)):
        test_slot_merge_matches_row_merge(dcp, heads, True)


@pytest.mark.parametrize(
    "dcp,name,old,new",
    [
        (2, "_beats", "(ba < bb)", "(ba > bb)"),
        (4, "_beats", "(ba < bb)", "(ba > bb)"),
        (8, "_merge_sorted", "for d in (4, 2, 1):", "for d in (4, 2):"),
        (4, "_merge_kernel_t", "best = jnp.min(", "best = jnp.max("),
        (2, "_rank_merge", "(b_j < b_i)", "(b_j > b_i)"),
    ],
    ids=["network-ties-2", "network-ties-4", "network-reverse-8", "extract", "dense"],
)
def test_slot_merge_fault_is_caught(monkeypatch, dcp, name, old, new):
    """Positive control: a merge that breaks ties to the higher block (or a
    network that does not reverse its second list) fails
    `test_slot_merge_matches_row_merge`."""
    if name in ("_merge_kernel_t", "_rank_merge"):
        monkeypatch.setattr(msa_dcp, "_network_merge", lambda topk: False)
    monkeypatch.setattr(msa_dcp, name, _faulty(getattr(msa_dcp, name), old, new))
    with pytest.raises(AssertionError):
        test_slot_merge_matches_row_merge(dcp, 1, True)


def test_candidate_kernels_emit_sorted_lists():
    """The network merge needs every rank's list sorted (score descending,
    ties to the lower block, misses last): `local_candidates`' picks are."""
    rng = np.random.default_rng(11)
    tokens, cols, dcp, topk = 256, 40, 4, 16
    scores = jnp.asarray(rng.integers(-3, 4, (tokens, 1, cols)), jnp.float32)
    pos = jnp.asarray(rng.integers(0, cols * dcp * PAGE, tokens), jnp.int32)
    s, b = msa_dcp.local_candidates(
        scores, pos, block_size=PAGE, topk=topk, init_blocks=1, local_blocks=1,
        rank=1, dcp=dcp, interpret=True,
    )
    s, b = np.asarray(s), np.asarray(b)
    ahead = (s[:, :-1] > s[:, 1:]) | ((s[:, :-1] == s[:, 1:]) & (b[:, :-1] < b[:, 1:]))
    miss = (s[:, :-1] == -np.inf) & (s[:, 1:] == -np.inf) & (b[:, 1:] == -1)
    assert np.all(ahead | miss)
    assert (b == -1).any() and (b >= 0).any()


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
        jax.ShapeDtypeStruct((dcp, 32, tokens), jnp.int32),
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
        jax.ShapeDtypeStruct((dcp, 32, tokens), jnp.int32),
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
