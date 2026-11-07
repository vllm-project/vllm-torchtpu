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
"""MiniMax Sparse Attention under decode context parallelism (DCP).

The ``dcp`` ranks that hold one replicated KV head split its context by
whole sparse blocks instead of each keeping all of it: with
``--cp-kv-cache-interleave-size`` equal to the block size, vLLM's logical
block ``j`` of a request covers ``dcp`` sparse blocks and rank ``r`` keeps
sparse block ``j * dcp + r`` in its page ``block_tables[j]``. So block ``b``
lives on rank ``b % dcp``, whole, and every block-level step stays local:

1. `msa_write_and_candidates_dcp`: each rank writes the K/V and index keys
   of its blocks' tokens, scores its blocks (the block max is over one
   block's keys, all on this rank) and keeps its local top-k candidates
   ``(score, block)`` under the selection's own order (forced blocks first,
   then score, ties to the lower block). The global top-k is a subset of the
   union of the ranks' local top-k, so exchanging ``dcp * topk`` candidates
   per token (an all-gather of ``[T, 2 * topk]`` f32, block ids as values)
   gives the exact non-DCP selection.
2. `msa_attend_dcp`: each rank merges the candidates into that selection,
   attends every query head of the KV head over the selected blocks it
   holds and returns the partial output with its log-sum-exp; the caller
   merges the ranks' partials (`cp_mla_attention.merge_lse_partials_scatter_heads`).

A DCP group wider than a KV head's TP replicas (e.g. DCP 8 at TP 8 with 4
KV heads) spans ``G = dcp / replicas`` KV heads: every rank then stores all
``G`` heads' K/V rows (and the one shared index-key head) for its blocks,
and runs both steps per head; the caller all-gathers the new K/V and index
queries of the group's heads along with the query heads.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_fp4,
    msa_select,
)

_NEG_INF = float("-inf")


def rank_array(rank: int | jax.Array) -> jax.Array:
    """The DCP rank as an ``int32[1]`` array.

    Callers pass it as a traced input (`ops.msa.get_msa_dcp_op` feeds a
    per-rank device buffer), so all ranks lower to one program: a Python
    ``int`` rank is baked into the HLO and gives every rank its own
    executable, compile and cache entry.
    """
    return jnp.asarray(rank, dtype=jnp.int32).reshape(1)


def owner_pages(
    block_tables_2d: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    valid: jax.Array,
    *,
    block_size: int,
    rank: int | jax.Array,
    dcp: int,
) -> jax.Array:
    """This rank's page of each token, dropped for tokens of other ranks."""
    block = pos // block_size
    page = block_tables_2d[seq_id, block // dcp]
    return jnp.where(valid & (block % dcp == rank), page, msa._DROPPED_PAGE)


def local_seq_lens(
    seq_lens: jax.Array, *, block_size: int, rank: int | jax.Array, dcp: int
) -> jax.Array:
    """A length whose page count is the number of blocks this rank holds:
    blocks ``rank, rank + dcp, ...`` below ``ceil(len / block_size)``."""
    blocks = -(-seq_lens // block_size)
    return jnp.maximum(blocks - rank + dcp - 1, 0) // dcp * block_size


def _candidates_kernel(
    rank_ref,
    scores_ref,
    pos_ref,
    score_out_ref,
    block_out_ref,
    *,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    dcp,
):
    """`msa_select._kernel` over this rank's columns (column ``c`` is block
    ``c * dcp + rank``), also returning the picked scores. The rank is read
    from SMEM so that every rank runs the same program."""
    rank = rank_ref[0]
    s = scores_ref[...]
    rows, num_cols = s.shape
    cols = jax.lax.broadcasted_iota(jnp.int32, (rows, num_cols), 1)
    blocks = cols * dcp + rank
    num_visible = pos_ref[...] // block_size + 1
    visible = blocks < num_visible
    s = jnp.where(visible, s, _NEG_INF)
    s = jnp.where(
        visible & (blocks >= num_visible - local_blocks),
        msa_select._FORCED_LOCAL_SCORE,
        s,
    )
    s = jnp.where(visible & (blocks < init_blocks), msa_select._FORCED_INIT_SCORE, s)
    picked_scores, picked_blocks = [], []
    for _ in range(topk):
        best_score = jnp.max(s, axis=1, keepdims=True)
        best = jnp.min(
            jnp.where(s == best_score, cols, num_cols), axis=1, keepdims=True
        )
        picked_scores.append(best_score)
        picked_blocks.append(jnp.where(best_score > _NEG_INF, best * dcp + rank, -1))
        s = jnp.where(cols == best, _NEG_INF, s)
    score_out_ref[...] = jnp.concatenate(picked_scores, axis=1)
    block_out_ref[...] = jnp.concatenate(picked_blocks, axis=1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "topk",
        "init_blocks",
        "local_blocks",
        "dcp",
        "block_rows",
        "interpret",
    ),
)
def local_candidates(
    scores: jax.Array,
    pos: jax.Array,
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: int | jax.Array,
    dcp: int,
    block_rows: int = msa_select.DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """This rank's top-k ``(scores [T, topk] f32, blocks [T, topk] i32)``.

    ``scores [T, 1, P]`` are this rank's block scores (column ``c`` is global
    block ``c * dcp + rank``); forced blocks carry `msa_select`'s forced
    scores and missing picks are ``(-inf, -1)``.
    """
    num_tokens, num_heads, num_cols = scores.shape
    if num_heads != 1:
        raise NotImplementedError(
            f"MSA block selection expects one index head per rank, got {num_heads}."
        )
    pad = -num_tokens % block_rows
    s = jnp.pad(scores[:, 0], ((0, pad), (0, 0)))
    p = jnp.pad(pos, (0, pad))[:, None]
    out_spec = pl.BlockSpec((block_rows, topk), lambda i: (i, 0))
    picked_scores, picked_blocks = pl.pallas_call(
        functools.partial(
            _candidates_kernel,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            dcp=dcp,
        ),
        grid=((num_tokens + pad) // block_rows,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec((block_rows, num_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, 1), lambda i: (i, 0)),
        ],
        out_specs=[out_spec, out_spec],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens + pad, topk), jnp.float32),
            jax.ShapeDtypeStruct((num_tokens + pad, topk), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=msa_select._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_dcp_candidates",
    )(rank_array(rank), s, p)
    return picked_scores[:num_tokens], picked_blocks[:num_tokens]


def pack_candidates(scores: jax.Array, blocks: jax.Array) -> jax.Array:
    """``[T, 2 * topk]`` f32: scores, then the block ids as f32 values, so one
    all-gather moves both.

    The ids travel as values (exact below 2**24; a block id is below
    max_model_len / block_size), not bitcast: a bitcast id is a subnormal
    f32 (and -1 a NaN), which a TPU collective may flush to zero or
    canonicalize, and every id would decode as block 0.
    """
    return jnp.concatenate([scores, blocks.astype(jnp.float32)], axis=-1)


def merge_candidates_legacy(candidates: jax.Array, topk: int) -> jax.Array:
    """The original merge, kept as the reference: ``topk`` rounds of max /
    min / mask over ``[T, dcp * topk]`` rows flattened from ``[dcp, T, 2 *
    topk]``.

    Served (via `merge_head_candidates` on ``[dcp, T, G * 2 * topk]``), XLA
    keeps the rows in their gathered ``[dcp, T * G, 2 * topk]`` shape: the
    16- and 32-wide minor dims fill 16 or 32 of 128 lanes, every round
    rematerializes the mask from all earlier picks, and each of its ~33
    fusions reads ~64 MB at 2048 tokens (j-adc17a5e: 26-31 us each, ~1 ms
    per layer).
    """
    dcp, num_tokens, _ = candidates.shape
    scores = candidates[..., :topk]
    blocks = candidates[..., topk:].astype(jnp.int32)
    s = scores.transpose(1, 0, 2).reshape(num_tokens, dcp * topk)
    b = blocks.transpose(1, 0, 2).reshape(num_tokens, dcp * topk)
    return _merge_rounds(s, b, topk)


def _merge_rounds(s: jax.Array, b: jax.Array, topk: int) -> jax.Array:
    """``topk`` rounds of: the best score, the lowest block holding it, then
    every copy of that block masked out. ``s, b [R, W]`` -> ``[R, topk]``."""
    no_block = jnp.iinfo(jnp.int32).max
    picked = []
    for _ in range(topk):
        best_score = jnp.max(s, axis=1, keepdims=True)
        best = jnp.min(jnp.where(s == best_score, b, no_block), axis=1, keepdims=True)
        picked.append(jnp.where(best_score > _NEG_INF, best, -1))
        s = jnp.where(b == best, _NEG_INF, s)
    return jnp.concatenate(picked, axis=1)


def head_rows(candidates: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """Every rank's candidates of a (KV head, token) side by side, lane
    dense: ``[dcp, T, G * 2 * topk]`` -> ``(scores, blocks) [G * T, dcp *
    topk]``, row ``g * T + t``, column ``rank * topk + i``.

    The rows of `merge_candidates_legacy`, head-major, built by
    concatenating lane slices of the gathered layout: one pass, where a
    reshape to ``[dcp, T * G, 2 * topk]`` first spreads 32 values over 128
    lanes.
    """
    dcp, num_tokens, width = candidates.shape
    num_heads = width // (2 * topk)

    def part(offset):
        return jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        candidates[
                            r, :, h * 2 * topk + offset : h * 2 * topk + offset + topk
                        ]
                        for r in range(dcp)
                    ],
                    axis=1,
                )
                for h in range(num_heads)
            ],
            axis=0,
        )

    return part(0), part(topk).astype(jnp.int32)


def _merge_sort(s: jax.Array, b: jax.Array, topk: int) -> jax.Array:
    """`_merge_rounds` as two row sorts: by (block, score descending), drop
    every copy of a block but its first, then by (score descending, block).
    A NaN score is never picked (as ``-inf``)."""
    # Sort keys ascend; the sort canonicalizes -0 to +0.
    neg = jnp.where(jnp.isnan(s), jnp.inf, -s)
    b, neg = jax.lax.sort((b, neg), dimension=1, num_keys=2)
    dup = jnp.concatenate(
        [jnp.zeros_like(b[:, :1], dtype=bool), b[:, 1:] == b[:, :-1]], axis=1
    )
    neg = jnp.where(dup, jnp.inf, neg)
    neg, b = jax.lax.sort((neg, b), dimension=1, num_keys=2)
    return jnp.where(neg[:, :topk] < jnp.inf, b[:, :topk], -1)


def _merge_kernel(s_ref, b_ref, out_ref, *, topk):
    """`_merge_rounds` on a tile of rows held in VMEM."""
    s = s_ref[...]
    s = jnp.where(jnp.isnan(s), _NEG_INF, s)
    out_ref[...] = _merge_rounds(s, b_ref[...], topk)


def _merge_pallas(
    s: jax.Array,
    b: jax.Array,
    topk: int,
    *,
    block_rows: int = msa_select.DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
) -> jax.Array:
    """`_merge_rounds` as one kernel: every round on a VMEM tile, the rows
    read once. A NaN score is never picked (as ``-inf``)."""
    rows, width = s.shape
    pad = -rows % block_rows
    s = jnp.pad(s, ((0, pad), (0, 0)), constant_values=_NEG_INF)
    b = jnp.pad(b, ((0, pad), (0, 0)), constant_values=-1)
    in_spec = pl.BlockSpec((block_rows, width), lambda i: (i, 0))
    out = pl.pallas_call(
        functools.partial(_merge_kernel, topk=topk),
        grid=((rows + pad) // block_rows,),
        in_specs=[in_spec, in_spec],
        out_specs=pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rows + pad, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge",
    )(s, b)
    return out[:rows]


MERGE_IMPLS = ("legacy", "serial", "sort", "pallas")
# The served merge. "legacy" until a TPU run of
# scripts/benchmark_msa_dcp_merge_candidates.py picks another.
DEFAULT_MERGE_IMPL = "legacy"


def merge_head_candidates(
    candidates: jax.Array,
    topk: int,
    *,
    impl: str = DEFAULT_MERGE_IMPL,
    interpret: bool = False,
) -> jax.Array:
    """Global selection per KV head from every rank's candidates ``[dcp, T,
    G * 2 * topk]`` (each rank's `pack_candidates` of its ``G`` heads,
    head-major) -> ``[T, G, topk]`` (``-1`` padded), in
    `msa_select.select_blocks`'s order: score descending, ties to the lower
    block. A block listed more than once counts once, at its highest score.

    Every `impl` returns the same blocks: "legacy" (the original rounds),
    "serial" (the same rounds on lane-dense `head_rows`), "sort" (two row
    sorts) and "pallas" (the rounds in one kernel). On a NaN score "sort"
    and "pallas" never pick it; the rounds follow the backend's max.
    """
    dcp, num_tokens, width = candidates.shape
    num_heads = width // (2 * topk)
    if impl == "legacy":
        rows = candidates.reshape(dcp, num_tokens * num_heads, 2 * topk)
        selected = merge_candidates_legacy(rows, topk)
        return selected.reshape(num_tokens, num_heads, topk)
    s, b = head_rows(candidates, topk)
    if impl == "serial":
        selected = _merge_rounds(s, b, topk)
    elif impl == "sort":
        selected = _merge_sort(s, b, topk)
    elif impl == "pallas":
        selected = _merge_pallas(s, b, topk, interpret=interpret)
    else:
        raise ValueError(f"Unknown merge impl {impl!r}; expected {MERGE_IMPLS}.")
    return selected.reshape(num_heads, num_tokens, topk).transpose(1, 0, 2)


def merge_candidates(
    candidates: jax.Array,
    topk: int,
    *,
    impl: str = DEFAULT_MERGE_IMPL,
    interpret: bool = False,
) -> jax.Array:
    """`merge_head_candidates` of one KV head: ``[dcp, T, 2 * topk]`` ->
    ``[T, topk]``."""
    return merge_head_candidates(candidates, topk, impl=impl, interpret=interpret)[:, 0]


def selected_page_keys(
    candidates: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    valid: jax.Array,
    *,
    topk: int,
    block_size: int,
    pages_per_seq: int,
    rank: jax.Array,
    dcp: int,
    merge_impl: str = DEFAULT_MERGE_IMPL,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attend_dcp`'s inputs to the attention kernel: ``(local_key [T],
    keys [T, G, topk])``, the page keys (seq * pages_per_seq + logical
    block, as `msa.sparse_attention`) of the token's own block and of the
    merged selection's blocks this rank holds, ``-1`` elsewhere."""
    selected = merge_head_candidates(
        candidates, topk, impl=merge_impl, interpret=interpret
    )
    own_block = pos // block_size
    local_key = jnp.where(
        valid & (own_block % dcp == rank),
        seq_id * pages_per_seq + own_block // dcp,
        -1,
    )
    mine = (selected >= 0) & (selected % dcp == rank) & valid[:, None, None]
    keys = jnp.where(mine, seq_id[:, None, None] * pages_per_seq + selected // dcp, -1)
    return local_key, keys


def msa_write_and_candidates_dcp(
    k: jax.Array,
    v: jax.Array,
    index_q: jax.Array,
    index_k: jax.Array,
    kv_cache: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: int | jax.Array,
    dcp: int,
    scale_cache: jax.Array | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """First half of the DCP forward: write this rank's tokens, score its
    blocks and pick its candidates.

    With 4-bit caches (`msa_fp4`, one KV head per rank) pass `scale_cache`;
    it is returned updated after `index_cache`.

    Shapes as `msa.msa_forward`, except that a rank may hold ``G >= 1`` KV
    heads (DCP wider than a KV head's replicas): ``k``/``v``/``index_q
    [T, G, D]``, cache rows ``[K_0 | V_0 | ... | K_G-1 | V_G-1]``, and the
    single index-key head shared by all. ``block_tables`` are vLLM's logical
    DCP blocks of ``dcp * block_size`` tokens. Returns ``(kv_cache,
    index_cache, candidates [T, G * 2 * topk] f32)``: `pack_candidates` per
    head, head-major.
    """
    num_tokens, num_heads, _ = k.shape
    rank = rank_array(rank)[0]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = owner_pages(
        block_tables_2d,
        seq_id,
        pos,
        valid,
        block_size=block_size,
        rank=rank,
        dcp=dcp,
    )
    write = functools.partial(
        msa.write_rows,
        pages=pages,
        pos=pos,
        max_num_seqs=max_num_seqs,
        interpret=interpret,
    )
    fp4 = scale_cache is not None
    scale_row = msa_fp4.index_scale_row(num_heads) if fp4 else None
    if fp4:
        kv_cache, index_cache, scale_cache = msa.write_fp4(
            kv_cache, index_cache, scale_cache, k, v, index_k, write, pages, pos
        )
        index_q = msa_fp4.rotate_queries(index_q)
    else:
        kv_rows = jnp.concatenate([k, v], axis=-1).reshape(num_tokens, -1)
        kv_cache = write(kv_cache, kv_rows)
        index_cache = write(index_cache, index_k)
    local_lens = local_seq_lens(seq_lens, block_size=block_size, rank=rank, dcp=dcp)
    if num_heads == 1:
        scores = msa._block_scores(
            index_q,
            index_cache,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            interpret,
            scale_cache,
            scale_row,
        )
    else:
        # All heads in one pass over this rank's index pages: [G, T, P],
        # head-major rows for the selection.
        scores = msa.partial_block_scores(
            index_q,
            index_cache,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            shards=1,
            interpret=interpret,
            scale_cache=scale_cache,
            scale_row=scale_row,
        ).reshape(num_heads * num_tokens, 1, -1)
    cand_scores, cand_blocks = local_candidates(
        scores,
        jnp.tile(pos, num_heads),
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=dcp,
        interpret=interpret,
    )
    packed = pack_candidates(cand_scores, cand_blocks)  # [G * T, 2 * topk]
    packed = packed.reshape(num_heads, num_tokens, -1).transpose(1, 0, 2)
    if fp4:
        return kv_cache, index_cache, scale_cache, packed.reshape(num_tokens, -1)
    return kv_cache, index_cache, packed.reshape(num_tokens, -1)


def msa_attend_dcp(
    q: jax.Array,
    kv_cache: jax.Array,
    candidates: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    sm_scale: float,
    topk: int,
    rank: int | jax.Array,
    dcp: int,
    scale_cache: jax.Array | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Second half of the DCP forward: attend over the selected blocks this
    rank holds.

    With a 4-bit `kv_cache` pass its `scale_cache`: queries are rotated here
    and the partial output rotated back, so callers see plain heads.

    ``q [T, G * Hg, D]`` holds every query head of the group's ``G`` KV heads
    (KV head ``h`` owns query heads ``[h * Hg, (h + 1) * Hg)``);
    ``candidates [dcp, T, G * 2 * topk]`` every rank's candidates. Returns
    the partial ``(out [T, G * Hg, D], lse [T, G * Hg] f32)``.
    """
    num_tokens = q.shape[0]
    rank = rank_array(rank)[0]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    pages_per_seq = block_tables_2d.shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    local_key, keys = selected_page_keys(
        candidates,
        seq_id,
        pos,
        valid,
        topk=topk,
        block_size=block_size,
        pages_per_seq=pages_per_seq,
        rank=rank,
        dcp=dcp,
    )  # keys [T, G, topk]
    num_heads = keys.shape[1]
    common = dict(
        sm_scale=sm_scale,
        block_q=msa_attention.DEFAULT_BLOCK_Q,
        pages_per_step=msa_attention.DEFAULT_PAGES_PER_STEP,
        interpret=interpret,
    )
    args = (local_key, pos % block_size + 1, block_tables.reshape(-1))
    if scale_cache is not None:
        q = msa_fp4.rotate_queries(q)
        if num_heads == 1:
            out, lse = msa_attention_lse.msa_sparse_attention_lse(
                q,
                kv_cache,
                keys[:, 0],
                *args,
                head=0,
                **common,
                **msa._fp4_attention_kwargs(scale_cache),
            )
        else:
            # KV head h's K / V scales are scale-page rows 2h / 2h + 1.
            out, lse = msa_attention_lse.msa_sparse_attention_lse_heads(
                q, kv_cache, keys, *args, **common, scale_cache=scale_cache
            )
        return msa_fp4.unrotate_output(out), lse
    if num_heads == 1:
        return msa_attention_lse.msa_sparse_attention_lse(
            q, kv_cache, keys[:, 0], *args, head=0, **common
        )
    # One kernel for all the group's KV heads (the head is a kernel input).
    return msa_attention_lse.msa_sparse_attention_lse_heads(
        q, kv_cache, keys, *args, **common
    )
