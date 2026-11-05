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
    msa_select,
)

_NEG_INF = float("-inf")


def owner_pages(
    block_tables_2d: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    valid: jax.Array,
    *,
    block_size: int,
    rank: int,
    dcp: int,
) -> jax.Array:
    """This rank's page of each token, dropped for tokens of other ranks."""
    block = pos // block_size
    page = block_tables_2d[seq_id, block // dcp]
    return jnp.where(valid & (block % dcp == rank), page, msa._DROPPED_PAGE)


def local_seq_lens(
    seq_lens: jax.Array, *, block_size: int, rank: int, dcp: int
) -> jax.Array:
    """A length whose page count is the number of blocks this rank holds:
    blocks ``rank, rank + dcp, ...`` below ``ceil(len / block_size)``."""
    blocks = -(-seq_lens // block_size)
    return jnp.maximum(blocks - rank + dcp - 1, 0) // dcp * block_size


def _candidates_kernel(
    scores_ref,
    pos_ref,
    score_out_ref,
    block_out_ref,
    *,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    rank,
    dcp,
):
    """`msa_select._kernel` over this rank's columns (column ``c`` is block
    ``c * dcp + rank``), also returning the picked scores."""
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
        "rank",
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
    rank: int,
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
            rank=rank,
            dcp=dcp,
        ),
        grid=((num_tokens + pad) // block_rows,),
        in_specs=[
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
    )(s, p)
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


def merge_candidates(candidates: jax.Array, topk: int) -> jax.Array:
    """Global selection ``[T, topk]`` (``-1`` padded) from every rank's
    packed candidates ``[dcp, T, 2 * topk]``, in `msa_select.select_blocks`'s
    order: score descending, ties to the lower block."""
    dcp, num_tokens, _ = candidates.shape
    scores = candidates[..., :topk]
    blocks = candidates[..., topk:].astype(jnp.int32)
    s = scores.transpose(1, 0, 2).reshape(num_tokens, dcp * topk)
    b = blocks.transpose(1, 0, 2).reshape(num_tokens, dcp * topk)
    no_block = jnp.iinfo(jnp.int32).max
    picked = []
    for _ in range(topk):
        best_score = jnp.max(s, axis=1, keepdims=True)
        best = jnp.min(jnp.where(s == best_score, b, no_block), axis=1, keepdims=True)
        picked.append(jnp.where(best_score > _NEG_INF, best, -1))
        s = jnp.where(b == best, _NEG_INF, s)
    return jnp.concatenate(picked, axis=1)


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
    rank: int,
    dcp: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """First half of the DCP forward: write this rank's tokens, score its
    blocks and pick its candidates.

    Shapes as `msa.msa_forward`; ``block_tables`` are vLLM's logical DCP
    blocks of ``dcp * block_size`` tokens. Returns ``(kv_cache, index_cache,
    candidates [T, 2 * topk] f32)`` (`pack_candidates`).
    """
    num_tokens = k.shape[0]
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
    kv_cache = write(kv_cache, jnp.concatenate([k, v], axis=-1)[:, 0])
    index_cache = write(index_cache, index_k)
    scores = msa._block_scores(
        index_q,
        index_cache,
        local_seq_lens(seq_lens, block_size=block_size, rank=rank, dcp=dcp),
        block_tables,
        cu_q_lens,
        distribution[2],
        interpret,
    )
    cand_scores, cand_blocks = local_candidates(
        scores,
        pos,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=dcp,
        interpret=interpret,
    )
    return kv_cache, index_cache, pack_candidates(cand_scores, cand_blocks)


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
    rank: int,
    dcp: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Second half of the DCP forward: attend over the selected blocks this
    rank holds.

    ``q [T, H, D]`` holds every query head of the group's KV head;
    ``candidates [dcp, T, 2 * topk]`` every rank's `pack_candidates`.
    Returns the partial ``(out [T, H, D], lse [T, H] f32)``.
    """
    num_tokens = q.shape[0]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    pages_per_seq = block_tables_2d.shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    selected = merge_candidates(candidates, topk)
    # Page keys (seq * pages_per_seq + logical block), as `msa.sparse_attention`.
    mine = (selected >= 0) & (selected % dcp == rank) & valid[:, None]
    keys = jnp.where(mine, seq_id[:, None] * pages_per_seq + selected // dcp, -1)
    own_block = pos // block_size
    local_key = jnp.where(
        valid & (own_block % dcp == rank),
        seq_id * pages_per_seq + own_block // dcp,
        -1,
    )
    return msa_attention_lse.msa_sparse_attention_lse(
        q,
        kv_cache,
        keys,
        local_key,
        pos % block_size + 1,
        block_tables.reshape(-1),
        sm_scale=sm_scale,
        block_q=msa_attention.DEFAULT_BLOCK_Q,
        pages_per_step=msa_attention.DEFAULT_PAGES_PER_STEP,
        interpret=interpret,
    )
