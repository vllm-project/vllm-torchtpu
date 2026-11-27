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
"""The attention half of the MSA DCP forward for a DCP group that spans
several KV heads (`msa_dcp`, ``G > 1``), as one SPMD program over the DCP
mesh axis: the per-rank body of a ``shard_map``.

The DCP attention gathers every query head of the group to every rank and
reduce-scatters the ranks' partial outputs back: ``2 * T * G * Hg * D``
bf16 per rank and layer, whatever the context. A prefill chunk's rows are
many and its context comparatively short (c64, DCP 8: 8K rows over a
149K-token context move 235 MB of queries and partials per rank, the
context's pages of one KV head 19 MB). So a long step attends prefill
requests head-parallel: each rank brings its own KV head's pages of those
requests from the other ranks (an all-to-all of whole pages) and attends
its own query heads over them, which needs no query gather and no merge.
Decode-sized requests (at most ``decode_rows`` query rows) keep the DCP
attention on their rows only, compacted to a static ``max_num_seqs *
decode_rows`` rows.

Both paths attend the same selection (`msa_dcp.merge_head_candidates` of
the ranks' candidates) over the same 4-bit pages; the head-parallel one
normalizes in one pass instead of merging bf16 partials.
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
    msa_attention_pairs,
    msa_dcp,
    msa_fp4,
)

# Steps of at least this many tokens attend their prefill requests
# head-parallel. Each round moves a static page buffer (~35 MB per rank at
# max_model_len 256K), which a step's query gather and merge exceed from
# about 2K tokens (~0.16 us per token at c64 DCP 8, j-0a66fa2e).
HEAD_PARALLEL_MIN_TOKENS = 2048
# Tiles and pages per kernel step of the head-parallel calls (every rank's
# pages): 64-token tiles as `msa_attention.PREFILL_BLOCK_Q_4BIT` (standalone
# 16 heads x 4096 tokens: 0.659 -> 0.385 ms at .999-local selections, 0.684 ->
# 0.587 at .995, r12 j-9c5505d9), wide groups to amortize a step's fixed cost.
_BLOCK_Q = 64
_PAGES_PER_STEP = 64
# A KV head's two replicas split a long step's rows instead of its query
# heads: each attends half the tokens with all 16 heads, so a tile's page
# union and its 4-bit unpacking serve twice the rows (kernel bench
# j-b3925dba, c64 shape: 8 heads x 8192 tokens 1.961 ms -> 16 heads x 4096
# tokens 1.248 ms at 32-page groups), for two permutes of half a step's
# queries and outputs between the replicas (one chip's cores at c64).
_SPLIT_PAGES_PER_STEP = 32
# The replicas take alternate blocks of this many tokens rather than the two
# halves: a step's halves differ in work (the decode rows a mixed step puts
# first attend nothing here; later chunk tokens see more pages), and the
# replica with less waits at the next exchange (c64 DCP 4 twin, per-core
# kernel time up to 2.2x, j-867fc577). A block is a whole 64-token tile.
_INTERLEAVE_TOKENS = 64
# Head-parallel rows attend page-major (`msa_attention_pairs`) instead of in
# tiles: "auto" when a step's mean page union per tile is over
# `_PAGE_MAJOR_MIN_RATIO` times its tokens' mean pages (a tile's cost grows
# with its union, page-major's with the pairs), "on" always, "off" never
# (`TPU_MINIMAX_M3_MSA_PAGE_MAJOR`). Op bench of a 4,096-row x 16-head call
# over a 130K prefix, each token's 15 other pages drawn from the last N, us
# tiles / page-major (j-42053fce): ratio 1.1: 321 / 535, 2.1: 440 / 535,
# 3.1: 598 / 538, 4.1: 721 / 537, 6.0: 998 / 546, 38.5: 5495 / 579. Tiles
# grow with the union, page-major stays near its pair count; the crossover
# is ~2.7 at 4,096 rows, ~3.1 at 2,048 and ~3.6 at 1,024 (j-9bc53a54).
PAGE_MAJOR = "auto"
# The ratio is estimated over a 1,024-bit bitmap per tile, and the head's
# exchanged page keys span dcp * num_pages: far pages alias and the estimate
# comes out low (uniform selections over a 253K context: 39 for a true 48);
# near the threshold (unions of a few dozen pages) it barely aliases.
_PAGE_MAJOR_MIN_RATIO = 3.0
# Decode steps of a DCP group of 4 over 2 KV heads attend each KV head in its
# own call so the cross-chip merge overlaps the second (`_pair_first_merge_by_head`).
PAIR_FIRST_HEAD_CALLS = True


def _interleave(x: jax.Array, axis: int = 0) -> jax.Array:
    """``x``'s `_INTERLEAVE_TOKENS`-token blocks along `axis`, even blocks
    first, then odd: each replica's half (`_replica_rows`) holds alternate
    blocks."""
    shape = x.shape
    n = shape[axis] // (2 * _INTERLEAVE_TOKENS)
    x = x.reshape(shape[:axis] + (n, 2, _INTERLEAVE_TOKENS) + shape[axis + 1 :])
    return jnp.swapaxes(x, axis, axis + 1).reshape(shape)


def merge_partials(
    out: jax.Array,
    lse: jax.Array,
    axis: str,
    *,
    rank: jax.Array,
    dcp: int,
    rs_out_precision: bool,
    interpret: bool = False,
) -> jax.Array:
    """`ops.msa.merge_head_partials` inside the SPMD program: head-major
    partials ``out [H, T, D]``, ``lse [H, T]`` -> this rank's heads,
    unrotated, ``[T, H / dcp, D]``."""
    if dcp == 4:
        # Pays when ranks r and r ^ 1 are one chip's cores (the DCP mesh's
        # contiguous ranks); correct for any device order.
        merged = _pair_first_merge(
            out, lse.astype(jnp.float32), axis, rank, rs_out_precision
        )
    else:
        lse = lse.astype(jnp.float32).T  # token-major across the gather
        lses = jax.lax.all_gather(lse, axis, axis=0)
        weighted = weigh_partials(
            out,
            lse,
            lses,
            out.dtype if rs_out_precision else jnp.float32,
            interpret=interpret,
        )
        merged = jax.lax.psum_scatter(weighted, axis, scatter_dimension=0, tiled=True)
    return msa_fp4.unrotate_output(merged.astype(out.dtype)).transpose(1, 0, 2)


def _combine(o1, l1, o2, l2):
    """Two partials of the same heads merged by their log-sum-exps: ``o
    [..., T, D]`` (normalized), ``l [..., T]`` f32 -> (f32 ``o``, ``l``). A
    zero-weight partial (no key attended, -inf LSE) drops out."""
    peak = jnp.maximum(l1, l2)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    w1, w2 = jnp.exp(l1 - peak), jnp.exp(l2 - peak)
    total = w1 + w2
    o = jnp.where(w1[..., None] > 0, o1.astype(jnp.float32) * w1[..., None], 0.0)
    o = o + jnp.where(w2[..., None] > 0, o2.astype(jnp.float32) * w2[..., None], 0.0)
    o = jnp.where(total[..., None] > 0, o / total[..., None], 0.0)
    return o, peak + jnp.log(total)


def _pair_first_merge(
    out: jax.Array, lse: jax.Array, axis: str, rank: jax.Array, rs_out_precision: bool
):
    """The LSE merge of a DCP group of 4 (rank ``r`` keeps head block ``r``)
    from head-major partials ``out [4 Hl, T, D]`` and ``lse [4 Hl, T]`` f32,
    in two exchanges that carry each block's LSEs with it: ranks ``r`` and
    ``r ^ 1`` (one chip's cores) swap the blocks of the other parity and
    combine them, then ranks ``r`` and ``r ^ 2`` swap the block the other
    keeps and combine. No LSE all-gather and no weighting pass, and one block
    per rank crosses between the chips, where an LSE all-gather, a weighting
    pass and a reduce-scatter sent three (c64 decode twin j-cd5ebe6f, per
    layer: LSE all-gather 8-15 us, weighting 4 us). Combines in f32; the
    pair's combined block crosses the chips in ``out.dtype``, or in f32
    without `TPU_DCP_RS_OUT_PRECISION`."""
    heads = out.shape[0] // 4
    blocks = out.reshape(4, heads, *out.shape[1:])
    lses = lse.reshape(4, heads, lse.shape[-1])
    parity, half = rank % 2 == 0, rank // 2 == 0
    pair = [(r, r ^ 1) for r in range(4)]
    o, lg = (
        jnp.where(parity, x[0::2], x[1::2]) for x in (blocks, lses)
    )  # this parity's blocks {p, p + 2}
    got_o, got_l = (
        jax.lax.ppermute(jnp.where(parity, x[1::2], x[0::2]), axis, pair)
        for x in (blocks, lses)
    )
    o, lg = _combine(o, lg, got_o, got_l)
    cross = [(r, r ^ 2) for r in range(4)]
    keep_o, keep_l = jnp.where(half, o[0], o[1]), jnp.where(half, lg[0], lg[1])
    other_o = jax.lax.ppermute(
        jnp.where(half, o[1], o[0]).astype(
            out.dtype if rs_out_precision else jnp.float32
        ),
        axis,
        cross,
    )
    other_l = jax.lax.ppermute(jnp.where(half, lg[1], lg[0]), axis, cross)
    return _combine(keep_o, keep_l, other_o, other_l)[0]


def _pair_first_merge_by_head(
    attend, axis: str, rank: jax.Array, rs_out_precision: bool, interpret: bool = False
) -> jax.Array:
    """`_pair_first_merge` of a DCP group of 4 over 2 KV heads (KV head ``h``
    owns head blocks ``2h`` and ``2h + 1``; ranks ``r`` and ``r ^ 1`` one
    chip's cores, both keeping blocks of KV head ``r // 2``), from
    ``attend(order)``, one kernel call per KV head in ``order``
    (`msa_attention_lse.msa_sparse_attention_lse_head_calls`). The other
    chip's KV head is attended first and its pair-combined block crosses
    the chips while this chip's own KV head is attended: the cross-chip
    permute (one link carries both cores' blocks) waited behind the whole
    attention (c64 decode twin j-6bf91132: ~18 us per layer at b47). Same
    exchanges and bytes as `_pair_first_merge`. Returns this rank's heads
    unrotated, token-major ``[T, Hl * D]``."""
    own, parity = rank // 2, rank % 2
    far = 1 - own
    far_call, own_call = attend(jnp.stack([far, own]))
    pair = [(r, r ^ 1) for r in range(4)]

    def pick(blocks, index):
        # Block ``index`` as ``[1, Hl, T, D]``, by selects of static blocks: a
        # 3D consumer (a dynamic slice, or selects of 3D blocks) gave the
        # kernel's output the 16-row bf16 tiling, where its 8-row tile stores
        # do not lower (Mosaic E2003). Selects, not arithmetic: the other KV
        # head's blocks of a call's output are never written.
        picked = blocks[3:]
        for b in (2, 1, 0):
            picked = jnp.where(index == b, blocks[b : b + 1], picked)
        return picked

    def pair_combine(out, lse, head):
        """Head ``head``'s block of this rank's parity combined over the chip."""
        out = out.reshape(4, -1, *out.shape[1:])
        lse = lse.astype(jnp.float32).reshape(4, -1, lse.shape[-1])
        keep, send = 2 * head + parity, 2 * head + 1 - parity
        got_o, got_l = (jax.lax.ppermute(pick(x, send), axis, pair) for x in (out, lse))
        return _combine(pick(out, keep), pick(lse, keep), got_o, got_l)

    out_far, lse_far = far_call()
    o_far, l_far = pair_combine(out_far, lse_far, far)
    o_far = o_far.astype(out_far.dtype if rs_out_precision else jnp.float32)
    cross = [(r, r ^ 2) for r in range(4)]
    other_o = jax.lax.ppermute(o_far, axis, cross)
    other_l = jax.lax.ppermute(l_far, axis, cross)
    out_own, lse_own = own_call(after=(o_far, l_far))
    if out_own.dtype != jnp.bfloat16 or other_o.dtype != jnp.bfloat16:
        o_own, l_own = pair_combine(out_own, lse_own, own)
        merged = _combine(o_own, l_own, other_o, other_l)[0][0]
        merged = msa_fp4.unrotate_output(merged.astype(out_own.dtype))
        return merged.transpose(1, 0, 2).reshape(merged.shape[1], -1)
    # This chip's head: the pair's combine, the cross-chip combine and the
    # unrotation in one kernel (as XLA ops, the f32 combines of [Hl, T, D]
    # and the unrotate cost more than the merge's collectives, twin
    # j-8a3254c0). The weights are XLA's: [Hl, T] f32 ops.
    lse = lse_own.astype(jnp.float32).reshape(4, -1, lse_own.shape[-1])
    keep, send = 2 * own + parity, 2 * own + 1 - parity
    blocks = out_own.reshape(4, -1, *out_own.shape[1:])
    got_o, got_l = (jax.lax.ppermute(pick(x, send), axis, pair) for x in (blocks, lse))
    return _merge_unrotate(
        out_own,
        keep,
        got_o[0],
        other_o[0],
        pick(lse, keep)[0],
        got_l[0],
        other_l[0],
        interpret=interpret,
    )


def _weights(l1, l2):
    """`_combine`'s weights of two partials' LSEs: ``(w1, w2, total, lse)``."""
    peak = jnp.maximum(l1, l2)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    w1, w2 = jnp.exp(l1 - peak), jnp.exp(l2 - peak)
    total = w1 + w2
    return w1, w2, total, peak + jnp.log(total)


def _merge_unrotate(out, keep, o2, o3, l1, l2, l3, *, interpret=False):
    """``_combine(_combine(out[keep block], l1, o2, l2), o3, l3)`` (bf16
    partials ``[Hl, T, D]``, f32 LSEs ``[Hl, T]``), cast to bf16 and
    unrotated (`msa_fp4.unrotate_output`), written token-major ``[T, Hl *
    D]``: op for op the XLA merge, in one kernel that reads block `keep`
    (traced) of the head-major ``out [4 Hl, T, D]`` in place."""
    heads, num_tokens, head_dim = o2.shape
    w1, w2, t1, l12 = _weights(l1, l2)
    v1, v2, t2, _ = _weights(l12, l3)
    weights = jnp.stack([w1, w2, t1, v1, v2, t2]).transpose(0, 2, 1)  # [6, T, Hl]

    def kernel(keep_ref, o1_ref, o2_ref, o3_ref, w_ref, h_ref, out_ref):
        del keep_ref
        for h in range(heads):
            w = [w_ref[k, :, h : h + 1] for k in range(6)]
            o = jnp.where(w[0] > 0, o1_ref[h].astype(jnp.float32) * w[0], 0.0)
            o = o + jnp.where(w[1] > 0, o2_ref[h].astype(jnp.float32) * w[1], 0.0)
            o = jnp.where(w[2] > 0, o / w[2], 0.0)
            o = jnp.where(w[3] > 0, o * w[3], 0.0)
            o = o + jnp.where(w[4] > 0, o3_ref[h].astype(jnp.float32) * w[4], 0.0)
            o = jnp.where(w[5] > 0, o / w[5], 0.0)
            y = jnp.dot(
                o.astype(jnp.bfloat16), h_ref[...], preferred_element_type=jnp.float32
            )
            out_ref[:, h * head_dim : (h + 1) * head_dim] = (
                y * (1.0 / head_dim)
            ).astype(out_ref.dtype)

    full = lambda shape: pl.BlockSpec(shape, lambda i, k: (0,) * len(shape))  # noqa: E731
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[
                pl.BlockSpec((heads, num_tokens, head_dim), lambda i, k: (k[0], 0, 0)),
                full(o2.shape),
                full(o3.shape),
                full(weights.shape),
                full((head_dim, head_dim)),
            ],
            out_specs=full((num_tokens, heads * head_dim)),
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, heads * head_dim), jnp.bfloat16),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge_unrotate",
    )(
        jnp.reshape(keep, (1,)).astype(jnp.int32),
        out,
        o2,
        o3,
        weights,
        jnp.asarray(msa_fp4.hadamard(head_dim).T, jnp.bfloat16),
    )


def merge_partials_all_to_all(
    out: jax.Array,
    lse: jax.Array,
    axis: str,
    *,
    dcp: int,
    interpret: bool = False,
) -> jax.Array:
    """`merge_partials` by one all-to-all: each rank's head block of the
    partial goes to its owner, its LSEs riding as one more 16-bit head row,
    and `msa_dcp.merge_partials` weighs, sums in f32 and unrotates in one
    pass. For the decode rows of a head-parallel step, whose LSE all-gather
    and reduce-scatter waited out late ranks (c64 mixed twin, j-741c6acc vs
    j-613b69cd: 288.6 -> 282.0 ms per step); a decode step keeps the
    reduce-scatter, which moves its partial faster than an all-to-all
    (30 vs 47 us per layer at c64 b46)."""
    num_heads, num_tokens, head_dim = out.shape
    heads = num_heads // dcp
    width = 2 * heads
    if out.dtype.itemsize != 2 or width > head_dim:
        raise ValueError(
            f"16-bit partials of at most {head_dim // 2} heads per rank, got "
            f"{out.dtype} {out.shape} over {dcp} ranks"
        )
    lse = lse.astype(jnp.float32).T.reshape(num_tokens, dcp, heads).transpose(1, 0, 2)
    bits = msa_dcp.words_to_u16_planes(lse)[:, None]  # [dcp, 1, T, 2 Hl]
    bits = jnp.pad(bits, ((0, 0), (0, 0), (0, 0), (0, head_dim - width)))
    send = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(out, jnp.uint16).reshape(
                dcp, heads, num_tokens, head_dim
            ),
            bits,
        ],
        axis=1,
    )  # [dcp, Hl + 1, T, D]: block r goes to rank r
    got = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)
    parts = jax.lax.bitcast_convert_type(got[:, :heads], out.dtype)
    lses = msa_dcp.u16_planes_to_words(got[:, heads], heads, jnp.float32)
    hadamard = jnp.asarray(msa_fp4.hadamard(head_dim), out.dtype)
    merged = msa_dcp.merge_partials(parts, lses, hadamard, interpret=interpret)
    return merged.reshape(num_tokens, heads, head_dim)


_WEIGH_BLOCK_TOKENS = 64


def _weigh_kernel(out_ref, lse_ref, lses_ref, o_ref):
    lses = lses_ref[...]  # [dcp, Bt, H]
    peak = jnp.max(lses, axis=0)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    total = peak + jnp.log(jnp.sum(jnp.exp(lses - peak[None]), axis=0))
    weight = jnp.exp(lse_ref[...] - total)  # [Bt, H]
    for h in range(out_ref.shape[0]):
        w = weight[:, h : h + 1]
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        o_ref[h] = jnp.where(w > 0, out_ref[h].astype(jnp.float32) * w, 0.0).astype(
            o_ref.dtype
        )


def weigh_partials(out, lse, lses, dtype, *, interpret=False):
    """Each head-major partial ``out [H, T, D]`` weighted by its share of the
    ranks' log-sum-exp, ``exp(lse - logsumexp(lses))`` (``lse [T, H]`` this
    rank's, ``lses [dcp, T, H]`` every rank's, f32), as `dtype`: one pass
    over the partials in place of XLA's f32 weighting of every head (~16 us
    per layer at c64 b50, j-5b748468)."""
    num_heads, num_tokens, head_dim = out.shape
    dcp = lses.shape[0]
    block = max(1, min(num_tokens, _WEIGH_BLOCK_TOKENS))
    return pl.pallas_call(
        _weigh_kernel,
        grid=(pl.cdiv(num_tokens, block),),
        in_specs=[
            pl.BlockSpec((num_heads, block, head_dim), lambda i: (0, i, 0)),
            pl.BlockSpec((block, num_heads), lambda i: (i, 0)),
            pl.BlockSpec((dcp, block, num_heads), lambda i: (0, i, 0)),
        ],
        out_specs=pl.BlockSpec((num_heads, block, head_dim), lambda i: (0, i, 0)),
        out_shape=jax.ShapeDtypeStruct(out.shape, dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_weigh_partials",
    )(out, lse, lses)


def _dcp_attend(
    q,
    kv_cache,
    scale_cache,
    candidates,
    seq_lens,
    block_tables,
    cu,
    *,
    axis,
    rank,
    dcp,
    sm_scale,
    topk,
    rs_out_precision,
    interpret,
    all_to_all=False,
):
    """The DCP attention of rows ``q [R, Hl, D]`` (this rank's rotated query
    heads) with this rank's packed ``candidates [R, G * 2 * topk]``, merged
    by `merge_partials` or, with `all_to_all`, `merge_partials_all_to_all`."""
    q_all = jax.lax.all_gather(q, axis, axis=1, tiled=True)
    cand_all = jax.lax.all_gather(candidates, axis, axis=0)
    attend = functools.partial(
        msa_dcp.msa_attend_dcp,
        q_all,
        kv_cache,
        cand_all,
        seq_lens,
        block_tables,
        cu,
        sm_scale=sm_scale,
        topk=topk,
        rank=rank,
        dcp=dcp,
        scale_cache=scale_cache,
        interpret=interpret,
    )
    kv_heads = candidates.shape[1] // (2 * topk)
    if (
        PAIR_FIRST_HEAD_CALLS
        and dcp == 4
        and kv_heads == 2
        and not all_to_all
        and q.shape[0] <= msa_dcp.DCP_SMALL_BATCH_TOKENS
    ):
        # [T, Hl * D], unrotated
        return _pair_first_merge_by_head(
            lambda order: attend(head_order=order),
            axis,
            rank,
            rs_out_precision,
            interpret,
        )
    out, lse = attend()
    if all_to_all:
        return merge_partials_all_to_all(out, lse, axis, dcp=dcp, interpret=interpret)
    return merge_partials(
        out,
        lse,
        axis,
        rank=rank,
        dcp=dcp,
        rs_out_precision=rs_out_precision,
        interpret=interpret,
    )


# Tokens per grid step of `unrotate_to_rows`.
_UNROTATE_BLOCK_TOKENS = 1024


def unrotate_to_rows(x: jax.Array, *, interpret: bool = False) -> jax.Array:
    """`msa_fp4.unrotate_output` of head-major ``x [H, T, D]`` (bf16), written
    token-major ``[T, H * D]``, the layout the output projection reads: as
    XLA ops the transpose, the dot and the reshape to rows each re-laid the
    output out (~4 copies of it, ~43 us per layer at c64, twin j-9a9fccc2)."""
    heads, num_tokens, head_dim = x.shape
    if x.dtype != jnp.bfloat16:
        out = msa_fp4.unrotate_output(x.transpose(1, 0, 2))
        return out.reshape(num_tokens, heads * head_dim)
    block_t = min(num_tokens, _UNROTATE_BLOCK_TOKENS)

    def kernel(x_ref, h_ref, o_ref):
        for h in range(heads):
            y = jnp.dot(x_ref[h], h_ref[...], preferred_element_type=jnp.float32)
            o_ref[:, h * head_dim : (h + 1) * head_dim] = (y * (1.0 / head_dim)).astype(
                o_ref.dtype
            )

    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=[
            pl.BlockSpec((heads, block_t, head_dim), lambda i: (0, i, 0)),
            pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_t, heads * head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, heads * head_dim), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_unrotate_rows",
    )(x, jnp.asarray(msa_fp4.hadamard(head_dim).T, jnp.bfloat16))


# Tokens of each replica's rows per grid step of the replica kernels.
_REPLICA_BLOCK_TOKENS = 512


def _replica_grid(rows: int, block: int) -> tuple[int, int]:
    """(pairs of `block`-token blocks per grid step, grid steps) for `rows`
    tokens per replica."""
    n = rows // block
    per_step = max(1, min(n, _REPLICA_BLOCK_TOKENS // block))
    while n % per_step:
        per_step -= 1
    return per_step, n // per_step


def replica_rows(
    q: jax.Array,
    theirs: jax.Array,
    j: jax.Array,
    *,
    block: int,
    rotate: bool,
    interpret: bool = False,
) -> jax.Array:
    """``[T / 2, 2 Hl, D]``: replica ``j``'s rows (the `block`-token blocks of
    parity ``j`` of ``q [T, Hl, D]``) with this rank's query heads first and
    the partner's ``theirs [T / 2, Hl, D]`` for the same rows after, rotated
    (`msa_fp4.rotate_queries`) with `rotate`. One pass that reads the blocks
    in place: as XLA ops, a transpose of the rotation's head-major output,
    the half picks and the concatenation cost ~5 copies of q (c64's
    8K-token steps: 56 us per layer, twin j-5c900b89)."""
    if q.dtype != jnp.bfloat16:
        raise ValueError(f"packs and rotates bf16 queries, got {q.dtype}")
    num_tokens, heads, head_dim = q.shape
    rows = num_tokens // 2
    per_step, steps = _replica_grid(rows, block)
    tokens = per_step * block
    width = heads * head_dim
    pairs = q.reshape(num_tokens // (2 * block), 2, block, width)
    theirs = theirs.reshape(rows // block, block, width)

    def kernel(j_ref, mine_ref, theirs_ref, h_ref, o_ref):
        del j_ref
        bits = []
        for src in (mine_ref, theirs_ref):
            for h in range(heads):
                x = src[:, :, h * head_dim : (h + 1) * head_dim]
                x = x.reshape(tokens, head_dim)
                if rotate:
                    x = jnp.dot(x, h_ref[...], preferred_element_type=jnp.float32)
                    x = (x * (1.0 / head_dim)).astype(jnp.bfloat16)
                bits.append(
                    jax.lax.bitcast_convert_type(x, jnp.uint16).astype(jnp.uint32)
                )
        # A token's heads 2k and 2k + 1 share a 32-bit row of its bf16 tile
        # (the even head in the low half): interleaving 8 rows of 32 bits
        # per token cost 31 us at 8K tokens, 16 bf16 rows 54 us (tpu7x).
        words = [bits[2 * k] | (bits[2 * k + 1] << 16) for k in range(heads)]
        o_ref.bitcast(jnp.uint32)[...] = jnp.stack(words, axis=1)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(steps,),
            in_specs=[
                pl.BlockSpec(
                    (per_step, None, block, width), lambda i, j: (i, j[0], 0, 0)
                ),
                pl.BlockSpec((per_step, block, width), lambda i, j: (i, 0, 0)),
                pl.BlockSpec((head_dim, head_dim), lambda i, j: (0, 0)),
            ],
            out_specs=pl.BlockSpec(
                (tokens, 2 * heads, head_dim), lambda i, j: (i, 0, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((rows, 2 * heads, head_dim), q.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_replica_rows",
    )(
        jnp.reshape(j, (1,)).astype(jnp.int32),
        pairs,
        theirs,
        jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16),
    )


def replica_rows_out(
    out: jax.Array,
    theirs: jax.Array,
    j: jax.Array,
    *,
    block: int,
    interpret: bool = False,
) -> jax.Array:
    """Inverse of `replica_rows` for the head-major attention output ``out
    [2 Hl, T / 2, D]`` of replica ``j``'s rows and ``theirs [Hl, T / 2, D]``
    (this rank's heads over the partner's rows): this rank's heads over every
    row, unrotated as `unrotate_to_rows`, written token-major ``[T, Hl * D]``
    in step order. Reads ``out``'s first ``Hl`` heads in place."""
    if out.dtype != jnp.bfloat16:
        raise ValueError(f"unrotates a bf16 output (one MXU pass), got {out.dtype}")
    _, rows, head_dim = out.shape
    heads = theirs.shape[0]
    per_step, steps = _replica_grid(rows, block)
    tokens = per_step * block
    width = heads * head_dim

    def kernel(j_ref, mine_ref, theirs_ref, h_ref, o_ref):
        j = j_ref[0]
        for parity, src in ((j, mine_ref), (1 - j, theirs_ref)):
            for h in range(heads):
                y = jnp.dot(src[h], h_ref[...], preferred_element_type=jnp.float32)
                y = (y * (1.0 / head_dim)).astype(o_ref.dtype)
                o_ref[:, pl.ds(parity, 1), :, h * head_dim : (h + 1) * head_dim] = (
                    y.reshape(per_step, 1, block, head_dim)
                )

    result = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(steps,),
            in_specs=[
                pl.BlockSpec((heads, tokens, head_dim), lambda i, j: (0, i, 0)),
                pl.BlockSpec((heads, tokens, head_dim), lambda i, j: (0, i, 0)),
                pl.BlockSpec((head_dim, head_dim), lambda i, j: (0, 0)),
            ],
            out_specs=pl.BlockSpec(
                (per_step, 2, block, width), lambda i, j: (i, 0, 0, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((rows // block, 2, block, width), out.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_replica_rows_out",
    )(
        jnp.reshape(j, (1,)).astype(jnp.int32),
        out,
        theirs,
        jnp.asarray(msa_fp4.hadamard(head_dim).T, jnp.bfloat16),
    )
    return result.reshape(2 * rows, width)


def _per_token(seq_id: jax.Array, table: jax.Array) -> jax.Array:
    """``table[seq_id]`` for a per-request table, by a one-hot over the
    requests: a gather of 8K tokens from a 64-entry table cost ~70 us."""
    onehot = seq_id[:, None] == jnp.arange(table.shape[0], dtype=jnp.int32)
    return jnp.sum(jnp.where(onehot, table[None, :].astype(jnp.int32), 0), axis=1)


# Pages whose head copies to every rank an exchange keeps in flight.
_EXCHANGE_PAGES_IN_FLIGHT = 8
# The exchange's barrier semaphore.
_EXCHANGE_COLLECTIVE_ID = 0


def live_part(live: jax.Array, replica, replicas: int, part: int) -> jax.Array:
    """Live pages of replica `replica`'s part when the first `live` logical
    blocks of a buffer are live and block ``l`` sits in part ``l % replicas``
    at index ``l // replicas``."""
    return jnp.clip((live - replica + replicas - 1) // replicas, 0, part)


def exchange_head_pages(
    kv_cache: jax.Array,
    pages: jax.Array,
    live: jax.Array,
    rank: jax.Array,
    *,
    axis: str,
    dcp: int,
    replicas: int,
    head_width: int,
    interpret: bool = False,
) -> jax.Array:
    """Inside a ``shard_map`` over `axis`: every rank's pages of the KV head
    this rank (``rank``, ``int32[1]``) owns, ``[replicas, dcp, L / replicas,
    R, P, head_width]``: ``[j, src, i]`` is rank ``src``'s page ``pages[j *
    L / replicas + i]``, for the `live_part` of ``live`` (``int32[1]``)
    pages of each part ``j``; the others are left unset.

    Rank ``src`` copies each live page's head-``h`` columns straight from its
    cache into rank ``h * replicas + j``'s output by remote DMA (one copy
    per page and destination, all destinations in flight at once), and
    every rank forwards what it received to the other replicas of its head
    (one chip's cores at c64). A static XLA gather, all-to-all and
    all-gather moved the whole buffer whatever the context, relaid the pages
    out around each collective and ran the all-to-all at one link's
    bandwidth."""
    _, rows, packing, _ = kv_cache.shape
    num_pages = pages.shape[0]
    part = num_pages // replicas
    # Slots past the live pages stay unset: harmless while no key reaches
    # them and while any byte pattern of the 4-bit cache dequantizes finite.
    if kv_cache.dtype != jnp.uint8:
        raise ValueError(f"4-bit (uint8) caches only, got {kv_cache.dtype}")

    def remote(src, dst, send_sem, recv_sem, device):
        return pltpu.make_async_remote_copy(
            src,
            dst,
            send_sem,
            recv_sem,
            device_id={axis: device},
            device_id_type=pl.DeviceIdType.MESH,
        )

    def kernel(
        pages_ref, live_ref, rank_ref, kv_ref, out_ref, send_sem, recv_sem, fwd_sem
    ):
        me = rank_ref[0]
        replica = me % replicas
        counts = [live_part(live_ref[0], j, replicas, part) for j in range(replicas)]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(dcp):
            pl.semaphore_signal(
                barrier,
                inc=1,
                device_id={axis: jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
        # Every rank has entered the call: its output buffer is live.
        pl.semaphore_wait(barrier, dcp)

        def sends(i, op):
            for d in range(dcp):
                head, j = divmod(d, replicas)

                def send(d=d, head=head, j=j):
                    columns = pl.ds(head * head_width, head_width)
                    src = kv_ref.at[pages_ref[j * part + i], :, :, columns]
                    op(remote(src, out_ref.at[j, me, i], send_sem, recv_sem, d))

                pl.when(i < counts[j])(send)

        def pipelined(n, copies):
            """Start ``copies(i, op)`` for ``i < n``, a bounded number in
            flight (equal-sized copies on one semaphore: any wait retires
            one)."""
            ahead = _EXCHANGE_PAGES_IN_FLIGHT

            def step(i, carry):
                copies(i, lambda c: c.start())
                pl.when(i >= ahead)(lambda: copies(i - ahead, lambda c: c.wait_send()))
                return carry

            def drain(i, carry):
                copies(i, lambda c: c.wait_send())
                return carry

            jax.lax.fori_loop(0, n, step, 0)
            jax.lax.fori_loop(jnp.maximum(n - ahead, 0), n, drain, 0)

        pipelined(counts[0], sends)

        def page_wait(sem):
            def wait(i, carry):
                page = out_ref.at[0, 0, 0]
                pltpu.make_async_copy(page, page, sem).wait()
                return carry

            return wait

        mine = jnp.where(replica == 0, counts[0], counts[-1])
        for j in range(1, replicas - 1):
            mine = jnp.where(replica == j, counts[j], mine)
        jax.lax.fori_loop(0, mine * dcp, page_wait(recv_sem), 0)

        # Forward the received pages to the head's other replicas.
        def forwards(i, op):
            for k in range(1, replicas):
                peer = me - replica + (replica + k) % replicas
                for src in range(dcp):
                    page = out_ref.at[replica, src, i]
                    op(remote(page, page, send_sem, fwd_sem, peer))

        pipelined(mine, forwards)
        theirs = sum(counts) - mine
        jax.lax.fori_loop(0, theirs * dcp, page_wait(fwd_sem), 0)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())] * 3,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (replicas, dcp, part, rows, packing, head_width), kv_cache.dtype
        ),
        compiler_params=pltpu.CompilerParams(collective_id=_EXCHANGE_COLLECTIVE_ID),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_exchange_head_pages",
    )(pages, live, rank, kv_cache)


def _exchange_pages(
    kv_cache,
    scale_cache,
    pages,
    live,
    rank,
    *,
    dcp,
    replicas,
    head_width,
    axis,
    interpret,
):
    """Every rank's ``pages [L]`` of the KV head each rank owns:
    ``(kv [L * dcp, S / 8, 4, 2 * D], scales [L * dcp, SCALE_ROWS, 128])``,
    rank ``src``'s page ``pages[j * L / replicas + i]`` at slot ``(j * dcp +
    src) * L / replicas + i``; only the pages `exchange_head_pages` calls
    live (`live` logical blocks) are set.

    The pages move by `exchange_head_pages`. The ``replicas`` ranks of a KV
    head each receive a ``1 / replicas`` part of the scales over an
    all-to-all and share them with an all-gather inside the replica group
    (one chip's cores at c64)."""
    num_pages = pages.shape[0]
    part = num_pages // replicas
    _, rows, packing, _ = kv_cache.shape
    heads = dcp // replicas
    if kv_cache.shape[3] != heads * head_width:
        raise ValueError(
            f"cache rows {kv_cache.shape} do not hold {heads} KV heads of "
            f"{head_width} columns"
        )
    recv = exchange_head_pages(
        kv_cache,
        pages,
        live.reshape(1),
        rank.reshape(1),
        axis=axis,
        dcp=dcp,
        replicas=replicas,
        head_width=head_width,
        interpret=interpret,
    )
    # Destination rank d = h * replicas + j gets part j of head h.
    sc = scale_cache[pages][:, : 2 * heads]  # K/V rows of every head
    sc = sc.reshape(replicas, part, heads, 2, sc.shape[-1])
    # [dcp, part * 2, lanes]: a [.., 2, lanes] block pads its (8, 128) tiles
    # four times over.
    send_sc = jnp.moveaxis(sc, 2, 0).reshape(dcp, part * 2, sc.shape[-1])
    recv_sc = jax.lax.all_to_all(send_sc, axis, 0, 0, tiled=True)
    if replicas > 1:
        groups = [list(range(h * replicas, (h + 1) * replicas)) for h in range(heads)]
        recv_sc = jax.lax.all_gather(
            recv_sc, axis, axis_index_groups=groups, tiled=True
        )
    recv = recv.reshape(num_pages * dcp, rows, packing, head_width)
    recv_sc = recv_sc.reshape(num_pages * dcp, 2, recv_sc.shape[-1])
    recv_sc = jnp.pad(recv_sc, ((0, 0), (0, msa_fp4.SCALE_ROWS - 2), (0, 0)))
    return recv, recv_sc


def _own_selection(
    candidates, *, dcp, replicas, topk, axis, interpret, share_only=False
):
    """The selection ``[T, topk]`` of this rank's KV head, from every rank's
    packed ``candidates [T, G * 2 * topk]``. Each replica of the head merges
    the candidates of a ``1 / replicas`` share of the tokens (an all-to-all
    of that share) and the replicas all-gather the merged selections, which
    are 1/8 the candidates' bytes. With `share_only` (the tokens split
    evenly), the replica's own share ``[T / replicas, topk]`` comes back
    instead, with no all-gather."""
    num_tokens = candidates.shape[0]
    heads = dcp // replicas
    split = replicas if num_tokens % replicas == 0 else 1
    part = num_tokens // split
    per_head = candidates.reshape(num_tokens, heads, 2 * topk).transpose(1, 0, 2)
    # Destination rank d = h * replicas + j gets head h's tokens of share j
    # (every share, when the tokens do not split).
    send = per_head.reshape(heads, split, part, 2 * topk)
    send = jnp.repeat(send, replicas // split, axis=1).reshape(dcp, part, 2 * topk)
    own = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)  # [dcp, T / split, 2K]
    selected = msa_dcp.merge_head_candidates(own, topk, interpret)[:, 0]
    if share_only:
        assert split == replicas, (num_tokens, replicas)
        return selected
    if split > 1:
        groups = [list(range(h * replicas, (h + 1) * replicas)) for h in range(heads)]
        selected = jax.lax.all_gather(
            selected, axis, axis_index_groups=groups, tiled=True
        )
    return selected


def _head_parallel_attend(
    q,
    kv_cache,
    scale_cache,
    candidates,
    seq_lens,
    block_tables,
    cu,
    hp_req,
    tokens,
    *,
    axis,
    rank,
    dcp,
    kv_head_replicas,
    sm_scale,
    topk,
    q_rotated,
    interpret,
):
    """Attention ``[T, Hl * D]`` (unrotated, token-major) of the rows of the
    requests in ``hp_req`` over their pages of this rank's KV head; other
    rows are zero. ``tokens`` is `msa.token_positions` of the step; ``q``
    is rotated (`msa_fp4.rotate_queries`) unless `q_rotated`.

    Requests are attended in rounds of at most ``pages_per_seq`` logical
    blocks (one max-length request): each round exchanges the live pages
    of one page buffer and attends the round's rows in windows of one
    kernel call."""
    num_tokens, num_q_heads, head_dim = q.shape
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    capacity = block_tables_2d.shape[1]  # logical blocks per round
    replicas = kv_head_replicas
    part = -(-capacity // replicas)  # each replica's share of the buffer
    # The two replicas of a KV head split the rows (`_replica_rows`): each
    # attends half the tokens, in alternate blocks, with all the head's query
    # heads.
    split = (
        replicas == 2
        and num_tokens % (2 * _BLOCK_Q) == 0
        and num_tokens % (2 * _INTERLEAVE_TOKENS) == 0
    )
    rows = num_tokens // replicas if split else num_tokens
    offset = rank % replicas * rows if split else 0
    if split:
        candidates = _interleave(candidates)
        tokens = tuple(_interleave(x) for x in tokens)

    selected = _own_selection(
        candidates,
        dcp=dcp,
        replicas=replicas,
        topk=topk,
        axis=axis,
        interpret=interpret,
        share_only=split,
    )

    seq_id, pos, valid = tokens
    if split:
        seq_id, pos, valid = (
            jax.lax.dynamic_slice_in_dim(x, offset, rows) for x in tokens
        )
        q = _replica_rows(
            q, rank, axis=axis, dcp=dcp, rotate=not q_rotated, interpret=interpret
        )
    elif not q_rotated:
        q = msa_fp4.rotate_queries(q)
    seq_blocks = -(-seq_lens // (block_size * dcp))
    blocks = jnp.where(hp_req, seq_blocks, 0)
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    # Rows per kernel call (the caller pads tokens to a block_q multiple).
    window = min(rows, msa_attention.MAX_TOKENS_PER_CALL)
    sel_block = jnp.maximum(selected, 0)
    own_block = pos // block_size
    limit = pos % block_size + 1

    def plan_round(first):
        csum = jnp.cumsum(jnp.where(seq_idx >= first, blocks, 0))
        # A request never exceeds the buffer (the block table's width); always
        # admitting `first` keeps the rounds moving if that ever broke.
        in_round = (
            hp_req & (seq_idx >= first) & ((csum <= capacity) | (seq_idx == first))
        )
        base = csum - blocks  # the request's first logical block in the buffer
        total = jnp.max(jnp.where(in_round, csum, 0))
        nxt = jnp.min(
            jnp.where((seq_idx >= first) & ~in_round & hp_req, seq_idx, max_num_seqs)
        )
        return csum, in_round, base, total, nxt

    def attend_round(plan, acc, merge):
        csum, in_round, base, total, _ = plan
        num_pages = part * replicas
        # Buffer page b holds logical block l = b % part * replicas + b //
        # part: logical block s - base[q] of request q.
        buf = jnp.arange(num_pages, dtype=jnp.int32)
        logical = buf % part * replicas + buf // part
        q_of = jnp.searchsorted(csum, logical, side="right", method="compare_all")
        q_of = jnp.minimum(q_of, max_num_seqs - 1)
        j = jnp.clip(logical - base[q_of], 0, capacity - 1)
        pages = jnp.where(logical < total, block_tables_2d[q_of, j], 0)
        kv, sc = _exchange_pages(
            kv_cache,
            scale_cache,
            pages,
            total,
            rank,
            dcp=dcp,
            replicas=replicas,
            head_width=2 * head_dim,
            axis=axis,
            interpret=interpret,
        )

        def slot(base, block):
            # Logical block l = base + block // dcp of rank block % dcp.
            logical = base + block // dcp
            j = logical % replicas
            return (j * dcp + block % dcp) * part + logical // replicas

        mine = valid & (_per_token(seq_id, in_round) > 0)
        tok_base = _per_token(seq_id, base)
        keys = jnp.where(
            mine[:, None] & (selected >= 0), slot(tok_base[:, None], sel_block), -1
        )
        local = jnp.where(mine, slot(tok_base, own_block), -1)
        lo = cu[jnp.min(jnp.where(in_round, seq_idx, max_num_seqs - 1))]
        hi = cu[jnp.max(jnp.where(in_round, seq_idx + 1, 0))]
        lo, hi = (jnp.clip(x - offset, 0, rows) for x in (lo, hi))
        if split:  # a request's rows are spread over the half
            lo, hi = 0, rows
        page_table = jnp.arange(dcp * num_pages, dtype=jnp.int32)

        attend = functools.partial(
            _attend_head,
            sm_scale=sm_scale,
            pages_per_step=_SPLIT_PAGES_PER_STEP if split else _PAGES_PER_STEP,
            scale_cache=sc,
            interpret=interpret,
        )
        if not merge:
            # One call over every row: rows outside the round see no page
            # and come back zero (cheap empty tiles, or no pairs).
            return attend(q, kv, keys, local, limit, page_table)

        def window_body(i, acc):
            start = jnp.minimum(lo + i * window, rows - window)

            def sl(x):
                return jax.lax.dynamic_slice_in_dim(x, start, window, 0)

            out = attend(sl(q), kv, sl(keys), sl(local), sl(limit), page_table)
            old = jax.lax.dynamic_slice_in_dim(acc, start, window, 1)
            out = jnp.where(sl(mine)[None, :, None], out, old)  # earlier rounds
            return jax.lax.dynamic_update_slice_in_dim(acc, out, start, 1)

        return jax.lax.fori_loop(0, -(-(hi - lo) // window), window_body, acc)

    first = jnp.min(jnp.where(hp_req, seq_idx, max_num_seqs))
    plan = plan_round(first)
    acc = attend_round(plan, None, merge=False)
    first = plan[4]

    # Further rounds sit under a cond: a while loop that reads the cache
    # makes XLA copy the whole cache ahead of it (0.45 ms per layer at c64).
    def more_rounds(state):
        def body(s):
            plan = plan_round(s[0])
            return plan[4], attend_round(plan, s[1], merge=True)

        return jax.lax.while_loop(lambda s: s[0] < max_num_seqs, body, state)

    _, acc = jax.lax.cond(first < max_num_seqs, more_rounds, lambda s: s, (first, acc))
    if split:
        return _replica_heads(acc, rank, axis=axis, dcp=dcp, interpret=interpret)
    return unrotate_to_rows(acc, interpret=interpret)


def _attend_head(
    q,
    kv,
    keys,
    local,
    limit,
    page_table,
    *,
    sm_scale,
    pages_per_step,
    scale_cache,
    interpret,
):
    """Head-major ``[H, T, D]`` attention of ``q [T, H, D]`` over one KV
    head's 4-bit pages: in tiles, or page-major (`PAGE_MAJOR`) where the
    page-major kernel takes the shape and the scores are max-free."""

    def tiles(operands):
        q, keys, local, limit = operands
        out, _ = msa_attention_lse.msa_sparse_attention_lse(
            q,
            kv,
            keys,
            local,
            limit,
            page_table,
            sm_scale=sm_scale,
            block_q=_BLOCK_Q,
            pages_per_step=pages_per_step,
            head=0,
            scale_cache=scale_cache,
            k_scale_row=0,
            v_scale_row=1,
            head_major=True,
            interpret=interpret,
        )
        return out

    def pairs(operands):
        q, keys, local, limit = operands
        return msa_attention_pairs.msa_sparse_attention_pairs(
            q,
            kv,
            keys,
            local,
            limit,
            page_table,
            sm_scale=sm_scale,
            scale_cache=scale_cache,
            interpret=interpret,
        )

    operands = (q, keys, local, limit)
    num_tokens, num_heads, head_dim = q.shape
    fits = msa_attention_pairs.supports(
        num_heads,
        head_dim,
        msa_attention.page_tokens(kv),
        page_table.shape[0],
        num_pages=scale_cache.shape[0],
    )
    if PAGE_MAJOR == "off" or not (fits and msa_attention_lse.MAX_FREE):
        return tiles(operands)
    if PAGE_MAJOR == "on":
        return pairs(operands)
    tiled = jnp.pad(keys, ((0, -num_tokens % _BLOCK_Q), (0, 0)), constant_values=-1)
    ratio = msa_attention_lse._union_ratio(tiled[:, None], _BLOCK_Q)
    return jax.lax.cond(ratio > _PAGE_MAJOR_MIN_RATIO, pairs, tiles, operands)


def _replica_rows(q, rank, *, axis, dcp, rotate, interpret):
    """This replica's rows of ``q [T, Hl, D]`` with both replicas' query
    heads, ``[T / 2, 2 Hl, D]`` (its own heads first): the replicas take
    alternate `_INTERLEAVE_TOKENS`-token blocks, and the partner (rank
    ``rank ^ 1``) sends its heads for this replica's blocks (`replica_rows`).
    Each head attends on its own, so the head order changes no value."""
    block = _INTERLEAVE_TOKENS
    j = rank % 2
    # Token rows of all heads: q's [T, Hl * D] layout as the rope leaves it.
    pairs = q.reshape(q.shape[0] // (2 * block), 2, block, -1)
    theirs = jax.lax.ppermute(
        jax.lax.dynamic_index_in_dim(pairs, 1 - j, axis=1, keepdims=False),
        axis,
        [(r, r ^ 1) for r in range(dcp)],
    )
    return replica_rows(q, theirs, j, block=block, rotate=rotate, interpret=interpret)


def _replica_heads(out, rank, *, axis, dcp, interpret):
    """`replica_rows_out` of this replica's head-major output ``[2 Hl, T / 2,
    D]``: the partner's rows of this rank's heads come from the partner."""
    heads = out.shape[0] // 2
    theirs = jax.lax.ppermute(out[heads:], axis, [(r, r ^ 1) for r in range(dcp)])
    return replica_rows_out(
        out, theirs, rank % 2, block=_INTERLEAVE_TOKENS, interpret=interpret
    )


def head_parallel(
    num_tokens: int,
    max_num_seqs: int,
    decode_rows: int,
    min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
) -> bool:
    """Whether `attend_and_merge` attends a step's prefill rows head-parallel."""
    return num_tokens >= max(min_tokens, 2 * max_num_seqs * decode_rows)


def attend_and_merge(
    rank: jax.Array,
    q: jax.Array,
    kv_cache: jax.Array,
    scale_cache: jax.Array,
    candidates: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    axis: str,
    dcp: int,
    kv_head_replicas: int,
    sm_scale: float,
    topk: int,
    decode_rows: int,
    rs_out_precision: bool,
    head_parallel_min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
    q_rotated: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """This rank's (``rank``, ``int32[1]``) attention output ``[T, Hl * D]``
    (unrotated) for its query heads ``q [T, Hl, D]``, rotated
    (`msa_fp4.rotate_queries`) unless `q_rotated`, and its packed
    ``candidates [T, G * 2 * topk]`` (`msa_dcp.msa_write_and_candidates_dcp`);
    4-bit caches.
    Runs inside a ``shard_map`` over `axis` (the DCP group)."""
    num_tokens = q.shape[0]
    # The rank is an input: torch_tpu's SPMD partitioner refuses the
    # partition-id that `lax.axis_index` lowers to.
    rank = msa_dcp.rank_array(rank)[0]
    common = dict(
        axis=axis, rank=rank, dcp=dcp, sm_scale=sm_scale, topk=topk, interpret=interpret
    )
    max_num_seqs = seq_lens.shape[0]
    rows = max_num_seqs * decode_rows
    rotate = (lambda x: x) if q_rotated else msa_fp4.rotate_queries
    if not head_parallel(
        num_tokens, max_num_seqs, decode_rows, head_parallel_min_tokens
    ):
        return _dcp_attend(
            rotate(q),
            kv_cache,
            scale_cache,
            candidates,
            seq_lens,
            block_tables,
            cu_q_lens,
            rs_out_precision=rs_out_precision,
            **common,
        ).reshape(num_tokens, -1)
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    live = (seq_idx < distribution[2]) & (q_lens > 0)
    dcp_req = live & (q_lens <= decode_rows)
    hp_req = live & (q_lens > decode_rows)

    # The decode requests' rows, compacted: their own cu_q_lens.
    cu_d = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(jnp.where(dcp_req, q_lens, 0))]
    ).astype(jnp.int32)
    p = jnp.arange(rows, dtype=jnp.int32)
    # compare_all: the default scan lowers to a loop of small ops.
    req = jnp.searchsorted(cu_d[1:], p, side="right", method="compare_all")
    req = jnp.minimum(req, max_num_seqs - 1)
    src_row = jnp.where(p < cu_d[-1], cu_q_lens[req] + p - cu_d[req], 0)
    # Rows of all heads, gathered in q's [T, Hl * D] layout as the rope
    # leaves it ([T, Hl, D] rows took a relayout of q first).
    q_rows = q.reshape(num_tokens, -1)[src_row].reshape(rows, *q.shape[1:])
    dcp_out = _dcp_attend(
        rotate(q_rows),
        kv_cache,
        scale_cache,
        candidates[src_row],
        seq_lens,
        block_tables,
        cu_d,
        rs_out_precision=rs_out_precision,
        all_to_all=True,
        **common,
    )
    tokens = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    hp_out = _head_parallel_attend(
        q,
        kv_cache,
        scale_cache,
        candidates,
        seq_lens,
        block_tables,
        cu_q_lens,
        hp_req,
        tokens,
        kv_head_replicas=kv_head_replicas,
        q_rotated=q_rotated,
        **common,
    )
    dcp_out = dcp_out.reshape(rows, -1)
    num_decode = cu_d[-1]

    def decode_prefix(out):
        # The runner orders decodes ahead of prefills: the decode rows are
        # then the step's first rows, one in-place update of `rows` rows.
        head = jnp.where((p < num_decode)[:, None], dcp_out, out[:rows])
        return jax.lax.dynamic_update_slice_in_dim(out, head, 0, 0)

    def scatter(out):
        # The decode rows go back to their places (the others are dropped):
        # a scatter of `rows` rows.
        dest = jnp.where(p < num_decode, src_row, num_tokens)
        return out.at[dest].set(dcp_out, mode="drop")

    prefix = jnp.all((p >= num_decode) | (src_row == p))
    # The barrier keeps the consumer's ops out of the branches: XLA sank the
    # output projection's f32 convert of the whole output into them (a 32 MB
    # f32 copy, ~20 us per layer at c64's 8K-token steps, twin j-a202588d).
    return jax.lax.optimization_barrier(
        jax.lax.cond(prefix, decode_prefix, scatter, hp_out)
    )
