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
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_fp4,
    msa_quantize,
)

# Steps of at least this many tokens attend their prefill requests
# head-parallel. Each round moves a static page buffer (~35 MB per rank at
# max_model_len 256K), which a step's query gather and merge exceed from
# about 2K tokens (~0.16 us per token at c64 DCP 8, j-0a66fa2e).
HEAD_PARALLEL_MIN_TOKENS = 2048
# Tiles and pages per kernel step of the head-parallel calls (every rank's
# pages): the tile of `msa_attention.PREFILL_BLOCK_Q_4BIT`, wide groups to
# amortize a step's fixed cost.
_BLOCK_Q = msa_attention.PREFILL_BLOCK_Q_4BIT
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
# kernel time up to 2.2x, j-867fc577). A block is whole tiles (`_BLOCK_Q`).
_INTERLEAVE_TOKENS = 64
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


# Tokens per replica per grid step of `rotate_replica_rows` and
# `unrotate_replica_rows` (8 heads x 4096 tokens on one v7x core: 17.1 and
# 16.6 us, 512-token steps 18.0 and 18.6 us; anvil, 3 reps).
_REPLICA_BLOCK_TOKENS = 1024


def _replica_grid(num_tokens: int) -> tuple[int, int]:
    """``(pairs, step)``: ``num_tokens`` as `_INTERLEAVE_TOKENS`-token block
    pairs (one block per replica), `step` pairs per grid step."""
    pairs = num_tokens // (2 * _INTERLEAVE_TOKENS)
    if num_tokens != 2 * _INTERLEAVE_TOKENS * pairs:
        raise ValueError(f"{num_tokens} tokens are no whole block pairs")
    return pairs, max(1, min(pairs, _REPLICA_BLOCK_TOKENS // _INTERLEAVE_TOKENS))


def rotate_replica_rows(
    q: jax.Array, rank: jax.Array, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """`msa_fp4.rotate_queries` of ``q [T, Hl, D]`` (bf16), written
    head-major in the replicas' row order (`_interleave`): ``(own, other)``,
    each ``[Hl, T / 2, D]``, the alternate `_INTERLEAVE_TOKENS`-token blocks
    replica ``rank % 2`` attends and those its partner attends. As XLA ops
    the rotation, the interleave and the replica's half each re-laid all of
    q out (c64 mixed twin j-c5ebe0af: ~50 us per layer)."""
    num_tokens, heads, head_dim = q.shape
    pairs, step = _replica_grid(num_tokens)
    block = _INTERLEAVE_TOKENS
    rows = step * block

    def kernel(rank_ref, own_in, other_in, h_ref, own_ref, other_ref):
        del rank_ref
        for x_ref, o_ref in ((own_in, own_ref), (other_in, other_ref)):
            x = x_ref[...].reshape(rows, heads * head_dim)
            for h in range(heads):
                y = jnp.dot(
                    x[:, h * head_dim : (h + 1) * head_dim],
                    h_ref[...],
                    preferred_element_type=jnp.float32,
                )
                o_ref[h] = (y * (1.0 / head_dim)).astype(o_ref.dtype)

    def half(other):
        return pl.BlockSpec(
            (step, None, block, heads * head_dim),
            lambda i, r: (i, (r[0] + other) % 2, 0, 0),
        )

    out = pl.BlockSpec((heads, rows, head_dim), lambda i, r: (0, i, 0))
    shape = jax.ShapeDtypeStruct((heads, num_tokens // 2, head_dim), q.dtype)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(pl.cdiv(pairs, step),),
            in_specs=[
                half(0),
                half(1),
                pl.BlockSpec((head_dim, head_dim), lambda i, r: (0, 0)),
            ],
            out_specs=[out, out],
        ),
        out_shape=[shape, shape],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_rotate_replica_rows",
    )(
        jnp.reshape(rank, (1,)).astype(jnp.int32),
        q.reshape(pairs, 2, block, heads * head_dim),
        q.reshape(pairs, 2, block, heads * head_dim),
        jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16),
    )


def unrotate_replica_rows(
    out: jax.Array, other: jax.Array, rank: jax.Array, *, interpret: bool = False
) -> jax.Array:
    """Inverse of `rotate_replica_rows` for the attention outputs, unrotated
    (`msa_fp4.unrotate_output`) as `unrotate_to_rows`: this replica's rows
    of its own heads (the first ``Hl`` of head-major ``out``, read in place)
    and the partner's rows of them (``other [Hl, T / 2, D]``) -> token-major
    ``[T, Hl * D]`` in step order. As XLA ops the replicas' halves and the
    de-interleave each re-laid the output out (c64 mixed twin j-c5ebe0af:
    ~30 us per layer)."""
    heads, half_tokens, head_dim = other.shape
    num_tokens = 2 * half_tokens
    pairs, step = _replica_grid(num_tokens)
    block = _INTERLEAVE_TOKENS
    rows = step * block

    def kernel(rank_ref, own_ref, other_ref, h_ref, o_ref):
        j = rank_ref[0] % 2
        for x_ref, pos in ((own_ref, j), (other_ref, 1 - j)):
            for h in range(heads):
                y = jnp.dot(x_ref[h], h_ref[...], preferred_element_type=jnp.float32)
                y = (y * (1.0 / head_dim)).astype(o_ref.dtype)
                o_ref[:, pos, :, h * head_dim : (h + 1) * head_dim] = y.reshape(
                    step, block, head_dim
                )

    rows_spec = pl.BlockSpec((heads, rows, head_dim), lambda i, r: (0, i, 0))
    merged = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(pl.cdiv(pairs, step),),
            in_specs=[
                rows_spec,
                rows_spec,
                pl.BlockSpec((head_dim, head_dim), lambda i, r: (0, 0)),
            ],
            out_specs=pl.BlockSpec(
                (step, 2, block, heads * head_dim), lambda i, r: (i, 0, 0, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (pairs, 2, block, heads * head_dim), other.dtype
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_unrotate_replica_rows",
    )(
        jnp.reshape(rank, (1,)).astype(jnp.int32),
        out,
        other,
        jnp.asarray(msa_fp4.hadamard(head_dim).T, jnp.bfloat16),
    )
    return merged.reshape(num_tokens, heads * head_dim)


# A gathered K/V word holds the nibble pairs of this many quarters of a
# replica's tokens (`gather_group_kv`).
_KV_QUARTERS = 4


def gather_group_kv(
    k: jax.Array,
    v: jax.Array,
    index_q: jax.Array,
    rank: jax.Array,
    *,
    axis: str,
    dcp: int,
    kv_head_replicas: int,
    interpret: bool = False,
) -> tuple[jax.Array | msa.QuantizedKV, jax.Array | None, jax.Array]:
    """Inside a ``shard_map`` over `axis`: this rank's KV head's ``k``/``v``/
    ``index_q [T, 1, D]`` -> the DCP group's KV heads' ``(k, v, index_q [T,
    G, D])`` for `msa_dcp.msa_write_and_candidates_dcp`. Rank ``i = h *
    kv_head_replicas + j`` holds KV head ``h``, as its other replicas do, so
    it sends only part ``j`` of the tokens.

    A step that quantizes its K/V in a kernel (`msa.fp4_targets`: bf16, at
    least `msa_quantize.MIN_TOKENS` rows) quantizes its own part before the
    gather and sends each K/V nibble pair as one byte, with the scales: the
    write's exact rows (`msa.QuantizedKV`, ``v`` None) for half the bytes
    of bf16 K/V rows (the bf16 gather of a c64 8K-token step: 34-106 us per
    layer, j-9364c1db / j-fa9b2cc7). In the program, unlike as a torch
    collective ahead of it, the gather can run beside work that does not
    need it (`prepare_queries`)."""
    num_tokens, _, head_dim = k.shape
    reps = kv_head_replicas
    heads = dcp // reps
    if num_tokens % reps:
        rows = jnp.concatenate([k, v, index_q], axis=1)  # [T, 3, D]
        per_rank = jax.lax.all_gather(rows, axis, axis=1, tiled=True)
        per_head = per_rank.reshape(num_tokens, dcp, 3, head_dim)[:, ::reps]
        return tuple(per_head[:, :, i] for i in range(3))
    part = num_tokens // reps
    rank = msa_dcp.rank_array(rank)[0]

    def mine(x):
        """This rank's part of the tokens of ``x [T, ...]``."""
        x = x.reshape(reps, part, *x.shape[1:])
        return jax.lax.dynamic_index_in_dim(x, rank % reps, keepdims=False)

    def per_head(x):
        """Every rank's gathered part ``[dcp, part, ...]`` -> ``[T, G, ...]``."""
        x = x.reshape(heads, reps, part, *x.shape[2:])
        return jnp.moveaxis(x, 0, 2).reshape(num_tokens, heads, *x.shape[3:])

    quantize = (
        heads > 1
        and num_tokens >= msa_quantize.MIN_TOKENS
        and k.dtype == jnp.bfloat16
        and part % (32 * _KV_QUARTERS) == 0  # whole 8-bit tiles per quarter
    )
    if not quantize:
        if num_tokens < HEAD_PARALLEL_MIN_TOKENS:
            # Decode-sized steps: [part, 3, D] rows (lane-dense rows cost
            # +4.4 us per decode layer at c64, twin j-899e03a2).
            rows = mine(jnp.concatenate([k, v, index_q], axis=1))
            per_rank = per_head(jax.lax.all_gather(rows, axis))  # [T, G, 3, D]
            return tuple(per_rank[:, :, i] for i in range(3))
        rows = jnp.concatenate([k, v, index_q], axis=-1).reshape(num_tokens, -1)
        per_rank = per_head(jax.lax.all_gather(mine(rows), axis))  # [T, G, 3 D]
        return tuple(per_rank[..., i * head_dim : (i + 1) * head_dim] for i in range(3))
    nibbles, scales = msa_quantize.quantize_kv_rows(
        mine(k), mine(v), interpret=interpret
    )  # [part, 2 D] ([K | V]), [part, 2]
    # A byte per K/V nibble pair, the bytes of four quarters of the tokens
    # in a word (lane-dense, each quarter whole), then the scales' bits: one
    # buffer.
    pairs = (nibbles[:, :head_dim] | (nibbles[:, head_dim:] << 4)).astype(jnp.uint32)
    n = part // _KV_QUARTERS
    kv_words = sum(
        pairs[b * n : (b + 1) * n] << (8 * b) for b in range(_KV_QUARTERS)
    )  # [part / 4, D]
    scale_words = jax.lax.bitcast_convert_type(scales, jnp.uint32).reshape(-1)
    scale_words = jnp.pad(scale_words, (0, -scale_words.shape[0] % head_dim))
    send = jnp.concatenate([kv_words, scale_words.reshape(-1, head_dim)])
    got = jax.lax.all_gather(send, axis)  # [dcp, rows, D]
    got_scales = jax.lax.bitcast_convert_type(
        got[:, n:].reshape(dcp, -1)[:, : 2 * part], jnp.float32
    ).reshape(dcp, part, 2)
    kv = msa.QuantizedKV(
        _unpack_kv_rows(got, part, heads, interpret=interpret),
        per_head(got_scales).reshape(num_tokens, -1),
    )
    index_q = per_head(jax.lax.all_gather(mine(index_q[:, 0]), axis))
    return kv, None, index_q


def _unpack_kv_rows(
    words: jax.Array, part: int, heads: int, *, interpret: bool = False
) -> jax.Array:
    """`gather_group_kv`'s gathered K/V words ``[dcp, >= part / 4, D]``
    (rank ``h * reps + j``'s part of KV head ``h``: byte ``b`` of word row
    ``i`` the nibble pair of its token ``b * part / 4 + i``) -> the write's
    uint8 nibble rows ``[T, 2 * G * D]`` (`msa_quantize.quantize_kv_rows`):
    as XLA ops the byte unpack and the head interleave re-laid the 8-bit
    rows out several times."""
    dcp, _, head_dim = words.shape
    reps = dcp // heads
    n = part // _KV_QUARTERS
    block = next(b for b in (512, 256, 128, 64, 32) if n % b == 0)
    per_quarter = n // block

    def kernel(w_ref, o_ref):
        shift = (8 * (pl.program_id(1) // per_quarter)).astype(jnp.uint32)
        for h in range(heads):
            pair = w_ref[h] >> shift
            for c in range(2):  # K, then V
                col = (2 * h + c) * head_dim
                o_ref[:, col : col + head_dim] = ((pair >> (4 * c)) & 0xF).astype(
                    o_ref.dtype
                )

    return pl.pallas_call(
        kernel,
        grid=(reps, part // block),
        in_specs=[
            pl.BlockSpec(
                (heads, None, block, head_dim),
                lambda j, i: (0, j, i % per_quarter, 0),
            )
        ],
        out_specs=pl.BlockSpec(
            (block, 2 * heads * head_dim), lambda j, i: (j * (part // block) + i, 0)
        ),
        out_shape=jax.ShapeDtypeStruct((reps * part, 2 * heads * head_dim), jnp.uint8),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
        name="minimax_m3_msa_dcp_unpack_kv_rows",
    )(words.reshape(heads, reps, -1, head_dim))


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


def partials_send_buffer(out: jax.Array, lse: jax.Array, dcp: int) -> jax.Array:
    """The all-to-all send buffer ``[dcp, Hl + 1, T, D]`` uint16 of
    head-major partials ``out [H, T, D]`` and ``lse [H, T]`` for
    `merge_received_partials`: block ``r`` goes to rank ``r``, its LSEs
    riding as one more head row. For the decode rows of a head-parallel
    step, whose LSE all-gather and reduce-scatter waited out late ranks
    (c64 mixed twin, j-741c6acc vs j-613b69cd: 288.6 -> 282.0 ms per step);
    a decode step keeps the reduce-scatter, which moves its partial faster
    than an all-to-all (30 vs 47 us per layer at c64 b46)."""
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
    return jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(out, jnp.uint16).reshape(
                dcp, heads, num_tokens, head_dim
            ),
            bits,
        ],
        axis=1,
    )


def merge_received_partials(
    got: jax.Array, dtype, *, interpret: bool = False
) -> jax.Array:
    """This rank's merged heads ``[T, Hl, D]`` (unrotated) from every rank's
    `partials_send_buffer` block for it, ``got [dcp, Hl + 1, T, D]``:
    `msa_dcp.merge_partials` weighs, sums in f32 and unrotates in one pass."""
    _, rows, num_tokens, head_dim = got.shape
    heads = rows - 1
    parts = jax.lax.bitcast_convert_type(got[:, :heads], dtype)
    lses = msa_dcp.u16_planes_to_words(got[:, heads], heads, jnp.float32)
    hadamard = jnp.asarray(msa_fp4.hadamard(head_dim), dtype)
    merged = msa_dcp.merge_partials(parts, lses, hadamard, interpret=interpret)
    return merged.reshape(num_tokens, heads, head_dim)


# The block exchanges' barrier semaphores (the page exchange's is 0).
_BLOCKS_COLLECTIVE_ID = 1
_GATHER_COLLECTIVE_ID = 2


class BlockExchange(NamedTuple):
    """`start_block_all_to_all`'s or `start_block_all_gather`'s blocks in
    flight, for `finish_block_exchange`."""

    out: jax.Array
    send_sem: jax.Array
    recv_sem: jax.Array
    # The start's copy of the blocks being sent: an input of the finish, so
    # XLA keeps it until the copies reading it are done.
    source: jax.Array


def start_block_all_to_all(
    x: jax.Array, rank: jax.Array, *, axis: str, dcp: int, interpret: bool = False
) -> BlockExchange:
    """``jax.lax.all_to_all(x, axis, 0, 0, tiled=True)`` of ``x [dcp, ...]``
    by remote DMA, returned in flight: block ``d`` goes to slot ``rank`` of
    rank ``d``'s output. XLA issues its own all-to-all just ahead of the
    result's first use, so one feeding work that runs much later stays
    exposed however early its operand is ready; these copies start where
    the call is ordered (by data or a barrier), and
    `finish_block_exchange` waits for them."""

    def kernel(rank_ref, x_ref, out_ref, source_ref, send_sem, recv_sem, copy_sem):
        # The copies read the blocks from this call's own output: XLA may keep
        # an operand that fits in VMEM and reuse that VMEM right after the
        # call, while the copies still read it.
        local = pltpu.make_async_copy(x_ref, source_ref, copy_sem)
        local.start()
        me = rank_ref[0]
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
        local.wait()
        for d in range(dcp):
            _remote(
                source_ref.at[d], out_ref.at[me], send_sem, recv_sem, d, axis
            ).start()

    any_ = pl.BlockSpec(memory_space=pl.ANY)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    started = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[any_],
            out_specs=(any_, any_, sem, sem),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())],
        ),
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
        ),
        # No side-effect flag: the finish consumes every output (as
        # `start_head_page_exchange`).
        compiler_params=pltpu.CompilerParams(collective_id=_BLOCKS_COLLECTIVE_ID),
        interpret=_interpret_params(interpret),
        name="minimax_m3_msa_dcp_block_all_to_all_start",
    )(rank.reshape(1), x)
    out, source, send_sem, recv_sem = started
    return BlockExchange(out, send_sem, recv_sem, source)


def start_block_all_gather(
    x: jax.Array,
    index: jax.Array,
    *,
    axes: tuple[str, str],
    shape: tuple[int, int],
    interpret: bool = False,
) -> BlockExchange:
    """``x`` gathered to every device of the ``axes`` mesh of ``shape``
    ``(lanes, ranks)`` by remote DMA, returned in flight, as
    `start_block_all_to_all` does an all-to-all: device ``(l, r)``'s output
    ``[lanes * ranks, *x.shape]`` gets it at slot ``index`` (``int32[1]``,
    this device's ``l * ranks + r``). `finish_block_exchange` waits."""
    lanes, ranks = shape
    copies = lanes * ranks

    def kernel(index_ref, x_ref, out_ref, source_ref, send_sem, recv_sem, copy_sem):
        devices = [
            {axes[0]: jnp.int32(lane), axes[1]: jnp.int32(r)}
            for lane in range(lanes)
            for r in range(ranks)
        ]
        # As `start_block_all_to_all`: the copies read this call's own output.
        local = pltpu.make_async_copy(x_ref, source_ref, copy_sem)
        local.start()
        barrier = pltpu.get_barrier_semaphore()
        for device in devices:
            pl.semaphore_signal(
                barrier, inc=1, device_id=device, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, copies)
        local.wait()
        for device in devices:
            pltpu.make_async_remote_copy(
                source_ref,
                out_ref.at[index_ref[0]],
                send_sem,
                recv_sem,
                device_id=device,
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

    any_ = pl.BlockSpec(memory_space=pl.ANY)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    out, source, send_sem, recv_sem = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[any_],
            out_specs=(any_, any_, sem, sem),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())],
        ),
        out_shape=(
            jax.ShapeDtypeStruct((copies, *x.shape), x.dtype),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=_GATHER_COLLECTIVE_ID),
        interpret=_interpret_params(interpret),
        name="minimax_m3_msa_dcp_block_all_gather_start",
    )(index.reshape(1), x)
    return BlockExchange(out, send_sem, recv_sem, source)


def finish_block_exchange(
    started: BlockExchange, copies: int, *, interpret: bool = False
) -> jax.Array:
    """The result of `start_block_all_to_all` or `start_block_all_gather`
    over ``copies`` devices: waits for this device's ``copies`` blocks and
    its own sends (equal-sized copies: any wait retires one)."""

    def kernel(buf_ref, send_ref, recv_ref, source_ref, out_ref):
        del buf_ref, source_ref  # aliased to out_ref; held for the sends
        block = out_ref.at[0]
        for sem in (recv_ref, send_ref):
            for _ in range(copies):
                pltpu.make_async_copy(block, block, sem).wait()

    any_ = pl.BlockSpec(memory_space=pl.ANY)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    return pl.pallas_call(
        kernel,
        grid=(1,),
        in_specs=[any_, sem, sem, any_],
        out_specs=any_,
        out_shape=jax.ShapeDtypeStruct(started.out.shape, started.out.dtype),
        input_output_aliases={0: 0},
        interpret=_interpret_params(interpret),
        name="minimax_m3_msa_dcp_block_exchange_finish",
    )(*started)


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
    partials=False,
):
    """The DCP attention of rows ``q [R, Hl, D]`` (this rank's rotated query
    heads) with this rank's packed ``candidates [R, G * 2 * topk]``, merged
    by `merge_partials`, or with `partials` its unmerged head-major ``(out,
    lse)`` for `partials_send_buffer`."""
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
        and not partials
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
    if partials:
        return out, lse
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


def _interpret_params(interpret: bool):
    return pltpu.InterpretParams(uninitialized_memory="zero") if interpret else False


class PageExchange(NamedTuple):
    """`start_head_page_exchange`'s pages in flight: the output buffer the
    other ranks copy into and the copies' semaphores, for
    `finish_head_page_exchange`. The forwards' receive semaphore is an
    output of the start too: a replica partner may forward before this rank
    enters its finish call, whose scratch would not be live yet."""

    out: jax.Array
    send_sem: jax.Array
    recv_sem: jax.Array
    fwd_sem: jax.Array


def _remote(src, dst, send_sem, recv_sem, device, axis):
    return pltpu.make_async_remote_copy(
        src,
        dst,
        send_sem,
        recv_sem,
        device_id={axis: device},
        device_id_type=pl.DeviceIdType.MESH,
    )


def _page_wait(sem, page):
    """A loop body that retires one page's copy on `sem` (equal-sized copies
    on one semaphore: any wait retires one)."""

    def wait(i, carry):
        pltpu.make_async_copy(page, page, sem).wait()
        return carry

    return wait


def _exchange_counts(live, rank, replicas, part):
    counts = [live_part(live, j, replicas, part) for j in range(replicas)]
    replica = rank % replicas
    mine = jnp.where(replica == 0, counts[0], counts[-1])
    for j in range(1, replicas - 1):
        mine = jnp.where(replica == j, counts[j], mine)
    return counts, mine


def start_head_page_exchange(
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
) -> PageExchange:
    """The sending half of `exchange_head_pages`: every rank starts the
    remote copies of its live pages and returns with them in flight, so the
    step's independent work (the indexer) runs while the pages move;
    `finish_head_page_exchange` waits for them. Arguments as
    `exchange_head_pages`."""
    _, rows, packing, _ = kv_cache.shape
    num_pages = pages.shape[0]
    part = num_pages // replicas
    if kv_cache.dtype != jnp.uint8:
        raise ValueError(f"4-bit (uint8) caches only, got {kv_cache.dtype}")

    def kernel(
        pages_ref, live_ref, rank_ref, kv_ref, out_ref, send_sem, recv_sem, fwd_sem
    ):
        del fwd_sem  # signalled by the partner's forwards (finish)
        me = rank_ref[0]
        counts, _ = _exchange_counts(live_ref[0], me, replicas, part)
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

        def send(i, carry):
            for d in range(dcp):
                head, j = divmod(d, replicas)

                def go(d=d, head=head, j=j):
                    columns = pl.ds(head * head_width, head_width)
                    src = kv_ref.at[pages_ref[j * part + i], :, :, columns]
                    dst = out_ref.at[j, me, i]
                    _remote(src, dst, send_sem, recv_sem, d, axis).start()

                pl.when(i < counts[j])(go)
            return carry

        jax.lax.fori_loop(0, jnp.max(jnp.stack(counts)), send, 0)

    any_ = pl.BlockSpec(memory_space=pl.ANY)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    started = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[any_],
            out_specs=(any_, sem, sem, sem),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (replicas, dcp, part, rows, packing, head_width), kv_cache.dtype
            ),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
        ),
        # No side-effect flag: the finish consumes every output, and torch_tpu's
        # SPMD partitioner refuses side-effect custom calls without a sharding.
        compiler_params=pltpu.CompilerParams(collective_id=_EXCHANGE_COLLECTIVE_ID),
        # Semaphores that outlive a call: interpret mode zero-initializes them.
        interpret=_interpret_params(interpret),
        name="minimax_m3_msa_dcp_exchange_head_pages_start",
    )(pages, live, rank, kv_cache)
    return PageExchange(*started)


def finish_head_page_exchange(
    started: PageExchange,
    pages: jax.Array,
    live: jax.Array,
    rank: jax.Array,
    *,
    axis: str,
    dcp: int,
    replicas: int,
    interpret: bool = False,
) -> jax.Array:
    """The receiving half of `exchange_head_pages`: waits for this rank's
    pages of `started`, forwards them to the head's other replicas (one
    chip's cores at c64), waits for theirs and for its own sends."""
    num_pages = pages.shape[0]
    part = num_pages // replicas

    def kernel(live_ref, rank_ref, buf_ref, send_ref, recv_ref, fwd_sem, out_ref):
        del buf_ref  # aliased to out_ref
        me = rank_ref[0]
        replica = me % replicas
        counts, mine = _exchange_counts(live_ref[0], me, replicas, part)
        page = out_ref.at[0, 0, 0]
        jax.lax.fori_loop(0, mine * dcp, _page_wait(recv_ref, page), 0)

        # Forward the received pages to the head's other replicas, a bounded
        # number in flight.
        def forwards(i, op):
            for k in range(1, replicas):
                peer = me - replica + (replica + k) % replicas
                for src in range(dcp):
                    p = out_ref.at[replica, src, i]
                    op(_remote(p, p, send_ref, fwd_sem, peer, axis))

        ahead = _EXCHANGE_PAGES_IN_FLIGHT

        def step(i, carry):
            forwards(i, lambda c: c.start())
            pl.when(i >= ahead)(lambda: forwards(i - ahead, lambda c: c.wait_send()))
            return carry

        def drain(i, carry):
            forwards(i, lambda c: c.wait_send())
            return carry

        jax.lax.fori_loop(0, mine, step, 0)
        jax.lax.fori_loop(jnp.maximum(mine - ahead, 0), mine, drain, 0)
        theirs = sum(counts) - mine
        jax.lax.fori_loop(0, theirs * dcp, _page_wait(fwd_sem, page), 0)
        # This rank's own sends (one per live page and destination).
        sent = sum(counts[j] * (dcp // replicas) for j in range(replicas))
        jax.lax.fori_loop(0, sent, _page_wait(send_ref, page), 0)

    any_ = pl.BlockSpec(memory_space=pl.ANY)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[any_, sem, sem, sem],
            out_specs=any_,
        ),
        out_shape=jax.ShapeDtypeStruct(started.out.shape, started.out.dtype),
        input_output_aliases={2: 0},
        # Semaphores that outlive a call: interpret mode zero-initializes them.
        interpret=_interpret_params(interpret),
        name="minimax_m3_msa_dcp_exchange_head_pages_finish",
    )(live, rank, *started)


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
    started=None,
):
    """Every rank's ``pages [L]`` of the KV head each rank owns:
    ``(kv [L * dcp, S / 8, 4, 2 * D], scales [L * dcp, SCALE_ROWS, 128])``,
    rank ``src``'s page ``pages[j * L / replicas + i]`` at slot ``(j * dcp +
    src) * L / replicas + i``; only the pages `exchange_head_pages` calls
    live (`live` logical blocks) are set.

    The pages move by `exchange_head_pages`. The ``replicas`` ranks of a KV
    head each receive a ``1 / replicas`` part of the scales over an
    all-to-all and share them with an all-gather inside the replica group
    (one chip's cores at c64). `started` (`start_head_page_exchange` of
    these `pages` and `live`) has the pages in flight already."""
    num_pages = pages.shape[0]
    part = num_pages // replicas
    _, rows, packing, _ = kv_cache.shape
    heads = dcp // replicas
    if kv_cache.shape[3] != heads * head_width:
        raise ValueError(
            f"cache rows {kv_cache.shape} do not hold {heads} KV heads of "
            f"{head_width} columns"
        )
    common = dict(axis=axis, dcp=dcp, replicas=replicas, interpret=interpret)
    if started is None:
        recv = exchange_head_pages(
            kv_cache,
            pages,
            live.reshape(1),
            rank.reshape(1),
            head_width=head_width,
            **common,
        )
    else:
        recv = finish_head_page_exchange(
            started, pages, live.reshape(1), rank.reshape(1), **common
        )
    # Destination rank d = h * replicas + j gets part j of head h.
    # K/V rows of every head, gathered as rows of the pages' tiles.
    scale_rows = scale_cache.shape[1]
    row = pages[:, None] * scale_rows + jnp.arange(2 * heads, dtype=jnp.int32)
    sc = scale_cache.reshape(-1, scale_cache.shape[-1])[row.reshape(-1)]
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
    are 1/8 the candidates' bytes. With `share_only` (two replicas that
    take the alternate blocks of `rotate_replica_rows`), the replica's own
    share ``[T / 2, topk]`` comes back instead, with no all-gather."""
    num_tokens = candidates.shape[0]
    heads = dcp // replicas
    if share_only:
        # Destination rank d = h * 2 + j gets head h's blocks of replica j,
        # in one transpose.
        blocks = candidates.reshape(-1, 2, _INTERLEAVE_TOKENS, heads, 2 * topk)
        send = blocks.transpose(3, 1, 0, 2, 4).reshape(dcp, num_tokens // 2, -1)
        own = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)
        return msa_dcp.merge_head_candidates(own, topk, interpret)[:, 0]
    split = replicas if num_tokens % replicas == 0 else 1
    part = num_tokens // split
    per_head = candidates.reshape(num_tokens, heads, 2 * topk).transpose(1, 0, 2)
    # Destination rank d = h * replicas + j gets head h's tokens of share j
    # (every share, when the tokens do not split).
    send = per_head.reshape(heads, split, part, 2 * topk)
    send = jnp.repeat(send, replicas // split, axis=1).reshape(dcp, part, 2 * topk)
    own = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)  # [dcp, T / split, 2K]
    selected = msa_dcp.merge_head_candidates(own, topk, interpret)[:, 0]
    if split > 1:
        groups = [list(range(h * replicas, (h + 1) * replicas)) for h in range(heads)]
        selected = jax.lax.all_gather(
            selected, axis, axis_index_groups=groups, tiled=True
        )
    return selected


def _plan_round(first, hp_req, blocks, capacity):
    """The requests of `_head_parallel_attend`'s round from request `first`
    on: ``(csum, in_round, base, total, next first)``."""
    max_num_seqs = hp_req.shape[0]
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    csum = jnp.cumsum(jnp.where(seq_idx >= first, blocks, 0))
    # A request never exceeds the buffer (the block table's width); always
    # admitting `first` keeps the rounds moving if that ever broke.
    in_round = hp_req & (seq_idx >= first) & ((csum <= capacity) | (seq_idx == first))
    base = csum - blocks  # the request's first logical block in the buffer
    total = jnp.max(jnp.where(in_round, csum, 0))
    nxt = jnp.min(
        jnp.where((seq_idx >= first) & ~in_round & hp_req, seq_idx, max_num_seqs)
    )
    return csum, in_round, base, total, nxt


def _round_pages(plan, block_tables_2d, replicas):
    """The page buffer of a round: buffer page b holds logical block ``l = b
    % part * replicas + b // part``: logical block ``s - base[q]`` of request
    ``q`` (0 past the round's blocks)."""
    csum, _, base, total, _ = plan
    max_num_seqs, capacity = block_tables_2d.shape
    part = -(-capacity // replicas)
    buf = jnp.arange(part * replicas, dtype=jnp.int32)
    logical = buf % part * replicas + buf // part
    q_of = jnp.searchsorted(csum, logical, side="right", method="compare_all")
    q_of = jnp.minimum(q_of, max_num_seqs - 1)
    j = jnp.clip(logical - base[q_of], 0, capacity - 1)
    return jnp.where(logical < total, block_tables_2d[q_of, j], 0)


def _request_classes(cu_q_lens, distribution, decode_rows):
    """``(dcp_req, hp_req)``: the step's live decode-sized requests (at most
    `decode_rows` rows) and the longer ones `attend_and_merge` attends
    head-parallel."""
    max_num_seqs = cu_q_lens.shape[0] - 1
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    live = (seq_idx < distribution[2]) & (q_lens > 0)
    return live & (q_lens <= decode_rows), live & (q_lens > decode_rows)


def head_parallel_step(
    num_tokens: int,
    max_num_seqs: int,
    decode_rows: int,
    min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
) -> bool:
    """Whether `attend_and_merge` attends a step of `num_tokens` rows' long
    requests head-parallel (else every row takes the DCP attention)."""
    return num_tokens >= max(min_tokens, 2 * max_num_seqs * decode_rows)


def start_first_round_exchange(
    kv_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    rank: jax.Array,
    *,
    num_tokens: int,
    axis: str,
    dcp: int,
    kv_head_replicas: int,
    decode_rows: int,
    head_parallel_min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
    interpret: bool = False,
) -> PageExchange | None:
    """The page exchange of `attend_and_merge`'s first head-parallel round,
    started (`start_head_page_exchange`) as soon as the step's rows are in
    the cache: it depends on the block tables only, and in the c64 mixed
    twin it ran exposed after the indexer (135 us per layer, j-c5ebe0af).
    None for a step `attend_and_merge` attends without head-parallel rounds.
    Pass the result to `attend_and_merge` as ``page_exchange``."""
    max_num_seqs = seq_lens.shape[0]
    if not head_parallel_step(
        num_tokens, max_num_seqs, decode_rows, head_parallel_min_tokens
    ):
        return None
    rank = msa_dcp.rank_array(rank)[0]
    _, hp_req = _request_classes(cu_q_lens, distribution, decode_rows)
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    blocks = jnp.where(hp_req, -(-seq_lens // (msa.page_size(kv_cache) * dcp)), 0)
    first = jnp.min(
        jnp.where(hp_req, jnp.arange(max_num_seqs, dtype=jnp.int32), max_num_seqs)
    )
    plan = _plan_round(first, hp_req, blocks, block_tables_2d.shape[1])
    return start_head_page_exchange(
        kv_cache,
        _round_pages(plan, block_tables_2d, kv_head_replicas),
        plan[3].reshape(1),
        rank.reshape(1),
        axis=axis,
        dcp=dcp,
        replicas=kv_head_replicas,
        head_width=kv_cache.shape[3] // (dcp // kv_head_replicas),
        interpret=interpret,
    )


def _head_parallel_attend(
    queries,
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
    interpret,
    first_exchange=None,
    alongside=None,
):
    """Token-major ``[T, Hl * D]`` attention of this rank's query heads
    (`StepQueries`, ``rows`` or ``pair``) for the rows of the requests in
    ``hp_req`` over their pages of this rank's KV head; other rows are zero.
    ``tokens`` is `msa.token_positions` of the step.

    Requests are attended in rounds of at most ``pages_per_seq`` logical
    blocks (one max-length request): each round exchanges the live pages
    of one page buffer and attends the round's rows in windows of one
    kernel call. `first_exchange` (`start_first_round_exchange`) has the
    first round's pages in flight. `alongside` (`attend_and_merge`) starts
    right before the first round's attention, after its page exchange and
    selection, whose collectives its copies would delay; the result is then
    ``(out, in flight)``."""
    num_tokens = tokens[0].shape[0]
    head_dim = (queries.rows if queries.pair is None else queries.pair[0]).shape[-1]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    capacity = block_tables_2d.shape[1]  # logical blocks per round
    replicas = kv_head_replicas
    part = -(-capacity // replicas)  # each replica's share of the buffer
    split = queries.pair is not None
    rows = num_tokens // replicas if split else num_tokens
    offset = rank % replicas * rows if split else 0
    pair = [(r, r ^ 1) for r in range(dcp)]
    if split:
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
        own, other = queries.pair
        # Two arrays the attention reads in place: no concatenated copy.
        q = (own, jax.lax.ppermute(other, axis, pair))
    else:
        q = queries.rows
    seq_blocks = -(-seq_lens // (block_size * dcp))
    blocks = jnp.where(hp_req, seq_blocks, 0)
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    # Rows per kernel call (the caller pads tokens to a block_q multiple).
    window = min(rows, msa_attention.MAX_TOKENS_PER_CALL)
    sel_block = jnp.maximum(selected, 0)
    own_block = pos // block_size
    limit = pos % block_size + 1

    def plan_round(first):
        return _plan_round(first, hp_req, blocks, capacity)

    more = None

    def attend_round(plan, acc, merge, started=None):
        nonlocal more
        _, in_round, base, total, _ = plan
        num_pages = part * replicas
        pages = _round_pages(plan, block_tables_2d, replicas)
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
            started=started,
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
            msa_attention_lse.msa_sparse_attention_lse,
            sm_scale=sm_scale,
            block_q=_BLOCK_Q,
            pages_per_step=_SPLIT_PAGES_PER_STEP if split else _PAGES_PER_STEP,
            head=0,
            scale_cache=sc,
            k_scale_row=0,
            v_scale_row=1,
            head_major=True,
            q_head_major=split,
            interpret=interpret,
        )
        if not merge:
            if alongside is not None:
                more, kv, keys = jax.lax.optimization_barrier(
                    (alongside((kv, keys)), kv, keys)
                )
            # One call over every row: the kernel chunks the rows and writes
            # its output in place; rows outside the round see no page and
            # come back zero (cheap empty tiles).
            out, _ = attend(q, kv, keys, local, limit, page_table)
            return out

        def window_body(i, acc):
            start = jnp.minimum(lo + i * window, rows - window)

            def sl(x, axis=0):
                return jax.lax.dynamic_slice_in_dim(x, start, window, axis)

            q_rows = jax.tree.map(lambda x: sl(x, 1 if split else 0), q)
            out, _ = attend(q_rows, kv, sl(keys), sl(local), sl(limit), page_table)
            old = jax.lax.dynamic_slice_in_dim(acc, start, window, 1)
            out = jnp.where(sl(mine)[None, :, None], out, old)  # earlier rounds
            return jax.lax.dynamic_update_slice_in_dim(acc, out, start, 1)

        return jax.lax.fori_loop(0, -(-(hi - lo) // window), window_body, acc)

    first = jnp.min(jnp.where(hp_req, seq_idx, max_num_seqs))
    plan = plan_round(first)
    acc = attend_round(plan, None, merge=False, started=first_exchange)
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
        heads = queries.pair[0].shape[0]
        other = jax.lax.ppermute(acc[heads:], axis, pair)
        out = unrotate_replica_rows(acc, other, rank, interpret=interpret)
    else:
        out = unrotate_to_rows(acc, interpret=interpret)
    return out if alongside is None else (out, more)


class StepQueries(NamedTuple):
    """A step's query heads, rotated (`msa_fp4.rotate_queries`) in the
    layouts `attend_and_merge` reads them (`prepare_queries`)."""

    # [T, Hl, D]: every row, for a step without head-parallel rounds or
    # whose rows the replicas do not split.
    rows: jax.Array | None
    # `rotate_replica_rows`: (own, partner's) [Hl, T / 2, D].
    pair: tuple[jax.Array, jax.Array] | None
    # [max_num_seqs * decode_rows, Hl, D]: a head-parallel step's decode rows.
    decode: jax.Array | None


def _decode_rows(cu_q_lens, distribution, decode_rows):
    """A head-parallel step's decode requests' rows, compacted: ``(cu_d,
    src_row, prefix)``, their own cu_q_lens, each row's step row, and whether
    they are the step's first rows (the runner orders decodes ahead of
    prefills)."""
    max_num_seqs = cu_q_lens.shape[0] - 1
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    dcp_req, _ = _request_classes(cu_q_lens, distribution, decode_rows)
    cu_d = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(jnp.where(dcp_req, q_lens, 0))]
    ).astype(jnp.int32)
    p = jnp.arange(max_num_seqs * decode_rows, dtype=jnp.int32)
    # compare_all: the default scan lowers to a loop of small ops.
    req = jnp.searchsorted(cu_d[1:], p, side="right", method="compare_all")
    req = jnp.minimum(req, max_num_seqs - 1)
    src_row = jnp.where(p < cu_d[-1], cu_q_lens[req] + p - cu_d[req], 0)
    prefix = jnp.all((p >= cu_d[-1]) | (src_row == p))
    return cu_d, src_row, prefix


def _split_rows(num_tokens: int, kv_head_replicas: int) -> bool:
    """Whether the two replicas of a KV head split a head-parallel step's
    rows (`rotate_replica_rows`): each attends alternate blocks of the tokens
    with all the head's query heads, its own first."""
    return kv_head_replicas == 2 and num_tokens % (2 * _INTERLEAVE_TOKENS) == 0


def prepare_queries(
    q: jax.Array,
    rank: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    kv_head_replicas: int,
    decode_rows: int,
    head_parallel_min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
    interpret: bool = False,
) -> StepQueries:
    """`attend_and_merge`'s rotated query heads (`StepQueries`) from this
    rank's ``q [T, Hl, D]``: they need no cache, so a step can compute them
    beside its K/V gather (`gather_group_kv`), which in the c64 mixed twin
    otherwise waited exposed (j-fa9b2cc7: 106 us per layer)."""
    num_tokens = q.shape[0]
    max_num_seqs = cu_q_lens.shape[0] - 1
    if not head_parallel_step(
        num_tokens, max_num_seqs, decode_rows, head_parallel_min_tokens
    ):
        return StepQueries(msa_fp4.rotate_queries(q), None, None)
    _, src_row, prefix = _decode_rows(cu_q_lens, distribution, decode_rows)
    rows = src_row.shape[0]
    # The decode rows' queries by a slice when they are the step's first: a
    # gather of `rows` rows of q cost ~16 us per layer (c64 mixed twin
    # j-c5ebe0af). Rows past the decodes' (attended, then dropped) differ.
    # The cond takes the rope's lane-dense rows: a [T, Hl, D] operand cost a
    # relayout of all of q.
    decode = jax.lax.cond(
        prefix, lambda x: x[:rows], lambda x: x[src_row], q.reshape(num_tokens, -1)
    ).reshape(rows, *q.shape[1:])
    decode = msa_fp4.rotate_queries(decode)
    if _split_rows(num_tokens, kv_head_replicas):
        rank = msa_dcp.rank_array(rank)[0]
        pair = rotate_replica_rows(q, rank, interpret=interpret)
        return StepQueries(None, pair, decode)
    return StepQueries(msa_fp4.rotate_queries(q), None, decode)


def attend_and_merge(
    rank: jax.Array,
    q: jax.Array | StepQueries,
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
    page_exchange: PageExchange | None = None,
    alongside: Callable[[jax.Array], Any] | None = None,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, Any]:
    """This rank's (``rank``, ``int32[1]``) attention output ``[T, Hl * D]``
    (unrotated) for its query heads ``q [T, Hl, D]`` (unrotated: this op
    rotates them, `msa_fp4.rotate_queries`; or `prepare_queries` of them)
    and its packed ``candidates [T, G * 2 * topk]``
    (`msa_dcp.msa_write_and_candidates_dcp`); 4-bit caches. `page_exchange`:
    `start_first_round_exchange` of this step. `alongside(ready)` starts
    work (a `BlockExchange`) to keep under the attention, once ``ready`` is
    computed (the step's last work ahead of the attention: its copies must
    not delay that work's collectives); it is returned after the attention:
    the result is then ``(out, in flight)``.
    Runs inside a ``shard_map`` over `axis` (the DCP group)."""
    num_tokens = candidates.shape[0]
    # The rank is an input: torch_tpu's SPMD partitioner refuses the
    # partition-id that `lax.axis_index` lowers to.
    rank = msa_dcp.rank_array(rank)[0]
    if not isinstance(q, StepQueries):
        q = prepare_queries(
            q,
            rank,
            cu_q_lens,
            distribution,
            kv_head_replicas=kv_head_replicas,
            decode_rows=decode_rows,
            head_parallel_min_tokens=head_parallel_min_tokens,
            interpret=interpret,
        )
    common = dict(
        axis=axis, rank=rank, dcp=dcp, sm_scale=sm_scale, topk=topk, interpret=interpret
    )
    if q.decode is None:  # no head-parallel rounds
        in_flight = None if alongside is None else alongside(candidates)
        in_flight, q = jax.lax.optimization_barrier((in_flight, q))
        out = _dcp_attend(
            q.rows,
            kv_cache,
            scale_cache,
            candidates,
            seq_lens,
            block_tables,
            cu_q_lens,
            rs_out_precision=rs_out_precision,
            **common,
        ).reshape(num_tokens, -1)
        out, in_flight = jax.lax.optimization_barrier((out, in_flight))
        return out if in_flight is None else (out, in_flight)
    _, hp_req = _request_classes(cu_q_lens, distribution, decode_rows)
    cu_d, src_row, prefix = _decode_rows(cu_q_lens, distribution, decode_rows)
    rows = src_row.shape[0]
    dcp_partials = _dcp_attend(
        q.decode,
        kv_cache,
        scale_cache,
        candidates[src_row],
        seq_lens,
        block_tables,
        cu_d,
        rs_out_precision=rs_out_precision,
        partials=True,
        **common,
    )
    # The decode rows attend first and their merge's all-to-all crosses
    # under the head-parallel attention, which never reads it: as XLA ops
    # they ran after it, the all-to-all and gathers exposed (~140 us per
    # layer, c64 mixed twin j-723bdab2).
    send = partials_send_buffer(*dcp_partials, dcp)
    in_flight = start_block_all_to_all(
        send, rank, axis=axis, dcp=dcp, interpret=interpret
    )
    in_flight, q = jax.lax.optimization_barrier((in_flight, q))
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
        first_exchange=page_exchange,
        alongside=alongside,
        **common,
    )
    hp_out, more = hp_out if alongside is not None else (hp_out, None)
    hp_out, in_flight, more = jax.lax.optimization_barrier((hp_out, in_flight, more))
    got = finish_block_exchange(in_flight, dcp, interpret=interpret)
    dcp_out = merge_received_partials(
        got, dcp_partials[0].dtype, interpret=interpret
    ).reshape(rows, -1)
    p = jnp.arange(rows, dtype=jnp.int32)
    num_decode = cu_d[-1]

    def decode_prefix(out):
        # One in-place update of `rows` rows.
        head = jnp.where((p < num_decode)[:, None], dcp_out, out[:rows])
        return jax.lax.dynamic_update_slice_in_dim(out, head, 0, 0)

    def scatter(out):
        # The decode rows go back to their places (the others are dropped):
        # a scatter of `rows` rows.
        dest = jnp.where(p < num_decode, src_row, num_tokens)
        return out.at[dest].set(dcp_out, mode="drop")

    # The barrier keeps the consumer's ops out of the branches: XLA sank the
    # output projection's f32 convert of the whole output into them (a 32 MB
    # f32 copy, ~20 us per layer at c64's 8K-token steps, twin j-a202588d).
    out = jax.lax.optimization_barrier(
        jax.lax.cond(prefix, decode_prefix, scatter, hp_out)
    )
    return out if more is None else (out, more)
