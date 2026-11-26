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
"""Torch entry point for the JAX MSA forward."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import torch
from torch_tpu._internal import pallas
from vllm.distributed import get_tp_group

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_dcp,
    msa_dcp_heads,
    msa_fp4,
    msa_indexer,
)

_KV_CACHE_ARG = 5
_INDEX_CACHE_ARG = 6
_SHARD_KV_CACHE_ARG = 4
_SHARD_INDEX_CACHE_ARG = 5
# 4-bit caches (`msa_fp4`): the scale cache follows the index cache.
_SCALE_CACHE_ARG = 7
_DCP_SCALE_CACHE_ARG = 6


@functools.cache
def get_msa_op(
    sm_scale: float, topk: int, init_blocks: int, local_blocks: int, fp4: bool = False
) -> Callable[..., torch.Tensor]:
    """Build (once per config) the op that runs `msa.msa_forward` in place.

    Built eagerly from module `__init__`: `pallas.jax_op` inspects the
    function signature, which Dynamo cannot trace inside a compiled forward.
    With `fp4` (4-bit caches, `msa_fp4`) the op also takes and updates the
    layer's ``scale_cache``.
    """
    if fp4:
        return _get_msa_fp4_op(sm_scale, topk, init_blocks, local_blocks)

    def _msa_jax(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        index_q: jax.Array,
        index_k: jax.Array,
        kv_cache: jax.Array,
        index_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return msa.msa_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            sm_scale=sm_scale,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )

    op_name = (
        f"pallas::minimax_m3_msa_t{topk}_i{init_blocks}_l{local_blocks}"
        f"_s{sm_scale:.8f}".replace(".", "p")
    )
    jax_op = pallas.jax_op(
        op_name, _msa_jax, donate_argnums=(_KV_CACHE_ARG, _INDEX_CACHE_ARG)
    )

    def _fake(q, k, v, index_q, index_k, kv_cache, index_cache, *args):
        return (
            torch.empty_like(kv_cache),
            torch.empty_like(index_cache),
            (torch.empty_like(q)),
        )

    jax_op.register_fake(_fake)

    def op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        new_kv_cache, new_index_cache, out = jax_op(
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            metadata.request_distribution,
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        return out

    return op


def _get_msa_fp4_op(
    sm_scale: float, topk: int, init_blocks: int, local_blocks: int
) -> Callable[..., torch.Tensor]:
    def _msa_jax(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        index_q: jax.Array,
        index_k: jax.Array,
        kv_cache: jax.Array,
        index_cache: jax.Array,
        scale_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return msa.msa_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            sm_scale=sm_scale,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            scale_cache=scale_cache,
        )

    op_name = (
        f"pallas::minimax_m3_msa_fp4_t{topk}_i{init_blocks}_l{local_blocks}"
        f"_s{sm_scale:.8f}".replace(".", "p")
    )
    jax_op = pallas.jax_op(
        op_name,
        _msa_jax,
        donate_argnums=(_KV_CACHE_ARG, _INDEX_CACHE_ARG, _SCALE_CACHE_ARG),
    )

    def _fake(q, k, v, index_q, index_k, kv_cache, index_cache, scale_cache, *args):
        return (
            torch.empty_like(kv_cache),
            torch.empty_like(index_cache),
            torch.empty_like(scale_cache),
            torch.empty_like(q),
        )

    jax_op.register_fake(_fake)

    def op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        metadata: AttentionMetadata,
        *,
        scale_cache: torch.Tensor,
    ) -> torch.Tensor:
        new_kv_cache, new_index_cache, new_scale_cache, out = jax_op(
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            scale_cache,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            metadata.request_distribution,
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        scale_cache.copy_(new_scale_cache)
        return out

    return op


@functools.cache
def get_msa_sharded_op(
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: int,
    shards: int,
) -> Callable[..., torch.Tensor]:
    """`get_msa_op` for an index cache sharded over the tensor-parallel group.

    Two JAX ops around one all-to-all: the first writes K/V and this rank's
    index keys and scores every index head on them; the all-to-all brings
    each rank the per-rank partials of its own head, whose max is the exact
    block score; the second selects blocks and attends. ``index_q`` passed
    to the op holds all ``shards`` heads.
    """

    def _score_jax(
        k: jax.Array,
        v: jax.Array,
        index_q: jax.Array,
        index_k: jax.Array,
        kv_cache: jax.Array,
        index_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return msa.msa_write_and_score_sharded(
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            rank=rank,
            shards=shards,
        )

    def _attend_jax(
        q: jax.Array,
        kv_cache: jax.Array,
        scores: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
    ) -> jax.Array:
        return msa.msa_attend_sharded(
            q,
            kv_cache,
            scores,
            seq_lens,
            block_tables,
            query_start_loc,
            sm_scale=sm_scale,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )

    suffix = (
        f"t{topk}_i{init_blocks}_l{local_blocks}_s{sm_scale:.8f}_r{rank}of{shards}"
    ).replace(".", "p")
    score_op = pallas.jax_op(
        f"pallas::minimax_m3_msa_shard_score_{suffix}",
        _score_jax,
        donate_argnums=(_SHARD_KV_CACHE_ARG, _SHARD_INDEX_CACHE_ARG),
    )
    attend_op = pallas.jax_op(
        f"pallas::minimax_m3_msa_shard_attend_{suffix}", _attend_jax
    )

    def _score_fake(
        k, v, index_q, index_k, kv_cache, index_cache, seq_lens, block_tables, *args
    ):
        # block_scores pads both tile classes to the wider page-group multiple.
        pages_per_seq = block_tables.shape[0] // seq_lens.shape[0]
        padded_pages = max(
            -(-pages_per_seq // step) * step
            for step in (
                msa_indexer.DEFAULT_PAGES_PER_STEP,
                msa_indexer.DEFAULT_LARGE_PAGES_PER_STEP,
                msa_indexer.DEFAULT_SMALL_PAGES_PER_STEP,
            )
        )
        return (
            torch.empty_like(kv_cache),
            torch.empty_like(index_cache),
            torch.empty(
                (shards, k.shape[0], padded_pages), dtype=torch.float32, device=k.device
            ),
        )

    score_op.register_fake(_score_fake)
    attend_op.register_fake(lambda q, *args: torch.empty_like(q))

    def op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        new_kv_cache, new_index_cache, partial = score_op(
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            metadata.request_distribution,
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        # Row r of the result came from rank r: its partial for this head.
        scores = _all_to_all(partial, get_tp_group()).amax(dim=0)
        return attend_op(
            q,
            kv_cache,
            scores[:, None, :],
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
        )

    return op


def _head_major_attend_fake(q, *args):
    """Shapes of a DCP attend op's head-major ``(out [H, T, D], lse [H, T])``
    for token-major ``q [T, H, D]`` (`msa_dcp.msa_attend_dcp`)."""
    num_tokens, num_heads, head_dim = q.shape
    return (
        torch.empty((num_heads, num_tokens, head_dim), dtype=q.dtype, device=q.device),
        torch.empty((num_heads, num_tokens), dtype=torch.float32, device=q.device),
    )


def _rotate_heads(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """``x @ H / n`` over the head dim, for the bf16 Sylvester ``hadamard``:
    `msa_fp4.rotate_queries` (and, ``H`` being symmetric,
    `msa_fp4.unrotate_output`), bit for bit (the bf16 dot accumulates in f32;
    the scale is a power of two)."""
    return torch.matmul(x, hadamard) * (1.0 / x.shape[-1])


def gather_query_heads(q: torch.Tensor, group: Any) -> torch.Tensor:
    """The DCP group's query heads ``[T, dcp * q_heads, D]``, gathered as
    they leave the rope: any op on `q` before the gather (a head-major
    transpose, a 4-bit rotation of this rank's heads) made XLA re-lay out
    the qkv / rope fusions that produce it (+0.25-0.3 ms per layer at 8K
    tokens, c48 / c64 twins j-4371707e, j-65bccede, j-c5c4bdfc); the
    attention op rotates the gathered queries itself."""
    return group.all_gather(q.contiguous(), dim=1)


def merge_head_partials(
    out: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    hadamard: torch.Tensor | None = None,
) -> torch.Tensor:
    """`cp_mla_attention.merge_lse_partials_scatter_heads` for head-major
    partials ``out [H, T, D]``, ``lse [H, T]``: the reduce-scatter keeps
    this rank's heads along the leading axis, and only that ``1 / DCP``
    share is transposed back to ``[T, H / DCP, D]`` (and, with a 4-bit
    cache's `hadamard`, unrotated: the rotation is linear, so the merged sum
    unrotates as its terms would, up to bf16 rounding order). Decode-sized
    steps of the 8/16-bit op merge over one all-to-all instead
    (`merge_head_partials_a2a`); `TPU_DCP_LSE_MERGE_A2A` (the MLA merge's
    switch) does not apply here."""
    # The LSEs cross the all-gather token-major ([T, H], as the MLA merge
    # sends them): gathered head-major they took 2x as long (c48 / c64
    # twins, j-fe812c28, j-c5c4bdfc).
    lse = lse.to(torch.float32).transpose(0, 1).contiguous()
    lses = group.all_gather(lse.unsqueeze(0), dim=0)
    weight = torch.exp(lse - torch.logsumexp(lses, dim=0)).transpose(0, 1)
    weight = weight.unsqueeze(-1)
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weighted = out.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    if envs.TPU_DCP_RS_OUT_PRECISION:
        weighted = weighted.to(out.dtype)
    merged = group.reduce_scatter(weighted.contiguous(), dim=0).to(out.dtype)
    if hadamard is not None:
        merged = _rotate_heads(merged, hadamard)
    return merged.transpose(0, 1)


# Steps of at most this many tokens (decode batches) exchange data in fewer,
# packed collectives: there each collective costs its ~2-3 us latency, not
# its bytes (`exchange_queries_and_candidates`, `merge_head_partials_a2a`).
PACKED_EXCHANGE_MAX_TOKENS = 64


def _halves(x: torch.Tensor) -> torch.Tensor:
    """16-bit `x` as int16, reinterpreted in place: a collective copies
    integers bit for bit, while a float collective may flush or canonicalize
    subnormal and NaN bit patterns. (An int32 view of bf16 pairs lowered to
    strided slices of the operand, 1.4 us per layer at 8 tokens, c4 twin
    j-42ceea6d.)"""
    return x.contiguous().view(torch.int16)


def _f32_halves(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """f32 ``[..., n]`` as int16 ``[..., 2n]``: every high half, then every
    low half, along `dim` (`_f32_from_halves`). An int16 view would
    interleave them, and joining interleaved halves back lowered to strided
    slices and relayout copies (~0.9 us per layer at 8 tokens, c4 twin
    j-73e59e3d); shifts and masks of whole blocks fuse into their
    neighbours."""
    words = x.contiguous().view(torch.int32)
    # Both halves sign-extended into int16 range, so the narrowing is exact.
    low = (words << 16) >> 16
    return torch.cat([(words >> 16).to(torch.int16), low.to(torch.int16)], dim=dim)


def _f32_from_halves(halves: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """f32 from `_f32_halves`' blocks of high, then low halves along `dim`
    (not an interleaved int16 view)."""
    high, low = halves.to(torch.int32).chunk(2, dim=dim)
    return ((high << 16) | (low & 0xFFFF)).view(torch.float32)


def exchange_queries_and_candidates(
    q: torch.Tensor, candidates: torch.Tensor, group: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """`gather_query_heads` and the candidates' all-gather in one all-gather
    of int16 halves: ``q [T, Hq, D]`` and ``candidates [T, C]`` f32 ->
    ``(q_all [T, dcp * Hq, D], candidates [dcp, T, C])``."""
    dcp = group.world_size
    num_tokens, num_heads, head_dim = q.shape
    q_halves = _halves(q.reshape(num_tokens, num_heads * head_dim))
    packed = torch.cat([q_halves, _f32_halves(candidates)], dim=1)
    gathered = group.all_gather(packed, dim=0).view(dcp, num_tokens, -1)
    split = q_halves.shape[1]
    q_all = (
        gathered[..., :split]
        .contiguous()
        .view(q.dtype)
        .view(dcp, num_tokens, num_heads, head_dim)
        .transpose(0, 1)
        .reshape(num_tokens, dcp * num_heads, head_dim)
    )
    return q_all, _f32_from_halves(gathered[..., split:])


def pack_head_partials(
    out: torch.Tensor, lse: torch.Tensor, ranks: int
) -> torch.Tensor:
    """Head-major 16-bit ``out [H, T, D]`` and ``lse [H, T]`` as one int16
    tensor of whole ``D``-wide rows, ``[ranks, H / ranks * (T + 2 * W / D),
    D]`` (`unpack_head_partials`): block ``r`` holds rank ``r``'s heads'
    output rows, then the high and then the low halves of their lses, each
    head's padded to ``W``, ``T`` rounded up to a multiple of ``D``. As two
    lanes past each output row (``[H, T, D + 2]``), the halves cost ~0.6 us
    of relayout copies per layer at 8 tokens (b1 twin, j-f351ea81)."""
    num_tokens, head_dim = out.shape[1:]
    width = -(-num_tokens // head_dim) * head_dim
    lse = torch.nn.functional.pad(lse.to(torch.float32), (0, width - num_tokens))
    return torch.cat(
        [
            _halves(out).view(ranks, -1, head_dim),
            _f32_halves(lse.reshape(ranks, -1, head_dim), dim=1),
        ],
        dim=1,
    )


def unpack_head_partials(
    packed: torch.Tensor, num_tokens: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """`pack_head_partials`' ``out [ranks, H / ranks, T, D]`` and ``lse
    [ranks, H / ranks, T]``."""
    ranks, rows, head_dim = packed.shape
    width = -(-num_tokens // head_dim) * head_dim
    num_heads = rows // (num_tokens + 2 * width // head_dim)
    split = num_heads * num_tokens
    out = packed[:, :split].contiguous().view(dtype)
    lse = _f32_from_halves(packed[:, split:], dim=1)
    return (
        out.view(ranks, num_heads, num_tokens, head_dim),
        lse.reshape(ranks, num_heads, width)[..., :num_tokens],
    )


def merge_received_partials(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """The log-sum-exp merge of the ranks' partials ``out [R, ..., D]``,
    ``lse [R, ...]`` over their leading axis, in f32, rounded once to
    `out`'s dtype. The ranks are summed term by term: a multiply-then-sum
    reduction lowers to a dot that rounds its f32 operands to bf16."""
    weight = torch.exp(lse - torch.logsumexp(lse, dim=0, keepdim=True)).unsqueeze(-1)
    merged = None
    for r in range(out.shape[0]):
        term = out[r].to(torch.float32) * weight[r]
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        term = torch.where(weight[r] > 0, term, torch.zeros_like(term))
        merged = term if merged is None else merged + term
    return merged.to(out.dtype)


def merge_head_partials_a2a(
    out: torch.Tensor, lse: torch.Tensor, group: Any
) -> torch.Tensor:
    """`merge_head_partials` (no 4-bit rotation) over one all-to-all: rank
    ``r`` receives every rank's partial and lse of its own heads, packed
    (`pack_head_partials`), and merges them locally, instead of an lse
    all-gather and a reduce-scatter. Returns ``[T, H / DCP, D]``."""
    recv = _all_to_all(pack_head_partials(out, lse, group.world_size), group)
    parts, lses = unpack_head_partials(recv, out.shape[1], out.dtype)
    return merge_received_partials(parts, lses).transpose(0, 1)


def gather_group_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    group: Any,
    dcp: int,
    kv_head_replicas: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The DCP group's query heads ``[T, dcp * q_heads, D]``
    (`gather_query_heads`) and its KV heads' ``k``/``v``/``index_q``
    (`gather_group_kv`).

    Two gathers: only the scoring op needs the K/V and index queries, and
    the attention op needs the queries after it, so the query gather (8 of a
    rank's 11 heads at DCP 8) can run while the scoring op computes instead
    of before it, and needs no strided split of a combined gather."""
    return gather_query_heads(q, group), *gather_group_kv(
        k, v, index_q, group, dcp, kv_head_replicas
    )


def gather_group_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    group: Any,
    dcp: int,
    kv_head_replicas: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``k``/``v``/``index_q [T, dcp // kv_head_replicas, D]`` of the DCP
    group's KV heads (rank ``i`` holds KV head ``i // kv_head_replicas``, so
    every ``kv_head_replicas``-th rank supplies the next head)."""
    num_tokens, _, head_dim = k.shape
    per_rank = group.all_gather(torch.cat([k, v, index_q], dim=1).contiguous(), dim=1)
    per_head = per_rank.view(num_tokens, dcp, 3, head_dim)[:, ::kv_head_replicas]
    return tuple(per_head[:, :, i].contiguous() for i in range(3))


@functools.cache
def get_msa_dcp_op(
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    dcp: int,
    kv_head_replicas: int = 1,
    fp4: bool = False,
    decode_rows: int = 1,
) -> Callable[..., torch.Tensor]:
    """`get_msa_op` for a KV head's context split over ``dcp`` ranks
    (`kernels.msa_dcp`).

    ``decode_rows`` is the most query rows of a decode request (1 + the
    speculative tokens): with 4-bit caches and a group of several KV heads,
    longer requests may attend head-parallel (`kernels.msa_dcp_heads`).

    With `fp4` (4-bit caches, one KV head per rank) the op also takes and
    updates the layer's ``scale_cache``.

    The returned op takes this rank's DCP rank as an ``int32[1]`` device
    tensor (``dcp_rank``, a module buffer of the caller), not a Python int:
    the rank then reaches the kernels as an input, every rank lowers to the
    same program, and one compile and one cache entry serve all ranks.

    ``kv_head_replicas`` is the TP ranks per KV head. A DCP group wider than
    that spans ``dcp // kv_head_replicas`` KV heads; all-gathers then also
    bring every rank the group's new K/V and index queries
    (`gather_group_heads`), and the LSE merge and reduce-scatter go out.

    Two JAX ops: the first writes this rank's tokens and returns its local
    top-k candidates; an all-gather over the DCP group brings every rank all
    ``dcp * topk`` candidates and the query heads of the whole group; the
    second merges the candidates into the exact selection and attends over
    the selected blocks this rank holds; a log-sum-exp merge (shared with the
    MLA DCP path) combines the ranks' partials and keeps this rank's heads.
    """
    from vllm_torchtpu.distributed.dcp import get_dcp_group

    if dcp % kv_head_replicas and kv_head_replicas % dcp:
        raise ValueError(
            f"DCP size {dcp} and the {kv_head_replicas} replicas of a KV head "
            "must divide one another."
        )
    kv_heads = max(1, dcp // kv_head_replicas)
    if fp4:
        return _get_msa_dcp_fp4_op(
            sm_scale,
            topk,
            init_blocks,
            local_blocks,
            dcp,
            kv_head_replicas,
            decode_rows,
        )

    def _score_jax(
        k: jax.Array,
        v: jax.Array,
        index_q: jax.Array,
        index_k: jax.Array,
        kv_cache: jax.Array,
        index_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        dcp_rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return msa_dcp.msa_write_and_candidates_dcp(
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            rank=dcp_rank,
            dcp=dcp,
        )

    def _attend_jax(
        q: jax.Array,
        kv_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return msa_dcp.msa_attend_dcp(
            q,
            kv_cache,
            candidates,
            seq_lens,
            block_tables,
            query_start_loc,
            sm_scale=sm_scale,
            topk=topk,
            rank=dcp_rank,
            dcp=dcp,
        )

    # No rank in the name: the op (and the program it lowers to) is the same
    # on every rank of the group.
    suffix = (
        f"t{topk}_i{init_blocks}_l{local_blocks}_s{sm_scale:.8f}_dcp{dcp}_g{kv_heads}"
    ).replace(".", "p")
    score_op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_score_{suffix}",
        _score_jax,
        donate_argnums=(_SHARD_KV_CACHE_ARG, _SHARD_INDEX_CACHE_ARG),
    )
    attend_op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_attend_{suffix}", _attend_jax
    )

    def _score_fake(k, v, index_q, index_k, kv_cache, index_cache, *args):
        return (
            torch.empty_like(kv_cache),
            torch.empty_like(index_cache),
            torch.empty(
                (k.shape[0], k.shape[1] * 2 * topk),
                dtype=torch.float32,
                device=k.device,
            ),
        )

    score_op.register_fake(_score_fake)
    attend_op.register_fake(_head_major_attend_fake)

    def op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        metadata: AttentionMetadata,
        *,
        dcp_rank: torch.Tensor,
    ) -> torch.Tensor:
        group = get_dcp_group()
        assert group is not None, "DCP group is not initialized"
        # Fixed while tracing: only the backbone reaches this op, and it is
        # traced per token-bucket group (compilation/shape_variants). A graph
        # traced once for every bucket would keep the first size's choice.
        packed = kv_heads == 1 and q.shape[0] <= PACKED_EXCHANGE_MAX_TOKENS
        if packed:
            q_all = None  # gathered with the candidates
        elif kv_heads == 1:
            q_all = gather_query_heads(q, group)
        else:
            q_all, k, v, index_q = gather_group_heads(
                q, k, v, index_q, group, dcp, kv_head_replicas
            )
        new_kv_cache, new_index_cache, candidates = score_op(
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            metadata.request_distribution,
            dcp_rank,
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        if packed:
            q_all, candidates = exchange_queries_and_candidates(q, candidates, group)
        else:
            candidates = group.all_gather(candidates.unsqueeze(0), dim=0)
        out, lse = attend_op(
            q_all,
            kv_cache,
            candidates,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            dcp_rank,
        )
        if packed:
            return merge_head_partials_a2a(out, lse, group)
        return merge_head_partials(out, lse, group)

    return op


def _get_msa_dcp_fp4_op(
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    dcp: int,
    kv_head_replicas: int,
    decode_rows: int,
) -> Callable[..., torch.Tensor]:
    """`get_msa_dcp_op` for 4-bit caches."""
    kv_heads = max(1, dcp // kv_head_replicas)
    from vllm_torchtpu.distributed.dcp import get_dcp_group

    def _score_jax(
        k: jax.Array,
        v: jax.Array,
        index_q: jax.Array,
        index_k: jax.Array,
        kv_cache: jax.Array,
        index_cache: jax.Array,
        scale_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        dcp_rank: jax.Array,
        q: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # This rank's queries are rotated here, a 1 / DCP share of the
        # group's heads: inside a JAX op the rotation leaves the rope
        # fusions that produce q alone (a torch matmul on q re-laid them
        # out, +0.3 ms per layer at 8K tokens, j-c5c4bdfc).
        return *msa_dcp.msa_write_and_candidates_dcp(
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            rank=dcp_rank,
            dcp=dcp,
            scale_cache=scale_cache,
        ), msa_fp4.rotate_queries(q)

    def _attend_jax(
        q: jax.Array,
        kv_cache: jax.Array,
        scale_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return msa_dcp.msa_attend_dcp(
            q,
            kv_cache,
            candidates,
            seq_lens,
            block_tables,
            query_start_loc,
            sm_scale=sm_scale,
            topk=topk,
            rank=dcp_rank,
            dcp=dcp,
            scale_cache=scale_cache,
        )

    suffix = (
        f"t{topk}_i{init_blocks}_l{local_blocks}_s{sm_scale:.8f}_dcp{dcp}_g{kv_heads}"
    ).replace(".", "p")
    score_op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_fp4_score_{suffix}",
        _score_jax,
        donate_argnums=(
            _SHARD_KV_CACHE_ARG,
            _SHARD_INDEX_CACHE_ARG,
            _DCP_SCALE_CACHE_ARG,
        ),
    )
    attend_op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_fp4_attend_{suffix}", _attend_jax
    )

    def _score_fake(k, v, index_q, index_k, kv_cache, index_cache, scale_cache, *args):
        return (
            torch.empty_like(kv_cache),
            torch.empty_like(index_cache),
            torch.empty_like(scale_cache),
            torch.empty(
                (k.shape[0], k.shape[1] * 2 * topk),
                dtype=torch.float32,
                device=k.device,
            ),
            torch.empty_like(args[-1]),
        )

    score_op.register_fake(_score_fake)
    attend_op.register_fake(_head_major_attend_fake)
    heads_op = (
        _get_msa_dcp_heads_op(
            sm_scale, topk, dcp, kv_head_replicas, decode_rows, suffix
        )
        if kv_heads > 1
        else None
    )

    def op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        metadata: AttentionMetadata,
        *,
        dcp_rank: torch.Tensor,
        scale_cache: torch.Tensor,
        hadamard: torch.Tensor,
    ) -> torch.Tensor:
        group = get_dcp_group()
        assert group is not None, "DCP group is not initialized"
        if kv_heads > 1:
            k, v, index_q = gather_group_kv(k, v, index_q, group, dcp, kv_head_replicas)
        new_kv_cache, new_index_cache, new_scale_cache, candidates, q = score_op(
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            scale_cache,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            metadata.request_distribution,
            dcp_rank,
            q.contiguous(),
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        scale_cache.copy_(new_scale_cache)
        if heads_op is not None:
            return heads_op(
                dcp_rank,
                q,
                kv_cache,
                scale_cache,
                candidates,
                metadata.seq_lens,
                metadata.block_tables,
                metadata.query_start_loc,
                metadata.request_distribution,
            )
        q_all = gather_query_heads(q, group)
        candidates = group.all_gather(candidates.unsqueeze(0), dim=0)
        out, lse = attend_op(
            q_all,
            kv_cache,
            scale_cache,
            candidates,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            dcp_rank,
        )
        return merge_head_partials(out, lse, group, hadamard)

    return op


def _get_msa_dcp_heads_op(
    sm_scale: float,
    topk: int,
    dcp: int,
    kv_head_replicas: int,
    decode_rows: int,
    suffix: str,
) -> Callable[..., torch.Tensor]:
    """The attention half of the 4-bit DCP op over several KV heads, as one
    SPMD op over the DCP mesh (`msa_dcp_heads.attend_and_merge`): its
    collectives run inside the program, so a long step can pick its paths
    and their sizes from the step's requests."""
    from jax.sharding import NamedSharding, PartitionSpec

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh

    mesh = get_or_create_dcp_mesh()
    sharded, replicated = PartitionSpec(DCP_AXIS_NAME), PartitionSpec()
    specs = (sharded,) * 5 + (replicated,) * 4
    body = jax.shard_map(
        functools.partial(
            msa_dcp_heads.attend_and_merge,
            axis=DCP_AXIS_NAME,
            dcp=dcp,
            kv_head_replicas=kv_head_replicas,
            sm_scale=sm_scale,
            topk=topk,
            decode_rows=decode_rows,
            rs_out_precision=envs.TPU_DCP_RS_OUT_PRECISION,
        ),
        mesh=mesh,
        in_specs=specs,
        out_specs=sharded,
        check_vma=False,
    )

    # torch_tpu reads the operand types from these annotations.
    def _attend_jax(
        dcp_rank: jax.Array,
        q: jax.Array,
        kv_cache: jax.Array,
        scale_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> jax.Array:
        args = (
            dcp_rank,
            q,
            kv_cache,
            scale_cache,
            candidates,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
        )
        return body(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_heads_attend_{suffix}_r{decode_rows}",
        _attend_jax,
        mesh=mesh,
        input_partition_specs=specs,
        # A sequence, one spec per output (a bare spec reads as its axes).
        output_partition_specs=(sharded,),
    )
    op.register_fake(lambda dcp_rank, q, *args: torch.empty_like(q))
    return op


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block ``d`` of ``x``'s leading axis goes to rank ``d``; block ``r`` of
    the result came from rank ``r``."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out
