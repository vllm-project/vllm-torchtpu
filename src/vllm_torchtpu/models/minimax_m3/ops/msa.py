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
    unrotates as its terms would, up to bf16 rounding order). It merges DCP
    groups over several KV heads, always by reduce-scatter
    (`TPU_DCP_LSE_MERGE_A2A`, the MLA merge's all-to-all for small steps,
    does not apply); over one KV head the attention op merges, in f32
    (`msa_dcp.attend_and_merge`; `TPU_DCP_RS_OUT_PRECISION` does not apply
    there)."""
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


def gather_group_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    group: Any,
    dcp: int,
    kv_head_replicas: int,
    dcp_rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The DCP group's query heads ``[T, dcp * q_heads, D]``
    (`gather_query_heads`) and its KV heads' ``k``/``v``/``index_q``
    (`gather_group_kv`).

    Two gathers: only the scoring op needs the K/V and index queries, and
    the attention op needs the queries after it, so the query gather (8 of a
    rank's 11 heads at DCP 8) can run while the scoring op computes instead
    of before it, and needs no strided split of a combined gather."""
    return gather_query_heads(q, group), *gather_group_kv(
        k, v, index_q, group, dcp, kv_head_replicas, dcp_rank
    )


def gather_group_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    group: Any,
    dcp: int,
    kv_head_replicas: int,
    dcp_rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``k``/``v``/``index_q [T, dcp // kv_head_replicas, D]`` of the DCP
    group's KV heads. Rank ``i = h * kv_head_replicas + j`` holds KV head
    ``h``, as its other replicas do, so it sends only part ``j`` of the
    tokens: a replica-split all-gather moves ``1 / kv_head_replicas`` of the
    bytes of gathering every rank's rows and keeping one replica's (c64
    decode twin j-c4bfc418: 17.3 us per layer for ``bf16[4096, 3, 128]``)."""
    num_tokens, _, head_dim = k.shape
    reps = kv_head_replicas
    if num_tokens % reps:
        rows = torch.cat([k, v, index_q], dim=1)  # [T, 3, D]
        per_rank = group.all_gather(rows.contiguous(), dim=1)
        per_head = per_rank.view(num_tokens, dcp, 3, head_dim)[:, ::reps]
        return tuple(per_head[:, :, i].contiguous() for i in range(3))
    part = num_tokens // reps
    if num_tokens < msa_dcp_heads.HEAD_PARALLEL_MIN_TOKENS:
        # Decode-sized steps: the lane-dense rows below cost +4.4 us per
        # decode layer at c64 (twin j-899e03a2).
        rows = torch.cat([k, v, index_q], dim=1)  # [T, 3, D]
        mine = torch.index_select(
            rows.view(reps, part, 3, head_dim), 0, dcp_rank % reps
        )
        parts = group.all_gather(mine.contiguous(), dim=0)  # [dcp, part, 3, D]
        per_head = (
            parts.view(dcp // reps, reps, part, 3, head_dim)
            .permute(1, 2, 0, 3, 4)
            .reshape(num_tokens, dcp // reps, 3, head_dim)
        )
        return tuple(per_head[:, :, i].contiguous() for i in range(3))
    # Lane-dense [T, 3 * D] rows: re-laying gathered [T, 3, D] rows out per
    # head cost ~88 us per layer at c64 (twin j-9a9fccc2), these ~0 (j-899e03a2).
    rows = torch.cat([k, v, index_q], dim=-1).reshape(num_tokens, 3 * head_dim)
    mine = torch.index_select(rows.view(reps, part, -1), 0, dcp_rank % reps)
    parts = group.all_gather(mine.contiguous(), dim=0)  # [dcp, part, 3 * D]
    return tuple(
        parts[..., i * head_dim : (i + 1) * head_dim]
        .reshape(dcp // reps, reps, part, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(num_tokens, dcp // reps, head_dim)
        .contiguous()
        for i in range(3)
    )


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
    if kv_heads == 1:
        attend_merge_op = _get_msa_dcp_attend_merge_op(
            sm_scale, topk, dcp, False, suffix
        )
    else:
        attend_merge_op = None
        attend_op = pallas.jax_op(
            f"pallas::minimax_m3_msa_dcp_attend_{suffix}", _attend_jax
        )
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
        if kv_heads > 1:
            q_all, k, v, index_q = gather_group_heads(
                q, k, v, index_q, group, dcp, kv_head_replicas, dcp_rank
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
        if attend_merge_op is not None:
            return attend_merge_op(
                dcp_rank,
                q,
                kv_cache,
                candidates,
                metadata.seq_lens,
                metadata.block_tables,
                metadata.query_start_loc,
            )
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
        return merge_head_partials(out, lse, group)

    return op


def fp4_dcp_rotates_queries(dcp: int, kv_head_replicas: int) -> bool:
    """Whether the 4-bit DCP op takes q and index_q already rotated
    (`msa_fp4.rotate_queries`) by the norm + rope kernel (`qk_norm_rope`'s
    ``rotate``): a group over one KV head. A group of several KV heads
    rotates them inside the op: kernel-rotated q and index_q cost relayout
    copies around the group gathers (c64 twin j-afe31cf0)."""
    return max(1, dcp // kv_head_replicas) == 1


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
    suffix = (
        f"t{topk}_i{init_blocks}_l{local_blocks}_s{sm_scale:.8f}_dcp{dcp}_g{kv_heads}"
    ).replace(".", "p")
    if kv_heads > 1:
        step_ops = {
            gather: _get_msa_dcp_heads_step_op(
                sm_scale,
                topk,
                init_blocks,
                local_blocks,
                dcp,
                kv_head_replicas,
                decode_rows,
                suffix,
                gather_weight=gather,
            )
            for gather in (False, True)
        }

        def heads_op(
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
            step_tokens: int | None = None,
            weight_block: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            """`weight_block`: (this rank's block of a weight, its rank index
            ``int32[1]``) to gather under the attention; the result is then
            ``(out, every rank's blocks)``."""
            # The step op rotates its own query heads; it picks its indexer
            # path itself (no StepHalves on this path).
            del hadamard, step_tokens
            new_kv_cache, new_index_cache, new_scale_cache, out, *gathered = step_ops[
                weight_block is not None
            ](
                dcp_rank,
                q.contiguous(),
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
                *(weight_block or ()),
            )
            kv_cache.copy_(new_kv_cache)
            index_cache.copy_(new_index_cache)
            scale_cache.copy_(new_scale_cache)
            return (out, *gathered) if gathered else out

        heads_op.gathers_weight = True
        return heads_op

    # One KV head per rank: the norm + rope kernel rotates the queries
    # (`fp4_dcp_rotates_queries`).
    def _scorer(prefill: bool | None) -> Callable[..., Any]:
        """The score op's JAX function; `prefill` overrides the indexer's
        prefill-vs-decode choice by row count (`msa_write_and_candidates_dcp`)."""

        def _write_and_score(
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
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
                scale_cache=scale_cache,
                index_q_rotated=True,
                prefill=prefill,
            )

        return _write_and_score

    def _score_op(prefill: bool | None):
        return pallas.jax_op(
            f"pallas::minimax_m3_msa_dcp_fp4_score_rot_{'pf_' if prefill else ''}"
            f"{suffix}",
            _scorer(prefill),
            donate_argnums=(
                _SHARD_KV_CACHE_ARG,
                _SHARD_INDEX_CACHE_ARG,
                _DCP_SCALE_CACHE_ARG,
            ),
        )

    score_op, prefill_score_op = _score_op(None), _score_op(True)

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
        )

    score_op.register_fake(_score_fake)
    prefill_score_op.register_fake(_score_fake)
    attend_merge_op = _get_msa_dcp_attend_merge_op(sm_scale, topk, dcp, True, suffix)

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
        step_tokens: int | None = None,
    ) -> torch.Tensor:
        """``step_tokens``: the whole step's token count when this call
        attends one token half of it (StepHalves): the half's candidates then
        take the indexer path the whole step would take."""
        score = score_op
        if step_tokens is not None and msa_indexer.is_prefill_step(
            k.shape[1] * step_tokens, metadata.seq_lens.shape[0]
        ):
            score = prefill_score_op
        new_kv_cache, new_index_cache, new_scale_cache, candidates = score(
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
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        scale_cache.copy_(new_scale_cache)
        return attend_merge_op(
            dcp_rank,
            q,
            kv_cache,
            candidates,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            scale_cache,
            hadamard,
        )

    return op


def _get_msa_dcp_attend_merge_op(
    sm_scale: float, topk: int, dcp: int, fp4: bool, suffix: str
) -> Callable[..., torch.Tensor]:
    """The attention half of the DCP op over one KV head, as one SPMD op over
    the DCP mesh (`msa_dcp.attend_and_merge`): this rank's head-major
    queries and its candidates in, its heads' merged output ``[T, Hl * D]``
    out. With `fp4` it also takes the scale cache and the hadamard."""
    from jax.sharding import NamedSharding, PartitionSpec

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh

    mesh = get_or_create_dcp_mesh()
    sharded, replicated = PartitionSpec(DCP_AXIS_NAME), PartitionSpec()
    # rank, q, kv_cache, candidates | seq_lens, block_tables, query_start_loc
    # | (scale_cache, hadamard)
    specs = (sharded,) * 4 + (replicated,) * 3 + ((sharded, replicated) if fp4 else ())
    body = jax.shard_map(
        functools.partial(
            msa_dcp.attend_and_merge,
            axis=DCP_AXIS_NAME,
            dcp=dcp,
            sm_scale=sm_scale,
            topk=topk,
        ),
        mesh=mesh,
        in_specs=specs,
        out_specs=sharded,
        check_vma=False,
    )

    def run(*args):
        return body(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # torch_tpu reads the operand types from these annotations.
    def _attend_jax(
        dcp_rank: jax.Array,
        q: jax.Array,
        kv_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
    ) -> jax.Array:
        return run(
            dcp_rank, q, kv_cache, candidates, seq_lens, block_tables, query_start_loc
        )

    def _attend_fp4_jax(
        dcp_rank: jax.Array,
        q: jax.Array,
        kv_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        scale_cache: jax.Array,
        hadamard: jax.Array,
    ) -> jax.Array:
        return run(
            dcp_rank,
            q,
            kv_cache,
            candidates,
            seq_lens,
            block_tables,
            query_start_loc,
            scale_cache,
            hadamard,
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_attend_merge_{suffix}_fp4{int(fp4)}",
        _attend_fp4_jax if fp4 else _attend_jax,
        mesh=mesh,
        input_partition_specs=specs,
        # A sequence, one spec per output (a bare spec reads as its axes).
        output_partition_specs=(sharded,),
    )

    def _fake(dcp_rank, q, *args):
        num_heads, num_tokens, head_dim = q.shape
        return q.new_empty((num_tokens, num_heads * head_dim))

    op.register_fake(_fake)
    return op


def _get_msa_dcp_heads_step_op(
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    dcp: int,
    kv_head_replicas: int,
    decode_rows: int,
    suffix: str,
    gather_weight: bool = False,
) -> Callable[..., torch.Tensor]:
    """The whole 4-bit DCP op over several KV heads as one SPMD op over the
    DCP mesh: the cache write, the indexer and candidates
    (`msa_dcp.msa_write_and_candidates_dcp`) and the attention and merge
    (`msa_dcp_heads.attend_and_merge`). One program for both halves lets a
    step with head-parallel rounds start the first round's page exchange
    right after the write, under the indexer
    (`msa_dcp_heads.start_first_round_exchange`), and the collectives run
    inside it, so a long step picks its paths and sizes from its requests:
    the group's K/V and index queries are gathered in it
    (`msa_dcp_heads.gather_group_kv`) beside the query rotation. Takes this
    rank's KV head's ``k``/``v``/``index_q [T, 1, D]``; returns the written
    caches and this rank's unrotated output ``[T, Hl * D]``.

    With `gather_weight` it also takes this rank's block of a weight and its
    index in rank order (``int32[1]``) and returns every rank's blocks,
    gathered by remote DMA under the attention (`start_block_all_gather`):
    the SP o_proj weight, whose XLA all-gather was issued after the
    attention and left exposed (~85 us per layer, c64 mixed twin)."""
    from jax.sharding import NamedSharding, PartitionSpec

    from vllm_torchtpu.distributed.dcp import (
        DCP_AXIS_NAME,
        DCP_LANE_AXIS_NAME,
        get_or_create_dcp_mesh,
    )

    mesh = get_or_create_dcp_mesh()
    sharded, replicated = PartitionSpec(DCP_AXIS_NAME), PartitionSpec()
    # rank, q, k, v, index_q, index_k, kv_cache, index_cache, scale_cache |
    # seq_lens, block_tables, query_start_loc, request_distribution
    # | (weight block, its index)
    specs = (sharded,) * 9 + (replicated,) * 4 + (sharded,) * (2 * gather_weight)
    num_outputs = 4 + gather_weight
    common = dict(axis=DCP_AXIS_NAME, dcp=dcp, kv_head_replicas=kv_head_replicas)
    mesh_shape = (mesh.shape[DCP_LANE_AXIS_NAME], mesh.shape[DCP_AXIS_NAME])

    def step(
        rank,
        q,
        k,
        v,
        index_q,
        index_k,
        kv_cache,
        index_cache,
        scale_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        distribution,
        *weight_block,
    ):
        num_tokens = k.shape[0]
        k, v, index_q = msa_dcp_heads.gather_group_kv(k, v, index_q, rank, **common)
        queries = msa_dcp_heads.prepare_queries(
            q,
            rank,
            cu_q_lens,
            distribution,
            kv_head_replicas=kv_head_replicas,
            decode_rows=decode_rows,
        )
        # The queries are ready by the time the gather lands, so XLA computes
        # them while it is in flight instead of after the indexer.
        (k, v, index_q), queries = jax.lax.optimization_barrier(
            ((k, v, index_q), queries)
        )
        after_write = None
        if msa_dcp_heads.head_parallel_step(num_tokens, seq_lens.shape[0], decode_rows):

            def after_write(kv_cache, scale_cache):
                del scale_cache  # the scales move with the round's pages
                return msa_dcp_heads.start_first_round_exchange(
                    kv_cache,
                    seq_lens,
                    block_tables,
                    cu_q_lens,
                    distribution,
                    rank,
                    num_tokens=num_tokens,
                    decode_rows=decode_rows,
                    **common,
                )

        kv_cache, index_cache, scale_cache, candidates, *started = (
            msa_dcp.msa_write_and_candidates_dcp(
                k,
                v,
                index_q,
                index_k,
                kv_cache,
                index_cache,
                seq_lens,
                block_tables,
                cu_q_lens,
                distribution,
                topk=topk,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                rank=rank,
                dcp=dcp,
                scale_cache=scale_cache,
                after_write=after_write,
            )
        )
        alongside = None
        if weight_block:
            weight, index = weight_block

            def alongside(ready):
                block, _ = jax.lax.optimization_barrier((weight, ready))
                return msa_dcp_heads.start_block_all_gather(
                    block,
                    index,
                    axes=(DCP_LANE_AXIS_NAME, DCP_AXIS_NAME),
                    shape=mesh_shape,
                )

        out = msa_dcp_heads.attend_and_merge(
            rank,
            queries,
            kv_cache,
            scale_cache,
            candidates,
            seq_lens,
            block_tables,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            topk=topk,
            decode_rows=decode_rows,
            rs_out_precision=envs.TPU_DCP_RS_OUT_PRECISION,
            page_exchange=started[0] if started else None,
            alongside=alongside,
            **common,
        )
        if alongside is None:
            return kv_cache, index_cache, scale_cache, out
        out, in_flight = out
        gathered = msa_dcp_heads.finish_block_exchange(
            in_flight, mesh_shape[0] * mesh_shape[1]
        )
        return kv_cache, index_cache, scale_cache, out, gathered

    body = jax.shard_map(
        step,
        mesh=mesh,
        in_specs=specs,
        out_specs=(sharded,) * num_outputs,
        check_vma=False,
    )

    # torch_tpu reads the operand types from these annotations.
    def _step_jax(
        dcp_rank: jax.Array,
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
    ) -> tuple[jax.Array, ...]:
        args = (
            dcp_rank,
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            scale_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
        )
        return body(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # torch_tpu reads the operand types from these annotations.
    def _step_weight_jax(
        dcp_rank: jax.Array,
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
        weight: jax.Array,
        weight_index: jax.Array,
    ) -> tuple[jax.Array, ...]:
        args = (
            dcp_rank,
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            scale_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            weight,
            weight_index,
        )
        return body(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_heads_step_{suffix}_r{decode_rows}"
        f"{'_w' if gather_weight else ''}",
        _step_weight_jax if gather_weight else _step_jax,
        mesh=mesh,
        input_partition_specs=specs,
        # A sequence, one spec per output (a bare spec reads as its axes).
        output_partition_specs=(sharded,) * num_outputs,
        donate_argnums=(6, 7, 8),
    )

    def _fake(
        dcp_rank, q, k, v, index_q, index_k, kv_cache, index_cache, scale_cache, *rest
    ):
        outs = (
            torch.empty_like(kv_cache),
            torch.empty_like(index_cache),
            torch.empty_like(scale_cache),
            q.new_empty((q.shape[0], q.shape[1] * q.shape[2])),
        )
        if not gather_weight:
            return outs
        weight = rest[-2]
        return (*outs, weight.new_empty((mesh_shape[0] * mesh_shape[1], *weight.shape)))

    op.register_fake(_fake)
    return op


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block ``d`` of ``x``'s leading axis goes to rank ``d``; block ``r`` of
    the result came from rank ``r``."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out
