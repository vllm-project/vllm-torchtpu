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

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_indexer

_KV_CACHE_ARG = 5
_INDEX_CACHE_ARG = 6
_SHARD_KV_CACHE_ARG = 4
_SHARD_INDEX_CACHE_ARG = 5


@functools.cache
def get_msa_op(
    sm_scale: float, topk: int, init_blocks: int, local_blocks: int
) -> Callable[..., torch.Tensor]:
    """Build (once per config) the op that runs `msa.msa_forward` in place.

    Built eagerly from module `__init__`: `pallas.jax_op` inspects the
    function signature, which Dynamo cannot trace inside a compiled forward.
    """

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


def split_group_heads(
    gathered: torch.Tensor, dcp: int, q_heads: int, kv_head_replicas: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the DCP all-gather of every rank's ``[q | k | v | index_q]``
    (``[T, dcp * (q_heads + 3), D]``, rank-major) into the group's query
    heads ``[T, dcp * q_heads, D]`` and the ``k``/``v``/``index_q
    [T, dcp // kv_head_replicas, D]`` of its KV heads: rank ``i`` of the
    group holds KV head ``i // kv_head_replicas``, so every
    ``kv_head_replicas``-th rank supplies the next head."""
    num_tokens, _, head_dim = gathered.shape
    gathered = gathered.view(num_tokens, dcp, q_heads + 3, head_dim)
    q_all = gathered[:, :, :q_heads].reshape(num_tokens, dcp * q_heads, head_dim)
    per_head = gathered[:, ::kv_head_replicas, q_heads:]
    return (
        q_all,
        per_head[:, :, 0].contiguous(),
        per_head[:, :, 1].contiguous(),
        per_head[:, :, 2].contiguous(),
    )


@functools.cache
def get_msa_dcp_op(
    sm_scale: float,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: int,
    dcp: int,
    kv_head_replicas: int = 1,
) -> Callable[..., torch.Tensor]:
    """`get_msa_op` for a KV head's context split over ``dcp`` ranks
    (`kernels.msa_dcp`).

    ``kv_head_replicas`` is the TP ranks per KV head. A DCP group wider than
    that spans ``dcp // kv_head_replicas`` KV heads; one all-gather then also
    brings every rank the group's new K/V and index queries (the GLM DCP
    pattern: one gather in, the LSE merge and reduce-scatter out).

    Two JAX ops: the first writes this rank's tokens and returns its local
    top-k candidates; an all-gather over the DCP group brings every rank all
    ``dcp * topk`` candidates and the query heads of the whole group; the
    second merges the candidates into the exact selection and attends over
    the selected blocks this rank holds; a log-sum-exp merge (shared with the
    MLA DCP path) combines the ranks' partials and keeps this rank's heads.
    """
    from vllm_torchtpu.distributed.dcp import get_dcp_group
    from vllm_torchtpu.layers.adapter.cp_mla_attention import (
        all_gather_heads,
        merge_lse_partials_scatter_heads,
    )

    if dcp % kv_head_replicas and kv_head_replicas % dcp:
        raise ValueError(
            f"DCP size {dcp} and the {kv_head_replicas} replicas of a KV head "
            "must divide one another."
        )
    kv_heads = max(1, dcp // kv_head_replicas)

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
            rank=rank,
            dcp=dcp,
        )

    def _attend_jax(
        q: jax.Array,
        kv_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
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
            rank=rank,
            dcp=dcp,
        )

    suffix = (
        f"t{topk}_i{init_blocks}_l{local_blocks}_s{sm_scale:.8f}_r{rank}of{dcp}"
        f"_g{kv_heads}"
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

    def _attend_fake(q, *args):
        return (
            torch.empty_like(q),
            torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
        )

    score_op.register_fake(_score_fake)
    attend_op.register_fake(_attend_fake)

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
        group = get_dcp_group()
        assert group is not None, "DCP group is not initialized"
        if kv_heads == 1:
            q_all = all_gather_heads(q)
        else:
            # One gather of [q | k | v | index_q] per rank; every
            # kv_head_replicas-th rank contributes the K/V of the next head.
            gathered = group.all_gather(
                torch.cat([q, k, v, index_q], dim=1).contiguous(), dim=1
            )
            q_all, k, v, index_q = split_group_heads(
                gathered, dcp, q.shape[1], kv_head_replicas
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
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        candidates = group.all_gather(candidates.unsqueeze(0), dim=0)
        out, lse = attend_op(
            q_all,
            kv_cache,
            candidates,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
        )
        return merge_lse_partials_scatter_heads(out, lse)

    return op


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block ``d`` of ``x``'s leading axis goes to rank ``d``; block ``r`` of
    the result came from rank ``r``."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out
