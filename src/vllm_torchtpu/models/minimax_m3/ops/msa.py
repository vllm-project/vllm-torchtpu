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
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3 import dcp_pack
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

logger = init_logger(__name__)


def _packed_merge(num_tokens: int) -> bool:
    """Whether a DCP step of `num_tokens` merges with one packed all-to-all
    (``TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS``). Compared while tracing:
    the backbone is re-traced per token-bucket group."""
    return 0 < num_tokens <= envs.TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS


def _packed_gather(num_tokens: int) -> bool:
    """Whether a DCP step of `num_tokens` gathers its query heads and
    candidates in one all-gather (``TPU_MINIMAX_M3_DCP_QCAND_GATHER_MAX_TOKENS``)."""
    return 0 < num_tokens <= envs.TPU_MINIMAX_M3_DCP_QCAND_GATHER_MAX_TOKENS


# The packed merge's XLA fusion fence, built by the op builder: building a
# `pallas.jax_op` inside the compiled forward fails under Dynamo.
_FENCE: list[Callable[[torch.Tensor], torch.Tensor] | None] = [None]


def _log_packed_collectives() -> None:
    """Once per process, from the op builders (outside the traced forward:
    a logging call under Dynamo breaks the graph); also builds `_FENCE`."""
    merge = envs.TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS
    gather = envs.TPU_MINIMAX_M3_DCP_QCAND_GATHER_MAX_TOKENS
    if merge and _FENCE[0] is None:
        from vllm_torchtpu.layers.adapter.custom_ops.optimization_barrier_op import (
            optimization_barrier_op,
        )

        _FENCE[0] = optimization_barrier_op()
    if merge:
        logger.info_once(
            "TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS=%d: MSA DCP partials merge "
            "with one packed all-to-all on steps of at most %d tokens.",
            merge,
            merge,
        )
    if gather:
        logger.info_once(
            "TPU_MINIMAX_M3_DCP_QCAND_GATHER_MAX_TOKENS=%d: MSA DCP query heads "
            "and candidates share one all-gather on steps of at most %d tokens.",
            gather,
            gather,
        )


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
    unrotates as its terms would, up to bf16 rounding order). The MSA merge
    always reduce-scatters: `TPU_DCP_LSE_MERGE_A2A` (the MLA merge's
    all-to-all for small steps) does not apply here; steps of at most
    ``TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS`` tokens take one packed
    all-to-all instead (`dcp_pack.merge_head_partials_all_to_all`)."""
    if _packed_merge(out.shape[1]):
        merged = dcp_pack.merge_head_partials_all_to_all(
            out, lse, group, _all_to_all, envs.TPU_DCP_RS_OUT_PRECISION, _FENCE[0]
        )
        if hadamard is not None:
            merged = _rotate_heads(merged, hadamard)
        return merged.transpose(0, 1)
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
    _log_packed_collectives()
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
        packed = kv_heads == 1 and _packed_gather(q.shape[0])
        if packed:
            q_all = None
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
            q_all, candidates = dcp_pack.gather_queries_and_candidates(
                q.contiguous(), candidates, group
            )
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

    def _attend_rot_jax(
        q: jax.Array,
        kv_cache: jax.Array,
        scale_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # `q`: the group's unrotated heads (early gather); per-row rotation.
        return _attend_jax(
            msa_fp4.rotate_queries(q),
            kv_cache,
            scale_cache,
            candidates,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp_rank,
        )

    def _decode_rows_q_jax(
        q: jax.Array, query_start_loc: jax.Array, request_distribution: jax.Array
    ) -> jax.Array:
        src_row = msa_dcp_heads.decode_row_plan(
            query_start_loc,
            request_distribution,
            query_start_loc.shape[0] - 1,
            decode_rows,
        )[3]
        return q[src_row]

    suffix = (
        f"t{topk}_i{init_blocks}_l{local_blocks}_s{sm_scale:.8f}_dcp{dcp}_g{kv_heads}"
    ).replace(".", "p")
    early = msa_dcp_heads.early_q_gather_enabled()
    attend_rot_op = None
    if early and kv_heads == 1:
        attend_rot_op = pallas.jax_op(
            f"pallas::minimax_m3_msa_dcp_fp4_attend_rot_{suffix}", _attend_rot_jax
        )
        attend_rot_op.register_fake(_head_major_attend_fake)
    decode_rows_q_op = None
    if early and kv_heads > 1:
        decode_rows_q_op = pallas.jax_op(
            f"pallas::minimax_m3_msa_dcp_decode_rows_q_r{decode_rows}",
            _decode_rows_q_jax,
        )
        decode_rows_q_op.register_fake(
            lambda q, qsl, dist: q.new_empty(
                ((qsl.shape[0] - 1) * decode_rows, q.shape[1], q.shape[2])
            )
        )
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
    early_q_op = early_dec_op = None
    if kv_heads > 1 and early:
        early_q_op = _get_msa_dcp_heads_op(
            sm_scale, topk, dcp, kv_head_replicas, decode_rows, suffix, q_gathered=True
        )
        early_dec_op = _get_msa_dcp_heads_op(
            sm_scale, topk, dcp, kv_head_replicas, decode_rows, suffix, q_dec=True
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
        # Gathers ahead of the scoring op, which they do not feed, so they
        # can overlap the indexer (`msa_dcp_heads.EARLY_Q_GATHER_ENV`).
        q_all = q_dec = None
        if early_q_op is not None:
            if msa_dcp_heads.uses_dcp_attention(
                q.shape[0], metadata.seq_lens.shape[0], decode_rows
            ):
                q_all = gather_query_heads(q, group)
            else:  # head-parallel step: only its decode rows take the gather
                q_dec = gather_query_heads(
                    decode_rows_q_op(
                        q.contiguous(),
                        metadata.query_start_loc,
                        metadata.request_distribution,
                    ),
                    group,
                )
        elif attend_rot_op is not None:
            q_all = gather_query_heads(q, group)
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
        if q_dec is not None:
            return early_dec_op(
                dcp_rank,
                q,
                kv_cache,
                scale_cache,
                candidates,
                metadata.seq_lens,
                metadata.block_tables,
                metadata.query_start_loc,
                metadata.request_distribution,
                q_dec,
            )
        if q_all is not None and early_q_op is not None:
            return early_q_op(
                dcp_rank,
                q_all,
                kv_cache,
                scale_cache,
                candidates,
                metadata.seq_lens,
                metadata.block_tables,
                metadata.query_start_loc,
                metadata.request_distribution,
            )
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
        if q_all is None and _packed_gather(q.shape[0]):
            q_all, candidates = dcp_pack.gather_queries_and_candidates(
                q, candidates, group
            )
            attend = attend_op
        else:
            candidates = group.all_gather(candidates.unsqueeze(0), dim=0)
            if q_all is None:
                q_all = gather_query_heads(q, group)
                attend = attend_op
            else:
                attend = attend_rot_op
        out, lse = attend(
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
    q_gathered: bool = False,
    q_dec: bool = False,
) -> Callable[..., torch.Tensor]:
    """The attention half of the 4-bit DCP op over several KV heads, as one
    SPMD op over the DCP mesh (`msa_dcp_heads.attend_and_merge`): its
    collectives run inside the program, so a long step can pick its paths
    and their sizes from the step's requests. `q_gathered`: the op takes the
    group's unrotated query heads, gathered by the caller. `q_dec`: the op
    takes one more input, the gathered unrotated heads of a head-parallel
    step's compacted decode rows."""
    from jax.sharding import NamedSharding, PartitionSpec

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh

    mesh = get_or_create_dcp_mesh()
    sharded, replicated = PartitionSpec(DCP_AXIS_NAME), PartitionSpec()
    specs = (sharded,) * 5 + (replicated,) * 4 + ((sharded,) if q_dec else ())
    fn = msa_dcp_heads.attend_and_merge
    if q_dec:

        def fn(*args, **kwargs):
            return msa_dcp_heads.attend_and_merge(*args[:9], q_dec=args[9], **kwargs)

    body = jax.shard_map(
        functools.partial(
            fn,
            axis=DCP_AXIS_NAME,
            dcp=dcp,
            kv_head_replicas=kv_head_replicas,
            sm_scale=sm_scale,
            topk=topk,
            decode_rows=decode_rows,
            rs_out_precision=envs.TPU_DCP_RS_OUT_PRECISION,
            q_gathered=q_gathered,
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

    def _attend_dec_jax(
        dcp_rank: jax.Array,
        q: jax.Array,
        kv_cache: jax.Array,
        scale_cache: jax.Array,
        candidates: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        q_dec: jax.Array,
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
            q_dec,
        )
        return body(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::minimax_m3_msa_dcp_heads_attend_{suffix}_r{decode_rows}"
        + ("_qg" if q_gathered else "")
        + ("_qd" if q_dec else ""),
        _attend_dec_jax if q_dec else _attend_jax,
        mesh=mesh,
        input_partition_specs=specs,
        # A sequence, one spec per output (a bare spec reads as its axes).
        output_partition_specs=(sharded,),
    )
    if q_gathered:
        op.register_fake(
            lambda dcp_rank, q, *args: q.new_empty(
                (q.shape[0], q.shape[1] // dcp, q.shape[2])
            )
        )
    else:
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
