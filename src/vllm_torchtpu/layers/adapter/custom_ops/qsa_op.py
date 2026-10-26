# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for sparse attention and its native pool."""

import functools
from collections.abc import Callable

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
    pooled_qsa_rows_step,
    pooled_qsa_shard_attend_rows_step,
    pooled_qsa_shard_attend_step,
    pooled_qsa_shard_merge,
    pooled_qsa_shard_pages_step,
    pooled_qsa_shard_select_rows_step,
    pooled_qsa_shard_select_step,
    pooled_qsa_shared_rows_step,
    pooled_qsa_shared_step,
    pooled_qsa_step,
    pooled_qsa_token_select_step,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import qsa_step


@functools.cache
def build_qsa_op() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    op = pallas.jax_op(
        "qwen4_exp::qsa",
        qsa_step,
        donate_argnums=(10, 11, 12, 13),
    )

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        packed_kv,
        compressed,
        ring,
        tags,
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
    ):
        return (
            *(torch.empty_like(x) for x in (packed_kv, compressed, ring, tags)),
            qgate.new_empty((qgate.shape[0], qgate.shape[1] // 2)),
            positions.new_empty((qgate.shape[0], token_topk + compress_ratio - 1)),
        )

    def run(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        packed_kv,
        compressed,
        ring,
        tags,
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
    ):
        planes = torch.stack((tags.to(torch.int32), (tags >> 32).to(torch.int32)), -1)
        packed, c, r, p, output, selected = op(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions.to(torch.int32),
            trig,
            packed_kv,
            compressed,
            ring,
            planes,
            starts.to(torch.int32),
            context.to(torch.int32),
            slots.to(torch.int32),
            main_table.to(torch.int32),
            compressed_table.to(torch.int32),
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            k_scale,
            v_scale,
        )
        for owner, updated in zip((packed_kv, compressed, ring), (packed, c, r)):
            owner.copy_(updated)
        tags.copy_(
            (p[..., 1].to(torch.int64) << 32) | (p[..., 0].to(torch.int64) & 0xFFFFFFFF)
        )
        return output, selected

    return run


@functools.cache
def build_pooled_qsa_op():
    op = pallas.jax_op("qwen4_exp::pooled_qsa", pooled_qsa_step, donate_argnums=(10,))

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
        compressed_slot=0,
        main_slot=-1,
        max_query_len=1,
    ):
        return (
            torch.empty_like(pool),
            qgate.new_empty((qgate.shape[0], qgate.shape[-1] // 2)),
            positions.new_empty(
                (qgate.shape[0], token_topk + compress_ratio - 1), dtype=torch.int32
            ),
        )

    def run(*args, **kwargs):
        args = list(args)
        for index in (8, 11, 12, 13, 14, 15):
            args[index] = args[index].to(torch.int32)
        updated, output, blocks = op(*args, **kwargs)
        args[10].copy_(updated)
        return output, blocks

    return run


@functools.cache
def build_pooled_qsa_select_op() -> Callable[..., tuple[torch.Tensor, ...]]:
    """``build_pooled_qsa_op`` up to the indexer, for an attention computed in the
    Qwen3.8 segment megakernel: returns ``(q, gate, pages, kv_lens)``."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import pooled_qsa_select_step

    op = pallas.jax_op(
        "qwen4_exp::pooled_qsa_select", pooled_qsa_select_step, donate_argnums=(10,)
    )

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
        compressed_slot=0,
        main_slot=-1,
        max_query_len=1,
    ):
        tokens, dim = qgate.shape[0], nq.shape[-1]
        heads = qgate.shape[-1] // 2 // dim
        budget = token_topk // compress_ratio
        return (
            torch.empty_like(pool),
            qgate.new_empty((tokens, heads, dim), dtype=torch.bfloat16),
            qgate.new_empty((tokens, heads, dim), dtype=torch.bfloat16),
            positions.new_empty((tokens, budget + 1), dtype=torch.int32),
            positions.new_empty((tokens,), dtype=torch.int32),
        )

    def run(*args, **kwargs):
        args = list(args)
        for index in (8, 11, 12, 13, 14, 15):
            args[index] = args[index].to(torch.int32)
        updated, q, gate, pages, kv_lens = op(*args, **kwargs)
        args[10].copy_(updated)
        return q, gate, pages, kv_lens

    return run


@functools.cache
def build_pooled_qsa_rows_op() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """``build_pooled_qsa_op`` attending only the trailing ``rows`` operand."""
    op = pallas.jax_op(
        "qwen4_exp::pooled_qsa_rows", pooled_qsa_rows_step, donate_argnums=(10,)
    )

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        rows,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
        compressed_slot=0,
        main_slot=-1,
        max_query_len=1,
    ):
        return (
            torch.empty_like(pool),
            qgate.new_empty((rows.shape[0], qgate.shape[-1] // 2)),
            positions.new_empty(
                (rows.shape[0], token_topk + compress_ratio - 1), dtype=torch.int32
            ),
        )

    def run(*args, **kwargs):
        args = list(args)
        for index in (8, 11, 12, 13, 14, 15, 16):
            args[index] = args[index].to(torch.int32)
        updated, output, blocks = op(*args, **kwargs)
        args[10].copy_(updated)
        return output, blocks

    return run


@functools.cache
def build_pooled_qsa_shared_op(with_rows: bool):
    """``build_pooled_qsa_op`` (``with_rows``: ``build_pooled_qsa_rows_op``)
    taking a trailing ``shared`` operand: another layer's ``selected`` for
    the attended rows, used in place of this layer's indexer selection."""
    suffix = "_rows" if with_rows else ""
    op = pallas.jax_op(
        f"qwen4_exp::pooled_qsa_shared{suffix}",
        pooled_qsa_shared_rows_step if with_rows else pooled_qsa_shared_step,
        donate_argnums=(10,),
    )

    def shapes(qgate, positions, pool, shared, compress_ratio, token_topk):
        rows = shared.shape[0]
        return (
            torch.empty_like(pool),
            qgate.new_empty((rows, qgate.shape[-1] // 2)),
            positions.new_empty(
                (rows, token_topk + compress_ratio - 1), dtype=torch.int32
            ),
        )

    if with_rows:

        @op.register_fake
        def fake_rows(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions,
            trig,
            pool,
            starts,
            context,
            ring_ids,
            main_table,
            compressed_table,
            rows,
            shared,
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            k_scale=1.0,
            v_scale=1.0,
            compressed_slot=0,
            main_slot=-1,
            max_query_len=1,
        ):
            return shapes(qgate, positions, pool, shared, compress_ratio, token_topk)

    else:

        @op.register_fake
        def fake(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions,
            trig,
            pool,
            starts,
            context,
            ring_ids,
            main_table,
            compressed_table,
            shared,
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            k_scale=1.0,
            v_scale=1.0,
            compressed_slot=0,
            main_slot=-1,
            max_query_len=1,
        ):
            return shapes(qgate, positions, pool, shared, compress_ratio, token_topk)

    ints = (8, 11, 12, 13, 14, 15, 16, 17) if with_rows else (8, 11, 12, 13, 14, 15, 16)

    def run(*args, **kwargs):
        args = list(args)
        for index in ints:
            args[index] = args[index].to(torch.int32)
        updated, output, blocks = op(*args, **kwargs)
        args[10].copy_(updated)
        return output, blocks

    return run


def _all_gather_ranks(x: torch.Tensor, group) -> torch.Tensor:
    """``[...]`` on every rank -> ``[ranks, ...]`` in rank order."""
    return group.all_gather(x.unsqueeze(0).contiguous(), dim=0)


def _all_to_all(x: torch.Tensor, group) -> torch.Tensor:
    """Block ``d`` of the leading axis goes to rank ``d``; block ``r`` of the
    result came from rank ``r``."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


@functools.cache
def _shard_ops(with_rows: bool):
    suffix = "_rows" if with_rows else ""
    select = pallas.jax_op(
        f"qwen4_exp::pooled_qsa_shard_select{suffix}",
        (
            pooled_qsa_shard_select_rows_step
            if with_rows
            else pooled_qsa_shard_select_step
        ),
        donate_argnums=(10,),
    )
    attend = pallas.jax_op(
        f"qwen4_exp::pooled_qsa_shard_attend{suffix}",
        (
            pooled_qsa_shard_attend_rows_step
            if with_rows
            else pooled_qsa_shard_attend_step
        ),
        donate_argnums=(7,),
    )

    def select_shapes(qgate, nq, positions, pool, starts, context, rows, ratio, topk):
        tokens, slots = qgate.shape[0], starts.shape[0]
        if rows is not None and rows.shape[0] <= context.shape[0]:
            tokens, slots = rows.shape[0], rows.shape[0] + 1
        head_dim = nq.shape[0]
        heads = qgate.shape[1] // (2 * head_dim)
        return (
            torch.empty_like(pool),
            qgate.new_empty((tokens, heads, head_dim)),
            qgate.new_empty((tokens, heads, head_dim)),
            positions.new_empty((tokens,), dtype=torch.int32),
            positions.new_empty((tokens,), dtype=torch.int32),
            positions.new_empty((slots,), dtype=torch.int32),
            positions.new_empty((tokens, topk // ratio, 2), dtype=torch.int32),
        )

    def attend_shapes(q, lengths, context, pool, rows, ratio, topk):
        tokens = q.shape[0]
        if rows is not None and rows.shape[0] > context.shape[0]:
            tokens = rows.shape[0]
        return (
            torch.empty_like(pool),
            q.new_empty((tokens, q.shape[1] * q.shape[2])),
            lengths.new_empty((tokens, topk + ratio - 1), dtype=torch.int32),
        )

    if with_rows:

        @select.register_fake
        def select_rows_fake(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions,
            trig,
            pool,
            starts,
            context,
            ring_ids,
            main_table,
            compressed_table,
            rows,
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            shard_rank,
            shard_size,
            k_scale=1.0,
            v_scale=1.0,
            compressed_slot=0,
            main_slot=-1,
            max_query_len=1,
        ):
            return select_shapes(
                qgate,
                nq,
                positions,
                pool,
                starts,
                context,
                rows,
                compress_ratio,
                token_topk,
            )

        @attend.register_fake
        def attend_rows_fake(
            q,
            gate,
            lengths,
            req,
            starts,
            main_table,
            context,
            pool,
            gathered,
            rows,
            kv_heads,
            compress_ratio,
            token_topk,
            k_scale=1.0,
            v_scale=1.0,
            main_slot=-1,
            max_query_len=1,
            merged=False,
        ):
            return attend_shapes(
                q, lengths, context, pool, rows, compress_ratio, token_topk
            )

    else:

        @select.register_fake
        def select_fake(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions,
            trig,
            pool,
            starts,
            context,
            ring_ids,
            main_table,
            compressed_table,
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            shard_rank,
            shard_size,
            k_scale=1.0,
            v_scale=1.0,
            compressed_slot=0,
            main_slot=-1,
            max_query_len=1,
        ):
            return select_shapes(
                qgate,
                nq,
                positions,
                pool,
                starts,
                context,
                None,
                compress_ratio,
                token_topk,
            )

        @attend.register_fake
        def attend_fake(
            q,
            gate,
            lengths,
            req,
            starts,
            main_table,
            context,
            pool,
            gathered,
            kv_heads,
            compress_ratio,
            token_topk,
            k_scale=1.0,
            v_scale=1.0,
            main_slot=-1,
            max_query_len=1,
            merged=False,
        ):
            return attend_shapes(
                q, lengths, context, pool, None, compress_ratio, token_topk
            )

    return select, attend


@functools.cache
def _merge_op():
    merge = pallas.jax_op("qwen4_exp::pooled_qsa_shard_merge", pooled_qsa_shard_merge)

    @merge.register_fake
    def merge_fake(received, compress_ratio, token_topk):
        return received.new_empty(
            (received.shape[1], token_topk // compress_ratio), dtype=torch.int32
        )

    return merge


def _exchange_candidates(candidates, tp_group, a2a_min_tokens, merge_op, ratio, topk):
    """Cross the TP group with every rank's local top-k candidates.

    Returns ``(gathered, merged)``: ``[ranks, tokens, k, 2]`` candidates, or,
    for steps of at least ``a2a_min_tokens`` tokens (0: never), ``[tokens, k]``
    blocks merged per 1/TP token block after an all-to-all and all-gathered.
    """
    world = int(tp_group.world_size)
    tokens = candidates.shape[0]
    # Fixed at trace time: each token bucket compiles its own graph.
    if a2a_min_tokens and tokens >= a2a_min_tokens and tokens % world == 0:
        received = _all_to_all(
            candidates.reshape(world, tokens // world, *candidates.shape[1:]),
            tp_group,
        )
        blocks = merge_op(received, ratio, topk)
        return _all_gather_ranks(blocks, tp_group).reshape(tokens, -1), True
    return _all_gather_ranks(candidates, tp_group), False


def build_pooled_qsa_shard_op(tp_group, a2a_min_tokens: int = 0):
    """``build_pooled_qsa_op`` with the indexer split across ``tp_group``.

    Each rank scores an interleaved 1/TP of every request's compressed keys
    and keeps its local top-k (scores and global positions). The candidates
    cross the TP group once, then every rank merges the exact global top-k
    and attends its own heads. Steps of at least ``a2a_min_tokens`` tokens
    (0: never) instead merge a 1/TP token block per rank after an
    all-to-all, then all-gather the merged blocks.

    The returned callable takes the pooled op's operands plus an optional
    trailing ``rows`` operand (as ``build_pooled_qsa_rows_op``) and a
    ``kv_heads`` keyword.
    """
    world = int(tp_group.world_size)
    rank = int(tp_group.rank_in_group)
    # Build the Pallas ops now, at layer construction: jax_op inspects the
    # step function's signature, which Dynamo cannot trace inside forward.
    shard_ops = {with_rows: _shard_ops(with_rows) for with_rows in (False, True)}
    merge_op = _merge_op()

    def run(*args, kv_heads, **kwargs):
        args = list(args)
        for index in (8, 11, 12, 13, 14, 15, 16):
            if index < len(args):
                args[index] = args[index].to(torch.int32)
        with_rows = len(args) == 17
        select, attend = shard_ops[with_rows]
        pool = args[10]
        updated, q, gate, lengths, req, starts, candidates = select(
            *args, shard_rank=rank, shard_size=world, **kwargs
        )
        pool.copy_(updated)
        ratio, topk = kwargs["compress_ratio"], kwargs["token_topk"]
        gathered, merged = _exchange_candidates(
            candidates, tp_group, a2a_min_tokens, merge_op, ratio, topk
        )
        attend_kwargs = {
            name: kwargs[name]
            for name in ("k_scale", "v_scale", "main_slot", "max_query_len")
            if name in kwargs
        }
        updated, output, selected = attend(
            q,
            gate,
            lengths,
            req,
            starts,
            args[14],
            args[12],
            pool,
            gathered,
            *args[16:],
            kv_heads=kv_heads,
            compress_ratio=ratio,
            token_topk=topk,
            merged=merged,
            **attend_kwargs,
        )
        pool.copy_(updated)
        return output, selected

    return run


@functools.cache
def _token_select_op():
    select = pallas.jax_op(
        "qwen4_exp::pooled_qsa_token_select",
        pooled_qsa_token_select_step,
        donate_argnums=(10,),
    )

    @select.register_fake
    def select_fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        shard_index,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        shard_size,
        k_scale=1.0,
        v_scale=1.0,
        compressed_slot=0,
        main_slot=-1,
        max_query_len=1,
    ):
        tokens, head_dim = qgate.shape[0], nq.shape[0]
        heads = qgate.shape[1] // (2 * head_dim)
        return (
            torch.empty_like(pool),
            qgate.new_empty((tokens, heads, head_dim)),
            qgate.new_empty((tokens, heads, head_dim)),
            positions.new_empty((tokens,), dtype=torch.int32),
            positions.new_empty((tokens,), dtype=torch.int32),
            positions.new_empty((starts.shape[0],), dtype=torch.int32),
            positions.new_empty(
                (tokens // shard_size, token_topk // compress_ratio), dtype=torch.int32
            ),
        )

    return select


def build_pooled_qsa_token_shard_op(tp_group):
    """``build_pooled_qsa_op`` with the indexer split over the step's tokens.

    For the prefill sequence-parallel backbone, whose QSA input is the whole
    gathered step on every rank: rank ``r`` indexes token rows
    ``[r * T / TP, (r + 1) * T / TP)``, the ``[T / TP, k]`` blocks are
    all-gathered (16 MB per layer at T = 8192, TP 8) and every rank attends its
    heads over all tokens. The selection is the replicated indexer's.

    The returned callable takes the pooled op's operands plus a trailing
    ``shard_index`` operand (int32 ``[1]``, the rank) and a ``kv_heads``
    keyword. Use it for every bucket of a graph or none (the buckets share
    traced artifacts, and the all-gather moves the graph's split points):
    each bucket's ``T`` must divide by TP and its shard must keep the ragged
    prefill indexer (more rows than request slots), as the module checks
    once for the graph's smallest bucket.
    """
    world = int(tp_group.world_size)
    select = _token_select_op()
    _, attend = _shard_ops(False)

    def run(*args, shard_index, kv_heads, **kwargs):
        tokens, slots = args[0].shape[0], args[12].shape[0]
        assert tokens % world == 0 and tokens // world > slots, (
            f"token-sharded indexer: {tokens} tokens do not shard over TP "
            f"{world} with more rows than {slots} request slots"
        )
        args = list(args)
        for index in (8, 11, 12, 13, 14, 15):
            args[index] = args[index].to(torch.int32)
        pool = args[10]
        updated, q, gate, lengths, req, starts, blocks = select(
            *args, shard_index, shard_size=world, **kwargs
        )
        pool.copy_(updated)
        attend_kwargs = {
            name: kwargs[name]
            for name in ("k_scale", "v_scale", "main_slot", "max_query_len")
            if name in kwargs
        }
        updated, output, selected = attend(
            q,
            gate,
            lengths,
            req,
            starts,
            args[14],
            args[12],
            pool,
            _all_gather_ranks(blocks, tp_group).reshape(tokens, -1),
            kv_heads=kv_heads,
            compress_ratio=kwargs["compress_ratio"],
            token_topk=kwargs["token_topk"],
            merged=True,
            **attend_kwargs,
        )
        pool.copy_(updated)
        return output, selected

    return run


@functools.cache
def _shard_pages_op():
    pages = pallas.jax_op(
        "qwen4_exp::pooled_qsa_shard_pages", pooled_qsa_shard_pages_step
    )

    @pages.register_fake
    def pages_fake(
        q,
        gate,
        lengths,
        req,
        main_table,
        context,
        gathered,
        compress_ratio,
        token_topk,
        page_size,
        merged=False,
    ):
        tokens = q.shape[0]
        return (
            q.new_empty(q.shape, dtype=torch.bfloat16),
            gate.new_empty(gate.shape, dtype=torch.bfloat16),
            lengths.new_empty((tokens, token_topk // compress_ratio + 1)),
            lengths.new_empty((tokens,)),
        )

    return pages


def build_pooled_qsa_shard_select_op(tp_group, a2a_min_tokens: int = 0):
    """``build_pooled_qsa_select_op`` with the indexer split across ``tp_group``
    (as ``build_pooled_qsa_shard_op``): returns ``(q, gate, pages, kv_lens)``
    for an attention computed in the Qwen3.8 segment megakernel."""
    world = int(tp_group.world_size)
    rank = int(tp_group.rank_in_group)
    # Build the Pallas ops now, at layer construction (see build_pooled_qsa_shard_op).
    select, _ = _shard_ops(False)
    merge_op = _merge_op()
    pages_op = _shard_pages_op()

    def run(*args, **kwargs):
        args = list(args)
        for index in (8, 11, 12, 13, 14, 15):
            args[index] = args[index].to(torch.int32)
        pool = args[10]
        updated, q, gate, lengths, req, starts, candidates = select(
            *args, shard_rank=rank, shard_size=world, **kwargs
        )
        pool.copy_(updated)
        ratio, topk = kwargs["compress_ratio"], kwargs["token_topk"]
        gathered, merged = _exchange_candidates(
            candidates, tp_group, a2a_min_tokens, merge_op, ratio, topk
        )
        return pages_op(
            q,
            gate,
            lengths,
            req,
            args[14],
            args[12],
            gathered,
            ratio,
            topk,
            pool.shape[1],
            merged=merged,
        )

    return run
