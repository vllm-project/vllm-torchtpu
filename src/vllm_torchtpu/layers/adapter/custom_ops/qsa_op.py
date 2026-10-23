# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for sparse attention and its native pool."""

import functools
from collections.abc import Callable

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
    pooled_qsa_rows_step,
    pooled_qsa_step,
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
