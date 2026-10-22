# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for sparse attention and its native pool."""

import functools
from collections.abc import Callable

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import pooled_qsa_step
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
            qgate.new_empty((qgate.shape[0], qgate.shape[-1] // 2)),
            positions.new_empty(
                (qgate.shape[0], token_topk + compress_ratio - 1), dtype=torch.int32
            ),
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
            tags,
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
        for owner, updated in zip(
            (packed_kv, compressed, ring, tags), (packed, c, r, p)
        ):
            owner.copy_(updated)
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
