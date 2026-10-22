# SPDX-License-Identifier: Apache-2.0
"""Torch boundaries for encoded PLE embedding lookup."""

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.ple import (
    decode_embeddings,
    resident_lookup_raw,
    short_conv,
)
from vllm_torchtpu.kernels.qwen4_exp.pool import pooled_short_conv


@functools.cache
def build_raw_lookup_op(vocab_start: int, local_rows: int, head_dim: int):
    def fn(table: jax.Array, ids: jax.Array) -> jax.Array:
        return resident_lookup_raw(
            table,
            ids,
            vocab_start=vocab_start,
            local_rows=local_rows,
            head_dim=head_dim,
        )

    op = pallas.jax_op(
        f"qwen4_exp::ple_raw_lookup_{vocab_start}_{local_rows}_{head_dim}", fn
    )

    @op.register_fake
    def fake(table, ids):
        dtype = torch.int16 if table.dtype == torch.int16 else torch.int8
        return table.new_empty((ids.shape[0], ids.shape[1] * head_dim), dtype=dtype)

    return op


@functools.cache
def build_decode_op():
    def fn(raw_embeddings: jax.Array, scale: jax.Array) -> jax.Array:
        return decode_embeddings(raw_embeddings, scale)

    op = pallas.jax_op("qwen4_exp::ple_decode", fn)

    @op.register_fake
    def fake(raw_embeddings, scale):
        return torch.empty_like(raw_embeddings, dtype=torch.bfloat16)

    return op


@functools.cache
def build_short_conv_op(dilation: int):
    def fn(
        x: jax.Array,
        weight: jax.Array,
        history: jax.Array,
        starts: jax.Array,
        slots: jax.Array,
        initial: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        output, updated = short_conv(
            x, weight, history, starts, slots, initial, dilation=dilation
        )
        return updated, output

    op = pallas.jax_op(f"qwen4_exp::ple_short_conv_{dilation}", fn, donate_argnums=(2,))

    @op.register_fake
    def fake(x, weight, history, starts, slots, initial):
        return torch.empty_like(history), torch.empty_like(x)

    def run(x, weight, history, starts, slots, initial):
        updated, output = op(x, weight, history, starts, slots, initial)
        history.copy_(updated)
        return output

    return run


@functools.cache
def build_pooled_short_conv_op(dilation, state_len):
    fn = functools.partial(pooled_short_conv, dilation=dilation, state_len=state_len)
    op = pallas.jax_op(
        f"qwen4_exp::pooled_short_conv_{dilation}_{state_len}", fn, donate_argnums=(2,)
    )

    @op.register_fake
    def fake(x, weight, pool, starts, ids, initial):
        return torch.empty_like(pool), torch.empty_like(x)

    def run(x, weight, pool, starts, ids, initial):
        updated, output = op(x, weight, pool, starts, ids, initial)
        pool.copy_(updated)
        return output

    return run
