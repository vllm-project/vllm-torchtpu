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
"""Torch boundaries for encoded PLE embedding lookup."""

import functools

import jax
import torch
from torch_tpu._internal import pallas
from vllm.models.qwen4_exp.common.hyperconnection import GroupedGemmaRMSNorm

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.ple import (
    decode_embeddings,
    resident_lookup_raw,
)
from vllm_torchtpu.kernels.qwen4_exp.ple_gate import normalized_gate_dot
from vllm_torchtpu.kernels.qwen4_exp.pool import pooled_short_conv


@functools.cache
def build_raw_lookup_op(local_rows: int, head_dim: int):
    def fn(table: jax.Array, ids: jax.Array, vocab_start: jax.Array) -> jax.Array:
        return resident_lookup_raw(
            table,
            ids,
            vocab_start=vocab_start,
            local_rows=local_rows,
            head_dim=head_dim,
        )

    op = pallas.jax_op(f"qwen4_exp::ple_raw_lookup_{local_rows}_{head_dim}", fn)

    @op.register_fake
    def fake(table, ids, vocab_start):
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


@functools.cache
def build_normalized_gate_dot_op(streams: int):
    def fn(
        key: jax.Array,
        query: jax.Array,
        key_inv: jax.Array,
        query_inv: jax.Array,
        key_weight: jax.Array,
        query_weight: jax.Array,
    ) -> jax.Array:
        return normalized_gate_dot(
            key,
            query,
            key_inv,
            query_inv,
            key_weight,
            query_weight,
            streams=streams,
        )

    op = pallas.jax_op(f"qwen4_exp::ple_gate_dot_s{streams}", fn)

    @op.register_fake
    def fake(key, query, key_inv, query_inv, key_weight, query_weight):
        return key.new_empty((key.shape[0], streams), dtype=torch.bfloat16)

    @op.register_kernel("cpu")
    def cpu(key, query, key_inv, query_inv, key_weight, query_weight):
        raise NotImplementedError("The fused PLE gate requires TPU tensors")

    return op


def build_normalized_gate_dot_dispatch(streams: int, hidden: int):
    """Select the PLE gate implementation when the layer is constructed."""
    if not envs.TPU_PLE_FUSED_GATE:

        def dot(
            key: torch.Tensor,
            query: torch.Tensor,
            norm_key: GroupedGemmaRMSNorm,
            norm_query: GroupedGemmaRMSNorm,
        ) -> torch.Tensor:
            key = norm_key(key).unflatten(-1, (streams, hidden))
            query = norm_query(query).unflatten(-1, (streams, hidden))
            return (key * query).float().sum(dim=-1, keepdim=True).to(key.dtype)

        return dot

    native = build_normalized_gate_dot_op(streams)

    def inverse_rms(raw: torch.Tensor, norm: GroupedGemmaRMSNorm) -> torch.Tensor:
        grouped = raw.float().unflatten(-1, (streams, hidden))
        return torch.rsqrt(grouped.square().mean(dim=-1) + norm.variance_epsilon)

    def dispatch(
        key: torch.Tensor,
        query: torch.Tensor,
        norm_key: GroupedGemmaRMSNorm,
        norm_query: GroupedGemmaRMSNorm,
    ) -> torch.Tensor:
        return native(
            key,
            query,
            inverse_rms(key, norm_key),
            inverse_rms(query, norm_query),
            norm_key.weight,
            norm_query.weight,
        ).unsqueeze(-1)

    return dispatch
