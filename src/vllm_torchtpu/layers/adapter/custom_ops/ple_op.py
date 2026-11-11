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
from torch.tpu import pallas
from vllm.models.qwen4_exp.common.hyperconnection import GroupedGemmaRMSNorm

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.host_lookup import (
    CHUNK_WORDS,
    ROW_CHUNKS,
    finish_host_lookup,
    start_host_lookup,
)
from vllm_torchtpu.kernels.qwen4_exp.ple import (
    decode_embeddings,
    resident_lookup_raw,
)
from vllm_torchtpu.kernels.qwen4_exp.ple_gate import normalized_gate_dot
from vllm_torchtpu.kernels.qwen4_exp.pool import pooled_short_conv


@functools.cache
def build_host_lookup_ops(local_rows: int, num_shards: int = 1):
    """Launch HOST DMA before the prefix and consume rows at the PLE layer."""
    device = pallas.get_pallas_mesh("ple_runtime").abstract_mesh.abstract_device
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("ple_host",),
        axis_types=(jax.sharding.AxisType.Auto,),
        abstract_device=device,
    )

    def start(table: jax.Array, ids: jax.Array) -> jax.Array:
        with jax.sharding.use_abstract_mesh(mesh):
            return start_host_lookup(table, ids, local_rows, num_shards=num_shards)

    start_op = pallas.jax_op(
        f"qwen4_exp::ple_host_start_{local_rows}_shards{num_shards}", start
    )

    @start_op.register_fake
    def start_fake(table, ids):
        return ids.new_empty((ids.numel() * ROW_CHUNKS * CHUNK_WORDS,))

    def start_with_host_marker(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # TorchTPU recovers HOST inputs from reload through slice/view nodes
        # when vLLM's bucket compiler supplies fresh FakeTensors. Keep a
        # 512-byte reload in the FX graph; XLA removes it and this integer
        # zero dependency, leaving only the asynchronous lookup.
        hint = torch.ops.ao.reload(table[:128], ids.device)
        return start_op(table, ids) + (hint[0] ^ hint[0])

    def finish(
        pending: jax.Array, ids: jax.Array, hidden: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        with jax.sharding.use_abstract_mesh(mesh):
            return finish_host_lookup(pending, ids, hidden, local_rows)

    finish_op = pallas.jax_op(
        f"qwen4_exp::ple_host_finish_{local_rows}_shards{num_shards}", finish
    )

    @finish_op.register_fake
    def finish_fake(pending, ids, hidden):
        return ids.new_empty(
            (*ids.shape, ROW_CHUNKS * CHUNK_WORDS), dtype=torch.int32
        ), torch.empty_like(hidden)

    return start_with_host_marker, finish_op


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
    def fake(
        x,
        weight,
        pool,
        starts,
        ids,
        initial,
        checkpoint_indices=None,
        read_offsets=None,
        request_distribution=None,
    ):
        return torch.empty_like(pool), torch.empty_like(x)

    def run(
        x,
        weight,
        pool,
        starts,
        ids,
        initial,
        checkpoint_indices=None,
        read_offsets=None,
        request_distribution=None,
    ):
        updated, output = op(
            x,
            weight,
            pool,
            starts,
            ids,
            initial,
            checkpoint_indices,
            read_offsets,
            request_distribution,
        )
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
