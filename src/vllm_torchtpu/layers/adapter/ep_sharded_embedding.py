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
"""Torch bridge for the token embedding split over every DP x TP rank
(``TPU_EMBED_EP_SHARD``).

Each DP engine otherwise keeps the whole vocabulary across its TP ranks. With
the knob on, token embedding tables (VocabParallelEmbedding, not LM heads) are
built with the EP group's size and rank, so each rank holds 1/(DP x TP) of the
rows and checkpoint loading reads only those rows;
``layers.core.ep_sharded_embedding`` serves the lookup.
Every engine runs the lookup in each forward (target and draft), which the EP
MoE layers already require of DP engines.
"""

import contextlib

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core.ep_sharded_embedding import (
    DP_AXIS,
    TP_AXIS,
    ep_embedding_lookup,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OP = None
# Attribute on an embedding module built with the EP group's sharding:
# (ep_rank, ep_size, tp_size it would have used otherwise).
EP_SHARD_ATTR = "_tpu_ep_shard"
# int32 [1] buffer: the first vocabulary row this rank holds.
EP_START_ATTR = "_tpu_ep_vocab_start"


def mark_ep_sharded(layer: torch.nn.Module, shard: tuple[int, int, int]) -> None:
    """Record the EP sharding of a freshly built embedding on the module."""
    setattr(layer, EP_SHARD_ATTR, shard)
    layer.register_buffer(
        EP_START_ATTR,
        torch.tensor([layer.shard_indices.org_vocab_start_index], dtype=torch.int32),
        persistent=False,
    )


@contextlib.contextmanager
def ep_group_as_tp(vpe_module):
    """Make VocabParallelEmbedding.__init__ shard over the EP group."""
    from vllm.distributed.parallel_state import get_ep_group, get_tp_group

    from vllm_torchtpu.layers.adapter.ragged_a2a_moe import get_dp_tp_mesh

    get_dp_tp_mesh()  # checks the rank-aligned DP x TP layout the lookup assumes
    ep, tp = get_ep_group(), get_tp_group()
    if int(tp.world_size) == 1:
        # The vocab-parallel lookup skips its all-reduce at TP=1, so it keeps
        # -0.0 rows that the DP reduce-scatter would turn into +0.0.
        raise NotImplementedError("TPU_EMBED_EP_SHARD needs tensor parallelism")
    saved = (
        vpe_module.get_tensor_model_parallel_rank,
        vpe_module.get_tensor_model_parallel_world_size,
    )
    vpe_module.get_tensor_model_parallel_rank = lambda: int(ep.rank_in_group)
    vpe_module.get_tensor_model_parallel_world_size = lambda: int(ep.world_size)
    try:
        yield int(ep.rank_in_group), int(ep.world_size), int(tp.world_size)
    finally:
        (
            vpe_module.get_tensor_model_parallel_rank,
            vpe_module.get_tensor_model_parallel_world_size,
        ) = saved


def _get_op():
    global _OP
    if _OP is not None:
        return _OP
    from vllm_torchtpu.layers.adapter.ragged_a2a_moe import get_dp_tp_mesh

    mesh = get_dp_tp_mesh()
    owned_rows = "ragged" if envs.TPU_EMBED_EP_ROWS else None

    # torch_tpu infers static argnums from annotations: every operand is jax.Array.
    def fn(ids: jax.Array, table: jax.Array, start: jax.Array) -> jax.Array:
        return ep_embedding_lookup(ids, table, start, mesh=mesh, owned_rows=owned_rows)

    spec = PartitionSpec(DP_AXIS, TP_AXIS)
    op = pallas.jax_op(
        "pallas::ep_sharded_embedding",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 3,
        output_partition_specs=(spec,),
    )
    op.register_fake(
        lambda ids, table, start: table.new_empty((*ids.shape, table.shape[-1]))
    )
    _OP = op
    logger.info(
        "EP-sharded embedding armed | dp=%d tp=%d | %s",
        mesh.shape[DP_AXIS],
        mesh.shape[TP_AXIS],
        "owned rows sent to their engines" if owned_rows else "DP reduce-scatter",
    )
    return op


def ep_embedding_forward(
    layer: torch.nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    """``layer``'s rows for this engine's ``input_ids`` (any leading shape)."""
    if (
        layer.num_added_embeddings
        or layer.org_vocab_size_padded != layer.num_embeddings_padded
    ):
        raise NotImplementedError("TPU_EMBED_EP_SHARD does not serve added vocab")
    weight = layer.weight
    ids = input_ids.reshape(-1).to(torch.int32)
    out = _get_op()(
        ids.reshape(1, 1, -1),
        weight.reshape(1, 1, *weight.shape),
        getattr(layer, EP_START_ATTR).reshape(1, 1, 1),
    )
    return out.reshape(*input_ids.shape, weight.shape[-1])


def prebuild_ep_embedding() -> None:
    """Build the mesh and op at weight load, outside the traced forward."""
    _get_op()


@torch.no_grad()
def restore_tp_dummy_rows(model: torch.nn.Module, low=-1e-3, high=1e-3, seed=1234):
    """Give EP-sharded dummy embeddings the rows of the vocab-parallel layout.

    Dummy loading draws each parameter from a generator seeded per tensor, so
    the values depend on the local shape: every TP shard of the vocab-parallel
    table holds the same draw. Redraw that TP-shard tensor and keep this
    rank's rows of the table it implies, so dummy-weight runs see the same
    embedding with or without the knob.
    """
    for module in model.modules():
        shard = getattr(module, EP_SHARD_ATTR, None)
        if shard is None:
            continue
        ep_rank, _, tp_size = shard
        weight = module.weight.data
        rows = weight.shape[0]
        tp_rows = module.num_embeddings_padded // tp_size
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(seed)
        tp_shard = torch.empty(
            (tp_rows, weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        tp_shard.uniform_(low, high, generator=generator)
        start = (ep_rank * rows) % tp_rows
        weight.copy_(tp_shard[start : start + rows])
        del tp_shard
        logger.info_once(
            "EP-sharded dummy embedding rows follow the vocab-parallel layout "
            "(%d of %d rows per rank)",
            rows,
            tp_rows,
        )
