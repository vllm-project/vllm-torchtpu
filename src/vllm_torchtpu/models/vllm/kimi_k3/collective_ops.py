# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch bridges for K3's sequence-parallel prefill collectives."""
from functools import lru_cache

import jax
import torch
from jax.sharding import PartitionSpec as P
from vllm.distributed import get_tp_group

from vllm_torchtpu.distributed.ep_mesh import (build_ep_mesh, ep_rank_order,
                                               get_ep_group)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

from vllm_torchtpu.kernels.kimi_k3.collectives import (  # isort: skip
    gather_moe_inputs, gather_project, mla_chip_layout, mla_preprocess,
    pack_mla_gate, project_reduce_scatter, ring_collective, ring_tables)

_OPS = {}


def _build_mla_ops(mesh, owners, layer, eps):
    key = ('mla_preprocess', layer, eps,
           tuple(d.id for d in mesh.devices.flat), tuple(owners))
    if key in _OPS:
        return _OPS[key]
    pairs, _, _ = mla_chip_layout(list(mesh.devices.flat), owners)

    def pack(weight: jax.Array) -> jax.Array:
        return pack_mla_gate(weight, mesh, pairs)

    pack_op = sharded_jax_op(f'pallas::kimi_mla_pack_gate_{layer}',
                             pack,
                             mesh=mesh,
                             input_partition_specs=(P('k3_sp', None), ),
                             output_partition_specs=P('k3_sp', None))
    pack_op.register_fake(lambda weight: weight.new_empty((7168, 6144)))

    def preprocess(
        x: jax.Array, weight: jax.Array, packed_gate: jax.Array,
        qnorm: jax.Array, kvnorm: jax.Array, sequence_parallel: bool
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return mla_preprocess(x,
                              weight,
                              packed_gate,
                              qnorm,
                              kvnorm,
                              mesh,
                              owners,
                              sequence_parallel=sequence_parallel,
                              eps=eps)

    op = sharded_jax_op(f'pallas::kimi_mla_preprocess_{layer}',
                        preprocess,
                        mesh=mesh,
                        input_partition_specs=(P('k3_sp', None), ) * 3 +
                        (P(), P(), P()),
                        output_partition_specs=(P('k3_sp', None), ) * 4)

    def fake(x, weight, packed_gate, qnorm, kvnorm, sequence_parallel):
        rows = x.shape[0] * 32 if sequence_parallel else x.shape[0]
        return tuple(
            x.new_empty((rows, width)) for width in (1536, 512, 64, 384))

    op.register_fake(fake)
    _OPS[key] = pack_op, op
    return pack_op, op


def _build_moe_gather_op(mesh, tables, collective_id):
    key = ('moe_gather', collective_id, tuple(d.id for d in mesh.devices.flat),
           tuple(tables.ravel()))
    if key in _OPS:
        return _OPS[key]

    def gather(x: jax.Array, weights: jax.Array,
               ids: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return gather_moe_inputs(x,
                                 weights,
                                 ids,
                                 mesh,
                                 tables,
                                 collective_id=collective_id)

    op = sharded_jax_op(f'pallas::kimi_moe_ag_{collective_id}',
                        gather,
                        mesh=mesh,
                        input_partition_specs=(P('k3_sp', None), ) * 3,
                        output_partition_specs=(P(), ) * 3)

    def fake(x, weights, ids):
        return tuple(
            a.new_empty((a.shape[0] * 32, a.shape[1]))
            for a in (x, weights, ids))

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_op(mesh, tables, gather, collective_id):
    key = (gather, collective_id, tuple(d.id for d in mesh.devices.flat),
           tuple(tables.ravel()))
    if key in _OPS:
        return _OPS[key]

    def collective(x: jax.Array) -> jax.Array:
        # The bridge assigns a partitioned global shape to this rank's input.
        # RS represents each rank's full-token partial as one leading slice.
        if not gather:
            x = x.reshape(32, x.shape[0] // 32, x.shape[1])
        return ring_collective(x,
                               mesh,
                               tables,
                               gather=gather,
                               collective_id=collective_id)

    name = f"pallas::kimi_ring_{'ag' if gather else 'rs'}_{collective_id}"
    op = sharded_jax_op(
        name,
        collective,
        mesh=mesh,
        input_partition_specs=(P('k3_sp', None), ),
        output_partition_specs=P() if gather else P('k3_sp', None))

    def fake(x):
        rows = x.shape[0] * 32 if gather else x.shape[0] // 32
        return torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_project_op(mesh, tables, collective_id):
    key = ('project', collective_id, tuple(d.id for d in mesh.devices.flat),
           tuple(tables.ravel()))
    if key in _OPS:
        return _OPS[key]

    def project(x: jax.Array, w: jax.Array) -> jax.Array:
        return project_reduce_scatter(x.reshape(32, x.shape[0] // 32, 384),
                                      w.reshape(32, 384, 7168),
                                      mesh,
                                      tables,
                                      collective_id=collective_id)

    op = sharded_jax_op(f"pallas::kimi_project_rs_{collective_id}",
                        project,
                        mesh=mesh,
                        input_partition_specs=(P('k3_sp',
                                                 None), P('k3_sp', None)),
                        output_partition_specs=P('k3_sp', None))

    def fake(x, w):
        return torch.empty((x.shape[0] // 32, w.shape[1]),
                           dtype=x.dtype,
                           device=x.device)

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_gather_project_op(mesh, tables, collective_id):
    key = ('gather_project', collective_id,
           tuple(d.id for d in mesh.devices.flat), tuple(tables.ravel()))
    if key in _OPS:
        return _OPS[key]

    def project(x: jax.Array, w: jax.Array) -> jax.Array:
        result = gather_project(x,
                                w.reshape(32, 7168, 1792),
                                mesh,
                                tables,
                                collective_id=collective_id)
        return result.reshape(-1, 1792)

    op = sharded_jax_op(f"pallas::kimi_gather_project_{collective_id}",
                        project,
                        mesh=mesh,
                        input_partition_specs=(P('k3_sp',
                                                 None), P('k3_sp', None)),
                        output_partition_specs=P('k3_sp', None))

    def fake(x, w):
        return torch.empty((x.shape[0] * 32, w.shape[1]),
                           dtype=x.dtype,
                           device=x.device)

    op.register_fake(fake)
    _OPS[key] = op
    return op


@lru_cache(maxsize=1)
def _mesh_and_tables():
    ep = get_ep_group()
    base = get_tp_group()
    if ep is None or list(ep.ranks) != list(base.ranks):
        raise ValueError('K3 rings require identical TP32 and EP32 rank order')
    mesh = build_ep_mesh(axis_name='k3_sp')
    owners = ep_rank_order()
    return mesh, ring_tables(list(mesh.devices.flat), owners)


@lru_cache(maxsize=4)
def _build_token_op(world_size, gather):
    mesh, tables = _mesh_and_tables()
    owners = tuple(int(i) for i in tables[0, 2])
    if world_size == 2:
        groups, _, _ = mla_chip_layout(list(mesh.devices.flat), owners)
    elif world_size == 32:
        groups = (tuple(sorted(range(32), key=lambda i: owners[i])), )
    else:
        raise ValueError("K3 SP token collectives require TP2 or TP32")

    def collective(x: jax.Array) -> jax.Array:

        def local(part):
            if gather:
                return jax.lax.all_gather(part,
                                          'k3_sp',
                                          axis=0,
                                          tiled=True,
                                          axis_index_groups=groups)
            return jax.lax.psum_scatter(part,
                                        'k3_sp',
                                        scatter_dimension=0,
                                        tiled=True,
                                        axis_index_groups=groups)

        return jax.shard_map(local,
                             mesh=mesh,
                             in_specs=P('k3_sp', None),
                             out_specs=P('k3_sp', None),
                             check_vma=False)(x)

    op = sharded_jax_op(
        f"pallas::kimi_token_{world_size}_{'ag' if gather else 'rs'}",
        collective,
        mesh=mesh,
        input_partition_specs=(P('k3_sp', None), ),
        output_partition_specs=P('k3_sp', None))

    def fake(x):
        rows = x.shape[0] * world_size if gather else x.shape[0] // world_size
        return x.new_empty((rows, x.shape[1]))

    op.register_fake(fake)
    return op


class TokenShardCollectives:
    """Gather/scatter token shards over TP2 or TP32 without concrete allocations."""

    def __init__(self, base):
        self.world_size = base.world_size
        self.rank_in_group = base.rank_in_group
        self.ag = _build_token_op(self.world_size, True)
        self.rs = _build_token_op(self.world_size, False)

    def all_gather(self, x, dim):
        assert dim == 0
        return self.ag(x)

    def reduce_scatter(self, x, dim):
        assert dim == 0
        return self.rs(x)


class FusedPrefillCollectives:
    """Fused prefill projections and MoE collectives, with standard fallbacks."""

    def __init__(self, prefix):
        self.base = get_tp_group()
        mesh, tables = _mesh_and_tables()
        # Fixed layer-derived IDs keep independent ring barriers distinct.
        components = prefix.split('.')
        layer = int(components[components.index('layers') + 1])
        self.layer = layer
        self.ag = _build_op(mesh, tables, True, 20000 + 2 * layer)
        self.rs = _build_op(mesh, tables, False, 20001 + 2 * layer)
        self.project_rs = _build_project_op(mesh, tables, 22000 + layer)
        self.gather_project = _build_gather_project_op(mesh, tables,
                                                       23000 + layer)
        self._moe_gather_op = _build_moe_gather_op(mesh, tables, 24000 + layer)

    def gather_moe_inputs(self, latent, weights, ids, *, num_experts):
        """Gather token activations and routes, packing compatible BF16 inputs."""
        weights, ids = weights.to(latent.dtype), ids.to(torch.int32)
        if num_experts <= 32768 and weights.dtype == torch.bfloat16:
            return self._moe_gather_op(latent, weights, ids)
        return tuple(self.all_gather(x, dim=0) for x in (latent, weights, ids))

    def mla_ops(self, eps):
        mesh, _ = _mesh_and_tables()
        return _build_mla_ops(mesh, ep_rank_order(), self.layer, eps)

    def all_gather(self, x, dim):
        if (dim == 0 and x.ndim == 2 and x.shape[0] > 0
                and x.shape[1] in (2176, 3584, 7168)
                and x.dtype == torch.bfloat16):
            return self.ag(x)
        return self.base.all_gather(x, dim=dim)

    def reduce_scatter(self, x, dim):
        if (dim == 0 and x.ndim == 2 and x.shape[0] > 0
                and x.shape[0] % 32 == 0 and x.shape[1] in (2176, 3584, 7168)
                and x.dtype == torch.bfloat16):
            return self.rs(x)
        return self.base.reduce_scatter(x, dim=dim)

    def project_reduce_scatter(self, x, projection):
        # Preserve the linear method for other weight formats.
        weight = getattr(projection, 'weight', None)
        if (x.shape[0] > 0 and x.shape[0] % 32 == 0 and x.shape[1] == 384
                and x.dtype == torch.bfloat16 and weight is not None
                and weight.dtype == torch.bfloat16
                and weight.shape == (384, 7168)
                and getattr(projection, WEIGHT_FLIPPED_ATTR, False)
                and getattr(projection, 'bias', None) is None):
            return self.project_rs(x, weight)
        return self.reduce_scatter(projection(x)[0], dim=0)

    def all_reduce(self, x):
        return self.base.all_reduce(x)
