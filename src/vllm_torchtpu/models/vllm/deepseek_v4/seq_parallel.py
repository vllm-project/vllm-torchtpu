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
"""Token-shard collectives over the TP group for DeepSeek-V4 sequence parallelism.

vLLM's ``GroupCoordinator.all_gather`` / ``reduce_scatter`` allocate their
output with ``torch.empty`` of the (symbolic) token count, which TorchTPU's
export refuses inside the dynamic trace. These are JAX ops over the EP mesh
instead, with the TP groups as ``axis_index_groups`` (the groups list each TP
group's mesh indices in TP-rank order, so block ``r`` of a gather is TP rank
``r``'s rows -- the block ``DeepseekV4Model`` hands that rank).

Built outside the trace (the mesh and the groups come from collectives), once
per process, and shared by every layer.
"""

from __future__ import annotations

import jax
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_token_replica_groups,
)

_OPS: dict[str, object] = {}


def _build(gather: bool, mesh, groups):
    width = len(groups[0])

    def collective(x: jax.Array) -> jax.Array:
        def local(part):
            if gather:
                return jax.lax.all_gather(
                    part, EP_AXIS_NAME, axis=0, tiled=True, axis_index_groups=groups
                )
            return jax.lax.psum_scatter(
                part,
                EP_AXIS_NAME,
                scatter_dimension=0,
                tiled=True,
                axis_index_groups=groups,
            )

        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=PartitionSpec(EP_AXIS_NAME, None),
            out_specs=PartitionSpec(EP_AXIS_NAME, None),
            check_vma=False,
        )(x)

    spec = PartitionSpec(EP_AXIS_NAME, None)
    op = jax_op(
        f"pallas::dsv4_sp_tokens_w{width}_{'ag' if gather else 'rs'}",
        collective,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=spec,
    )

    def fake(x):
        rows = x.shape[0] * width if gather else x.shape[0] // width
        return x.new_empty((rows, *x.shape[1:]))

    op.register_fake(fake)
    return op


def build_sp_ops() -> dict[str, object]:
    """{'ag', 'rs'}: row all-gather / reduce-scatter of a [rows, cols] tensor
    over this rank's TP group. Must first be called outside any trace."""
    if not _OPS:
        mesh = build_ep_mesh()
        if mesh is None:
            raise RuntimeError("TPU_DSV4_SEQ_PARALLEL needs an EP mesh")
        groups = ep_token_replica_groups(is_sequence_parallel=False)
        if len(groups[0]) < 2:
            raise RuntimeError("TPU_DSV4_SEQ_PARALLEL needs a TP group")
        _OPS["ag"] = _build(True, mesh, groups)
        _OPS["rs"] = _build(False, mesh, groups)
    return _OPS
