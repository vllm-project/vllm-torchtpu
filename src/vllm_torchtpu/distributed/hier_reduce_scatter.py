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
"""The EP MoE combine's reduce-scatter on the hierrs_tc kernels.

The EP mesh is in ascending device id, but `group.reduce_scatter` hands rank r
the r-th row block, so mesh index i must receive the block of the rank there.
"""

from typing import Any, NamedTuple

import numpy as np
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

OP_NAME = "pallas::hier_reduce_scatter"

STREAM_MIN_BYTES = 16 << 20
# Below this many input bytes per device the FP8 quantize pass costs more than
# the halved wire bytes save on tpu7x-8, so the call keeps the BF16 wire.
FP8_MIN_BYTES = 16 << 20


class _Armed(NamedTuple):
    op: Any
    ranks: tuple[int, ...]


_ARMED: _Armed | None = None


def _unsupported_world_reason() -> str | None:
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.distributed.ep_mesh import get_ep_group
    from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
        supports_num_devices,
    )

    ep = get_ep_group()
    if ep is None:
        return "expert parallelism is off"
    if not supports_num_devices(int(ep.world_size)):
        return f"the kernel does not support an EP group of {ep.world_size}"
    if int(ep.world_size) != int(dist.get_world_size()):
        return "the EP group does not span the whole torch.distributed world"
    if list(get_dp_group().ranks) != list(ep.ranks):
        return "the DP group's ranks differ from the EP group's"
    return None


def make_reduce_scatter_fn(
    mesh: Any,
    order: tuple[int, ...],
    *,
    axis_name: str,
    fp8_comm: bool = False,
    fp8_static_scale: float | None = None,
    fp8_min_bytes: int = FP8_MIN_BYTES,
    stream: bool = True,
    stream_min_bytes: int = STREAM_MIN_BYTES,
):
    """Reduce-scatter in which mesh index i gets the block of rank `order[i]`."""
    import jax
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.collectives.hierrs_tc.kernel_stream import (
        reduce_scatter_stream_local,
        supports_stream,
    )
    from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
        hierarchical_reduce_scatter_local,
    )

    num_devices = len(order)
    perm = np.asarray(order, dtype=np.int32)
    is_identity = bool(np.array_equal(perm, np.arange(num_devices)))

    def permute(part: jax.Array) -> jax.Array:
        if is_identity:
            return part
        rows, hidden = part.shape
        blocks = part.reshape(num_devices, rows // num_devices, hidden)
        return blocks[perm].reshape(rows, hidden)

    if fp8_static_scale is not None and fp8_static_scale <= 0:
        raise ValueError(f"fp8_static_scale must be > 0, got {fp8_static_scale}")

    def local(part: jax.Array) -> jax.Array:
        rows = part.shape[0]
        fp8 = fp8_comm and part.nbytes >= fp8_min_bytes
        use_stream = (
            stream and part.nbytes >= stream_min_bytes and supports_stream(rows)
        )
        if use_stream and not fp8:
            return reduce_scatter_stream_local(part, axis_name=axis_name, order=order)
        # The row-streaming kernel's FP8 wire is static-scale only.
        if use_stream and fp8_static_scale is not None:
            return reduce_scatter_stream_local(
                part,
                axis_name=axis_name,
                order=order,
                fp8_comm=True,
                fp8_static_scale=fp8_static_scale,
            )
        return hierarchical_reduce_scatter_local(
            permute(part),
            num_devices=num_devices,
            axis_name=axis_name,
            fp8_comm=fp8,
            fp8_static_scale=fp8_static_scale,
        )

    def reduce_scatter(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=False,
        )(x)

    return reduce_scatter


def prebuild_hier_reduce_scatter() -> bool:
    """Returns whether the op is armed. Call outside Dynamo: it runs a collective."""
    global _ARMED
    if _ARMED is not None or not envs.TPU_MOE_EP_HIER_RS:
        return _ARMED is not None
    fp8 = envs.TPU_MOE_EP_HIER_RS_ICI_DTYPE == "fp8"
    scale = envs.TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE
    if scale is not None and scale <= 0:
        raise ValueError(
            f"TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE must be a positive number, got "
            f"{scale}. Unset it for per-chunk dynamic scaling."
        )
    reason = _unsupported_world_reason()
    if reason is not None:
        logger.warning_once(
            "TPU_MOE_EP_HIER_RS is set, but the MoE combine keeps the group's "
            "reduce-scatter: %s.",
            reason,
        )
        return False

    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.ep_mesh import (
        EP_AXIS_NAME,
        build_ep_mesh,
        ep_rank_order,
        get_ep_group,
    )
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    mesh = build_ep_mesh(axis_name=EP_AXIS_NAME)
    order = ep_rank_order()
    ep = get_ep_group()
    assert mesh is not None and order is not None and ep is not None
    reduce_scatter = make_reduce_scatter_fn(
        mesh,
        order,
        axis_name=EP_AXIS_NAME,
        fp8_comm=fp8,
        fp8_static_scale=scale,
    )
    op = sharded_jax_op(
        OP_NAME,
        reduce_scatter,
        mesh=mesh,
        input_partition_specs=(P(EP_AXIS_NAME, None),),
        output_partition_specs=P(EP_AXIS_NAME, None),
    )
    num_devices = len(order)
    op.register_fake(lambda x: x.new_empty((x.shape[0] // num_devices, x.shape[1])))
    _ARMED = _Armed(op=op, ranks=tuple(int(r) for r in ep.ranks))
    logger.info_once(
        "MoE combine reduce-scatter runs on hierrs_tc: %d devices, ici_dtype=%s%s, "
        "rank at each mesh index=%s.",
        num_devices,
        envs.TPU_MOE_EP_HIER_RS_ICI_DTYPE,
        ""
        if not fp8
        else f", static scale {scale}"
        if scale is not None
        else ", dynamic scale (micro-batched kernel only)",
        order,
    )
    return True


def moe_combine_reduce_scatter(group: Any, x: torch.Tensor) -> torch.Tensor:
    """`group.reduce_scatter(x, dim=0)`, on hierrs_tc when armed and eligible."""
    from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
        is_supported_by_hierrs_tc,
    )

    armed = _ARMED
    if (
        armed is not None
        and tuple(int(r) for r in group.ranks) == armed.ranks
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and is_supported_by_hierrs_tc(len(armed.ranks), x.shape[0], x.shape[1])
    ):
        return armed.op(x)
    return group.reduce_scatter(x, dim=0)
