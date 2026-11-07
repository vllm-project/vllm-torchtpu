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
"""All-gather over a DP pair as an XLA collective-permute.

XLA keeps a torch all-gather synchronous on TPU, so MoE dispatch waits for it
at the point the tokens are gathered. A collective-permute is issued as an
async start/done pair instead, which lets XLA start the exchange as soon as
its operand exists and finish it where the result is first read, running the
work that does not need the peer's rows (router logits, top-k, the shared
expert) in between. With two DP ranks the permute moves exactly the rows the
all-gather would, and the op orders them by DP rank as the all-gather does.

The op runs over the EP mesh, so it shares a DeviceAssignment with the other
EP-mesh ops a module may hold.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op
from vllm.distributed.parallel_state import get_dp_group

from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_dp_groups,
    get_ep_group,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Integer dtype of each element size: the rows move as integers, whose
# selects and concatenations keep every bit pattern.
_BITS_DTYPE = {2: torch.int16, 4: torch.int32}

_OP: Any = None


def exchange_fn(mesh, dp_groups: tuple[tuple[int, int], ...], dp_rank: int):
    """The JAX body: each rank's [T, ...] block in, its DP pair's [2T, ...]
    out, DP rank 0's rows first. ``dp_groups`` are mesh indices.

    The order is fixed by this process's ``dp_rank`` rather than read from
    ``lax.axis_index``: that lowers to partition-id, which XLA refuses to
    SPMD-partition in a torch_tpu module. Each rank builds its own op and runs
    only its own device's shard of it.
    """
    perm = [(a, b) for a, b in dp_groups] + [(b, a) for a, b in dp_groups]
    spec = PartitionSpec(EP_AXIS_NAME)

    def local(block: jax.Array) -> jax.Array:
        peer = jax.lax.ppermute(block, EP_AXIS_NAME, perm)
        return jnp.concatenate([block, peer] if dp_rank == 0 else [peer, block])

    def exchange(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
        )(x)

    return exchange


def prepare_dp_pair_exchange(parallel_config: Any) -> None:
    """Build the op when MoE dispatch gathers over a DP pair inside the EP mesh.

    Expert parallelism on and one pipeline stage is where the EP mesh spans
    the world, as `build_ep_mesh` needs. Collective over the EP group: call it
    on every rank after the model is loaded and before anything is traced.
    The decision reads only the shared config and gathered groups, so every
    rank takes it the same way.
    """
    global _OP
    if (
        _OP is not None
        or not parallel_config.enable_expert_parallel
        or parallel_config.data_parallel_size != 2
        or parallel_config.pipeline_parallel_size != 1
        or get_ep_group() is None
    ):
        return
    mesh = build_ep_mesh()
    assert mesh is not None
    dp_groups = ep_dp_groups()
    if sorted(i for g in dp_groups for i in g) != list(range(mesh.size)):
        logger.info(
            "DP pair exchange not built: DP groups %s do not tile the EP mesh "
            "of %d; dispatch keeps the all-gather.",
            dp_groups,
            mesh.size,
        )
        return
    dp_rank = get_dp_group().rank_in_group
    spec = PartitionSpec(EP_AXIS_NAME)
    # torch_tpu types the partition specs as tuples of axis names, but takes
    # jax PartitionSpecs (see fused_moe_decode._build_op).
    # pyrefly: ignore[no-matching-overload]
    op = jax_op(
        # The order is baked into the body, so name it into the graph: peer
        # replay shares compiled graphs between ranks whose code matches.
        f"vllm_torchtpu::dp_pair_exchange_dp{dp_rank}",
        exchange_fn(mesh, dp_groups, dp_rank),
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=spec,
    )
    op.register_fake(lambda x: x.new_empty((2 * x.shape[0], *x.shape[1:])))
    logger.info("DP pair exchange over the EP mesh | dp_groups=%s", dp_groups)
    _OP = op


def dp_pair_exchange_ready() -> bool:
    return _OP is not None


def dp_pair_all_gather(t: torch.Tensor) -> torch.Tensor:
    """``get_dp_group().all_gather(t, dim=0)``, bit for bit."""
    assert _OP is not None, "prepare_dp_pair_exchange() has not built the op"
    bits = t.contiguous().view(_BITS_DTYPE[t.element_size()])
    return _OP(bits).view(t.dtype)
