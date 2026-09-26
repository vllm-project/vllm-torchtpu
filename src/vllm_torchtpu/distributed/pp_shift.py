# SPDX-License-Identifier: Apache-2.0
"""Hand one pipeline stage's activations to the next over ICI.

The op is an XLA collective-permute over a mesh of the pipeline group's
devices: every stage's block lands on the stage after it, and the last stage's
block lands on the first stage, which ignores it, so the permutation is one
closed cycle. Every stage must call it at the same point in its launch
sequence; ``pp_wave`` arranges that.
"""

import inspect
from typing import Any

import jax
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu.distributed.mesh_utils import (
    _collect_rank_to_device_id,
    _get_current_global_rank,
    _get_tpu_global_device_id,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

AXIS = "pp"
_MESH: Mesh | None = None
_POS_OF_RANK: tuple[int, ...] = ()
_OPS: dict[tuple, object] = {}


def get_or_create_pp_mesh() -> Mesh:
    """A JAX mesh over the pipeline group's devices in device-id order.

    The runtime addresses collective peers by device order, so mesh positions
    follow device ids; ``pp_positions()`` maps pipeline ranks onto them.
    """
    global _MESH, _POS_OF_RANK
    if _MESH is not None:
        return _MESH
    from vllm.distributed.parallel_state import get_pp_group

    group = get_pp_group()
    ranks = tuple(int(rank) for rank in group.ranks)
    rank_to_device = _collect_rank_to_device_id(
        group, _get_current_global_rank(), _get_tpu_global_device_id()
    )
    devices_by_id = {int(device.id): device for device in jax.devices()}
    device_ids = sorted(rank_to_device[rank] for rank in ranks)
    _MESH = Mesh(np.asarray([devices_by_id[d] for d in device_ids]), axis_names=(AXIS,))
    _POS_OF_RANK = tuple(device_ids.index(rank_to_device[rank]) for rank in ranks)
    logger.info(
        "PP hand-off mesh (device order): %s",
        ", ".join(
            f"rank{r}->pos{_POS_OF_RANK[i]}/dev{rank_to_device[r]}"
            for i, r in enumerate(ranks)
        ),
    )
    return _MESH


def pp_positions() -> tuple[int, ...]:
    """Mesh position of each pipeline rank, in rank order."""
    get_or_create_pp_mesh()
    return _POS_OF_RANK


def pp_permute_op(mesh: Mesh, num_tensors: int = 1):
    """Torch op carrying ``num_tensors`` [tokens, width] tensors one stage
    around the cycle in a single collective-permute, routed by mesh position.

    ``num_tensors == 1`` is ``x_out = pp_permute(x)``
    ``num_tensors > 1`` for the model that must also pass side data between stages
    e.g. a DSA model's shared top-k indices, takes a wider op: one program, one
    permute per tensor, so a hand-off is still one launch and the launch
    numbers of ``pp_wave`` keep pairing across stages. The tensors may differ
    in width and dtype; they are separate operands, not a packed buffer.
    """
    if num_tensors < 1:
        raise ValueError(f"a hand-off carries at least one tensor; got {num_tensors}")
    key = ("permute", int(num_tensors)) + tuple(int(d.id) for d in mesh.devices)
    op = _OPS.get(key)
    if op is not None:
        return op
    pos = pp_positions()
    perm = [(pos[r], pos[(r + 1) % len(pos)]) for r in range(len(pos))]
    spec = PartitionSpec(AXIS)

    if num_tensors == 1:

        def shift(x: jax.Array) -> jax.Array:
            return shard_map(
                lambda xb: jax.lax.ppermute(xb, AXIS, perm),
                mesh=mesh,
                in_specs=(spec,),
                out_specs=spec,
                check_rep=False,
            )(x)

        name = "vllm_torchtpu::pp_permute"
        output_specs: Any = spec
    else:

        def shift(*xs: jax.Array) -> tuple:
            return shard_map(
                lambda *xb: tuple(jax.lax.ppermute(x, AXIS, perm) for x in xb),
                mesh=mesh,
                in_specs=(spec,) * num_tensors,
                out_specs=(spec,) * num_tensors,
                check_rep=False,
            )(*xs)

        # The op registry reads the signature to type the torch schema, and a
        # ``*args`` one carries no count; spell the operands out.
        shift.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [
                inspect.Parameter(
                    f"x{i}",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=jax.Array,
                )
                for i in range(num_tensors)
            ],
            return_annotation=tuple[(jax.Array,) * num_tensors],
        )
        name = f"vllm_torchtpu::pp_permute{num_tensors}"
        output_specs = (spec,) * num_tensors

    op = sharded_jax_op(
        name,
        shift,
        mesh=mesh,
        input_partition_specs=(spec,) * num_tensors,
        output_partition_specs=output_specs,
    )
    _OPS[key] = op
    return op
