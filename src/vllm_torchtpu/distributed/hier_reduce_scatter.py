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
"""Arms the EP MoE combine's reduce-scatter on the hierrs_tc kernels.

`TPUModelRunner.load_model` arms it once per process from the env vars
TPU_MOE_COMBINE_HIER_RS, TPU_MOE_COMBINE_HIER_RS_FP8_WIRE and
TPU_MOE_COMBINE_HIER_RS_FP8_SCALE (all default off); every rank must make the
same decision.

Two scopes, picked from the groups:
- "ep": the EP group is the DP group (DP=8, TP=1 + EP). vLLM's MoE combine
  (`TpuDeviceCommunicator.combine`, the single-chunk pipelined MoE) calls
  `moe_combine_reduce_scatter`. Refused with a draft model: the draft graph is
  traced once for every bucket, and a call's eligibility depends on its rows.
- "tp": the EP group is the TP group (TP=8 + EP, DP=1). Only a model that
  reduce-scatters its own FFN output over TP (MiniMax-M3
  TPU_MINIMAX_M3_SP_MODE=full) calls `tp_reduce_scatter`; vLLM's combine
  keeps its collective.
  A draft model does not refuse: it never calls `tp_reduce_scatter`.

The op runs on the EP mesh. The kernel exchanges between hypercube chips
2k / 2k+1 neighbours, so when the mesh's chips are not in physical order (the
rank-ordered mesh with DCP / PCP / MOE_FUSED_EP_SORTED_GATHER) it gets a chip
permutation from the chip topology (`_chip_perm`) instead of being refused.
`group.reduce_scatter` hands rank r the r-th row block, so mesh index i
receives the block of the rank there (`ep_rank_order`). Kernel selection lives
in `kernels/collectives/hierrs_tc/dispatch.py`.
"""

from typing import Any, NamedTuple

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

OP_NAME = "pallas::hier_reduce_scatter"


class HierRsConfig(NamedTuple):
    """`wire` "off", "bf16" or "fp8" (the dtype between chips; sums stay
    bf16); `fp8_static_scale` the multiplier before the fp8 cast."""

    wire: str = "off"
    fp8_static_scale: float | None = None


def hier_rs_config() -> HierRsConfig:
    """The TPU_MOE_COMBINE_HIER_RS* env vars, checked."""
    on = envs.TPU_MOE_COMBINE_HIER_RS
    fp8 = envs.TPU_MOE_COMBINE_HIER_RS_FP8_WIRE
    scale = envs.TPU_MOE_COMBINE_HIER_RS_FP8_SCALE
    if fp8 and not on:
        raise ValueError(
            "TPU_MOE_COMBINE_HIER_RS_FP8_WIRE=1 needs TPU_MOE_COMBINE_HIER_RS=1."
        )
    if not on:
        return HierRsConfig()
    if not fp8:
        return HierRsConfig("bf16")
    if scale <= 0:
        raise ValueError(
            "TPU_MOE_COMBINE_HIER_RS_FP8_WIRE=1 needs a positive "
            f"TPU_MOE_COMBINE_HIER_RS_FP8_SCALE (got {scale}): the row-streaming "
            "kernel has no dynamic scale. Calibrate it on the model: partial "
            "sums above 448 / scale clip."
        )
    return HierRsConfig("fp8", float(scale))


class _Armed(NamedTuple):
    op: Any
    ranks: tuple[int, ...]
    scope: str = "ep"


_ARMED: _Armed | None = None
# Why arming refused, once it has; arming runs at most once per process.
_REFUSED: str | None = None


def _chip_version() -> Any:
    from jax.experimental.pallas import tpu as pltpu

    # The worker owns the TPU by the time the model loads.
    return pltpu.get_tpu_info().chip_version


def _tp_ranks() -> list[int] | None:
    try:
        from vllm.distributed.parallel_state import get_tp_group

        return [int(r) for r in get_tp_group().ranks]
    except (AssertionError, AttributeError, RuntimeError):
        return None


def _unsupported_world_reason(draft_model: bool) -> tuple[str | None, str]:
    """(refusal reason or None, scope "ep" / "tp")."""
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group
    from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
        supports_chip_topology,
        supports_num_devices,
    )

    ep = get_ep_group()
    if ep is None:
        return "expert parallelism is off", "ep"
    if not supports_num_devices(int(ep.world_size)):
        return f"the kernel does not support an EP group of {ep.world_size}", "ep"
    if int(ep.world_size) != int(dist.get_world_size()):
        return "the EP group does not span the whole torch.distributed world", "ep"
    ep_ranks = [int(r) for r in ep.ranks]
    if [int(r) for r in get_dp_group().ranks] == ep_ranks:
        scope = "ep"
        if draft_model:
            return "a draft model (MTP) is configured", scope
    elif _tp_ranks() == ep_ranks:
        scope = "tp"
    else:
        return "neither the DP nor the TP group's ranks are the EP group's", "ep"
    topology = get_chip_topology()
    if topology is None:
        return "the chip topology is unknown", scope
    if not supports_chip_topology(topology.num_chips, topology.cores_per_chip):
        return (
            f"the kernel needs 4 chips of 2 TensorCores, not {topology.num_chips} "
            f"of {topology.cores_per_chip}"
        ), scope
    from jax.experimental.pallas import tpu as pltpu

    chip = _chip_version()
    if chip != pltpu.ChipVersion.TPU_7X:
        # The VMEM limits and the kernel selection are measured on tpu7x.
        return f"the kernel is tuned for tpu7x, not {chip}", scope
    return None, scope


def _chip_perm(
    mesh: Any, chips: list[list[int]] | None
) -> tuple[tuple[int, ...] | None, str | None]:
    """(chip_perm, refusal reason): `chip_perm[k]` is the mesh chip (mesh
    indices 2m, 2m+1) on physical chip `chips[k]`; None when that is the
    identity. `chips` is `ChipTopology.chips` (sorted by ICI coordinates, so
    hypercube chips k and k ^ (1 << d) are neighbours); None takes chip k =
    devices 2k and 2k+1, which needs the mesh in ascending device id."""
    ids = [int(d.id) for d in mesh.devices.flat]
    if chips is None:
        if ids != sorted(ids):
            return None, f"the EP mesh is not in ascending device id: {ids}"
        return None, None
    chip_of = {int(i): k for k, members in enumerate(chips) for i in members}
    mesh_chips = []
    for m in range(len(ids) // 2):
        pair = {chip_of.get(ids[2 * m]), chip_of.get(ids[2 * m + 1])}
        if len(pair) != 1 or None in pair:
            return None, (
                f"EP mesh indices {2 * m} and {2 * m + 1} are not one chip's "
                f"two cores: device ids {ids}, chips {chips}"
            )
        mesh_chips.append(pair.pop())
    perm = tuple(mesh_chips.index(k) for k in range(len(mesh_chips)))
    return (None if perm == tuple(range(len(perm))) else perm), None


def _build_op(
    mesh: Any,
    order: tuple[int, ...],
    axis_name: str,
    *,
    fp8_comm: bool = False,
    fp8_static_scale: float | None = None,
    chip_perm: tuple[int, ...] | None = None,
) -> Any:
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.collectives.hierrs_tc.dispatch import (
        make_reduce_scatter_fn,
    )

    reduce_scatter = make_reduce_scatter_fn(
        mesh,
        order,
        axis_name=axis_name,
        fp8_comm=fp8_comm,
        fp8_static_scale=fp8_static_scale,
        chip_perm=chip_perm,
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(x: jax.Array) -> jax.Array:
        return reduce_scatter(x)

    spec = P(axis_name, None)
    # torch_tpu annotates the partition specs as tuples of axis-name tuples, but
    # jax_op takes PartitionSpecs at runtime, as other jax_op call sites pass them.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        OP_NAME,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=(spec,),
    )
    num_devices = len(order)
    # vLLM traces with a dynamic token dimension.
    op.register_fake(lambda x: x.new_empty((x.shape[0] // num_devices, x.shape[1])))
    return op


def _gather_reasons(reason: str | None) -> list[str | None]:
    """Every rank's refusal reason, over the world's CPU group."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [reason]
    from vllm.distributed.parallel_state import get_world_group

    seen: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(seen, reason, group=get_world_group().cpu_group)
    return seen


def _agreed_reason(reason: str | None) -> str | None:
    """Refuse on every rank if one refuses, so no rank waits on the others in
    build_ep_mesh or in the kernel."""
    seen = _gather_reasons(reason)
    refused = {r: why for r, why in enumerate(seen) if why is not None}
    if not refused:
        return None
    if len(refused) == len(seen) and len(set(refused.values())) == 1:
        return seen[0]
    return f"ranks disagree: {refused}"


def _topology_chips() -> list[list[int]] | None:
    from vllm_torchtpu.distributed.chip_topology import get_chip_topology

    topology = get_chip_topology()
    return None if topology is None else [list(c) for c in topology.chips]


def prebuild_hier_reduce_scatter(
    config: HierRsConfig | None = None, draft_model: bool = False
) -> bool:
    """Returns whether the op is armed. Call once on every rank, outside Dynamo:
    it runs collectives. `config` None is off."""
    global _ARMED, _REFUSED
    config = config or HierRsConfig()
    if _ARMED is not None or _REFUSED is not None or config.wire == "off":
        return _ARMED is not None
    fp8 = config.wire == "fp8"
    scale = config.fp8_static_scale
    if fp8 and (scale is None or scale <= 0):
        raise ValueError(
            "TPU_MOE_COMBINE_HIER_RS_FP8_WIRE needs a positive "
            f"TPU_MOE_COMBINE_HIER_RS_FP8_SCALE, got {scale}."
        )

    from vllm_torchtpu.distributed.ep_mesh import (
        EP_AXIS_NAME,
        build_ep_mesh,
        ep_rank_order,
        get_ep_group,
    )

    reason, scope = _unsupported_world_reason(draft_model)
    reason = _agreed_reason(reason)
    mesh = order = ep = perm = None
    if reason is None:
        mesh = build_ep_mesh(axis_name=EP_AXIS_NAME)
        order = ep_rank_order()
        ep = get_ep_group()
        assert mesh is not None and order is not None and ep is not None
        # build_ep_mesh gives every rank the same mesh, so they decide alike.
        perm, reason = _chip_perm(mesh, _topology_chips())
    if reason is not None:
        _REFUSED = reason
        logger.warning_once(
            "TPU_MOE_COMBINE_HIER_RS (wire %s) is set, but the MoE "
            "combine keeps the group's reduce-scatter: %s.",
            config.wire,
            reason,
        )
        return False
    assert mesh is not None and order is not None and ep is not None

    op = _build_op(
        mesh,
        order,
        EP_AXIS_NAME,
        fp8_comm=fp8,
        fp8_static_scale=scale if fp8 else None,
        chip_perm=perm,
    )
    _ARMED = _Armed(op=op, ranks=tuple(int(r) for r in ep.ranks), scope=scope)
    logger.info_once(
        "MoE combine reduce-scatter runs on hierrs_tc (%s group): %d devices, "
        "ici_dtype=%s%s, rank at each mesh index=%s, chip_perm=%s.%s",
        scope.upper(),
        len(order),
        config.wire,
        f", static scale {scale}" if fp8 else "",
        order,
        perm,
        " Layers the fused EP kernel serves combine inside it instead."
        if envs.USE_MOE_FUSED_EP_KERNEL and scope == "ep"
        else "",
    )
    return True


@torch.compiler.assume_constant_result
def _log_fallback(reason: str) -> None:
    """Trace-time log; takes no symbolic values."""
    logger.warning_once(
        "TPU_MOE_COMBINE_HIER_RS: a reduce-scatter keeps the group's collective: %s.",
        reason,
    )


def _ineligible_reason(num_devices: int, x: torch.Tensor) -> str | None:
    from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
        is_supported_by_hierrs_tc,
        supports_hidden,
    )

    if x.dim() != 2:
        return f"the input has {x.dim()} dims, not 2"
    if x.dtype != torch.bfloat16:
        return f"the input is {x.dtype}, not bfloat16"
    hidden = int(x.shape[1])
    if not supports_hidden(num_devices, hidden):
        return f"hidden size {hidden} does not split over {num_devices} devices"
    if not is_supported_by_hierrs_tc(num_devices, x.shape[0], hidden):
        return f"the row count does not give {num_devices} devices 8 rows or more each"
    return None


def _dispatch(armed: _Armed | None, group: Any, x: torch.Tensor) -> torch.Tensor:
    if armed is not None and tuple(int(r) for r in group.ranks) == armed.ranks:
        reason = _ineligible_reason(len(armed.ranks), x)
        if reason is None:
            return armed.op(x)
        _log_fallback(reason)
    return group.reduce_scatter(x, dim=0)


def moe_combine_reduce_scatter(group: Any, x: torch.Tensor) -> torch.Tensor:
    """`group.reduce_scatter(x, dim=0)`, on hierrs_tc when armed for the EP=DP
    scope and eligible."""
    armed = _ARMED
    return _dispatch(
        armed if armed is not None and armed.scope == "ep" else None, group, x
    )


def is_armed_for_tp() -> bool:
    armed = _ARMED
    return armed is not None and armed.scope == "tp"


def tp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """`tensor_model_parallel_reduce_scatter(x, 0)`, on hierrs_tc when armed
    for the TP=EP scope and eligible (a model-owned MoE combine)."""
    from vllm.distributed.parallel_state import get_tp_group

    armed = _ARMED
    return _dispatch(
        armed if armed is not None and armed.scope == "tp" else None, get_tp_group(), x
    )
