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
"""Torch bridge for the ragged all-to-all EP exchange (``TPU_MOE_RAGGED_A2A``).

With the knob on, the quant method owns the EP exchange
(``supports_internal_mk``): vLLM hands it this engine's tokens, routing runs
locally, and ``layers.core.ragged_a2a_moe`` moves each token only to the DP
group members that host its experts. vLLM's TP all-reduce still finishes the
reduction.
"""

import functools

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu import decode_switches
from vllm_torchtpu.layers.adapter.fused_moe import _routed_rows_per_decode_step
from vllm_torchtpu.layers.core.fused_moe_gmm import resolve_onehot_permute_threshold
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    DP_AXIS,
    TP_AXIS,
    exchange_capacity,
    ragged_a2a_moe,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MESH = None
_OPS: dict[tuple, object] = {}


def ragged_a2a_enabled() -> bool:
    return envs.TPU_MOE_RAGGED_A2A


def get_dp_tp_mesh(knob: str = "TPU_MOE_RAGGED_A2A"):
    """A (dp, tp) mesh over the torch.distributed world, partition = rank.

    The exchange assumes vLLM's layout: global rank ``dp_rank * tp + tp_rank``
    and EP rank equal to the global rank. Checked here, outside any trace;
    ``knob`` names the switch a refusal reports.
    """
    global _MESH
    if _MESH is not None:
        return _MESH
    import torch.distributed as dist
    from torch_tpu._internal.pallas import get_pallas_mesh
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

    from vllm_torchtpu.distributed.mesh_utils import get_pipeline_stage

    dp, tp, ep = get_dp_group(), get_tp_group(), get_ep_group()
    dp_size, tp_size = int(dp.world_size), int(tp.world_size)
    rank, world = int(dist.get_rank()), int(dist.get_world_size())
    _, stages = get_pipeline_stage()
    dp_rank, tp_rank = int(dp.rank_in_group), int(tp.rank_in_group)
    expected = {
        "world": (world, dp_size * tp_size),
        "pipeline stages": (stages, 1),
        "rank": (rank, dp_rank * tp_size + tp_rank),
        "dp group": (tuple(dp.ranks), tuple(range(tp_rank, world, tp_size))),
        "tp group": (
            tuple(tp.ranks),
            tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size)),
        ),
        "ep group": (tuple(ep.ranks), tuple(range(world))),
    }
    bad = {k: v for k, v in expected.items() if v[0] != v[1]}
    if bad:
        raise NotImplementedError(f"{knob} needs a rank-aligned DP x TP layout: {bad}")
    _MESH = get_pallas_mesh(
        axis_names=(DP_AXIS, TP_AXIS), mesh_shape=(dp_size, tp_size)
    )
    logger.info("Ragged A2A MoE mesh | dp=%d tp=%d", dp_size, tp_size)
    return _MESH


def capacity_factors(dp_size: int, *, placed: bool) -> tuple[float, float]:
    """The ``(rows, routes)`` capacity factors of a MoE layer's exchange: the
    env knobs where set, else the defaults for ``dp_size`` DP members.

    Up to 8 members: x2.5 / x3, or x2 / x2 with TPU_MOE_EXPERT_PLACEMENT.
    Larger groups hold fewer experts per rank, so real routing loads the
    busiest rank further above the mean: a layer the file places
    (``placed``) takes x4 / x4 up to 16 members and x7 / x7 above, a layer in
    linear placement x4 / x5 and x7 / x8.5.
    """
    if dp_size <= 8:
        rows, routes = (2.0, 2.0) if envs.TPU_MOE_EXPERT_PLACEMENT else (2.5, 3.0)
    elif dp_size <= 16:
        rows, routes = (4.0, 4.0) if placed else (4.0, 5.0)
    else:
        rows, routes = (7.0, 7.0) if placed else (7.0, 8.5)
    if envs.TPU_MOE_A2A_CAPACITY is not None:
        rows = envs.TPU_MOE_A2A_CAPACITY
    if envs.TPU_MOE_A2A_ROUTE_CAPACITY is not None:
        routes = envs.TPU_MOE_A2A_ROUTE_CAPACITY
    return rows, routes


def _check_replicas(switches, *, scaled, owner_combine):
    """Refuse a layer with redundant expert slots that the exchange cannot
    serve them in: they need scaled experts, the layer's owner combine and the
    sorted plan."""
    what = "Expert replicas (TPU_MOE_EXPERT_PLACEMENT with redundant slots) need"
    if not scaled:
        raise ValueError(f"{what} scaled experts")
    if not owner_combine:
        if switches.path("a2a_owner_combine"):
            raise ValueError(
                f"{what} the owner combine of the layer, which is not armed "
                "for it (see decode_switches.owner_combine_layer)"
            )
        raise ValueError(
            f"{what} a2a_owner_combine: set TPU_MOE_RAGGED_A2A and remove "
            f"a2a_owner_combine from {decode_switches.FAST_PATHS_OFF}"
        )
    if not switches.path("a2a_sorted_plan"):
        raise ValueError(
            f"{what} a2a_sorted_plan: remove a2a_sorted_plan from "
            f"{decode_switches.FAST_PATHS_OFF}"
        )


def _get_op(
    *,
    topk,
    activation,
    rhs_quant_dtype,
    skip_padded_tokens,
    shared=False,
    scaled=True,
    placed=False,
    owner_combine=False,
    replicas=False,
):
    key = (
        topk,
        activation,
        rhs_quant_dtype,
        skip_padded_tokens,
        shared,
        scaled,
        placed,
        owner_combine,
        replicas,
    )
    op = _OPS.get(key)
    if op is not None:
        return op
    switches = decode_switches.get()
    keywords = decode_switches.exchange_keywords(switches, owner_combine=owner_combine)
    if replicas:
        _check_replicas(switches, scaled=scaled, owner_combine=owner_combine)
    mesh = get_dp_tp_mesh()
    # The full-size exchange receives into one region per DP member, sized for
    # all of that member's tokens: the local permute sees the all-gather
    # path's rows, so the threshold follows the full decode step.
    step_routed_rows = _routed_rows_per_decode_step(topk)
    onehot_threshold = resolve_onehot_permute_threshold(step_routed_rows)
    capacity_factor, route_factor = capacity_factors(mesh.shape[DP_AXIS], placed=placed)
    run = functools.partial(
        ragged_a2a_moe,
        mesh=mesh,
        topk=topk,
        activation=activation,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_moe_permute_threshold=onehot_threshold,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
        capacity_factor=capacity_factor,
        route_factor=route_factor,
        **keywords,
    )

    # torch_tpu infers static argnums from annotations: every operand is jax.Array.
    def fn(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        experts_start: jax.Array,
    ) -> jax.Array:
        return run(x, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, experts_start)

    def fn_shared(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        experts_start: jax.Array,
        shared_output: jax.Array,
    ) -> jax.Array:
        return run(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            experts_start,
            shared=shared_output,
        )

    def fn_unscaled(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        experts_start: jax.Array,
    ) -> jax.Array:
        return run(x, w1, w2, None, None, topk_weights, topk_ids, experts_start)

    def fn_unscaled_shared(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        experts_start: jax.Array,
        shared_output: jax.Array,
    ) -> jax.Array:
        return run(
            x,
            w1,
            w2,
            None,
            None,
            topk_weights,
            topk_ids,
            experts_start,
            shared=shared_output,
        )

    def fn_replicas(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        experts_start: jax.Array,
        replica_table: jax.Array,
    ) -> jax.Array:
        return run(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            experts_start,
            replicas=replica_table,
        )

    def fn_replicas_shared(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        experts_start: jax.Array,
        replica_table: jax.Array,
        shared_output: jax.Array,
    ) -> jax.Array:
        return run(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            experts_start,
            replicas=replica_table,
            shared=shared_output,
        )

    if replicas:
        body = fn_replicas_shared if shared else fn_replicas
    elif scaled:
        body = fn_shared if shared else fn
    else:
        body = fn_unscaled_shared if shared else fn_unscaled
    # One mesh axis per dimension: the operands carry two leading unit axes.
    spec = PartitionSpec(DP_AXIS, TP_AXIS)
    op = pallas.jax_op(
        f"pallas::ragged_a2a_moe_{len(_OPS)}",
        body,
        mesh=mesh,
        input_partition_specs=(spec,) * (6 + 2 * scaled + shared + replicas),
        # A tuple of per-output specs: a bare PartitionSpec is not a tuple, so
        # torch_tpu would read its first axis name as the whole spec.
        output_partition_specs=(spec,),
    )
    # vLLM traces the token dimension as dynamic; the output matches x.
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _OPS[key] = op
    logger.info_once(
        "Ragged A2A MoE armed | topk=%d activation=%s dp=%d tp=%d | one-hot "
        "permute threshold %d routed rows (full decode step: %d routed rows) | "
        "streamed combine %s, owner combine %s, row gather %s, row scatter %s, "
        "owner tail %s, "
        "routing dma %s, send index rows minor %s, plan kernel %s",
        topk,
        activation,
        mesh.shape[DP_AXIS],
        mesh.shape[TP_AXIS],
        onehot_threshold,
        step_routed_rows,
        "on" if keywords["streamed_combine"] else "off",
        "on" if keywords["owner_combine"] else "off",
        "on" if keywords["row_gather"] else "off",
        "on" if keywords["row_scatter"] else "off",
        "on" if keywords["owner_tail"] else "off",
        "on" if keywords["routing_dma"] else "off",
        "on" if keywords["send_index_rows_minor"] else "off",
        "on" if keywords["plan_kernel"] else "off",
    )
    _log_capacity(
        mesh, topk, onehot_threshold, step_routed_rows, capacity_factor, route_factor
    )
    return op


def _log_capacity(
    mesh, topk, onehot_threshold, step_routed_rows, factor, route_factor
) -> None:
    """Log the capacity sizes of a full decode step (at op build, untraced)."""
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    text = getattr(getattr(cfg, "model_config", None), "hf_text_config", None)
    num_experts = getattr(text, "num_experts", None) or getattr(
        text, "n_routed_experts", None
    )
    dp_size, tp_size = mesh.shape[DP_AXIS], mesh.shape[TP_AXIS]
    if not num_experts or not step_routed_rows:
        return
    decode_tokens = step_routed_rows // (dp_size * topk)
    rows, routes = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=decode_tokens,
        topk=topk,
        local_num_experts=num_experts // (dp_size * tp_size),
        factor=factor,
        onehot_moe_permute_threshold=onehot_threshold,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        route_factor=route_factor or None,
        row_tile=decode_switches.A2A_ROW_TILE,
    )
    if rows >= dp_size * decode_tokens:
        logger.info(
            "Ragged A2A MoE capacity x%g | decode step: full-size exchange "
            "(capacity %d rows >= %d group tokens)",
            factor,
            rows,
            dp_size * decode_tokens,
        )
        return
    logger.info(
        "Ragged A2A MoE capacity x%g (routes x%g) | decode step: %d of %d rows, "
        "%d of %d routes per rank",
        factor,
        route_factor or factor,
        rows,
        dp_size * decode_tokens,
        routes,
        step_routed_rows,
    )


def prebuild_ragged_a2a_moe(
    *,
    topk,
    activation,
    rhs_quant_dtype=None,
    scaled=True,
    placed=False,
    owner_combine=False,
    replicas=False,
) -> None:
    """Build the mesh and op at weight load, outside the traced forward.
    ``scaled=False``: the op for unscaled (bf16) experts; ``placed``: for a
    layer TPU_MOE_EXPERT_PLACEMENT places; ``owner_combine``: for a layer
    whose owner combine is armed (``decode_switches.arm_owner_combine``);
    ``replicas``: one whose file has redundant slots.

    Padded tokens reach the op unrouted (TPU_MOE_SKIP_PADDED_TOKENS
    drops their routes at top-k), so the op never skips routes by weight."""
    for shared in (False, True) if owner_combine else (False,):
        _get_op(
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=False,
            shared=shared,
            scaled=scaled,
            placed=placed,
            owner_combine=owner_combine,
            replicas=replicas,
        )


def ragged_a2a_fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor,
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
    shared_output: torch.Tensor | None = None,
    placed: bool = False,
    owner_combine: bool = False,
    replicas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same contract as ``fused_moe_gmm`` with local tokens in and out; the
    result still needs vLLM's TP all-reduce, except with ``owner_combine``
    (the owner combine armed for the layer), where it is already
    summed over TP and includes ``shared_output`` (this rank's shared-expert
    partial) when given. Unscaled (bf16) experts pass ``None`` for both
    scales; ``placed``: the layer's experts are placed by
    TPU_MOE_EXPERT_PLACEMENT; ``replicas``: its redundant slots
    (``layer._expert_replicas``) when the file has them."""
    if w1_bias is not None or w2_bias is not None:
        raise NotImplementedError("TPU_MOE_RAGGED_A2A does not take expert biases")
    if experts_start is None:
        raise NotImplementedError("TPU_MOE_RAGGED_A2A needs expert parallelism")
    scaled = w1_scale is not None
    if scaled != (w2_scale is not None):
        raise ValueError("TPU_MOE_RAGGED_A2A needs both expert scales or neither")
    if shared_output is not None and not owner_combine:
        raise ValueError("shared_output needs the owner combine of the layer")
    op = _get_op(
        topk=topk,
        activation=activation,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=False,
        shared=shared_output is not None,
        scaled=scaled,
        placed=placed,
        owner_combine=owner_combine,
        replicas=replicas is not None,
    )
    scales = (w1_scale, w2_scale) if scaled else ()
    stacked = (
        hidden_states,
        w1,
        w2,
        *scales,
        topk_weights,
        topk_ids.to(torch.int32),
        experts_start.reshape(1),
    )
    if replicas is not None:
        stacked += (replicas,)
    if shared_output is not None:
        stacked += (shared_output.to(hidden_states.dtype),)
    out = op(*(t.reshape(1, 1, *t.shape) for t in stacked))
    return out.reshape(hidden_states.shape)
