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
"""Torch bridge for the fused expert-parallel MoE kernel.

The default MoE path pays the expert-parallel exchange as two separate
collectives around a per-rank compute chain: vLLM all-gathers the tokens, this
rank routes them, permutes, runs two grouped matmuls, combines, and then vLLM
reduce-scatters the result. A phased profile of Qwen3.5-397B at 8k1k/c256 puts
that reduce-scatter at 1190 us per MoE layer and shows it fully exposed --
nothing in the layer is left to overlap it with -- while the symmetric
all-gather of the same 134 MB is hidden behind the compute at 36 us.

This path hands the whole layer to one SPMD program instead. The kernel selects
its own top-k, pushes each routed row straight to the rank that owns its
expert, runs the expert matmuls, and pushes results back, so the transport
overlaps the matmuls rather than bracketing them. It replaces, per layer: the
router's full 512-wide sort, both routed-row argsorts, the SparseCore permute,
both grouped matmuls, the combine, and both collectives.

``MOE_FUSED_EP_KERNEL_MIN_TOKENS`` decides once, at weight load, against the
scheduler's largest batch, whether to arm this for the deployment. Where it is
armed it serves every step, decode included.

Upstream's recipe instead picks per step and leaves decode on the grouped-matmul
path, on the reasoning that decode is weight-bandwidth bound with no transport
to hide. That is right for their configuration and wrong for Qwen3.5-397B at
DP8+EP, measured: arming everything gives 33,304 tok/s against 30,888 for the
grouped-matmul path, and decode gets *faster* too, 54.87 ms median TPOT against
56.86. At hidden 4096 the combine this replaces is worth paying a kernel for
even at decode. The 35B proxy predicts the opposite (TPOT +3.9%), which is the
usual story for anything that scales with hidden size.
"""

import functools
import json
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    ep_sorted_gather_group,
    ep_token_replica_groups,
)
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK,
    PACK4,
    U32_SUBLANE_TILE,
    token_gather_smem_bytes,
)
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)

# The tile height the kernel slabs rows into, and the unit its ragged stride
# rounds to. The kernel's routing tables are built against this value, so it
# is not free to change here alone.
_TILE_M = 128

# SMEM used by the resident routing tables and Mosaic's scalar working set,
# excluding the two fixed token-gather windows. Measured by subtracting the
# former full token-gather scalar-prefetch table at the allocation boundary.
_SMEM_OVERHEAD_BYTES = 236 * 1024

# One op for every shape: `fused_ep_moe_v2` derives its routing block and ragged
# stride from the operands when they are not given, so nothing here has to vary
# with the token count -- which is what keeps the token count out of the traced
# graph.
FUSED_MOE_EP_OP_NAME = "pallas::fused_moe_ep"

# Ops are built at weight-load time and the compiled forward may only look them
# up: a `jax.sharding.Mesh` holds a numpy object array of TpuDevice, and Dynamo
# tries to turn anything it sees into a tensor, so the mesh must never appear in
# a traced frame.
#
# Keyed by what the op closes over rather than held in one slot, because
# `prebuild_fused_moe_ep` runs once per MoE layer and a single slot is
# last-writer-wins: a model whose layers disagree on topk, renormalize or the
# activation would run every layer through whichever one loaded last. Keyed
# rather than one object per layer because the op object is what torch_tpu
# exports and compiles, so a per-layer object re-exports the same program once
# per MoE layer -- 60 times on Qwen3.5-397B.
#
# WHICH layers are armed is a separate question, and not one this key can
# answer: two of prebuild's refusals -- the SMEM bound and the weight contract
# -- read things that are not in it. So prebuild hands the op back to the caller
# and the caller records it on the layer's own quant method; see
# `fused_moe_ep_supported`.
_OPS: dict[tuple, Any] = {}
# This is a fact about the mesh, not about a layer, so one copy for the process
# is what it is: every layer in a deployment sees the same EP group.
# This rank's mesh index, as a device tensor. The kernel needs it as data;
# see the note in fused_ep_moe_v2 on why a value computed inside the traced
# function turns into a `partition-id` XLA will not accept.
_RANK_BUFFER: Any = None

# The attribute prebuild's result is recorded under, on the quant method that
# owns the layer. vLLM builds one quant method per MoE layer
# (`Fp8Config.get_quant_method`), so this is per-layer state without a registry
# that has to be torn down between models in one process.
FUSED_MOE_EP_OP_ATTR = "_fused_moe_ep_op"


def _squeeze_channel_scale(
    scale: torch.Tensor, experts: int, channels: int
) -> torch.Tensor | None:
    """Our GMM scales are 4-D ``[E, blocks, 1, N]``; the kernel takes ``[E, N]``.

    Only the per-output-channel form (one block) converts: a real block-scaled
    weight is a different kernel format and is refused rather than silently
    flattened.
    """
    if scale is None:
        return None
    s = scale
    while s.ndim > 2:
        if s.shape[1] != 1:
            return None
        s = s.squeeze(1)
    if s.shape != (experts, channels):
        return None
    return s.contiguous()


def fused_moe_ep_supported(owner: Any) -> bool:
    """Whether `prebuild_fused_moe_ep` armed the fused path for this layer.

    `owner` is the layer's quant method -- one per MoE layer -- or anything
    else prebuild's result was recorded on. Asked per layer rather than per
    process because prebuild refuses on things that vary by layer, so a
    process-wide answer would let a refused layer run through an armed one's
    program.

    Takes no tensor on purpose. A per-step choice would have to branch on the
    token count, and vLLM traces the backbone once with that dimension dynamic,
    so the first bucket traced freezes the decision for every other bucket:
    measured, gating that way fused one of eleven buckets and gave back the
    whole win (30,667 tok/s against 33,304). vllm-torchtpu #474 exists to fix
    exactly that, by reading Dynamo's shape guards and retracing per bucket. It
    was tried here and reverted -- its warmup ladder goes out of step with a
    partition that falls mid-ladder ("artifact_compile_range_8_8 was left to a
    later trace (s18 >= 128) and never compiled"), and every request failed.
    Revisit if that PR lands with the ladder issue resolved; the gain on offer
    is decode steps skipping this kernel's ~180 us per-call fixed cost.
    """
    return getattr(owner, FUSED_MOE_EP_OP_ATTR, None) is not None


def _build_op(
    mesh,
    topk: int,
    renormalize: bool,
    activation: str,
    mesh_expert_order: tuple[int, ...] | None,
    sharded_plan: bool,
    weight_format="fp8",
    rhs_qb=None,
    scoring_fn="softmax",
    has_score_bias=False,
    routed_scaling_factor=1.0,
    has_hash_ids=False,
    *,
    token_replica_groups=None,
    sorted_gather=None,
    dedup_cfg=None,
    has_l2p=False,
    shared_cfg=None,
):
    """The op for this closure, built once per distinct closure and reused.

    `sorted_gather` is `ep_sorted_gather_group()`'s (group, positions), or
    None for the single fused program. With it the op is a `_SortedGatherOp`.

    `dedup_cfg` (G3-LOCAL) is `(waves, BoardTopology | None)` to serve the
    layer with tokamax's De-Dup MoE (`dedup_ep_moe`) instead of
    `fused_ep_moe_v2`; None keeps v2.

    `shared_cfg` (G3-LOCAL) is `(has_shared_gate, shared_overlap,
    w8a16_max_rows)` when fusing the TP-sharded shared expert into De-Dup MoE;
    None keeps external shared experts.
    """
    from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2

    key = (
        topk,
        renormalize,
        activation,
        mesh_expert_order,
        sharded_plan,
        weight_format,
        rhs_qb,
        scoring_fn,
        has_score_bias,
        routed_scaling_factor,
        has_hash_ids,
        token_replica_groups,
        None if sorted_gather is None else sorted_gather[1],
        dedup_cfg,
        has_l2p,
        shared_cfg,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached

    # Each distinct closure gets its own torch op name: two `CustomOpDef`s
    # registered under one qualname do not raise, they silently share a
    # dispatch entry, and the second closure would then serve the first.
    name = FUSED_MOE_EP_OP_NAME if not _OPS else f"{FUSED_MOE_EP_OP_NAME}_{len(_OPS)}"

    routing_kw = dict(
        scoring_fn=scoring_fn, routed_scaling_factor=routed_scaling_factor
    )

    # Any: the layer, route or compute result depending on bias_kw's stage;
    # each wrapper below declares the one it returns.
    def call(x, w1, w2, w1_scale, w2_scale, gating, rank, **bias_kw) -> Any:
        # The router's top-k selector writes into an f32 accumulator ref, so a
        # bf16 gate output -- which is what the model's gate matmul produces --
        # fails inside the kernel on a dtype mismatch. Widen at the boundary;
        # softmax runs in f32 on this path anyway.
        if dedup_cfg is not None and dedup_cfg[0] == "hybrid":
            # G3-LOCAL hybrid: t_local is static per exported token bucket.
            _, inner_cfg, v2_max_t = dedup_cfg
            t_local = _moe_t_local(
                x.shape[0], mesh.shape[EP_AXIS_NAME], token_replica_groups
            )
            pick = _hybrid_pick(t_local, v2_max_t)
            logger.info(
                "[VLLM_TT] Hybrid MoE plan | rows=%d t_local=%d impl=%s "
                "v2_max_t_local=%d",
                x.shape[0] // mesh.shape[EP_AXIS_NAME],
                t_local,
                pick,
                v2_max_t,
            )
            layer_cfg = inner_cfg if pick == "direct_w4" else None
        else:
            layer_cfg = dedup_cfg
        if layer_cfg is not None:
            # G3-LOCAL: the De-Dup MoE layer takes the same operands, layout
            # and routing knobs as fused_ep_moe_v2 (it shares v2's router,
            # routing tables and expert FFN through tokamax's copy of v2).
            from tokamax._src.ops.experimental.dedup_moe.layer import (
                monolithic as dedup_monolithic,
            )

            assert "stage" not in bias_kw, "dedup has no split route/compute"
            waves, dedup_topology = layer_cfg
            fwd_kw = {}
            # G3-LOCAL: per-bucket waves and kernel/combine options from env
            # (DEDUP_MOE_DECODE_WAVES / _DECODE_MAX_T / _KERNEL_OPTS /
            # _COMBINE_OPTS). t_local is static per exported token bucket.
            # Unset env leaves the call exactly as before.
            t_local = _moe_t_local(
                x.shape[0], mesh.shape[EP_AXIS_NAME], token_replica_groups
            )
            ovr = _dedup_overrides()
            if dedup_topology is not None and t_local <= ovr.direct_max_t:
                # G3-LOCAL hybrid: buckets at or below DEDUP_MOE_DIRECT_MAX_T
                # (decode) take the direct path, the same kernel BOARD_SIZE=0
                # builds; larger (prefill) buckets keep De-Dup.
                dedup_topology = None
                logger.info(
                    "[VLLM_TT] De-Dup hybrid | t_local=%d <= direct_max_t=%d "
                    "-> direct",
                    t_local,
                    ovr.direct_max_t,
                )
            tile_m = (
                ovr.prefill_capacity
                if (
                    ovr.prefill_capacity
                    and t_local >= ovr.prefill_capacity_min_t
                )
                else _TILE_M
            )
            if ovr.decode_waves and t_local <= ovr.decode_max_t:
                waves = ovr.decode_waves
            kopts = dict(ovr.kernel_options)
            if ovr.decode_kernel_options and t_local <= ovr.decode_max_t:
                kopts.update(dict(ovr.decode_kernel_options))
            if kopts:
                fwd_kw["kernel_options"] = kopts
            if ovr.combine_options:
                fwd_kw["combine_options"] = dict(ovr.combine_options)
            if ovr.any_set:
                logger.info(
                    "[VLLM_TT] De-Dup call | t_local=%d waves=%d capacity=%d "
                    "kernel_options=%s combine_options=%s",
                    t_local,
                    waves,
                    tile_m,
                    kopts,
                    dict(ovr.combine_options),
                )
            if dedup_topology is not None:
                fwd_kw["dedup_fwd_max_runs"] = _dedup_fwd_max_runs(
                    dedup_topology, waves, t_local
                )
                logger.info(
                    "[VLLM_TT] De-Dup plan | t_local=%d waves=%d "
                    "dedup_fwd_max_runs=%d",
                    t_local,
                    waves,
                    fwd_kw["dedup_fwd_max_runs"],
                )
            if shared_cfg is not None:
                _, shared_overlap, shared_w8a16_max_rows = shared_cfg
                eff_shared_overlap = (
                    "dispatch"
                    if (shared_overlap == "split" and waves > 1)
                    else shared_overlap
                )
                fwd_kw["shared_expert_sharded"] = True
                fwd_kw["shared_expert_act_fn"] = "silu"
                fwd_kw["shared_expert_overlap"] = eff_shared_overlap
                fwd_kw["shared_matmul_fn"] = _shared_matmul_fn(shared_w8a16_max_rows)
            return dedup_monolithic.dedup_ep_moe(
                x,
                w1,
                w2,
                w1_scale,
                w2_scale,
                gating.astype(jnp.float32),
                topk=topk,
                renormalize=renormalize,
                mesh=mesh,
                capacity=tile_m,
                weight_format=dedup_monolithic.WeightFormat(weight_format),
                act_fn=activation,
                rank=rank,
                mesh_ep_ranks=mesh_expert_order,
                token_replica_groups=token_replica_groups,
                waves=waves,
                dedup_topology=dedup_topology,
                **fwd_kw,
                **bias_kw,
                **routing_kw,
            )
        return fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating.astype(jnp.float32),
            rank=rank,
            topk=topk,
            renormalize=renormalize,
            mesh=mesh,
            capacity=_TILE_M,
            block=None,
            ragged_stride=None,
            weight_format=WeightFormat(weight_format),
            rhs_qb=rhs_qb,
            act_fn=activation,
            mesh_ep_ranks=mesh_expert_order,
            sharded_plan=sharded_plan,
            token_replica_groups=token_replica_groups,
            **bias_kw,
            **routing_kw,
        )

    # Every operand is annotated jax.Array: torch_tpu infers static argnums
    # from the signature and refuses an unannotated one.
    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return call(x, w1, w2, w1_scale, w2_scale, gating, rank)

    def moe_with_score_bias(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        score_bias: jax.Array,
    ) -> jax.Array:
        return call(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            rank,
            score_bias=score_bias.astype(jnp.float32),
        )

    def moe_with_hash_ids(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        hash_ids: jax.Array,
    ) -> jax.Array:
        return call(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            rank,
            hash_topk_ids=hash_ids.astype(jnp.int32),
        )

    # G3-LOCAL (DEDUP_MOE_PLACEMENT_DIR): the static expert placement's
    # logical->physical map is an operand, not a closure constant, because it
    # differs per layer and one op serves every layer (see `_OPS`).
    def moe_with_l2p(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        l2p: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            logical_to_physical_map=l2p.astype(jnp.int32),
        )

    def moe_with_score_bias_l2p(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        score_bias: jax.Array,
        l2p: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            score_bias=score_bias.astype(jnp.float32),
            logical_to_physical_map=l2p.astype(jnp.int32),
        )

    # G3-LOCAL (DEDUP_MOE_FUSE_SHARED): shared-expert operands (ws1, ws2,
    # ws1_scale, ws2_scale [, ws_gate]) precede optional l2p so `_PlacedOp`
    # still appends l2p last without changing its wrapper contract.
    def moe_shared(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
        )

    def moe_shared_gated(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
        ws_gate: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
            ws_gate=ws_gate,
        )

    def moe_shared_l2p(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
        l2p: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
            logical_to_physical_map=l2p.astype(jnp.int32),
        )

    def moe_shared_gated_l2p(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
        ws_gate: jax.Array,
        l2p: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
            ws_gate=ws_gate,
            logical_to_physical_map=l2p.astype(jnp.int32),
        )

    def moe_bias_shared(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        score_bias: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            score_bias=score_bias.astype(jnp.float32),
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
        )

    def moe_bias_shared_gated(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        score_bias: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
        ws_gate: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            score_bias=score_bias.astype(jnp.float32),
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
            ws_gate=ws_gate,
        )

    def moe_bias_shared_l2p(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        score_bias: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
        l2p: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            score_bias=score_bias.astype(jnp.float32),
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
            logical_to_physical_map=l2p.astype(jnp.int32),
        )

    def moe_bias_shared_gated_l2p(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        score_bias: jax.Array,
        ws1: jax.Array,
        ws2: jax.Array,
        ws1_scale: jax.Array,
        ws2_scale: jax.Array,
        ws_gate: jax.Array,
        l2p: jax.Array,
    ) -> jax.Array:
        return call(
            x, w1, w2, w1_scale, w2_scale, gating, rank,
            score_bias=score_bias.astype(jnp.float32),
            ws1=ws1, ws2=ws2, ws1_scale=ws1_scale, ws2_scale=ws2_scale,
            ws_gate=ws_gate,
            logical_to_physical_map=l2p.astype(jnp.int32),
        )

    assert not (has_score_bias and has_hash_ids)
    assert not (has_l2p and (has_hash_ids or sorted_gather is not None))
    assert not (shared_cfg is not None and (has_hash_ids or sorted_gather is not None))
    has_shared = shared_cfg is not None
    has_shared_gate = bool(shared_cfg[0]) if shared_cfg is not None else False
    spec = PartitionSpec(EP_AXIS_NAME)
    # The score-correction bias is indexed by GLOBAL expert id and read by
    # every shard for its own tokens, so it is replicated where the other
    # The score-correction bias is indexed by GLOBAL expert id and read by
    # every shard for its own tokens, so it is replicated where the other
    # seven are sharded on axis 0. The spec tuple goes straight to shard_map,
    # so a heterogeneous tuple is what it wants; the bias is NOT closed over
    # as a constant because it differs per layer and `_OPS` would then key one
    # exported program per MoE layer.
    input_specs = (
        (spec,) * 7
        + ((PartitionSpec(),) if has_score_bias else ())
        + ((spec,) if has_hash_ids else ())
        + ((spec,) * 4 if has_shared else ())
        + ((PartitionSpec(),) if has_shared_gate else ())
        + ((PartitionSpec(),) if has_l2p else ())
    )
    # Stock jax_op. It used to size its outputs from the export's avals, where
    # a shard_map result is recorded as replicated, so this rank's 2048 rows
    # came back claiming the mesh-wide 16384; `sharded_jax_op`, a local copy of
    # JaxCallable with that one line replaced, was the way around it. Torchtpu
    # takes output_partition_specs itself now, which is the same information
    # without a copy of anyone's source.
    if has_shared:
        _shared_fns = {
            (False, False, False): moe_shared,
            (False, True, False): moe_shared_gated,
            (False, False, True): moe_shared_l2p,
            (False, True, True): moe_shared_gated_l2p,
            (True, False, False): moe_bias_shared,
            (True, True, False): moe_bias_shared_gated,
            (True, False, True): moe_bias_shared_l2p,
            (True, True, True): moe_bias_shared_gated_l2p,
        }
        fn = _shared_fns[(bool(has_score_bias), has_shared_gate, bool(has_l2p))]
    elif has_l2p:
        fn = moe_with_score_bias_l2p if has_score_bias else moe_with_l2p
    elif has_score_bias:
        fn = moe_with_score_bias
    elif has_hash_ids:
        fn = moe_with_hash_ids
    else:
        fn = moe
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        # x, w1, w2, w1_scale, w2_scale, gating, rank [, score_bias | hash_ids]
        # -- and the single output is sharded on axis 0 like x.
        input_partition_specs=input_specs,
        output_partition_specs=spec,
    )

    # Replaces the fake jax_op installs, which resolves the output aval by
    # running a real `jax.export` every time Dynamo traces the op. This kernel
    # returns the rows it was given, so the answer is known without exporting.
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    if sorted_gather is not None:
        op = _build_split_ops(
            name,
            call,
            mesh,
            input_specs,
            spec,
            topk,
            weight_format,
            has_score_bias,
            has_hash_ids,
            token_replica_groups,
            sorted_gather,
        )
    _OPS[key] = op
    return op


class _SortedGatherOp:
    """The fused layer as route, a torch all-gather, compute.

    Called exactly like the single op. The gather runs over
    `ep_sorted_gather_group`'s device-id ordered group while both programs
    keep the rank-ordered mesh; compute is told where each rank's rows
    landed. Holds no mesh, so it is safe to reach from a traced forward.
    """

    def __init__(self, route, compute, group):
        self.route = route
        self.compute = compute
        self.group = group

    def __call__(self, x, w1, w2, w1_scale, w2_scale, gating, rank, *extra):
        from torch.distributed import _functional_collectives as funcol

        operands = (x, w1, w2, w1_scale, w2_scale, gating, rank, *extra)
        q, router_weights, payload = self.route(*operands)
        q_g = funcol.wait_tensor(funcol.all_gather_tensor(q, 0, self.group))
        return self.compute(*operands, q_g, router_weights, payload)


def _build_split_ops(
    name,
    call,
    mesh,
    input_specs,
    spec,
    topk,
    weight_format,
    has_score_bias,
    has_hash_ids,
    token_replica_groups,
    sorted_gather,
):
    """`_SortedGatherOp` for the closure `call` was built for."""
    from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat
    from vllm_torchtpu.kernels.fused_moe.v2.host import weight_form
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _exchange_layout
    from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import replica_wire_rows

    group, positions = sorted_gather
    ep = mesh.shape[EP_AXIS_NAME]
    quantized = weight_form(WeightFormat(weight_format)).quantized_activations

    def extra_kw(extra):
        if has_score_bias:
            return dict(score_bias=extra.astype(jnp.float32))
        if has_hash_ids:
            return dict(hash_topk_ids=extra.astype(jnp.int32))
        return {}

    def route(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return call(x, w1, w2, w1_scale, w2_scale, gating, rank, stage="route")

    def route_extra(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        extra: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return call(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            rank,
            stage="route",
            **extra_kw(extra),
        )

    def compute(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        q_g: jax.Array,
        router_weights: jax.Array,
        payload: jax.Array,
    ) -> jax.Array:
        return call(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            rank,
            stage="compute",
            routed=(q_g, router_weights, payload),
            gather_positions=positions,
        )

    def compute_extra(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        extra: jax.Array,
        q_g: jax.Array,
        router_weights: jax.Array,
        payload: jax.Array,
    ) -> jax.Array:
        return call(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            rank,
            stage="compute",
            routed=(q_g, router_weights, payload),
            gather_positions=positions,
            **extra_kw(extra),
        )

    has_extra = has_score_bias or has_hash_ids
    route_op = jax_op(
        f"{name}_route",
        route_extra if has_extra else route,
        mesh=mesh,
        input_partition_specs=input_specs,
        output_partition_specs=(spec,) * 3,
    )
    # q_g is the whole gather on every rank: one block of mesh-wide rows per
    # shard, so it is sharded on axis 0 like the rest.
    compute_op = jax_op(
        f"{name}_compute",
        compute_extra if has_extra else compute,
        mesh=mesh,
        input_partition_specs=input_specs + (spec,) * 3,
        output_partition_specs=spec,
    )

    replica_width = len(token_replica_groups[0]) if token_replica_groups else 1

    # x.shape[0] is symbolic here, and must stay unguarded: see
    # replica_wire_rows, the padded row count route returns for that reason.
    def _route_fake(x, w1, *_args, **_kwargs):
        t_wire = replica_wire_rows(x.shape[0], replica_width, U32_SUBLANE_TILE)
        width = _exchange_layout(t_wire, topk, w1.shape[0] * ep, quantized)[-1]
        q_dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
        return (
            x.new_empty(
                (t_wire, x.shape[1] // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK),
                dtype=q_dtype,
            ),
            x.new_empty((t_wire, topk), dtype=torch.float32),
            x.new_empty((1, width), dtype=torch.int32),
        )

    def _compute_fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    route_op.register_fake(_route_fake)
    compute_op.register_fake(_compute_fake)
    logger.info_once(
        "Fused EP MoE: activation all-gather runs over the device-id ordered "
        "group | gather_positions=%s",
        positions,
    )
    return _SortedGatherOp(route_op, compute_op, group)


def fused_moe_ep(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
    score_bias: torch.Tensor | None = None,
    hash_ids: torch.Tensor | None = None,
    shared: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run one MoE layer through the fused EP kernel.

    Takes this rank's own tokens and returns this rank's own tokens: the
    dispatch and combine happen inside the program, so the caller must not
    all-gather the input or reduce-scatter the output. `owner` is the layer's
    quant method, the same object `fused_moe_ep_supported` was asked about;
    only call this when that said yes.

    `score_bias` is the layer's `e_score_correction_bias`, [global experts]
    f32, and `hash_ids` is `hash_topk_ids_operand`'s [tokens, topk] i32 table
    lookup. Each must be passed exactly when prebuild saw the corresponding
    thing on the layer -- prebuild built the op's signature around that.
    Getting it wrong is a torch arity error at the first call, not a wrong
    answer.
    """
    op = getattr(owner, FUSED_MOE_EP_OP_ATTR)
    if w1.dtype == torch.float4_e2m1fn_x2:
        # FP4 scales are already [E, K/block, N]. Even a single block must
        # retain its axis; the per-channel helper would incorrectly drop it.
        scales = (w1_scale, w2_scale)
    else:
        scales = (
            _squeeze_channel_scale_checked(w1_scale),
            _squeeze_channel_scale_checked(w2_scale),
        )
    # Simulate before masking: the padding mask writes NaN rows, which would
    # make the simulation's mean anchor NaN and leave every row unrouted.
    router_logits, score_bias, hash_ids = _simulated_routing_operands(
        router_logits, score_bias, hash_ids
    )
    if envs.TPU_MOE_SKIP_PADDED_TOKENS:
        router_logits = token_padding.mask_router_logits_for_padding(router_logits)
    extra = tuple(t for t in (score_bias, hash_ids) if t is not None) + shared
    return op(hidden_states, w1, w2, *scales, router_logits, _RANK_BUFFER, *extra)


def _simulated_routing_operands(
    router_logits: torch.Tensor,
    score_bias: torch.Tensor | None,
    hash_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """The kernel's routing operands, under `uniform_random` routing simulation.

    The fused kernel scores and selects experts itself, so
    `moe_routing.maybe_simulate_routing` never runs on this path. To make
    `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` mean here what it
    means on the grouped-matmul path -- every token's experts drawn uniformly
    at random, so expert load is balanced in expectation -- the kernel is
    handed operands whose top-k IS that draw:

    - gating: `moe_routing.uniform_random_logits`, the f32 noise anchored to
      the real logits' mean that `moe_routing._uniform_random_routing` draws
      from.
    - score bias: zeros, so the correction bias cannot shift the draw.
    - hash ids (hash-routed layers): top-k of the same noise.

    The gate weights are the kernel's scoring of the noise rather than the
    grouped-matmul path's ones; the expert choice, which is what the simulation
    exists to control, is the same distribution. Profiling only: the model's
    output is meaningless, as `moe_routing` already warns at import.

    Returns the operands unchanged when no simulation is set.
    """
    from vllm_torchtpu.layers.adapter import moe_routing

    if not moe_routing.is_uniform_random_simulation():
        return router_logits, score_bias, hash_ids
    noise = moe_routing.uniform_random_logits(router_logits)
    if hash_ids is not None:
        hash_ids = torch.topk(noise, k=hash_ids.shape[-1], dim=-1).indices.to(
            torch.int32
        )
    if score_bias is not None:
        score_bias = torch.zeros_like(score_bias)
    return noise, score_bias, hash_ids


def register_fused_scale_buffers(layer) -> None:
    """Stage the fp4 weight scales as the `[E, K/block, N]` tables the op takes.

    The loader's tables are `[E, K/block, 1, N]`. A squeeze of one is a view,
    and torch_tpu materializes a view graph input with a program of its own
    before every step, in a layout the kernel then copies out of again inside
    the step. So each table is copied once into a base tensor of its own, and
    the 4-D parameter becomes a view of that copy rather than a second one.
    The tables are synced here because the loaders' own sync has already run.
    """
    tables = []
    for param, buf in (
        ("w13_weight_scale", "_tpu_fused_w13_scale"),
        ("w2_weight_scale", "_tpu_fused_w2_scale"),
    ):
        table = fresh_buffer(getattr(layer, param).squeeze(2))
        layer.register_buffer(buf, table, persistent=False)
        setattr(
            layer, param, torch.nn.Parameter(table.unsqueeze(2), requires_grad=False)
        )
        tables.append(table)
    if tables[0].device.type == "tpu":
        synchronize_tensors(tables)


# The buffer `register_score_bias_buffer` stages the bias operand under, and
# the name callers read back to hand to `fused_moe_ep`.
FUSED_MOE_EP_SCORE_BIAS_ATTR = "_tpu_fused_score_bias"


def register_score_bias_buffer(layer) -> torch.Tensor | None:
    """Stage `e_score_correction_bias` as the f32 operand the op takes.

    Done once at weight load for the same reason the scale buffers are: the
    parameter is f32 already on every checkpoint seen so far, but `.float()`
    inside the compiled forward would be a cast per layer per step on a
    tensor that never changes. Returns the buffer, or None when the layer has
    no bias -- which is also what `fused_moe_ep` wants to be handed.
    """
    bias = getattr(layer, "e_score_correction_bias", None)
    if bias is None:
        return None
    staged = bias.detach().to(torch.float32).contiguous()
    layer.register_buffer(FUSED_MOE_EP_SCORE_BIAS_ATTR, staged, persistent=False)
    return staged


def score_bias_operand(layer) -> torch.Tensor | None:
    """The staged bias for this layer, or None. Safe inside a traced forward."""
    return getattr(layer, FUSED_MOE_EP_SCORE_BIAS_ATTR, None)


def hash_topk_ids_operand(layer, input_ids: torch.Tensor | None) -> torch.Tensor | None:
    """DeepSeek-V4's per-token expert ids, or None for a top-k layer.

    The table lookup stays out here rather than moving into the kernel: it is
    one gather of `[tokens, topk]` int32 against a table the kernel would
    otherwise have to hold, and it is the same expression the grouped-matmul
    path routes with (`moe_routing._hash_moe_select`), so the two paths cannot
    drift into selecting different experts.

    `input_ids` must be this rank's OWN ids, one per row of the hidden states
    the kernel is about to be handed -- an armed layer owns its dispatch, so
    vLLM does not DP-gather the rows and the ids must not be gathered either.
    """
    table = getattr(layer, "hash_indices_table", None)
    if table is None:
        return None
    if input_ids is None:
        raise ValueError(
            "the fused EP MoE kernel is armed for a hash-routed layer, which "
            "selects its experts from input_ids, and none were supplied"
        )
    return table[input_ids.long().flatten()].to(torch.int32)


def _mesh_expert_order(ep: int) -> tuple[int, ...] | None:
    """Which EP rank's expert block belongs at each mesh index, or None.

    The tuple is static configuration closed into the JAX op. The router keeps
    its original EP-rank order; only the selected expert IDs are mapped inside
    the op after top-k.
    """
    order = ep_rank_order()
    if order is None:
        return None
    order = tuple(order)
    if tuple(sorted(order)) != tuple(range(ep)):
        raise ValueError(f"EP rank order must permute range({ep}); got {order}")
    if order == tuple(range(ep)):
        return None
    # A tuple, not a list: `info_once` keys its cache on the arguments.
    logger.info_once(
        "Fused EP MoE: EP ranks are not in device-id order, so the expert "
        "IDs are relabelled after top-k | ep_rank_at_mesh_index=%s",
        order,
    )
    return order


def _squeeze_channel_scale_checked(scale: torch.Tensor) -> torch.Tensor:
    """`_squeeze_channel_scale` without the shape arguments, for the hot path.

    Prebuild already proved the layout converts for THIS layer, so this only
    has to do it. The `shape[1] == 1` condition is what prebuild proved and is
    kept here anyway: without it a non-singleton axis makes `squeeze` a no-op
    and the loop cannot terminate, and the only in-trace way to say so --
    raising -- reaches the operator as `Unsupported: Observed exception` with
    the message discarded, because vLLM compiles the backbone with
    `fullgraph=True`.
    """
    s = scale
    while s.ndim > 2 and s.shape[1] == 1:
        s = s.squeeze(1)
    return s


def prebuild_fused_moe_ep(
    layer,
    topk: int,
    renormalize: bool,
    activation: str,
    *,
    weight_format="fp8",
    rhs_qb=None,
    weights=None,
    supplies_hash_ids=False,
) -> Any | None:
    """Resolve the mesh and build the fused-EP op for one layer.

    Called once per layer from `process_weights_after_loading`. Everything here
    is forbidden inside a compiled forward: mapping EP ranks to TPU device ids
    is an `all_gather_object`, and a `Mesh` cannot survive Dynamo's tracing.

    Returns the op to serve this layer with, or None on any refusal. The caller
    records it, which is what makes arming per layer rather than per process --
    several of the refusals below read things that vary by layer.

    `supplies_hash_ids` says the caller will pass `hash_topk_ids_operand` on
    every forward. Hash routing is a property of the LAYER, but whether it can
    be served is a property of the quant method calling this: only the
    DeepSeek-V4 MXFP4 method is handed `input_ids`, and arming a hash-routed
    layer under a method that is not would route every token by top-k instead,
    silently. So the layer's table alone does not admit it; the caller's
    promise does.

    `weights` optionally supplies (w1, w2, w1_scale, w2_scale) before they are
    installed on the layer. NVFP4 uses meta tensors describing the proposed
    requantized layout so admission precedes any lossy weight conversion.
    The caller must validate the actual prepared tensors before installing
    them; the built op closes over configuration, never these tensors.
    """
    if not envs.USE_MOE_FUSED_EP_KERNEL:
        return None
    if weight_format == "fp4" and not envs.MOE_FUSED_EP_ENABLE_W4A8:
        return None

    # The experts have to actually be expert-parallel. vLLM builds an EP group
    # for every MoE model whether or not `--enable-expert-parallel` was passed
    # (`initialize_model_parallel` gates only on `model_config.is_moe`), so the
    # mesh alone does not answer this. With the experts replicated, each rank
    # holds the WHOLE expert set, and the kernel would read its shard as one
    # 1/ep block of a global set ep times too large -- a mis-addressed expert
    # lookup, or a late shape error, never a clear refusal.
    if not layer.moe_config.moe_parallel_config.use_ep:
        logger.info_once(
            "Fused EP MoE not engaged: the experts are replicated rather than "
            "expert-parallel, so each shard would hold the whole expert set."
        )
        return None

    # The token-gather table remains in HBM and is streamed through two fixed
    # SMEM windows, so TP and a 4K bucket do not exceed the compile-time SMEM
    # limit. The fused EP program assigns disjoint slices of replicated TP
    # inputs and restores their outputs using native TP groups. The TPU MoERunner
    # patch suppresses its explicit PCP all-gather/reduce-scatter only for a
    # layer this function actually arms; refused layers keep the old fallback.
    # Consequently the kernel is the sole dispatch/combine owner under PCP,
    # independent of whether the separate GMM chunk pipeline is enabled.
    pcp = _pcp_size(layer)
    mesh = build_ep_mesh()
    if mesh is None:
        logger.info_once(
            "Fused EP MoE requested but there is no EP group; "
            "the grouped-matmul path serves every step."
        )
        return None
    ep = mesh.shape[EP_AXIS_NAME]
    # Collective over the world, so before any refusal that could differ by
    # rank; cached after the first layer.
    sorted_gather = (
        ep_sorted_gather_group() if envs.MOE_FUSED_EP_SORTED_GATHER else None
    )

    gather_bytes = token_gather_smem_bytes(_TILE_M)
    needed_smem = _SMEM_OVERHEAD_BYTES + gather_bytes
    smem = _smem_capacity_bytes()
    if needed_smem > smem:
        logger.warning_once(
            "Fused EP MoE not engaged: its fixed routing working set needs "
            "%d B of this chip's %d B SMEM (%d B for two streamed "
            "token-gather windows and about %d B for resident tables and "
            "compiler working memory).",
            needed_smem,
            smem,
            gather_bytes,
            _SMEM_OVERHEAD_BYTES,
        )
        return None
    global _RANK_BUFFER
    # The kernel's index is this rank's position in the mesh, which is ordered
    # by device id unless PCP or DCP is on, so it is not necessarily the EP rank;
    # `ep_rank_order` has the mapping.
    _RANK_BUFFER = torch.tensor(
        [[ep_mesh_index()]], dtype=torch.int32, device=layer.w13_weight.device
    )
    mesh_expert_order = _mesh_expert_order(ep)

    if weights is None:
        if weight_format == "fp4":
            weights = (
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
            )
        else:
            weights = (
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
            )
    w1, w2, w1_scale, w2_scale = weights
    local_experts, hidden = w1.shape[0], w1.shape[1]
    # Torch stores two FP4 output elements per byte; JAX sees the logical
    # width after the bridge converts the native FP4 tensor.
    inter = w1.shape[2] if weight_format == "fp4" else w1.shape[2] // 2

    # The threshold is applied here rather than per step: the op is shape
    # agnostic, so the only place a token count is known outside the traced
    # forward is the scheduler's cap. A deployment whose largest batch is below
    # the threshold never builds the op at all; one above it uses the kernel
    # for every step, decode included.
    max_node_tokens = _max_node_tokens(ep, pcp)
    if max_node_tokens < envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS:
        logger.info_once(
            "Fused EP MoE not engaged: the largest node-wide batch is %d "
            "tokens, below MOE_FUSED_EP_KERNEL_MIN_TOKENS=%d.",
            max_node_tokens,
            envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS,
        )
        return None

    # The kernel reads a token's expert id as `mesh_index * (e_total // ep) +
    # j`, so it assumes every shard owns the same contiguous block. vLLM does
    # not: `determine_expert_map` hands the first `global % ep` ranks one extra
    # expert, and EPLB gives a rank more experts still. The grouped-matmul path
    # copes because it subtracts a per-rank `experts_start`; this one would
    # address the wrong shard for every id past the first uneven boundary,
    # silently. `local_experts * ep` is exactly the `e_total` the kernel will
    # derive from the global weight, so requiring the gate width to match it is
    # the same check the kernel would make if it made one.
    if layer.global_num_experts != local_experts * ep:
        logger.warning_once(
            "Fused EP MoE not engaged: the kernel needs one equal contiguous "
            "expert block per shard, but %d global experts over ep=%d does "
            "not divide into this rank's %d.",
            layer.global_num_experts,
            ep,
            local_experts,
        )
        return None

    reason = fused_moe_ep_unsupported_reason(
        layer,
        weights,
        activation,
        weight_format=weight_format,
        rhs_qb=rhs_qb,
        supplies_hash_ids=supplies_hash_ids,
    )
    if reason is not None:
        logger.warning_once("Fused EP MoE not engaged: %s", reason)
        return None

    sharded_plan = envs.MOE_FUSED_EP_V2_SHARDED_PLAN
    token_replica_groups = ep_token_replica_groups(
        is_sequence_parallel=layer.moe_config.moe_parallel_config.is_sequence_parallel
    )
    scoring_fn, has_score_bias, routed_scale = _routing_config(layer)
    has_hash_ids = getattr(layer, "hash_indices_table", None) is not None
    dedup_cfg = None
    if envs.MOE_FUSED_EP_KERNEL_IMPL in ("dedup", "hybrid"):
        dedup_cfg, reason = _dedup_config(
            mesh, ep, local_experts, weight_format, sorted_gather
        )
        if dedup_cfg is None:
            logger.warning_once("De-Dup MoE not engaged: %s", reason)
            return None
        if envs.MOE_FUSED_EP_KERNEL_IMPL == "hybrid":
            dedup_cfg = ("hybrid", dedup_cfg, envs.VLLM_TT_HYBRID_V2_MAX_T_LOCAL)
    # G3-LOCAL: static expert placement (EPLB). Every refusal above is behind
    # us, so the layer is armed and only the fused op reads these weights.
    l2p = None
    if os.environ.get("DEDUP_MOE_PLACEMENT_DIR") and weight_format == "fp8":
        l2p = _apply_dedup_placement(
            layer, mesh, ep, mesh_expert_order, dedup_cfg, sorted_gather,
            has_hash_ids,
        )
    op = _build_op(
        mesh,
        topk,
        renormalize,
        activation,
        mesh_expert_order,
        sharded_plan,
        weight_format,
        rhs_qb,
        scoring_fn,
        has_score_bias,
        routed_scale,
        has_hash_ids,
        token_replica_groups=token_replica_groups,
        sorted_gather=sorted_gather,
        dedup_cfg=dedup_cfg,
        has_l2p=l2p is not None,
    )
    if l2p is not None:
        op = _PlacedOp(op, l2p)
    if os.environ.get("DEDUP_MOE_ROUTING_HIST_DIR"):
        op = _maybe_route_hist(layer, op, topk, has_score_bias)
    if dedup_cfg is not None and dedup_cfg[0] == "hybrid":
        logger.info_once(
            "Hybrid MoE armed | v2 for t_local < %d, else De-Dup waves=%d "
            "topology=%s",
            dedup_cfg[2],
            dedup_cfg[1][0],
            "direct (no De-Dup)"
            if dedup_cfg[1][1] is None
            else dedup_cfg[1][1].tables().describe(),
        )
    elif dedup_cfg is not None:
        logger.info_once(
            "De-Dup MoE armed | waves=%d topology=%s",
            dedup_cfg[0],
            "direct (no De-Dup)"
            if dedup_cfg[1] is None
            else dedup_cfg[1].tables().describe(),
        )
    logger.info_once(
        "Fused EP MoE armed | hidden=%d inter=%d local_experts=%d ep=%d "
        "pcp=%d topk=%d capacity=%d sharded_plan=%s format=%s rhs_qb=%s "
        "act=%s scoring=%s score_bias=%s hash_routing=%s routed_scale=%s "
        "token_replica_groups=%s",
        hidden,
        inter,
        local_experts,
        ep,
        pcp,
        topk,
        _TILE_M,
        sharded_plan,
        weight_format,
        rhs_qb,
        activation,
        scoring_fn,
        has_score_bias,
        has_hash_ids,
        routed_scale,
        token_replica_groups,
    )
    # Chunked collection stops applying the moment this arms: the fused call
    # returns before `apply_monolithic` reaches the chunked path, and the
    # kernel exchanges inside its own program.
    if envs.TPU_MOE_COLLECTION_CHUNK_SIZE:
        logger.warning_once(
            "Fused EP MoE armed, so TPU_MOE_COLLECTION_CHUNK_SIZE=%s is not "
            "in effect for this layer: the kernel routes and exchanges inside "
            "its own program.",
            envs.TPU_MOE_COLLECTION_CHUNK_SIZE,
        )
    if (
        dedup_cfg is not None
        and isinstance(dedup_cfg[0], int)
        and sorted_gather is None
        and not has_hash_ids
    ):
        setattr(
            layer,
            _DEDUP_EP_META_ATTR,
            _DedupEpMeta(
                mesh=mesh,
                topk=topk,
                renormalize=renormalize,
                activation=activation,
                mesh_expert_order=mesh_expert_order,
                sharded_plan=sharded_plan,
                weight_format=weight_format,
                rhs_qb=rhs_qb,
                scoring_fn=scoring_fn,
                has_score_bias=has_score_bias,
                routed_scale=routed_scale,
                token_replica_groups=token_replica_groups,
                dedup_cfg=dedup_cfg,
                l2p=l2p,
                route_hist=op._hist if isinstance(op, _RouteHistOp) else None,
            ),
        )
    return op


def fused_moe_ep_unsupported_reason(
    layer,
    weights,
    activation,
    *,
    weight_format="fp8",
    rhs_qb=None,
    supplies_hash_ids=False,
) -> str | None:
    """Validate prepared (or meta) weights and routing before installation."""
    w1, w2, w1_scale, w2_scale = weights
    if w1.ndim != 3 or w2.ndim != 3 or w2.shape[0] != w1.shape[0]:
        return f"weight rank/expert mismatch {tuple(w1.shape)} {tuple(w2.shape)}"
    local_experts, hidden = w1.shape[:2]
    inter = w1.shape[2] if weight_format == "fp4" else w1.shape[2] // 2
    packing = 2 if weight_format == "fp4" else 1
    if w2.shape[1] != inter or w2.shape[2] * packing != hidden:
        return f"w2 {tuple(w2.shape)} is not [E, {inter}, {hidden}]"
    from vllm_torchtpu.kernels.fused_moe.v2.kernel import ACT_FNS

    if activation not in ACT_FNS:
        return (
            f"activation {activation!r} is not wired to the kernel's ACT_FNS {ACT_FNS}"
        )
    if weight_format == "fp4":
        if w1.dtype != torch.float4_e2m1fn_x2 or w2.dtype != w1.dtype:
            return "the FP4 form requires native float4_e2m1fn_x2 weights"
        if not isinstance(rhs_qb, int) or rhs_qb <= 0:
            return "NVFP4 fused EP requires a positive integer block size"
        # Match the packed-row geometry used by the kernel's FP4 readers.
        packed_row_tile = U32_SUBLANE_TILE * PACK4
        if rhs_qb % packed_row_tile:
            return (
                f"NVFP4 fused EP block-{rhs_qb} must be a multiple of "
                f"the packed-weight row tile ({packed_row_tile})"
            )
        if hidden % rhs_qb or inter % rhs_qb:
            return (
                f"NVFP4 fused EP block-{rhs_qb} must divide both "
                f"hidden={hidden} and inter={inter}"
            )
        for name, scale, k, n in (
            ("w1_scale", w1_scale, hidden, 2 * inter),
            ("w2_scale", w2_scale, inter, hidden),
        ):
            if tuple(scale.shape) != (local_experts, k // rhs_qb, 1, n):
                return f"{name} is not the block-{rhs_qb} K-major scale layout"
    elif weight_format == "fp8":
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != w1.dtype:
            return "the kernel's fp8 form takes float8_e4m3fn expert weights"
        if _squeeze_channel_scale(w1_scale, local_experts, 2 * inter) is None:
            return "w1_scale is not the per-output-channel form the kernel takes"
        if _squeeze_channel_scale(w2_scale, local_experts, hidden) is None:
            return "w2_scale is not the per-output-channel form the kernel takes"
    else:
        return f"unsupported fused EP weight format {weight_format!r}"
    # The kernel takes expert biases (`has_w1_bias`/`has_w2_bias`) but this
    # bridge does not pass them, and dropping a bias is silent: the output is
    # simply wrong by the bias term on every routed token. Refuse until they
    # are wired through as operands.
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "expert biases are not wired through the fused EP op"
    return _unsupported_routing_reason(layer, supplies_hash_ids=supplies_hash_ids)


def _routing_config(layer) -> tuple[str, bool, float]:
    """The three routing knobs the op closes over, read off the layer.

    Read here rather than taken from the caller so that the values baked into
    the op and the values `_unsupported_routing_reason` admitted are the same
    read of the same layer. A caller that passed its own could disagree with
    what was admitted and route every token differently, silently.
    """
    scoring_fn = layer.scoring_func
    has_bias = layer.e_score_correction_bias is not None
    scale = float(layer.routed_scaling_factor)
    return scoring_fn, has_bias, scale


def _unsupported_routing_reason(layer, *, supplies_hash_ids=False) -> str | None:
    """Why this layer's ROUTING cannot go through the kernel, or None.

    The fused path never calls `moe_routing.route`: the kernel scores with a
    plain softmax and selects top-k itself, with `renormalize` as its only
    option. Everything else `select_experts` can do -- a different scoring
    function, an expert-score correction bias, expert grouping, a supplied
    `custom_routing_function` -- is not that, and taking the fused path anyway
    routes tokens to a DIFFERENT set of experts with different gate weights,
    with nothing raised and nothing logged. Each of these is a refusal rather
    than a silent divergence.

    Hash routing is the one that is now served rather than refused: the
    caller looks the ids up and hands them to the kernel in place of its own
    selection. It is still refused for a caller that did not promise to supply
    them -- see `supplies_hash_ids` on `prebuild_fused_moe_ep`.
    """
    from vllm_torchtpu.kernels.fused_moe.v2.layer import SCORING_FNS
    from vllm_torchtpu.layers.adapter import moe_routing

    if layer.custom_routing_function is not None:
        return (
            "the layer supplies a custom_routing_function; the kernel "
            "routes with its own top-k"
        )
    scoring_fn, has_bias, _ = _routing_config(layer)
    if scoring_fn not in SCORING_FNS:
        return (
            f"scoring_func {scoring_fn!r}; the kernel's router scores "
            f"with {SCORING_FNS}"
        )
    if has_bias:
        bias = layer.e_score_correction_bias
        # The bias is an operand, not a constant, so it has to be a real
        # per-global-expert vector this rank can hand over whole.
        if bias.ndim != 1 or bias.shape[0] != layer.global_num_experts:
            return (
                f"e_score_correction_bias {tuple(bias.shape)} is not one "
                f"value per global expert [{layer.global_num_experts}]"
            )
    if (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    ):
        return "grouped top-k routing is not implemented in the kernel"
    table = getattr(layer, "hash_indices_table", None)
    if table is not None:
        if not supplies_hash_ids:
            return (
                "this layer routes by hash table and its quant method "
                "does not supply input_ids to the fused op"
            )
        if has_bias:
            # Nothing selects with the biased scores on this path, so a bias
            # here would be dropped rather than applied.
            return (
                "hash routing with an e_score_correction_bias: the table "
                "makes the selection the bias exists to shift"
            )
        topk = layer.moe_config.experts_per_token
        if table.ndim != 2 or table.shape[1] != topk:
            return (
                f"hash_indices_table {tuple(table.shape)} is not one row "
                f"per token id with the layer's topk={topk} expert ids"
            )
    # The routing simulator replaces the routing output wholesale, so a run
    # that asked for it and got real routing is a profiling result that is
    # quietly not the thing it claims to measure. The kernel routes by itself,
    # so it can only honour a strategy whose draw can be handed to it as
    # routing operands: `uniform_random` can (`_simulated_routing_operands`),
    # the others are refused.
    if (
        moe_routing._SIMULATION_STRATEGY is not None
        and not moe_routing.is_uniform_random_simulation()
    ):
        return (
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set to a strategy other "
            "than uniform_random and the kernel does its own routing, so the "
            "simulation would not be in effect"
        )
    return None


class _DedupOverrides(NamedTuple):
    """G3-LOCAL per-call De-Dup knobs read from the environment."""

    decode_waves: int
    decode_max_t: int
    direct_max_t: int
    prefill_capacity: int
    prefill_capacity_min_t: int
    kernel_options: tuple
    decode_kernel_options: tuple
    combine_options: tuple

    @property
    def any_set(self):
        return bool(
            self.decode_waves
            or self.prefill_capacity
            or self.kernel_options
            or self.decode_kernel_options
            or self.combine_options
        )


def _json_options(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object; got {raw!r}")
    return tuple(sorted(value.items()))


@functools.cache
def _dedup_overrides():
    """Process-wide De-Dup call overrides (G3-LOCAL).

    DEDUP_MOE_DECODE_WAVES (0 = off): waves for buckets with
    t_local <= DEDUP_MOE_DECODE_MAX_T (default 96, as tokamax DECODE_RULES).
    DEDUP_MOE_KERNEL_OPTS / DEDUP_MOE_COMBINE_OPTS: JSON objects passed to
    `dedup_ep_moe` as `kernel_options` / `combine_options`; unset passes
    neither kwarg, so the call is unchanged.
    """
    decode_waves = int(os.environ.get("DEDUP_MOE_DECODE_WAVES", "0") or 0)
    decode_max_t = int(os.environ.get("DEDUP_MOE_DECODE_MAX_T", "96") or 96)
    # DEDUP_MOE_DIRECT_MAX_T (0 = off): buckets with t_local <= this run the
    # direct path even when DEDUP_MOE_BOARD_SIZE arms De-Dup (G3-LOCAL hybrid).
    direct_max_t = int(os.environ.get("DEDUP_MOE_DIRECT_MAX_T", "0") or 0)
    if direct_max_t < 0:
        raise ValueError(f"DEDUP_MOE_DIRECT_MAX_T={direct_max_t} must be >= 0")
    prefill_capacity = int(
        os.environ.get("DEDUP_MOE_PREFILL_CAPACITY", "0") or 0
    )
    prefill_capacity_min_t = int(
        os.environ.get("DEDUP_MOE_PREFILL_CAPACITY_MIN_T", "512") or 512
    )
    if decode_waves < 0 or decode_max_t < 0 or prefill_capacity_min_t < 0:
        raise ValueError(
            f"DEDUP_MOE_DECODE_WAVES={decode_waves}, "
            f"DEDUP_MOE_DECODE_MAX_T={decode_max_t}, and "
            f"DEDUP_MOE_PREFILL_CAPACITY_MIN_T={prefill_capacity_min_t} "
            "must be >= 0"
        )
    if prefill_capacity < 0 or (prefill_capacity and prefill_capacity % 128):
        raise ValueError(
            f"DEDUP_MOE_PREFILL_CAPACITY={prefill_capacity} must be 0 or a "
            "positive multiple of 128"
        )
    return _DedupOverrides(
        decode_waves,
        decode_max_t,
        direct_max_t,
        prefill_capacity,
        prefill_capacity_min_t,
        _json_options("DEDUP_MOE_KERNEL_OPTS"),
        _json_options("DEDUP_MOE_DECODE_KERNEL_OPTS"),
        _json_options("DEDUP_MOE_COMBINE_OPTS"),
    )


def _moe_t_local(total_rows, ep, token_replica_groups):
    """Distinct tokens per EP device for a (static) global row count."""
    rows = total_rows // ep
    width = len(token_replica_groups[0]) if token_replica_groups else 1
    t_local = -(-rows // width)
    if width > 1:
        align = min(8, 1 << (t_local - 1).bit_length())
        t_local = -(-t_local // align) * align
    return t_local


def _hybrid_pick(t_local, v2_max_t_local):
    """G3-LOCAL hybrid: `v2` below the threshold, else `direct_w4`."""
    return "v2" if t_local < v2_max_t_local else "direct_w4"


def _dedup_fwd_max_runs(topology, waves, t_local):
    """De-Dup forward-table depth for this static bucket, sized like tokamax
    layer.plan.static_fwd_max_runs (max(128, next_pow2(factor * t_local)),
    capped by the SMEM table budget)."""
    from tokamax._src.ops.experimental.dedup_moe.layer import (
        config as dedup_layer_config,
        plan as dedup_layer_plan,
    )

    runs = dedup_layer_plan.static_fwd_max_runs(
        topology.tables().num_slots,
        waves,
        t_local,
        factor=dedup_layer_config.UnifiedRules().fwd_runs_factor,
    )
    # G3-LOCAL: optional cap. A deeper forward table is a larger unrolled
    # program; at t_local >= 256 its compile can exceed host RAM on a 4x4x1
    # head host. Overflow runs are demoted to direct sends (bit-exact).
    cap = int(os.environ.get("DEDUP_MOE_FWD_MAX_RUNS_CAP", "0") or 0)
    return min(runs, cap) if cap > 0 else runs


# G3-LOCAL: De-Dup MoE (MOE_FUSED_EP_KERNEL_IMPL=dedup) support.
#
# TensorCores per chip. A TPU8i tray is four chips in a row along x
# (slices are 4xKx1, one tray per y), each with two TensorCores, so a
# board of 8 EP devices is one tray (x // 4) and a board of 4 is half a tray
# (x // 2, one Borg task).
_DEDUP_CORES_PER_CHIP = 2


def _device_board_keys(device_ids, board_size):
    """Board key of every device in `device_ids`, or (None, reason).

    "coords" groups by physical chip coordinates from torch_tpu's world
    topology (`global_device_attributes`, which is cached process state, not
    a collective): board = (slice, x // chips_per_board, y, z). "id" uses
    id // board_size.
    """
    if envs.DEDUP_MOE_BOARD_SOURCE == "id":
        return [int(d) // board_size for d in device_ids], None
    chips_x = max(1, board_size // _DEDUP_CORES_PER_CHIP)
    try:
        from torch_tpu._internal.distributed import tpu_distributed

        attrs = tpu_distributed.global_device_attributes()
    except Exception as error:  # noqa: BLE001 - optional capability probe
        return None, f"cannot read the device coordinates: {error}"
    by_id = {}
    for d in attrs:
        coords = tuple(int(c) for c in (d.coords or ()))
        if len(coords) < 3:
            return None, f"device {d.id} reports coords {coords}"
        by_id[int(d.id)] = (
            int(d.slice_index or 0),
            coords[0] // chips_x,
            coords[1],
            coords[2],
            coords,
            int(d.core_on_chip or 0),
        )
    missing = [d for d in device_ids if int(d) not in by_id]
    if missing:
        return None, f"no coordinates for EP devices {missing}"
    logger.info_once(
        "De-Dup MoE device coords in mesh order | %s",
        tuple((int(d), by_id[int(d)][4], by_id[int(d)][5]) for d in device_ids),
    )
    return [by_id[int(d)][:4] for d in device_ids], None


class _PlacedOp:
    """The fused op with this layer's logical->physical map appended (G3-LOCAL)."""

    def __init__(self, op, l2p):
        self._op = op
        self._l2p = l2p

    def __call__(self, *args):
        return self._op(*args, self._l2p)


class _RouteHistOp:
    """Counts this rank's top-k expert ids per layer, then calls the op (G3-LOCAL).

    The ids come from torch's top-k on the router logits (plus the score bias,
    when the layer has one): the same selection the kernel makes, with ties
    possibly broken differently. This runs outside the op, so the MoE output is
    unchanged. Rows whose logits are constant (padding: a zero hidden state) or
    NaN (masked padding) are not counted.
    """

    def __init__(self, op, hist, topk, has_score_bias):
        self._op = op
        self._hist = hist
        self._topk = topk
        self._has_score_bias = has_score_bias

    def __call__(self, *args):
        sel = args[5].float()
        if self._has_score_bias:
            sel = sel + args[7].float()
        ids = torch.topk(sel, self._topk, dim=-1).indices
        valid = (sel.amax(dim=-1) > sel.amin(dim=-1)).to(torch.int32)
        weight = valid.unsqueeze(-1).expand_as(ids).reshape(-1)
        self._hist.scatter_add_(0, ids.reshape(-1).to(torch.int64), weight)
        return self._op(*args)


_ROUTE_HISTS: list[tuple[int, Any]] = []
_ROUTE_HIST_THREAD: Any = None


def _dump_route_hists():
    import numpy as np

    directory = os.environ.get("DEDUP_MOE_ROUTING_HIST_DIR", "")
    if not directory or not _ROUTE_HISTS:
        return
    os.makedirs(directory, exist_ok=True)
    me = ep_mesh_index()
    for idx, hist in list(_ROUTE_HISTS):
        path = os.path.join(directory, f"layer_{idx:03d}.mesh{me:02d}.npy")
        np.save(path + ".tmp.npy", hist.detach().to("cpu").numpy())
        os.replace(path + ".tmp.npy", path)


def _maybe_route_hist(layer, op, topk, has_score_bias):
    """Wrap `op` with a per-layer routing histogram (DEDUP_MOE_ROUTING_HIST_DIR).

    The `[global_num_experts]` int32 counts are written to
    `<dir>/layer_<N:03d>.mesh<MM>.npy` every DEDUP_MOE_ROUTING_HIST_EVERY_S
    seconds (default 300) by a daemon thread, and at exit. Ray workers can be
    killed without running atexit, which is why there is a periodic dump.
    """
    global _ROUTE_HIST_THREAD
    from tokamax._src.ops.experimental.dedup_moe.layer import placement

    idx = placement.layer_index(getattr(layer, "layer_name", "") or "")
    if idx is None:
        return op
    hist = torch.zeros(
        int(layer.global_num_experts), dtype=torch.int32, device=layer.w13_weight.device
    )
    layer.register_buffer("_tpu_route_hist", hist, persistent=False)
    _ROUTE_HISTS.append((idx, hist))
    if _ROUTE_HIST_THREAD is None:
        import atexit
        import threading
        import time

        every = float(os.environ.get("DEDUP_MOE_ROUTING_HIST_EVERY_S", "300") or 300)

        def loop():
            while True:
                time.sleep(every)
                try:
                    _dump_route_hists()
                except Exception as error:  # noqa: BLE001 - best effort
                    logger.warning("routing histogram dump failed: %s", error)

        _ROUTE_HIST_THREAD = threading.Thread(
            target=loop, name="dedup_route_hist", daemon=True
        )
        _ROUTE_HIST_THREAD.start()
        atexit.register(_dump_route_hists)
        logger.info_once(
            "De-Dup routing histogram armed | dir=%s every=%.0fs",
            os.environ["DEDUP_MOE_ROUTING_HIST_DIR"],
            every,
        )
    return _RouteHistOp(op, hist, topk, has_score_bias)


# One gather program per (row count, operand shape, dtype), shared by layers.
_PLACE_OPS: dict[tuple, Any] = {}
FUSED_MOE_EP_L2P_ATTR = "_tpu_fused_l2p"
_PLACED_PARAMS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)


def _place_rows_op(mesh, n_rows, shape, dtype):
    key = (n_rows, tuple(shape), dtype)
    op = _PLACE_OPS.get(key)
    if op is None:
        from tokamax._src.ops.experimental.dedup_moe.layer import placement

        def place(w: jax.Array, rows: jax.Array) -> jax.Array:
            return placement.place_rows(w, rows, mesh=mesh, axis=EP_AXIS_NAME)

        op = jax_op(
            f"pallas::dedup_moe_place_{len(_PLACE_OPS)}",
            place,
            mesh=mesh,
            input_partition_specs=(PartitionSpec(EP_AXIS_NAME), PartitionSpec()),
            output_partition_specs=PartitionSpec(EP_AXIS_NAME),
        )
        _PLACE_OPS[key] = op
    return op


def _apply_dedup_placement(
    layer, mesh, ep, mesh_expert_order, dedup_cfg, sorted_gather, has_hash_ids
):
    """Place this layer's experts per DEDUP_MOE_PLACEMENT_DIR (G3-LOCAL).

    Reads `<dir>/layer_<idx:03d>.npy` (`slot_to_orig[S]`, EP-rank-major slots;
    see tokamax dedup_moe/layer/placement.py). Rewrites w13/w2 and their scales
    as the placed `[S/ep, ...]` shards -- a ring gather over the EP mesh, a pure
    copy -- and returns the `[e_logical]` / `[e_logical, nb, 32]` map to pass on
    every call, or None when the layer stays unplaced (no file, or refused).
    Bit-exact vs unplaced: the router scores logical experts; only the selected
    ids are relabelled.
    """
    from tokamax._src.ops.experimental.dedup_moe.layer import placement

    directory = os.environ["DEDUP_MOE_PLACEMENT_DIR"]
    name = getattr(layer, "layer_name", "") or ""
    idx = placement.layer_index(name)
    if idx is None:
        logger.warning_once("De-Dup placement skipped: no layer index in %r", name)
        return None
    s2o = placement.load_layer_placement(directory, idx)
    if s2o is None:
        # DEDUP_MOE_PLACEMENT_STRICT=1: every armed MoE layer must have a file
        # (catches a trace whose layer indexing does not match the model).
        if os.environ.get("DEDUP_MOE_PLACEMENT_STRICT", "0") == "1":
            raise RuntimeError(
                f"DEDUP_MOE_PLACEMENT_STRICT: no placement for {name} "
                f"(layer {idx}) in {directory}"
            )
        logger.info("De-Dup placement | layer=%d unplaced (no file)", idx)
        return None
    if dedup_cfg is None or not isinstance(dedup_cfg[0], int):
        logger.warning_once(
            "De-Dup placement refused: needs MOE_FUSED_EP_KERNEL_IMPL=dedup "
            "(not v2 or hybrid)"
        )
        return None
    if sorted_gather is not None or has_hash_ids:
        logger.warning_once(
            "De-Dup placement refused: sorted gather / hash routing"
        )
        return None
    e_logical = int(layer.global_num_experts)
    try:
        s2o = placement.validate(s2o, e_logical=e_logical, ep=ep)
        local = s2o.shape[0] // ep
        waves = [dedup_cfg[0]] + [w for w in (_dedup_overrides().decode_waves,) if w]
        bad = [w for w in waves if local % w]
        if bad:
            raise ValueError(f"{local} slots per device must divide by waves {bad}")
        rows = placement.source_rows(
            s2o, e_logical=e_logical, ep=ep, mesh_ep_ranks=mesh_expert_order
        )
        board_size = dedup_cfg[1].board_size if dedup_cfg[1] is not None else None
        l2p_np = placement.logical_to_physical_map(
            s2o, e_logical=e_logical, ep=ep, board_size=board_size
        )
    except ValueError as error:
        logger.warning("De-Dup placement refused | layer=%d: %s", idx, error)
        return None
    device = layer.w13_weight.device
    rows_t = torch.tensor(rows, dtype=torch.int32, device=device)
    placed = []
    for attr in _PLACED_PARAMS:
        t = getattr(layer, attr)
        op = _place_rows_op(mesh, rows.shape[0], t.shape[1:], t.dtype)
        placed.append((attr, op(t, rows_t)))
    for attr, new in placed:
        setattr(layer, attr, torch.nn.Parameter(new, requires_grad=False))
    l2p = torch.tensor(l2p_np, dtype=torch.int32, device=device)
    layer.register_buffer(FUSED_MOE_EP_L2P_ATTR, l2p, persistent=False)
    if device.type == "tpu":
        synchronize_tensors([new for _, new in placed] + [l2p])
    logger.info(
        "De-Dup placement applied | layer=%d slots=%d per_device=%d kind=%s "
        "l2p=%s dir=%s",
        idx,
        s2o.shape[0],
        local,
        "perm" if s2o.shape[0] == e_logical else "rep",
        tuple(l2p_np.shape),
        directory,
    )
    return l2p


def _dedup_config(mesh, ep, local_experts, weight_format, sorted_gather):
    """(waves, BoardTopology | None) for the De-Dup op, or (None, reason)."""
    if sorted_gather is not None:
        return None, "MOE_FUSED_EP_SORTED_GATHER splits the program"
    if weight_format != "fp8":
        return None, f"weight format {weight_format!r}; De-Dup takes fp8 here"
    if ep < 3:
        return None, f"wave dispatch needs ep >= 3; got {ep}"
    waves = envs.DEDUP_MOE_WAVES
    if waves < 1 or local_experts % waves:
        return None, (
            f"DEDUP_MOE_WAVES={waves} must divide the {local_experts} experts "
            "per shard"
        )
    try:
        ovr = _dedup_overrides()
    except ValueError as error:
        return None, str(error)
    if ovr.decode_waves and local_experts % ovr.decode_waves:
        return None, (
            f"DEDUP_MOE_DECODE_WAVES={ovr.decode_waves} must divide the "
            f"{local_experts} experts per shard"
        )
    if ovr.any_set:
        # Strings, not dicts: `info_once` keys its `@lru_cache` on the arguments.
        logger.info_once(
            "De-Dup MoE overrides | decode_waves=%d decode_max_t=%d "
            "prefill_capacity=%d kernel_options=%s decode_kernel_options=%s "
            "combine_options=%s",
            ovr.decode_waves,
            ovr.decode_max_t,
            ovr.prefill_capacity,
            str(dict(ovr.kernel_options)),
            str(dict(ovr.decode_kernel_options)),
            str(dict(ovr.combine_options)),
        )
    board_size = envs.DEDUP_MOE_BOARD_SIZE
    if board_size <= 0 or ep // board_size < 2:
        return (waves, None), None
    if ep % board_size:
        return None, f"ep={ep} is not a multiple of board size {board_size}"
    from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology

    device_ids = [int(d.id) for d in mesh.devices.flat]
    keys, reason = _device_board_keys(device_ids, board_size)
    if keys is None:
        return None, reason
    # The kernel needs board b at the contiguous mesh positions
    # [b * size, (b + 1) * size); the mesh order is build_ep_mesh's. Verify the
    # physical grouping is exactly that rather than assume it from the ids.
    groups = {}
    for pos, key in enumerate(keys):
        groups.setdefault(key, []).append(pos)
    blocks = sorted(groups.values(), key=min)
    want = [list(range(b * board_size, (b + 1) * board_size)) for b in range(ep // board_size)]
    if blocks != want:
        return None, (
            f"physical boards are not contiguous {board_size}-device blocks of "
            f"the EP mesh: mesh device ids {device_ids}, board keys {keys}"
        )
    topo = topology.BoardTopology(tuple(tuple(b) for b in blocks))
    return (waves, topo), None


def _max_node_tokens(ep: int, pcp: int = 1) -> int:
    """The largest node-wide token count a step can carry, or 0 if unknown.

    A DP scheduler cap is per independent batch, so every DP rank contributes
    another cap's worth of tokens.  PCP ranks partition one scheduler batch;
    multiplying by the full flattened EP width would count the same logical
    tokens once per PCP shard. ``ep // pcp`` is the number of independent DP
    batches represented here.
    """
    from vllm.config import get_current_vllm_config

    try:
        cfg = get_current_vllm_config()
    except Exception:  # noqa: BLE001 - no config outside a served model
        return 0
    cap = cfg.scheduler_config.max_num_batched_tokens
    if not isinstance(cap, int):
        return 0
    pcp = max(int(pcp), 1)
    return int(cap) * max(ep // pcp, 1)


def _smem_capacity_bytes() -> int:
    """This chip's scalar memory, or tpu7x's outside a device process."""
    try:
        from jax.experimental.pallas import tpu as pltpu

        return int(pltpu.get_tpu_info().smem_capacity_bytes)
    except Exception:  # noqa: BLE001 - CPU host, or an unknown device kind
        return 1024 * 1024


def _pcp_size(layer) -> int:
    """Returns the layer's configured prefill-context-parallel width."""
    return int(layer.moe_config.moe_parallel_config.pcp_size)


# G3-LOCAL (DEDUP_MOE_FUSE_SHARED): fuse TP-sharded shared_expert (+ optional
# sigmoid expert_gate + TP reduction) into De-Dup MoE so it overlaps with
# routing all-gather and arrival combine instead of running serially on TC.
_DEDUP_EP_META_ATTR = "_tpu_dedup_ep_meta"
_DEDUP_SHARED_BUFFERS = (
    "_dedup_ws1",
    "_dedup_ws2",
    "_dedup_ws1_scale",
    "_dedup_ws2_scale",
    "_dedup_ws_gate",
)


class _DedupEpMeta(NamedTuple):
    mesh: Any
    topk: int
    renormalize: bool
    activation: str
    mesh_expert_order: tuple[int, ...] | None
    sharded_plan: bool
    weight_format: str
    rhs_qb: Any
    scoring_fn: str
    has_score_bias: bool
    routed_scale: float
    token_replica_groups: Any
    dedup_cfg: tuple
    l2p: Any
    route_hist: Any


@functools.cache
def _shared_matmul_fn(w8a16_max_rows: int):
    """Cached JAX dense FP8 matmul matching linear_common._quantized_matmul_jax."""
    from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax

    if w8a16_max_rows == 0:
        return _quantized_matmul_jax

    def _mm(x: jax.Array, w_q: jax.Array, w_s: jax.Array) -> jax.Array:
        return _quantized_matmul_jax(x, w_q, w_s, w8a16_max_rows=w8a16_max_rows)

    return _mm


def shared_operands(layer) -> tuple[torch.Tensor, ...]:
    """Staged shared-expert tensors for a fused layer, or (). Safe in trace."""
    ws1 = getattr(layer, _DEDUP_SHARED_BUFFERS[0], None)
    if ws1 is None:
        return ()
    tensors = (
        ws1,
        getattr(layer, _DEDUP_SHARED_BUFFERS[1]),
        getattr(layer, _DEDUP_SHARED_BUFFERS[2]),
        getattr(layer, _DEDUP_SHARED_BUFFERS[3]),
    )
    ws_gate = getattr(layer, _DEDUP_SHARED_BUFFERS[4], None)
    return (*tensors, ws_gate) if ws_gate is not None else tensors


def _parse_dedup_fuse_shared_env() -> str | None:
    """Overlap mode for DEDUP_MOE_FUSE_SHARED, or None when disabled."""
    raw = os.environ.get("DEDUP_MOE_FUSE_SHARED", "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return None
    if raw in ("1", "true", "yes", "on", "routing_ag"):
        return "routing_ag"
    if raw in ("dispatch", "split"):
        return raw
    raise ValueError(
        f"Invalid DEDUP_MOE_FUSE_SHARED={raw!r}; expected "
        "0/1/routing_ag/dispatch/split."
    )


def _valid_fp8_linear_scale(scale: torch.Tensor, k_dim: int, n_dim: int) -> bool:
    if scale.dtype != torch.float32:
        return False
    if scale.ndim == 4:
        return (
            scale.shape[0] == 1
            and scale.shape[2] == 1
            and scale.shape[3] == n_dim
            and scale.shape[1] > 0
            and k_dim % scale.shape[1] == 0
        )
    if scale.ndim == 1:
        return scale.shape[0] in (n_dim, 1)
    return False


def _dedup_shared_slice(
    runner, *, hidden: int, replica_width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int] | str:
    """This rank's shared-expert tensors for De-Dup fusion, or refusal reason."""
    from vllm.model_executor.layers.activation import SiluAndMul

    from vllm_torchtpu.layers.adapter.linear_common import (
        KEEP_VLLM_LAYOUT_ATTR,
        WEIGHT_FLIPPED_ATTR,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    shared = getattr(runner, "_shared_experts", None)
    mlp = getattr(shared, "_layer", None)
    if mlp is None:
        return "no runner._shared_experts._layer"
    if not isinstance(getattr(mlp, "act_fn", None), SiluAndMul):
        return f"unsupported shared act_fn {type(getattr(mlp, 'act_fn', None)).__name__}"
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if gate_up is None or down is None:
        return "shared MLP lacks gate_up_proj/down_proj"
    for name, lin in (("gate_up_proj", gate_up), ("down_proj", down)):
        if not isinstance(getattr(lin, "quant_method", None), VllmFp8LinearMethodTPU):
            return f"{name} is not VllmFp8LinearMethodTPU"
        if not getattr(lin, WEIGHT_FLIPPED_ATTR, False) or getattr(
            lin, KEEP_VLLM_LAYOUT_ATTR, False
        ):
            return f"{name} is not in canonical K-major layout"
        if getattr(lin, "bias", None) is not None:
            return f"{name} has a bias"
        if lin.weight.dtype != torch.float8_e4m3fn:
            return f"{name} weight dtype {lin.weight.dtype} is not float8_e4m3fn"
    if replica_width > 1 and getattr(down, "reduce_results", True):
        return "shared down_proj reduces its own output"
    # The op sums the shared partials over the token-replica group only, so
    # the linears' TP width must equal that group width (1 when replicated,
    # e.g. disable_tp under sequence-parallel MoE).
    from vllm.distributed import get_tensor_model_parallel_world_size

    for name, lin in (("gate_up_proj", gate_up), ("down_proj", down)):
        lin_tp = int(getattr(lin, "tp_size", get_tensor_model_parallel_world_size()))
        if lin_tp != replica_width:
            return (
                f"{name} tp_size={lin_tp} != token replica width {replica_width}"
            )
    ws1, ws1_scale = gate_up.weight.data, gate_up.weight_scale.data
    ws2, ws2_scale = down.weight.data, down.weight_scale.data
    if ws1.ndim != 2 or ws2.ndim != 2:
        return "shared weights are not 2-D"
    fs = ws2.shape[0]
    if tuple(ws1.shape) != (hidden, 2 * fs) or tuple(ws2.shape) != (fs, hidden):
        return f"shared shapes {tuple(ws1.shape)}/{tuple(ws2.shape)} mismatch hidden={hidden}"
    if not _valid_fp8_linear_scale(ws1_scale, hidden, 2 * fs):
        return f"unsupported ws1_scale shape={tuple(ws1_scale.shape)} dtype={ws1_scale.dtype}"
    if not _valid_fp8_linear_scale(ws2_scale, fs, hidden):
        return f"unsupported ws2_scale shape={tuple(ws2_scale.shape)} dtype={ws2_scale.dtype}"
    w8a16_1 = int(getattr(gate_up.quant_method, "w8a16_max_rows", 0))
    w8a16_2 = int(getattr(down.quant_method, "w8a16_max_rows", 0))
    if w8a16_1 != w8a16_2:
        return f"mismatched w8a16_max_rows ({w8a16_1} vs {w8a16_2})"
    gate = getattr(mlp, "expert_gate", None)
    ws_gate = None
    if gate is not None:
        if getattr(gate, "bias", None) is not None:
            return "expert_gate has a bias"
        w = getattr(gate, "weight", None)
        if (
            w is None
            or w.ndim != 2
            or tuple(w.shape) not in ((1, hidden), (hidden, 1))
            or w.dtype not in (torch.bfloat16, torch.float32)
        ):
            return f"unsupported expert_gate weight {None if w is None else (tuple(w.shape), w.dtype)}"
        ws_gate = w.data
    elif type(mlp).__name__ not in ("DeepseekV2MLP", "Qwen2MoeMLP", "Qwen3NextMLP"):
        return f"unrecognized ungated shared MLP {type(mlp).__name__}"
    # Hand over parameter `.data` directly, never views (avoids per-step
    # `tt_jit_as_strided` re-materialization).
    return (ws1, ws2, ws1_scale, ws2_scale, ws_gate, w8a16_1)


def _dedup_shared_admit(runner) -> tuple[_DedupEpMeta, tuple, float] | str:
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    if not isinstance(runner._quant_method, VllmFp8MoEMethodTPU):
        return f"quant method {type(runner._quant_method).__name__} is not VllmFp8MoEMethodTPU"
    layer = runner.routed_experts
    meta = getattr(layer, _DEDUP_EP_META_ATTR, None)
    if meta is None or not fused_moe_ep_supported(runner._quant_method):
        return "layer is not armed with De-Dup EP MoE"
    cfg = runner.moe_config
    if getattr(cfg, "skip_final_all_reduce", False):
        return "skip_final_all_reduce is set"
    if (
        runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
    ):
        return "routed input/output transform is set"
    if runner.shared_expert_gate is not None:
        return "runner.shared_expert_gate is set externally"
    runner_scale = float(getattr(runner, "routed_scaling_factor", 1.0))
    if runner_scale != 1.0 and getattr(cfg, "in_dtype", None) == torch.float16:
        return "fp16 routed_scaling_factor != 1.0"
    replica_width = (
        len(meta.token_replica_groups[0]) if meta.token_replica_groups else 1
    )
    hidden = int(layer.w13_weight.shape[1])
    shared = _dedup_shared_slice(
        runner, hidden=hidden, replica_width=replica_width
    )
    if isinstance(shared, str):
        return shared
    return meta, shared, runner_scale


def _check_same_shared_arming_on_every_rank(signature: tuple, what: str) -> None:
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group

    group = get_ep_group()
    if group is None:
        return
    gathered: list = [None] * int(group.world_size)
    torch.distributed.all_gather_object(gathered, signature, group=group.cpu_group)
    if any(g != gathered[0] for g in gathered):
        raise RuntimeError(
            f"De-Dup MoE fused shared expert ({what}): EP ranks disagree on "
            f"armed layers: {sorted({str(g)[:160] for g in gathered})}"
        )


def arm_dedup_fuse_shared(model: torch.nn.Module, what: str = "model") -> int:
    """Fuse eligible layers' shared_expert (+ gate + TP reduce) into De-Dup MoE."""
    shared_overlap = _parse_dedup_fuse_shared_env()
    if shared_overlap is None:
        return 0
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    if not runners:
        return 0
    plan = []
    refusals: set[str] = set()
    for runner in runners:
        res = _dedup_shared_admit(runner)
        if isinstance(res, str):
            refusals.add(res)
        else:
            plan.append((runner, *res))
    signature = (
        shared_overlap,
        tuple(
            (
                getattr(r, "layer_name", "?"),
                shared[4] is not None,
                meta.l2p is not None,
                shared[2].ndim,
                shared[5],
            )
            for r, meta, shared, _ in plan
        ),
    )
    _check_same_shared_arming_on_every_rank(signature, what)
    armed = 0
    for runner, meta, shared, runner_scale in plan:
        ws1, ws2, ws1_scale, ws2_scale, ws_gate, w8a16_max_rows = shared
        layer = runner.routed_experts
        op = _build_op(
            meta.mesh,
            meta.topk,
            meta.renormalize,
            meta.activation,
            meta.mesh_expert_order,
            meta.sharded_plan,
            meta.weight_format,
            meta.rhs_qb,
            meta.scoring_fn,
            meta.has_score_bias,
            meta.routed_scale * runner_scale,
            False,
            token_replica_groups=meta.token_replica_groups,
            sorted_gather=None,
            dedup_cfg=meta.dedup_cfg,
            has_l2p=meta.l2p is not None,
            shared_cfg=(ws_gate is not None, shared_overlap, w8a16_max_rows),
        )
        if meta.l2p is not None:
            op = _PlacedOp(op, meta.l2p)
        if meta.route_hist is not None:
            op = _RouteHistOp(op, meta.route_hist, meta.topk, meta.has_score_bias)
        setattr(runner._quant_method, FUSED_MOE_EP_OP_ATTR, op)
        for buf_name, tensor in zip(
            _DEDUP_SHARED_BUFFERS[:4], (ws1, ws2, ws1_scale, ws2_scale)
        ):
            layer.register_buffer(buf_name, tensor, persistent=False)
        if ws_gate is not None:
            layer.register_buffer(_DEDUP_SHARED_BUFFERS[4], ws_gate, persistent=False)
        runner._shared_experts = None
        runner.routed_scaling_factor = 1.0
        runner._forward_entry = runner._select_forward()
        armed += 1
    logger.info(
        "De-Dup MoE fused shared expert armed on %d of %d MoE layers of the %s "
        "(overlap=%s, gated=%s, scale_ndim=%s, refusals=%s)",
        armed,
        len(runners),
        what,
        shared_overlap,
        sorted({s[1] for s in signature[1]}),
        sorted({s[3] for s in signature[1]}),
        sorted(refusals),
    )
    return armed


def arm_dedup_fuse_shared_models(model_runner: Any) -> int:
    """Arm the target model and distinct draft model (MTP) after weight load."""
    armed = arm_dedup_fuse_shared(model_runner.model, "target model")
    draft = getattr(getattr(model_runner, "drafter", None), "draft_model", None)
    if isinstance(draft, torch.nn.Module) and draft is not model_runner.model:
        armed += arm_dedup_fuse_shared(draft, "draft model")
    return armed
