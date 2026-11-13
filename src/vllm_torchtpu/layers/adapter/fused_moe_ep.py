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

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec

# Stock jax_op, as upstream main. torch_tpu >= 0.2.0 takes
# `output_partition_specs` itself and sizes each output for this rank's
# shard. The fork's `sharded_jax_op` shim re-derived JaxCallable.__call__
# from torch_tpu's source to get the same effect; against 0.2.0 its needle
# no longer matches, so it silently falls back to stock-with-global-sizing
# ("sharded outputs will be sized globally"), and the fused EP op then
# returned ep*T rows (6144 for DP8 x 768) where the runner expected T.
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    ep_sorted_gather_group,
    ep_token_replica_groups,  # noqa: F401  (tests patch it on this module)
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


def _prefill_role() -> bool:
    """P/D merge role selector for the fused EP op's tuning knobs.

    True only on the prefill node (VLLM_DSV41_PURE_20L_PREFILL=1): the small-T
    capacity ladder, the BF16 wire, sub-channel FP8 activations and the small-T
    FP8 shard_map path follow their env knobs exactly as in the prefill tree.
    Otherwise the op keeps the deployed decode behaviour: capacity=_TILE_M
    (unless VLLM_DSV41_MOE_V2_CAPACITY is set), FP8 wire, per-row FP8
    activations, per-expert (uncoalesced) peer push, no small-T FP8 path.
    """
    return bool(envs.VLLM_DSV41_PURE_20L_PREFILL)


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
    precomputed_routing=False,
    *,
    token_replica_groups=None,
    sorted_gather=None,
):
    """The op for this closure, built once per distinct closure and reused.

    `sorted_gather` is `ep_sorted_gather_group()`'s (group, positions), or
    None for the single fused program. With it the op is a `_SortedGatherOp`.
    """
    from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2

    (ax,) = mesh.axis_names
    ep = mesh.shape[ax]
    prefill_role = _prefill_role()
    # Decode role: the deployed decode kernel has no sub-channel activation
    # scales, no BF16 wire and no coalesced peer push; pin them off so the
    # env defaults (SUBCHANNEL_FP8=1, WIRE_BF16_MAX_TOKENS=768) cannot leak in.
    subchannel_fp8 = (
        bool(envs.VLLM_DSV41_MOE_V2_SUBCHANNEL_FP8) if prefill_role else False
    )
    requantize_mid = bool(envs.VLLM_DSV41_MOE_V2_REQUANTIZE)
    wire_bf16_max_tokens = (
        int(envs.VLLM_DSV41_MOE_V2_WIRE_BF16_MAX_TOKENS) if prefill_role else -1
    )
    coalesce_kw = (
        {}
        if (prefill_role or envs.VLLM_DSV41_MOE_V2_COALESCED_PUSH)
        else {"coalesce_peer_push": False}
    )
    fp4_max_t = int(envs.VLLM_DSV41_MOE_FP4_PATH_C_MAX_TOKENS)
    fp8_max_t = int(envs.VLLM_DSV41_MOE_FP8_PATH_C_MAX_TOKENS)
    max_v2_t_local = (
        int(envs.VLLM_DSV41_MOE_EP_V2_MAX_LOCAL_TOKENS)
        if envs.VLLM_DSV41_MOE_EP_V2_MAX_LOCAL_TOKENS is not None
        else (32768 if ep <= 2 else 8192)
    )
    cap_override = envs.VLLM_DSV41_MOE_V2_CAPACITY

    def _capacity(rows: int) -> int:
        if cap_override is not None:
            return int(cap_override)
        return 32 if rows <= 128 else (64 if rows <= 1024 else _TILE_M)

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
        precomputed_routing,
        token_replica_groups,
        None if sorted_gather is None else sorted_gather[1],
        subchannel_fp8,
        requantize_mid,
        wire_bf16_max_tokens,
        fp4_max_t,
        fp8_max_t,
        max_v2_t_local,
        cap_override,
        prefill_role,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached

    name = FUSED_MOE_EP_OP_NAME if not _OPS else f"{FUSED_MOE_EP_OP_NAME}_{len(_OPS)}"

    routing_kw = dict(
        scoring_fn=scoring_fn,
        routed_scaling_factor=routed_scaling_factor,
        subchannel_fp8=subchannel_fp8,
        requantize_mid=requantize_mid,
    )

    def _unpack_fp4_u8(w1, w2):
        if weight_format == "fp4" and w1.dtype == jnp.uint8:
            w1 = jax.lax.bitcast_convert_type(w1, jnp.float4_e2m1fn).reshape(
                *w1.shape[:-1], w1.shape[-1] * 2
            )
            w2 = jax.lax.bitcast_convert_type(w2, jnp.float4_e2m1fn).reshape(
                *w2.shape[:-1], w2.shape[-1] * 2
            )
        return w1, w2

    # Any: the layer, route or compute result depending on bias_kw's stage;
    # each wrapper below declares the one it returns.
    def call(x, w1, w2, w1_scale, w2_scale, gating, rank, **bias_kw) -> Any:
        w1, w2 = _unpack_fp4_u8(w1, w2)
        w1_scale = w1_scale.squeeze(-2) if w1_scale.ndim == 4 else w1_scale
        w2_scale = w2_scale.squeeze(-2) if w2_scale.ndim == 4 else w2_scale
        _cap = _capacity(x.shape[0])
        # The router's top-k selector writes into an f32 accumulator ref, so a
        # bf16 gate output -- which is what the model's gate matmul produces --
        # fails inside the kernel on a dtype mismatch. Widen at the boundary;
        # softmax runs in f32 on this path anyway.
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
            capacity=_cap,
            block=None,
            ragged_stride=None,
            weight_format=WeightFormat(weight_format),
            rhs_qb=rhs_qb,
            act_fn=activation,
            mesh_ep_ranks=mesh_expert_order,
            sharded_plan=sharded_plan,
            token_replica_groups=token_replica_groups,
            wire_bf16=(x.shape[0] <= wire_bf16_max_tokens),
            **coalesce_kw,
            **bias_kw,
            **routing_kw,
        )

    def _small_t_fp8_gated(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        *,
        score_bias: jax.Array | None = None,
        hash_ids: jax.Array | None = None,
    ) -> jax.Array:
        w1, w2 = _unpack_fp4_u8(w1, w2)
        from jax import lax

        from vllm_torchtpu.kernels.fused_moe.v2.layer import (
            _apply_scoring_fn,
        )
        from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select
        from vllm_torchtpu.layers.core.fused_moe_gmm import (
            fused_moe_func_fp8_dispatch,
            quantize_moe_dispatch_packed_rows,
            resolve_onehot_permute_threshold,
        )

        T_global = x.shape[0]
        (ax,) = mesh.axis_names
        ep = mesh.shape[ax]
        t_local = T_global // ep
        e_total = w1.shape[0]
        g_local = e_total // ep
        onehot_thr = resolve_onehot_permute_threshold()
        _fn_dispatch = getattr(
            fused_moe_func_fp8_dispatch,
            "__wrapped__",
            fused_moe_func_fp8_dispatch,
        )
        sel_rows = (
            t_local
            if t_local <= 256
            else next(d for d in range(256, 7, -8) if t_local % d == 0)
        )
        skip_pad = T_global > 16

        def _shard_fn(x_l, w1_l, w2_l, w1s_l, w2s_l, gate_l, rank_l, *extra_l):
            scores = _apply_scoring_fn(gate_l, scoring_fn)
            if hash_ids is not None:
                ids_l = extra_l[0].astype(jnp.int32)
                w_l = jnp.take_along_axis(scores, ids_l, axis=-1)
            elif score_bias is not None:
                sb_l = extra_l[0].astype(jnp.float32)
                w_l, ids_l = pallas_select(
                    scores + sb_l[None, :],
                    topk=topk,
                    block_rows=sel_rows,
                    weight_scores=scores,
                )
            else:
                w_l, ids_l = pallas_select(scores, topk=topk, block_rows=sel_rows)
            row_routes = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
            if renormalize:
                denom = jnp.maximum(w_l.sum(axis=-1, keepdims=True), 1e-20)
                w_l = w_l / jnp.where(row_routes, denom, 1.0)
            if routed_scaling_factor != 1.0:
                w_l = w_l * jnp.float32(routed_scaling_factor)
            w_l = jnp.where(row_routes, w_l, 0.0)
            ids_l = ids_l.astype(jnp.int32)
            packed_l = quantize_moe_dispatch_packed_rows(
                x_l, w_l.astype(x_l.dtype), ids_l
            )
            packed_g = lax.all_gather(packed_l, ax, axis=0, tiled=True)
            me = rank_l[0, 0]
            ep_rank = (
                jnp.asarray(mesh_expert_order, dtype=jnp.int32)[me]
                if mesh_expert_order is not None
                else me
            )
            out_g = _fn_dispatch(
                packed_g,
                jnp.empty((0,), dtype=x_l.dtype),
                w1_l,
                w2_l,
                w1s_l[:, None, None, :] if w1s_l is not None else None,
                w2s_l[:, None, None, :] if w2s_l is not None else None,
                None,
                None,
                jnp.empty((0, topk), dtype=x_l.dtype),
                jnp.empty((0, topk), dtype=jnp.int32),
                experts_start=ep_rank * g_local,
                topk=topk,
                activation=activation,
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=onehot_thr,
                skip_padded_tokens=skip_pad,
            )
            return lax.psum_scatter(
                out_g.astype(x_l.dtype), ax, scatter_dimension=0, tiled=True
            )

        extra_args = ()
        extra_specs = ()
        if hash_ids is not None:
            extra_args = (hash_ids.astype(jnp.int32),)
            extra_specs = (PartitionSpec(ax),)
        elif score_bias is not None:
            extra_args = (score_bias.astype(jnp.float32),)
            extra_specs = (PartitionSpec(),)

        return jax.shard_map(
            _shard_fn,
            mesh=mesh,
            in_specs=(PartitionSpec(ax),) * 7 + extra_specs,
            out_specs=PartitionSpec(ax),
            check_vma=False,
        )(x, w1, w2, w1_scale, w2_scale, gating, rank, *extra_args)

    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        if (
            prefill_role
            and x.shape[0] <= 384
            and weight_format == "fp8"
            and token_replica_groups is None
        ):
            return _small_t_fp8_gated(x, w1, w2, w1_scale, w2_scale, gating, rank)
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
        if (
            prefill_role
            and x.shape[0] <= 384
            and weight_format == "fp8"
            and token_replica_groups is None
        ):
            return _small_t_fp8_gated(
                x, w1, w2, w1_scale, w2_scale, gating, rank, score_bias=score_bias
            )
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
        if (
            prefill_role
            and x.shape[0] <= 384
            and weight_format == "fp8"
            and token_replica_groups is None
        ):
            return _small_t_fp8_gated(
                x, w1, w2, w1_scale, w2_scale, gating, rank, hash_ids=hash_ids
            )
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

    def moe_with_precomputed_routing(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_ids: jax.Array,
        topk_weights: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        w1_scale = w1_scale.squeeze(-2) if w1_scale.ndim == 4 else w1_scale
        w2_scale = w2_scale.squeeze(-2) if w2_scale.ndim == 4 else w2_scale
        T_global, hidden = x.shape
        (ax,) = mesh.axis_names
        ep = mesh.shape[ax]
        t_local = T_global // ep
        e_total = w1.shape[0]
        g_local = e_total // ep

        # fp4_max_t and fp8_max_t closed over from envs and recorded in _OPS key
        use_path_c = T_global <= (
            fp4_max_t if weight_format == "fp4" else fp8_max_t
        ) and (t_local % 8 == 0)
        if use_path_c and weight_format in ("fp8", "fp4"):
            from jax import lax

            from vllm_torchtpu.layers.core.fused_moe_gmm import (
                fused_moe_func_fp8_dispatch,
                quantize_moe_dispatch_packed_rows,
                resolve_onehot_permute_threshold,
            )

            onehot_thr = resolve_onehot_permute_threshold()
            _fn_dispatch = getattr(
                fused_moe_func_fp8_dispatch,
                "__wrapped__",
                fused_moe_func_fp8_dispatch,
            )
            rhs_qdtype = (
                None
                if w1.dtype == jnp.float4_e2m1fn
                else (jnp.float4_e2m1fn if weight_format == "fp4" else None)
            )

            def _local_path_c(x_l, w1_l, w2_l, w1s_l, w2s_l, ids_l, w_l, rank_l):
                packed_l = quantize_moe_dispatch_packed_rows(
                    x_l, w_l.astype(x_l.dtype), ids_l.astype(jnp.int32)
                )
                packed_g = lax.all_gather(packed_l, ax, axis=0, tiled=True)
                me = rank_l[0, 0]
                ep_rank = (
                    jnp.asarray(mesh_expert_order, dtype=jnp.int32)[me]
                    if mesh_expert_order is not None
                    else me
                )
                exp_start = ep_rank * g_local
                w1s_4d = (
                    w1s_l[:, :, None, :]
                    if (w1s_l is not None and w1s_l.ndim == 3)
                    else (w1s_l[:, None, None, :] if w1s_l is not None else None)
                )
                w2s_4d = (
                    w2s_l[:, :, None, :]
                    if (w2s_l is not None and w2s_l.ndim == 3)
                    else (w2s_l[:, None, None, :] if w2s_l is not None else None)
                )
                out_g = _fn_dispatch(
                    packed_g,
                    jnp.empty((0,), dtype=x_l.dtype),
                    w1_l,
                    w2_l,
                    w1s_4d,
                    w2s_4d,
                    None,
                    None,
                    jnp.empty((0, topk), dtype=x_l.dtype),
                    jnp.empty((0, topk), dtype=jnp.int32),
                    experts_start=exp_start,
                    topk=topk,
                    activation=activation,
                    use_ep=True,
                    rhs_quant_dtype=rhs_qdtype,
                    use_sparse_core=True,
                    onehot_moe_permute_threshold=onehot_thr,
                    skip_padded_tokens=(T_global > 16),
                )
                return lax.psum_scatter(
                    out_g.astype(x_l.dtype), ax, scatter_dimension=0, tiled=True
                )

            return jax.shard_map(
                _local_path_c,
                mesh=mesh,
                in_specs=(PartitionSpec(ax),) * 8,
                out_specs=PartitionSpec(ax),
                check_vma=False,
            )(
                x,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_ids.astype(jnp.int32),
                topk_weights.astype(jnp.float32),
                rank,
            )

        def _run_v2(x_s, ids_s, w_s):
            _cap_s = _capacity(x_s.shape[0])
            return fused_ep_moe_v2(
                x_s,
                w1,
                w2,
                w1_scale,
                w2_scale,
                None,
                rank=rank,
                topk=topk,
                renormalize=False,
                mesh=mesh,
                capacity=_cap_s,
                block=None,
                ragged_stride=None,
                weight_format=WeightFormat(weight_format),
                rhs_qb=rhs_qb,
                act_fn=activation,
                mesh_ep_ranks=mesh_expert_order,
                sharded_plan=sharded_plan,
                hash_topk_ids=ids_s.astype(jnp.int32),
                topk_weights_override=w_s.astype(jnp.float32),
                routed_scaling_factor=1.0,
                token_replica_groups=(
                    None if precomputed_routing else token_replica_groups
                ),
                subchannel_fp8=subchannel_fp8,
                requantize_mid=requantize_mid,
                wire_bf16=(x_s.shape[0] <= wire_bf16_max_tokens),
                **coalesce_kw,
            )

        _max_v2_t_local = max_v2_t_local
        if t_local <= _max_v2_t_local:
            return _run_v2(x, topk_ids, topk_weights)

        x_sh = x.reshape(ep, t_local, hidden)
        ids_sh = topk_ids.reshape(ep, t_local, topk)
        w_sh = topk_weights.reshape(ep, t_local, topk)
        out_chunks = []
        for s in range(0, t_local, _max_v2_t_local):
            e = min(s + _max_v2_t_local, t_local)
            c_len = e - s
            out_c = _run_v2(
                x_sh[:, s:e, :].reshape(ep * c_len, hidden),
                ids_sh[:, s:e, :].reshape(ep * c_len, topk),
                w_sh[:, s:e, :].reshape(ep * c_len, topk),
            )
            out_chunks.append(out_c.reshape(ep, c_len, hidden))
        return jnp.concatenate(out_chunks, axis=1).reshape(T_global, hidden)

    spec = PartitionSpec(EP_AXIS_NAME)
    if precomputed_routing:
        input_specs = (spec,) * 8
        fn = moe_with_precomputed_routing
    else:
        assert not (has_score_bias and has_hash_ids)
        input_specs = (
            (spec,) * 7
            + ((PartitionSpec(),) if has_score_bias else ())
            + ((spec,) if has_hash_ids else ())
        )
        if has_score_bias:
            fn = moe_with_score_bias
        elif has_hash_ids:
            fn = moe_with_hash_ids
        else:
            fn = moe
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=input_specs,
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    if sorted_gather is not None and not precomputed_routing:
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
    elif sorted_gather is not None:
        logger.info_once(
            "Fused EP MoE: MOE_FUSED_EP_SORTED_GATHER is not applied to the "
            "precomputed-routing op; it runs as the single fused program."
        )
    op._jax_fn = fn
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
    *,
    precomputed_topk_ids: torch.Tensor | None = None,
    precomputed_topk_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    op = getattr(owner, FUSED_MOE_EP_OP_ATTR)
    if w1.dtype == torch.float4_e2m1fn_x2:
        # P/D merge: the decode tree hands native float4_e2m1fn_x2 weights to
        # the op unchanged; the prefill tree re-views them as u8 and unpacks
        # inside the op (`_unpack_fp4_u8`). Prefill's MXFP4 path supplies int32
        # words, so this branch only matters on decode -- keep D's form there.
        if _prefill_role():
            w1 = w1.view(torch.uint8)
            w2 = w2.view(torch.uint8)
        scales = (w1_scale, w2_scale)
    elif w1.dtype in (torch.uint8, torch.int32):
        scales = (w1_scale, w2_scale)
    else:
        scales = (
            _squeeze_channel_scale_checked(w1_scale),
            _squeeze_channel_scale_checked(w2_scale),
        )
    if precomputed_topk_ids is not None and precomputed_topk_weights is not None:
        return op(
            hidden_states,
            w1,
            w2,
            *scales,
            precomputed_topk_ids.to(torch.int32),
            precomputed_topk_weights.to(torch.float32),
            _RANK_BUFFER,
        )
    # Simulate before masking: the padding mask writes NaN rows, which would
    # make the simulation's mean anchor NaN and leave every row unrouted.
    router_logits, score_bias, hash_ids = _simulated_routing_operands(
        router_logits, score_bias, hash_ids
    )
    if envs.TPU_MOE_SKIP_PADDED_TOKENS:
        router_logits = token_padding.mask_router_logits_for_padding(router_logits)
    extra = tuple(t for t in (score_bias, hash_ids) if t is not None)
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
    precomputed_routing=False,
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

    if not layer.moe_config.moe_parallel_config.use_ep:
        logger.info_once(
            "Fused EP MoE not engaged: the experts are replicated rather than "
            "expert-parallel, so each shard would hold the whole expert set."
        )
        return None

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
        [[ep_mesh_index()]],
        dtype=torch.int32,
        device=(
            layer.w13_weight.device
            if getattr(layer, "w13_weight", None) is not None
            else next(layer.parameters()).device
        ),
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
    local_experts = w1.shape[0]
    if weight_format == "fp4" and w1.dtype == torch.int32:
        hidden = w1.shape[1] * PACK4
        inter = w1.shape[2] // 2
    else:
        hidden = w1.shape[1]
        inter = w1.shape[2] if weight_format == "fp4" else w1.shape[2] // 2

    max_node_tokens = _max_node_tokens(ep, pcp)
    if max_node_tokens < envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS:
        logger.info_once(
            "Fused EP MoE not engaged: the largest node-wide batch is %d "
            "tokens, below MOE_FUSED_EP_KERNEL_MIN_TOKENS=%d.",
            max_node_tokens,
            envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS,
        )
        return None

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
        precomputed_routing=precomputed_routing,
    )
    if reason is not None:
        logger.warning_once("Fused EP MoE not engaged: %s", reason)
        return None

    sharded_plan = envs.MOE_FUSED_EP_V2_SHARDED_PLAN
    # P/D merge: no token replica groups. The prefill fix (cd4eedbcb) dropped
    # them under precomputed routing because token-sharded TP ranks hold
    # DISTINCT rows; the decode tree (1d02e4ab9) dropped them unconditionally.
    # DeepseekV4MoE feeds the fused op distinct per-TP-rank rows on every
    # fused path, so replica dedup is never valid there; at TP=1 the groups
    # are singletons, which TokenReplicaLayout treats exactly like None.
    token_replica_groups = None
    scoring_fn, has_score_bias, routed_scale = _routing_config(layer)
    has_hash_ids = getattr(layer, "hash_indices_table", None) is not None
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
        precomputed_routing,
        token_replica_groups=token_replica_groups,
        sorted_gather=sorted_gather,
    )
    logger.info_once(
        "Fused EP MoE armed | hidden=%d inter=%d local_experts=%d ep=%d "
        "pcp=%d topk=%d capacity=%d sharded_plan=%s format=%s rhs_qb=%s "
        "act=%s scoring=%s score_bias=%s hash_routing=%s precomputed_routing=%s "
        "routed_scale=%s token_replica_groups=%s",
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
        precomputed_routing,
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
    return op


def fused_moe_ep_unsupported_reason(
    layer,
    weights,
    activation,
    *,
    weight_format="fp8",
    rhs_qb=None,
    supplies_hash_ids=False,
    precomputed_routing=False,
) -> str | None:
    """Validate prepared (or meta) weights and routing before installation."""
    w1, w2, w1_scale, w2_scale = weights
    if w1.ndim != 3 or w2.ndim != 3 or w2.shape[0] != w1.shape[0]:
        return f"weight rank/expert mismatch {tuple(w1.shape)} {tuple(w2.shape)}"
    local_experts = w1.shape[0]
    if weight_format == "fp4" and w1.dtype == torch.int32:
        hidden = w1.shape[1] * PACK4
        inter = w1.shape[2] // 2
        if w2.shape != (local_experts, inter // PACK4, hidden):
            return f"w2 {tuple(w2.shape)} is not [E, {inter // PACK4}, {hidden}]"
    else:
        hidden = w1.shape[1]
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
        if (
            w1.dtype not in (torch.float4_e2m1fn_x2, torch.uint8, torch.int32)
            or w2.dtype != w1.dtype
        ):
            return (
                "the FP4 form requires native float4_e2m1fn_x2, uint8, or int32 weights"
            )
        if not isinstance(rhs_qb, int) or rhs_qb <= 0:
            return "NVFP4 fused EP requires a positive integer block size"
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
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "expert biases are not wired through the fused EP op"
    return _unsupported_routing_reason(
        layer,
        supplies_hash_ids=supplies_hash_ids,
        precomputed_routing=precomputed_routing,
    )


def _routing_config(layer) -> tuple[str, bool, float]:
    scoring_fn = layer.scoring_func
    has_bias = layer.e_score_correction_bias is not None
    scale = float(layer.routed_scaling_factor)
    return scoring_fn, has_bias, scale


def _unsupported_routing_reason(
    layer, *, supplies_hash_ids=False, precomputed_routing=False
) -> str | None:
    if precomputed_routing:
        table = getattr(layer, "hash_indices_table", None)
        if table is not None and not supplies_hash_ids:
            return (
                "this layer routes by hash table and its quant method "
                "does not supply input_ids to the fused op"
            )
        return None
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
