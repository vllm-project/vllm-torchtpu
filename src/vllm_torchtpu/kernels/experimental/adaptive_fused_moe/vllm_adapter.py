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
"""Strict Torch/JAX bridge for an explicitly selected experimental kernel.

The FP8/NVFP4 quantization methods select this bridge at weight-load time with
MOE_FUSED_EP_KERNEL_IMPL=adaptive. Callers provide prepared K-major weights.
Native FP4 selects W4A8 automatically; the
experimental path always builds sharded routing and never returns a GMM fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec
from vllm.config import get_current_vllm_config

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    ep_token_replica_groups,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental import adaptive_fused_moe as adaptive_kernel
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import host
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.host import (
    PACK4,
    U32_SUBLANE_TILE,
)
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

PreparedWeights = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

logger = init_logger(__name__)
_TILE_M = 128
_SMEM_OVERHEAD_BYTES = 236 * 1024
FUSED_MOE_EP_OP_NAME = "pallas::adaptive_fused_moe"
FUSED_MOE_EP_OP_ATTR = "_adaptive_fused_moe_op"
_OPS: dict[tuple, torch.library.CustomOpDef] = {}
_RANK_BUFFER: Any = None


def _device_info():
    return pltpu.get_tpu_info()


def _max_node_tokens(ep: int, pcp: int) -> int:
    config = get_current_vllm_config()
    cap = config.scheduler_config.max_num_batched_tokens
    if not isinstance(cap, int) or cap <= 0 or pcp <= 0 or ep % pcp:
        raise ValueError(
            "adaptive fused MoE requires a valid scheduler cap and PCP size"
        )
    return cap * (ep // pcp)


def prebuild_adaptive_fused_moe(
    layer: RoutedExperts,
    topk: int,
    renormalize: bool,
    activation: str,
    *,
    weights: PreparedWeights | None = None,
    rhs_qb: int | None = None,
) -> torch.library.CustomOpDef:
    """Build the selected experimental path, raising on every refusal.

    Weights must already be in the K-major runtime layout. FP4 format and
    contraction block size are inferred from their dtype and scale shapes.
    No W4A8 opt-in or routing-plan environment variable is read here.
    """
    parallel = layer.moe_config.moe_parallel_config
    if not parallel.use_ep:
        raise ValueError("adaptive fused MoE requires expert-parallel weights")
    mesh = build_ep_mesh()
    if mesh is None:
        raise ValueError("adaptive fused MoE requires an EP group")
    ep = mesh.shape[EP_AXIS_NAME]
    pcp = int(parallel.pcp_size)
    node_tokens = _max_node_tokens(ep, pcp)
    if node_tokens < envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS:
        raise ValueError(
            "maximum node token count is below MOE_FUSED_EP_KERNEL_MIN_TOKENS"
        )
    if weights is None:
        w1, w2 = layer.w13_weight, layer.w2_weight
        if w1.dtype == torch.float4_e2m1fn_x2:
            weights = (w1, w2, layer.w13_weight_scale, layer.w2_weight_scale)
        else:
            weights = (w1, w2, layer.w13_weight_scale_inv, layer.w2_weight_scale_inv)
    w1, w2, s1, s2 = weights
    if w1.dtype == torch.float4_e2m1fn_x2:
        weight_format = "fp4"
        if rhs_qb is None:
            if s1.ndim != 4 or s1.shape[1] <= 0 or w1.shape[1] % s1.shape[1]:
                raise ValueError(
                    "cannot infer FP4 block size from prepared scale layout"
                )
            rhs_qb = w1.shape[1] // s1.shape[1]
    elif w1.dtype == torch.float8_e4m3fn:
        weight_format = "fp8"
    else:
        raise ValueError(f"unsupported adaptive fused MoE weight dtype {w1.dtype}")
    reason = fused_moe_ep_unsupported_reason(
        layer, weights, activation, weight_format=weight_format, rhs_qb=rhs_qb
    )
    if reason is not None:
        raise ValueError(f"adaptive fused MoE: {reason}")
    local_experts, hidden = w1.shape[:2]
    inter = w1.shape[2] if weight_format == "fp4" else w1.shape[2] // 2
    if layer.global_num_experts != local_experts * ep:
        raise ValueError("adaptive fused MoE requires equal contiguous expert blocks")
    if not 1 <= topk <= layer.global_num_experts:
        raise ValueError("topk must be between one and the expert count")
    if hidden <= 0 or hidden % host.HIDDEN_LANE_BLOCK or inter < host.HIDDEN_LANE_BLOCK:
        raise ValueError(
            "adaptive fused MoE weight dimensions do not satisfy lane alignment"
        )
    if weight_format == "fp8" and inter % host.HIDDEN_LANE_BLOCK:
        raise ValueError("FP8 intermediate dimension must be lane aligned")
    info = _device_info()
    if host.chip_generation(info) < host.MIN_GENERATION:
        raise ValueError(
            f"adaptive fused MoE requires TPU generation >= {host.MIN_GENERATION}; "
            f"got generation {host.chip_generation(info)}"
        )
    smem_bytes = _SMEM_OVERHEAD_BYTES + host.token_gather_smem_bytes(_TILE_M)
    if smem_bytes > info.smem_capacity_bytes:
        raise ValueError("adaptive fused MoE routing working set exceeds SMEM")
    plan = host.select_weight_buffers(
        local_experts,
        _TILE_M,
        hidden,
        inter,
        weight_format=weight_format,
        rhs_qb=host.QB4 if rhs_qb is None else rhs_qb,
        info=info,
    )
    order = _mesh_expert_order(ep)
    global _RANK_BUFFER
    _RANK_BUFFER = torch.tensor(
        [[ep_mesh_index()]], dtype=torch.int32, device=layer.w13_weight.device
    )
    groups = ep_token_replica_groups(is_sequence_parallel=parallel.is_sequence_parallel)
    op = _build_op(
        mesh,
        topk,
        renormalize,
        activation,
        order,
        weight_format,
        rhs_qb,
        token_replica_groups=groups,
    )
    logger.info_once(
        "Experimental adaptive fused MoE armed | format=%s W1=%d W2=%d "
        "hidden=%d inter=%d local_experts=%d rhs_qb=%s estimated_vmem_bytes=%d",
        weight_format,
        plan.w1_nbuf,
        plan.w2_nbuf,
        hidden,
        inter,
        local_experts,
        rhs_qb,
        plan.vmem_bytes,
    )
    return op


def adaptive_fused_moe_supported(owner: Any) -> bool:
    """Only a successfully prebuilt op owns vLLM's dispatch/combine."""
    return getattr(owner, FUSED_MOE_EP_OP_ATTR, None) is not None


def prebuild_adaptive_nvfp4(
    layer: RoutedExperts,
    activation: str,
    configured_block: int | None = None,
) -> tuple[torch.library.CustomOpDef, int]:
    """Validate checkpoint geometry before committing runtime requantization."""
    block = U32_SUBLANE_TILE * PACK4 if configured_block is None else configured_block
    hidden = layer.w13_weight.shape[-1] * 2
    inter = layer.w2_weight.shape[-1] * 2
    if not isinstance(block, int) or block <= 0 or hidden % block:
        raise ValueError(
            f"adaptive NVFP4 block-{block} must be positive and divide hidden={hidden}"
        )
    inter = host.align_up(inter, block)
    experts = layer.w13_weight.shape[0]
    # Meta tensors describe the prepared packed K-major layout without
    # allocating another copy of the checkpoint. Strict prebuild also checks
    # routing, EP placement, alignment and memory before weights are replaced.
    weights = (
        torch.empty(
            (experts, hidden, inter), dtype=torch.float4_e2m1fn_x2, device="meta"
        ),
        torch.empty(
            (experts, inter, hidden // 2), dtype=torch.float4_e2m1fn_x2, device="meta"
        ),
        torch.empty(
            (experts, hidden // block, 1, 2 * inter), dtype=torch.float32, device="meta"
        ),
        torch.empty(
            (experts, inter // block, 1, hidden), dtype=torch.float32, device="meta"
        ),
    )
    op = prebuild_adaptive_fused_moe(
        layer,
        topk=layer.moe_config.experts_per_token,
        renormalize=layer.renormalize,
        activation=activation,
        weights=weights,
        rhs_qb=block,
    )
    return op, block


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


def fused_moe_ep_unsupported_reason(
    layer: RoutedExperts,
    weights: PreparedWeights,
    activation: str,
    *,
    weight_format: str = "fp8",
    rhs_qb: int | None = None,
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
    if activation != "silu":
        return f"activation {activation!r} is not wired to the kernel's ACT_FNS"
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
    return _unsupported_routing_reason(layer)


def _unsupported_routing_reason(layer: RoutedExperts) -> str | None:
    """Why this layer's ROUTING cannot go through the kernel, or None.

    The fused path never calls `moe_routing.route`: the kernel scores with a
    plain softmax and selects top-k itself, with `renormalize` as its only
    option. Everything else `select_experts` can do -- a different scoring
    function, an expert-score correction bias, expert grouping, hash routing,
    a routed scaling factor, a supplied `custom_routing_function` -- is not
    that, and taking the fused path anyway routes tokens to a DIFFERENT set of
    experts with different gate weights, with nothing raised and nothing
    logged. Each of these is a refusal rather than a silent divergence.
    """
    if layer.custom_routing_function is not None:
        return (
            "the layer supplies a custom_routing_function; the kernel "
            "routes with its own softmax top-k"
        )
    scoring_fn = layer.scoring_func
    if scoring_fn != "softmax":
        return f"scoring_func {scoring_fn!r}; the kernel scores with softmax"
    if layer.e_score_correction_bias is not None:
        return "e_score_correction_bias is not applied inside the kernel"
    if (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    ):
        return "grouped top-k routing is not implemented in the kernel"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing is not implemented in the kernel"
    scale = float(layer.routed_scaling_factor)
    if scale != 1.0:
        return f"routed_scaling_factor {scale} is not applied by the kernel"
    # The routing simulator replaces the routing output wholesale, so a run
    # that asked for it and got real routing is a profiling result that is
    # quietly not the thing it claims to measure.
    if moe_routing._SIMULATION_STRATEGY is not None:
        return (
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set and the kernel "
            "does its own routing, so the simulation would not be in "
            "effect"
        )
    return None


def _build_op(
    mesh: Mesh,
    topk: int,
    renormalize: bool,
    activation: str,
    mesh_expert_order: tuple[int, ...] | None,
    weight_format: str = "fp8",
    rhs_qb: int | None = None,
    *,
    token_replica_groups: tuple[tuple[int, ...], ...] | None = None,
) -> torch.library.CustomOpDef:
    """The op for this closure, built once per distinct closure and reused."""
    key = (
        mesh,
        topk,
        renormalize,
        activation,
        mesh_expert_order,
        weight_format,
        rhs_qb,
        token_replica_groups,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached

    # Each distinct closure gets its own torch op name: two `CustomOpDef`s
    # registered under one qualname do not raise, they silently share a
    # dispatch entry, and the second closure would then serve the first.
    name = FUSED_MOE_EP_OP_NAME if not _OPS else f"{FUSED_MOE_EP_OP_NAME}_{len(_OPS)}"

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
        # The router's top-k selector writes into an f32 accumulator ref, so a
        # bf16 gate output -- which is what the model's gate matmul produces --
        # fails inside the kernel on a dtype mismatch. Widen at the boundary;
        # softmax runs in f32 on this path anyway.
        # A bare array, not a one-tuple: a one-tuple result gives the op the
        # schema `-> ((Tensor))`, and torch guard_ints every symbolic input dim
        # of an op with that schema, which specializes the model's dynamic token
        # count to whichever bucket compiled first.
        return adaptive_kernel.adaptive_fused_moe(
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
            weight_format=host.WeightFormat(weight_format),
            rhs_qb=rhs_qb,
            act_fn=activation,
            mesh_ep_ranks=mesh_expert_order,
            token_replica_groups=token_replica_groups,
        )

    spec = PartitionSpec(EP_AXIS_NAME)
    # Not stock `pallas.jax_op`: it sizes its outputs from the export's avals,
    # where a shard_map result is recorded as replicated, so this rank's 2048
    # rows come back claiming the mesh-wide 16384. `sharded_jax_op` is stock
    # with only that line replaced.
    op = sharded_jax_op(
        name,
        moe,
        mesh=mesh,
        # x, w1, w2, w1_scale, w2_scale, gating, rank -- every operand is
        # sharded on axis 0 over the EP axis, and so is the single output.
        input_partition_specs=(spec,) * 7,
        output_partition_specs=spec,
    )

    # Deliberately replaces the shard-aware fake `sharded_jax_op` installed,
    # which resolves the output aval by running a real `jax.export` every time
    # Dynamo traces the op. This kernel returns the rows it was given, so the
    # answer is known without exporting for it.
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def run_adaptive_fused_moe(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Execute a successfully prebuilt experimental op, or raise."""
    op = getattr(owner, FUSED_MOE_EP_OP_ATTR, None)
    if op is None:
        raise RuntimeError("adaptive fused MoE was not prebuilt for this layer")
    if w1.dtype == torch.float4_e2m1fn_x2:
        # FP4 scales are already [E, K/block, N]. Even a single block must
        # retain its axis; the per-channel helper would incorrectly drop it.
        def block_scale(scale):
            if scale.ndim == 4 and scale.shape[2] == 1:
                return scale.squeeze(2)
            if scale.ndim != 3:
                raise ValueError("FP4 runtime scales require a contraction-block axis")
            return scale

        return op(
            hidden_states,
            w1,
            w2,
            block_scale(w1_scale),
            block_scale(w2_scale),
            router_logits,
            _RANK_BUFFER,
        )
    return op(
        hidden_states,
        w1,
        w2,
        _squeeze_channel_scale_checked(w1_scale),
        _squeeze_channel_scale_checked(w2_scale),
        router_logits,
        _RANK_BUFFER,
    )
