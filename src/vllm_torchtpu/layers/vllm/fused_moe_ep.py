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

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (EP_AXIS_NAME, build_ep_mesh,
                                               ep_mesh_index, ep_rank_order)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.fused_moe.v2.host import (PACK4, U32_SUBLANE_TILE,
                                                     token_gather_smem_bytes)
from vllm_torchtpu.logger import init_logger

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


def _squeeze_channel_scale(scale: torch.Tensor, experts: int,
                           channels: int) -> torch.Tensor | None:
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


def _build_op(mesh,
              topk: int,
              renormalize: bool,
              activation: str,
              mesh_expert_order: tuple[int, ...] | None,
              sharded_plan: bool,
              weight_format="fp8",
              rhs_qb=None):
    """The op for this closure, built once per distinct closure and reused."""
    from vllm_torchtpu.kernels.fused_moe.v2 import (WeightFormat,
                                                    fused_ep_moe_v2)

    key = (topk, renormalize, activation, mesh_expert_order, sharded_plan,
           weight_format, rhs_qb)
    cached = _OPS.get(key)
    if cached is not None:
        return cached

    # Each distinct closure gets its own torch op name: two `CustomOpDef`s
    # registered under one qualname do not raise, they silently share a
    # dispatch entry, and the second closure would then serve the first.
    name = (FUSED_MOE_EP_OP_NAME
            if not _OPS else f"{FUSED_MOE_EP_OP_NAME}_{len(_OPS)}")

    # Every operand is annotated jax.Array: torch_tpu infers static argnums
    # from the signature and refuses an unannotated one.
    def moe(x: jax.Array, w1: jax.Array, w2: jax.Array, w1_scale: jax.Array,
            w2_scale: jax.Array, gating: jax.Array,
            rank: jax.Array) -> jax.Array:
        # The router's top-k selector writes into an f32 accumulator ref, so a
        # bf16 gate output -- which is what the model's gate matmul produces --
        # fails inside the kernel on a dtype mismatch. Widen at the boundary;
        # softmax runs in f32 on this path anyway.
        # A bare array, not a one-tuple: a one-tuple result gives the op the
        # schema `-> ((Tensor))`, and torch guard_ints every symbolic input dim
        # of an op with that schema, which specializes the model's dynamic token
        # count to whichever bucket compiled first.
        return fused_ep_moe_v2(x,
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
                               sharded_plan=sharded_plan)

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
        input_partition_specs=(spec, ) * 7,
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


def fused_moe_ep(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Run one MoE layer through the fused EP kernel.

    Takes this rank's own tokens and returns this rank's own tokens: the
    dispatch and combine happen inside the program, so the caller must not
    all-gather the input or reduce-scatter the output. `owner` is the layer's
    quant method, the same object `fused_moe_ep_supported` was asked about;
    only call this when that said yes.
    """
    op = getattr(owner, FUSED_MOE_EP_OP_ATTR)
    if w1.dtype == torch.float4_e2m1fn_x2:
        # FP4 scales are already [E, K/block, N]. Even a single block must
        # retain its axis; the per-channel helper would incorrectly drop it.
        return op(hidden_states, w1, w2, w1_scale, w2_scale, router_logits,
                  _RANK_BUFFER)
    return op(hidden_states, w1, w2, _squeeze_channel_scale_checked(w1_scale),
              _squeeze_channel_scale_checked(w2_scale), router_logits,
              _RANK_BUFFER)


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
        raise ValueError(
            f"EP rank order must permute range({ep}); got {order}")
    if order == tuple(range(ep)):
        return None
    # A tuple, not a list: `info_once` keys its cache on the arguments.
    logger.info_once(
        "Fused EP MoE: EP ranks are not in device-id order, so the expert "
        "IDs are relabelled after top-k | ep_rank_at_mesh_index=%s", order)
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


def prebuild_fused_moe_ep(layer,
                          topk: int,
                          renormalize: bool,
                          activation: str,
                          *,
                          weight_format="fp8",
                          rhs_qb=None,
                          weights=None) -> Any | None:
    """Resolve the mesh and build the fused-EP op for one layer.

    Called once per layer from `process_weights_after_loading`. Everything here
    is forbidden inside a compiled forward: mapping EP ranks to TPU device ids
    is an `all_gather_object`, and a `Mesh` cannot survive Dynamo's tracing.

    Returns the op to serve this layer with, or None on any refusal. The caller
    records it, which is what makes arming per layer rather than per process --
    several of the refusals below read things that vary by layer.

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
            "expert-parallel, so each shard would hold the whole expert set.")
        return None

    # The former full token-gather scalar-prefetch table made this path look
    # token-count limited under TP and at a 4K/rank DP8 bucket. The table now
    # remains in HBM and is streamed through two fixed SMEM windows, so that
    # compile-time limit is gone. Tensor parallelism is still refused for an
    # independent reason: the kernel deadlocks during warmup, and a replicated
    # batch would also make every rank repeat the full expert work.
    #
    # That hang is NOT specific to this kernel: the fused reduce-scatter MoE
    # kernel, written independently, fails at the same phase with the same
    # signature under TP8, while both work under DP8. Ruled out for it, and
    # so presumably for this one: receive-count mismatch, a ragged last tile,
    # a missing drain, barrier re-entry across layers, tiny buckets, routing
    # skew, collective-id collision, and a Pallas barrier interleaved with
    # XLA all-reduces. The untested candidate is GDN's `ragged_all_to_all`, a
    # data-dependent collective in 45 of this model's 60 layers.
    #
    # There is another thing worth knowing before anyone lifts it: under
    # replicated TP each rank hands the op a full copy of the batch as its
    # shard, so the kernel does ep_size times the expert work for one batch
    # of output. Lifting the refusal usefully needs the bridge to hand each
    # rank a distinct 1/ep slice, which also removes the SMEM inflation.
    # PCP is not replicated TP: every PCP rank already presents a distinct
    # token shard, exactly like DP does for this kernel.  The TPU MoERunner
    # patch suppresses its explicit PCP all-gather/reduce-scatter only for a
    # layer this function actually arms; refused layers keep the old fallback.
    # Consequently the kernel is the sole dispatch/combine owner under PCP,
    # independent of whether the separate GMM chunk pipeline is enabled.
    pcp = _pcp_size(layer)
    tp = _tensor_parallel_size()
    if tp > 1:
        logger.warning_once(
            "Fused EP MoE not engaged: under tensor parallelism (tp_size=%d) "
            "it arms and compiles but then deadlocks during warmup, and with "
            "the batch replicated it would do %dx the expert work per batch "
            "anyway. The grouped-matmul path serves every step.", tp, tp)
        return None

    mesh = build_ep_mesh()
    if mesh is None:
        logger.info_once("Fused EP MoE requested but there is no EP group; "
                         "the grouped-matmul path serves every step.")
        return None
    ep = mesh.shape[EP_AXIS_NAME]

    gather_bytes = token_gather_smem_bytes(_TILE_M)
    needed_smem = _SMEM_OVERHEAD_BYTES + gather_bytes
    smem = _smem_capacity_bytes()
    if needed_smem > smem:
        logger.warning_once(
            "Fused EP MoE not engaged: its fixed routing working set needs "
            "%d B of this chip's %d B SMEM (%d B for two streamed "
            "token-gather windows and about %d B for resident tables and "
            "compiler working memory).", needed_smem, smem, gather_bytes,
            _SMEM_OVERHEAD_BYTES)
        return None
    global _RANK_BUFFER
    # The kernel's index is this rank's position in the mesh, which is ordered
    # by device id, not its EP rank.
    _RANK_BUFFER = torch.tensor([[ep_mesh_index()]],
                                dtype=torch.int32,
                                device=layer.w13_weight.device)
    mesh_expert_order = _mesh_expert_order(ep)

    if weights is None:
        if weight_format == "fp4":
            weights = (layer.w13_weight, layer.w2_weight,
                       layer.w13_weight_scale, layer.w2_weight_scale)
        else:
            weights = (layer.w13_weight, layer.w2_weight,
                       layer.w13_weight_scale_inv, layer.w2_weight_scale_inv)
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
            max_node_tokens, envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS)
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
            "not divide into this rank's %d.", layer.global_num_experts, ep,
            local_experts)
        return None

    reason = fused_moe_ep_unsupported_reason(layer,
                                             weights,
                                             activation,
                                             weight_format=weight_format,
                                             rhs_qb=rhs_qb)
    if reason is not None:
        logger.warning_once("Fused EP MoE not engaged: %s", reason)
        return None

    sharded_plan = envs.MOE_FUSED_EP_V2_SHARDED_PLAN
    if weight_format == "fp8":
        op = _build_op(mesh, topk, renormalize, activation, mesh_expert_order,
                       sharded_plan)
    else:
        op = _build_op(mesh, topk, renormalize, activation, mesh_expert_order,
                       sharded_plan, weight_format, rhs_qb)
    logger.info_once(
        "Fused EP MoE armed | hidden=%d inter=%d local_experts=%d ep=%d "
        "pcp=%d topk=%d capacity=%d sharded_plan=%s format=%s rhs_qb=%s",
        hidden, inter, local_experts, ep, pcp, topk, _TILE_M, sharded_plan,
        weight_format, rhs_qb)
    # Two knobs stop applying the moment this arms, and neither would say so on
    # its own: the fused call returns before `apply_monolithic` reaches either
    # the padding mask or the chunked path. Padding costs expert work and
    # transport rather than correctness -- and it is not free at warmup, where
    # `_dummy_run` marks the whole batch as padding.
    if envs.TPU_MOE_SKIP_PADDED_TOKENS or envs.TPU_MOE_COLLECTION_CHUNK_SIZE:
        logger.warning_once(
            "Fused EP MoE armed, so TPU_MOE_SKIP_PADDED_TOKENS=%s and "
            "TPU_MOE_COLLECTION_CHUNK_SIZE=%s are not in effect for this "
            "layer: the kernel routes and exchanges inside its own program.",
            envs.TPU_MOE_SKIP_PADDED_TOKENS,
            envs.TPU_MOE_COLLECTION_CHUNK_SIZE)
    return op


def fused_moe_ep_unsupported_reason(layer,
                                    weights,
                                    activation,
                                    *,
                                    weight_format="fp8",
                                    rhs_qb=None) -> str | None:
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
            return (f"NVFP4 fused EP block-{rhs_qb} must be a multiple of "
                    f"the packed-weight row tile ({packed_row_tile})")
        if hidden % rhs_qb or inter % rhs_qb:
            return (f"NVFP4 fused EP block-{rhs_qb} must divide both "
                    f"hidden={hidden} and inter={inter}")
        for name, scale, k, n in (("w1_scale", w1_scale, hidden, 2 * inter),
                                  ("w2_scale", w2_scale, inter, hidden)):
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
    if (getattr(layer, "w13_bias", None) is not None
            or getattr(layer, "w2_bias", None) is not None):
        return "expert biases are not wired through the fused EP op"
    return _unsupported_routing_reason(layer)


def _unsupported_routing_reason(layer) -> str | None:
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
    from vllm_torchtpu.layers.vllm import moe_routing

    if layer.custom_routing_function is not None:
        return ("the layer supplies a custom_routing_function; the kernel "
                "routes with its own softmax top-k")
    scoring_fn = layer.scoring_func
    if scoring_fn != "softmax":
        return f"scoring_func {scoring_fn!r}; the kernel scores with softmax"
    if layer.e_score_correction_bias is not None:
        return "e_score_correction_bias is not applied inside the kernel"
    if (layer.use_grouped_topk and layer.num_expert_group > 1
            and layer.topk_group < layer.num_expert_group):
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
        return ("VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set and the kernel "
                "does its own routing, so the simulation would not be in "
                "effect")
    return None


def _tensor_parallel_size() -> int:
    """This deployment's tensor-parallel width, or 1 if it cannot be read.

    Read off the config, not off `get_tp_group()`. The group coordinator is a
    mutable singleton and `spec_decode.utils._force_draft_tp1` rewrites its
    `world_size` to 1 for the duration of the draft model's weight load --
    which is exactly where this runs. On a TP>1 target with an MTP draft, the
    target's layers would refuse and the draft's would arm on a spoofed 1.
    """
    try:
        from vllm.config import get_current_vllm_config
        return int(
            get_current_vllm_config().parallel_config.tensor_parallel_size)
    except Exception:  # noqa: BLE001 - no config outside a served model
        return 1


def _max_node_tokens(ep: int, pcp: int = 1) -> int:
    """The largest node-wide token count a step can carry, or 0 if unknown.

    A DP scheduler cap is per independent batch, so every DP rank contributes
    another cap's worth of tokens.  PCP ranks partition one scheduler batch;
    multiplying by the full flattened EP width would count the same logical
    tokens once per PCP shard.  TP is refused before this helper is called, so
    ``ep // pcp`` is the number of independent DP batches represented here.
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
