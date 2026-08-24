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
scheduler's largest batch, whether to build this operation for the deployment.
When it is built, ``TPU_MOE_FUSED_EP_STEP_MIN_TOKENS`` can keep smaller compiled
token shapes on the gather/grouped-matmul/combine path while larger shapes use
this fully fused path. Its default of zero preserves fused EP for every shape.
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
from vllm_torchtpu.kernels.fused_moe.v2.host import ragged_stride_bound
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The tile height the kernel slabs rows into, and the unit its ragged stride
# rounds to. Matches the value tpu-inference's serving adapter asks for.
_TILE_M = 128

# SMEM the kernel's prefetched routing tables have to fit in, in bytes. The
# tables are i32 and `ragged_stride_bound` long; see the gate in
# `prebuild_fused_moe_ep` for why this is the real limit on the token count.
_ROUTING_TABLE_SMEM_BUDGET = 1024 * 1024

# One op for every shape: `fused_ep_moe_v2` derives its routing block and ragged
# stride from the operands when they are not given, so nothing here has to vary
# with the token count -- which is what keeps the token count out of the traced
# graph. The name is also what `TpuPlatform` puts in `splitting_ops`.
FUSED_MOE_EP_OP_NAME = "pallas::fused_moe_ep"

# Ops are registered at weight-load time, keyed by this rank's token count.
# The compiled forward may only look them up: a `jax.sharding.Mesh` holds a
# numpy object array of TpuDevice, and Dynamo tries to turn anything it sees
# into a tensor, so the mesh must never appear in a traced frame.
_OP: Any = None
_EP_SIZE: int | None = None
# This rank's mesh index, as a device tensor. The kernel needs it as data;
# see the note in fused_ep_moe_v2 on why a value computed inside the traced
# function turns into a `partition-id` XLA will not accept.
_RANK_BUFFER: Any = None
# Which EP rank's expert block sits at each mesh index, or None for the
# identity. See `_to_mesh_expert_order`.
_EXPERT_ORDER: Any = None


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


def fused_moe_ep_supported() -> bool:
    """Whether `prebuild_fused_moe_ep` armed the fused path for this layer.

    Shape selection belongs to the FP8 method, which owns the EP exchange and
    selects fused EP or gather/GMM/combine for each compiled static shape.
    This helper reports only whether the fused operation is available.
    """
    return _OP is not None


def _build_op(mesh, topk: int, renormalize: bool, activation: str):
    from vllm_torchtpu.kernels.fused_moe.v2 import (WeightFormat,
                                                    fused_ep_moe_v2)

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
                               weight_format=WeightFormat.FP8,
                               act_fn=activation)

    spec = PartitionSpec(EP_AXIS_NAME)
    # Not stock `pallas.jax_op`: it sizes its outputs from the export's avals,
    # where a shard_map result is recorded as replicated, so this rank's 2048
    # rows come back claiming the mesh-wide 16384. `sharded_jax_op` is stock
    # with only that line replaced.
    op = sharded_jax_op(
        FUSED_MOE_EP_OP_NAME,
        moe,
        mesh=mesh,
        # x, w1, w2, w1_scale, w2_scale, gating, rank -- every operand is
        # sharded on axis 0 over the EP axis, and so is the single output.
        input_partition_specs=(spec, ) * 7,
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    global _OP
    _OP = op
    return op


def fused_moe_ep(
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
    all-gather the input or reduce-scatter the output. Only call it when
    `fused_moe_ep_supported` said yes.
    """
    return _OP(hidden_states, w1, w2, _squeeze_channel_scale_checked(w1_scale),
               _squeeze_channel_scale_checked(w2_scale),
               _to_mesh_expert_order(router_logits), _RANK_BUFFER)


def retain_fused_ep_graph_inputs(output: torch.Tensor) -> torch.Tensor:
    """Keep fused-EP rank buffers in the GMM shape-specialized graph.

    The subtraction is identically zero and XLA removes it, but its dataflow
    keeps the runtime rank buffers in the FX subgraph's lifted input list.
    """
    marker = None
    for graph_input in (_RANK_BUFFER, _EXPERT_ORDER):
        if graph_input is None:
            continue
        value = graph_input.reshape(-1)[0].to(output.dtype)
        zero = value - value
        marker = zero if marker is None else marker + zero
    return output if marker is None else output + marker


def _to_mesh_expert_order(router_logits: torch.Tensor) -> torch.Tensor:
    """Reindex the expert axis from EP-rank order to mesh order.

    The kernel reads a token's expert id as `mesh_index * experts_per_shard +
    j`, but vLLM numbers experts by EP rank, and the mesh is ordered by device
    id (see `ep_mesh.ep_rank_order`). Where those two orders differ, the block
    of logits for EP rank r has to move to the position of the mesh index that
    rank sits at. The output is per token, so nothing has to be undone
    afterwards. `_EXPERT_ORDER` is None when the orders already agree, which is
    every deployment whose EP ranks come up in device-id order.
    """
    if _EXPERT_ORDER is None:
        return router_logits
    # Split and rejoin the expert axis without naming the token axis: reading
    # `shape[0]` here would put the token count in the traced graph.
    blocks = router_logits.unflatten(-1, (_EP_SIZE, -1))
    return blocks.index_select(-2, _EXPERT_ORDER).flatten(-2)


def _expert_block_permutation(ep: int, device) -> torch.Tensor | None:
    """Which EP rank's expert block belongs at each mesh index, or None.

    None means the identity, so the hot path can skip the gather entirely.
    """
    order = ep_rank_order()
    if order is None or tuple(order) == tuple(range(ep)):
        return None
    # A tuple, not a list: `info_once` keys its cache on the arguments.
    logger.info_once(
        "Fused EP MoE: EP ranks are not in device-id order, so the expert "
        "axis is reindexed to mesh order | ep_rank_at_mesh_index=%s", order)
    return torch.tensor(order, dtype=torch.int32, device=device)


def _squeeze_channel_scale_checked(scale: torch.Tensor) -> torch.Tensor:
    """`_squeeze_channel_scale` without the shape arguments, for the hot path.

    Prebuild already proved the layout converts, so this only has to do it.
    """
    s = scale
    while s.ndim > 2:
        s = s.squeeze(1)
    return s


def prebuild_fused_moe_ep(layer, topk: int, renormalize: bool,
                          activation: str) -> None:
    """Resolve the mesh and register a fused-EP op per compiled token bucket.

    Called once per layer from `process_weights_after_loading`. Everything here
    is forbidden inside a compiled forward: mapping EP ranks to TPU device ids
    is an `all_gather_object`, and a `Mesh` cannot survive Dynamo's tracing.
    """
    global _EP_SIZE
    if not envs.USE_MOE_FUSED_EP_KERNEL:
        return
    # Two separate things stop this kernel serving under tensor parallelism,
    # and they were conflated for a long time. Both are gated here.
    #
    # FIRST, a compile-time SMEM bound, driven by the TOKEN COUNT rather than
    # by TP.
    #
    # The kernel prefetches its routing tables into SMEM at the head of the
    # program, and they are sized by `ragged_stride_bound(T, topk, e_total,
    # capacity)` i32 entries against a 1 MiB budget. For Qwen3.5-397B
    # (topk 10, 512 experts, capacity 128) that crosses 1 MiB between
    # T = 25,843 and T = 32,768: `ragged_stride_bound(32768, ...) * 4` is
    # 1,325,568, which is exactly the size the compiler reported when every
    # worker died. So the failure is a token-count crossing.
    #
    # TP only LOOKED causal because under plain tensor parallelism the batch
    # is replicated, so each rank hands the op a full copy as its shard and
    # the kernel's global T becomes node_tokens * tp_size. At the benchmark's
    # 16,384-token chunk that is 131,072 tokens and 5.25 MB of tables -- five
    # times over -- while the same chunk under DP8 is 16,384 tokens and fits.
    # "tp > 1" and "T too large" coincided at the batch we happened to run.
    #
    # The bound is gated below, once the mesh gives us the EP width. It is
    # not TP-specific -- a DP deployment above ~3,200 tokens per engine
    # crosses it too, and used to die the same way.
    #
    # SECOND, and independently, the kernel DEADLOCKS under tensor
    # parallelism. With the SMEM bound respected (chunk 2048, so 16,384
    # global tokens and 670 KB of tables) it now arms, compiles every bucket
    # and enters the warmup ladder -- and then every worker hangs inside
    # `_dummy_run` and is killed by a libtpu telemetry abort. Measured
    # 2026-08-20 on TP8; chunk 3072 does not even compile ("Program smem
    # requirement 1.11M", the tables plus ~115 KB of other prefetched
    # operands).
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
    # So TP is still refused, but now for the reason that actually holds.
    # There is a third thing worth knowing before anyone lifts it: under
    # replicated TP each rank hands the op a full copy of the batch as its
    # shard, so the kernel does ep_size times the expert work for one batch
    # of output. Lifting the refusal usefully needs the bridge to hand each
    # rank a distinct 1/ep slice, which also removes the SMEM inflation.
    # Prefill context parallelism has to be refused for a different reason,
    # and it is the more dangerous of the two because it is silent.
    # `do_naive_dispatch_combine` is not what runs the PCP exchange:
    # `MoERunner._maybe_dispatch`/`_maybe_combine` bracket the layer with a
    # PCP all-gather and reduce-scatter whenever `pcp_size > 1 and not
    # use_all2all_kernels`, and this tree's own
    # `_patch_moe_explicit_pcp_collectives` guarantees that condition for
    # `use_ep and pcp_size > 1 and dp_size == 1`. At pcp>1/dp1/tp1 the TP
    # refusal below does not fire -- `_tensor_parallel_size()` is 1 -- so the
    # kernel would arm, return each rank's tokens already combined over every
    # expert, and then have that reduce-scattered against seven identical
    # copies: every routed MoE output multiplied by pcp_size, with nothing
    # raised and nothing logged.
    pcp = _pcp_size(layer)
    if pcp > 1:
        logger.warning_once(
            "Fused EP MoE not engaged: prefill context parallelism "
            "(pcp_size=%d) keeps vLLM's own all-gather/reduce-scatter around "
            "the layer, and this kernel already combines over every expert, "
            "so arming both would multiply the MoE output by %d.", pcp, pcp)
        return

    tp = _tensor_parallel_size()
    if tp > 1:
        logger.warning_once(
            "Fused EP MoE not engaged: under tensor parallelism (tp_size=%d) "
            "it arms and compiles but then deadlocks during warmup, and with "
            "the batch replicated it would do %dx the expert work per batch "
            "anyway. The grouped-matmul path serves every step.", tp, tp)
        return

    mesh = build_ep_mesh()
    if mesh is None:
        logger.info_once("Fused EP MoE requested but there is no EP group; "
                         "the grouped-matmul path serves every step.")
        return
    ep = mesh.shape[EP_AXIS_NAME]
    _EP_SIZE = ep

    # `_max_node_tokens(ep)` is already the GLOBAL token count this kernel
    # sees: under DP it is the per-engine cap times the engines, and under
    # replicated TP it is the per-rank batch times the ranks, because every
    # rank hands the op a full copy as its shard.
    global_tokens = _max_node_tokens(ep)
    if global_tokens:
        table_bytes = ragged_stride_bound(
            global_tokens, topk, layer.global_num_experts, _TILE_M) * 4
        if table_bytes > _ROUTING_TABLE_SMEM_BUDGET:
            logger.warning_once(
                "Fused EP MoE not engaged: at %d global tokens its routing "
                "tables need %d B of SMEM, over the %d B budget. Cap "
                "--max-num-batched-tokens at %d, or shard the token axis so "
                "the kernel sees fewer tokens.", global_tokens, table_bytes,
                _ROUTING_TABLE_SMEM_BUDGET,
                _max_node_tokens_fitting_smem(topk, layer.global_num_experts,
                                              ep))
            return
    global _RANK_BUFFER, _EXPERT_ORDER
    # The kernel's index is this rank's position in the mesh, which is ordered
    # by device id, not its EP rank.
    _RANK_BUFFER = torch.tensor([[ep_mesh_index()]],
                                dtype=torch.int32,
                                device=layer.w13_weight.device)
    _EXPERT_ORDER = _expert_block_permutation(ep, layer.w13_weight.device)

    w1, w2 = layer.w13_weight, layer.w2_weight
    w1_scale = layer.w13_weight_scale_inv
    w2_scale = layer.w2_weight_scale_inv
    local_experts, hidden = w1.shape[0], w1.shape[1]
    inter = w1.shape[2] // 2

    # This load-time threshold decides whether the fused operation is worth
    # building at all. TPU_MOE_FUSED_EP_STEP_MIN_TOKENS independently keeps
    # smaller compiled shapes on gather/GMM/combine at execution time.
    max_node_tokens = _max_node_tokens(ep)
    if max_node_tokens < envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS:
        logger.info_once(
            "Fused EP MoE not engaged: the largest node-wide batch is %d "
            "tokens, below MOE_FUSED_EP_KERNEL_MIN_TOKENS=%d.",
            max_node_tokens, envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS)
        return

    reason = _unsupported_reason(w1, w2, w1_scale, w2_scale, local_experts,
                                 hidden, inter, activation)
    if reason is not None:
        logger.warning_once("Fused EP MoE not engaged: %s", reason)
        return

    _build_op(mesh, topk, renormalize, activation)
    logger.info_once(
        "Fused EP MoE armed | hidden=%d inter=%d local_experts=%d ep=%d "
        "topk=%d capacity=%d", hidden, inter, local_experts, ep, topk, _TILE_M)


def _unsupported_reason(w1, w2, w1_scale, w2_scale, local_experts, hidden,
                        inter, activation) -> str | None:
    """Why this layer cannot use the fused kernel, or None if it can."""
    if w1.ndim != 3 or w2.ndim != 3 or w2.shape[0] != local_experts:
        return f"weight rank/expert mismatch {tuple(w1.shape)} {tuple(w2.shape)}"
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return (f"the kernel's fp8 form takes float8_e4m3fn expert weights, "
                f"got {w1.dtype} / {w2.dtype}")
    if w2.shape[1] != inter or w2.shape[2] != hidden:
        return f"w2 {tuple(w2.shape)} is not [E, {inter}, {hidden}]"
    if activation != "silu":
        return f"activation {activation!r} is not wired to the kernel's ACT_FNS"
    if _squeeze_channel_scale(w1_scale, local_experts, 2 * inter) is None:
        return (
            "w1_scale is not the per-output-channel form the kernel takes; "
            "a block-scaled weight is a different kernel format")
    if _squeeze_channel_scale(w2_scale, local_experts, hidden) is None:
        return "w2_scale is not the per-output-channel form the kernel takes"
    return None


def _tensor_parallel_size() -> int:
    """This deployment's tensor-parallel width, or 1 if it cannot be read."""
    try:
        from vllm.distributed.parallel_state import get_tp_group
        return int(get_tp_group().world_size)
    except Exception:  # noqa: BLE001 - no group outside a served model
        return 1


def _max_node_tokens(ep: int) -> int:
    """The largest node-wide token count a step can carry, or 0 if unknown."""
    from vllm.config import get_current_vllm_config

    try:
        cfg = get_current_vllm_config()
    except Exception:  # noqa: BLE001 - no config outside a served model
        return 0
    cap = getattr(cfg.scheduler_config, "max_num_batched_tokens", None)
    return int(cap) * ep if isinstance(cap, int) else 0


def _max_node_tokens_fitting_smem(topk: int, e_total: int, ep: int) -> int:
    """Largest per-step node-wide token count whose routing tables fit SMEM.

    Reported in the refusal so the operator is told the number to set rather
    than left to bisect --max-num-batched-tokens by hand. Divided by the
    tensor-parallel width because that is the factor by which a replicated
    batch inflates the global token count the kernel sees.
    """
    lo, hi = 1, 1 << 20
    while lo < hi:
        mid = (lo + hi + 1) // 2
        fits = ragged_stride_bound(mid, topk, e_total,
                                   _TILE_M) * 4 <= _ROUTING_TABLE_SMEM_BUDGET
        lo, hi = (mid, hi) if fits else (lo, mid - 1)
    return lo // max(ep, 1)


def _pcp_size(layer) -> int:
    """This layer's prefill-context-parallel width, or 1 if it cannot be read.

    Read off the layer rather than the global config because that is where
    `fp8.py` already reads `use_ep` from, and a layer built for a different
    parallel shape should answer for itself.
    """
    cfg = getattr(getattr(layer, "moe_config", None), "moe_parallel_config",
                  None)
    try:
        return int(getattr(cfg, "pcp_size", 1) or 1)
    except (TypeError, ValueError):
        return 1
