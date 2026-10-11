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
# ruff: noqa: E501
"""The v3 fused expert-parallel MoE kernel behind vLLM's MoE layer: the torch bridge and the kernel binding.

The default MoE path pays the expert-parallel exchange as two separate
collectives around a per-rank compute chain: vLLM all-gathers the tokens, this
rank routes them, permutes, runs two grouped matmuls, combines, and then vLLM
reduce-scatters the result, with that reduce-scatter fully exposed while the
all-gather hides behind the compute.

This path hands the whole layer to one SPMD program instead. The kernel selects
its own top-k, pushes each routed row straight to the rank that owns its
expert, runs the expert matmuls, and pushes results back, so the transport
overlaps the matmuls rather than bracketing them. It replaces, per layer: the
router's full sort, both routed-row argsorts, the SparseCore permute, both
grouped matmuls, the combine, and both collectives.

``MOE_FUSED_EP_KERNEL_MIN_TOKENS`` decides once, at weight load, against the
scheduler's largest batch, whether to build this operation for the deployment.
When it is built, ``TPU_MOE_FUSED_EP_STEP_MIN_TOKENS`` can keep smaller compiled
token shapes on the few-token kernel or the grouped-matmul path while larger
shapes use this fully fused path. Its default of zero preserves fused EP for every shape.

The kernel binding. `kernels/fused_moe/v3` is the kernel's own package, whose
entry point is `fused_ep_moe`. `_build_op` below gives the op a `run_moe`
over global arrays sharded on the EP axis: x [tokens, hidden], w1 [E, hidden,
2 * inter], w2 [E, inter, hidden], the per-output-channel fp8 scales [E, N],
the router logits [tokens, E], this rank's mesh index and the per-rank
valid-token prefix, returning one [tokens, hidden] output sharded like x. The
scale squeeze and PCP width helpers come from `fused_moe_ep`. The binding
uses the kernel's `top_k`, `tile_rows`, and `activation` arguments, with
these export and padding constraints:
- the rank tensor is handed to the kernel as its shard-index operand,
  because `lax.axis_index` lowers to a partition-id the export path refuses;
- there is no valid-token operand: a padded row is routed nowhere by handing
  the kernel a NaN logit row (`drop_nan_rows`), and its input row is zeroed
  so nothing of it travels; the kernel returns zero for such a row. A rank
  with no valid rows still takes part in every collective.

PyTorch/XLA custom-op adapter bridging vLLM FusedMoE layers to the Pallas Fused EP v3 pipeline.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe
from vllm_torchtpu.kernels.fused_moe.v3 import layer as _latest_layer
from vllm_torchtpu.kernels.fused_moe.v3 import layout as _latest_layout
from vllm_torchtpu.layers.adapter.fused_moe_ep import (
    _squeeze_channel_scale,
    _squeeze_channel_scale_checked,
    _unsupported_routing_reason,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The FFN tile height (the kernel's `tile_rows`); smaller tiles are far
# slower with the kernel's DMA strategy, so it is not a knob.
TILE_ROWS = 128
WEIGHT_FORMAT = "fp8"


def mask_local_rows(x: jax.Array, gating: jax.Array, valid_tokens: jax.Array):
    """One rank's rows: row r is valid when r < valid_tokens[0]. An invalid
    row gets NaN logits, which the kernel's selector drops (`drop_nan_rows`),
    and a zero input row."""
    local = jnp.arange(x.shape[0], dtype=jnp.int32)
    valid = (local < valid_tokens.astype(jnp.int32)[0])[:, None]
    gating = jnp.where(valid, gating, jnp.nan)
    x = jnp.where(valid, x, jnp.zeros_like(x))
    return x, gating


def mask_padded_rows(
    x: jax.Array, gating: jax.Array, valid_tokens: jax.Array, ep: int, mesh=None
):
    """Rows past each rank's valid prefix route nowhere and carry nothing.

    `x` and `gating` are the global arrays, `ep * t_local` rows in rank
    order; `valid_tokens` is int32 [ep]. Row r of rank r // t_local is valid
    when r % t_local < valid_tokens[rank]. An invalid row gets NaN logits,
    which the kernel's selector drops (`drop_nan_rows`), and a zero input row.

    With `mesh`, each rank masks its own rows inside a shard_map. The global
    form below positions rows with an iota over the whole token axis, which
    XLA's partitioner rewrites as partition-id * t_local + local row, and the
    torch bridge compiles the exported program in a mode that refuses the
    partition-id instruction ("partitionId instruction is not supported for
    SPMD partitioning"). The local form has no global position to compute.
    """
    if mesh is not None:
        spec = PartitionSpec(MESH_AXIS)
        return jax.shard_map(
            mask_local_rows,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=(spec, spec),
        )(x, gating, valid_tokens)
    tokens = x.shape[0]
    t_local = tokens // ep
    local = jnp.arange(tokens, dtype=jnp.int32) % t_local
    limit = jnp.repeat(valid_tokens.astype(jnp.int32), t_local)
    valid = (local < limit)[:, None]
    gating = jnp.where(valid, gating, jnp.nan)
    x = jnp.where(valid, x, jnp.zeros_like(x))
    return x, gating


def served_config(
    *, hidden: int, inter: int, experts_per_shard: int, top_k: int
) -> Config:
    """The kernel's own settings for this shape (Config.for_shape), logged
    once so every precision and structural choice is on record."""
    config = Config.for_shape(
        hidden=hidden, inter=inter, experts_per_shard=experts_per_shard, top_k=top_k
    )
    logger.info_once(
        "Fused EP MoE (latest kernel) settings for hidden=%d inter=%d "
        "experts_per_shard=%d top_k=%d: tile_rows=%d weight_format=%s "
        "token_rows=%s intermediate=%s result_rows=%s elementwise=%s "
        "combine=%s row_scale_dtype=%s sorted_rows=%s "
        "stream_block=%s token_block=%s weight_slots=%d weight_prefetch=%d "
        "result_slots=%d row_fetch_ahead=%d vmem_fraction=%s",
        hidden,
        inter,
        experts_per_shard,
        top_k,
        TILE_ROWS,
        WEIGHT_FORMAT,
        config.token_rows,
        config.intermediate,
        config.result_rows,
        config.elementwise_dtype,
        config.combine_dtype,
        config.row_scale_dtype,
        config.sorted_rows,
        config.stream_block,
        config.token_block,
        config.weight_slots,
        config.weight_prefetch,
        config.result_slots,
        config.row_fetch_ahead,
        config.vmem_fraction,
    )
    return config


def run_moe_v3(
    mesh,
    topk: int,
    renormalize: bool,
    activation: str,
    *,
    hidden: int,
    inter: int,
    experts_per_shard: int,
):
    """The `run_moe` the adapter's op wraps, on the latest kernel."""
    # The torch bridge exports the whole op as one jitted function; the
    # layer's own inner jit is what the export path's SPMD partitioner
    # refuses. The same partitioner wants a sharding on every side-effecting
    # custom call (these programs' outputs carry their dependencies), and it
    # refuses `lax.axis_index` as a partition-id, so the shard's index goes in
    # as data (the rank tensor). Set here, on the kernel package, for every
    # program this binding builds.
    _latest_layer.WRAP_IN_JIT = False
    _latest_layout.SIDE_EFFECTS = False
    _latest_layout.SHARD_INDEX_OPERAND = True
    if MESH_AXIS not in mesh.axis_names:
        raise ValueError(
            f"the latest kernel splits the experts over the mesh axis named "
            f"{MESH_AXIS!r}; the adapter's mesh has {mesh.axis_names}"
        )
    ep = mesh.shape[MESH_AXIS]  # noqa: F841
    config = served_config(
        hidden=hidden, inter=inter, experts_per_shard=experts_per_shard, top_k=topk
    )

    def run_moe(x, w1, w2, w1_scale, w2_scale, gating, rank, valid_tokens):
        # rank: int32 [ep, 1], each shard's mesh index, the adapter's device
        # tensor; it becomes the kernel's shard-index operand.
        drop_nan_rows = valid_tokens is not None
        return fused_ep_moe(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            top_k=topk,
            renormalize=renormalize,
            mesh=mesh,
            tile_rows=TILE_ROWS,
            weight_format=WEIGHT_FORMAT,
            activation=activation,
            drop_nan_rows=drop_nan_rows,
            config=config,
            shard_index=rank,
            experts_per_shard=experts_per_shard,
            valid_tokens=valid_tokens,
        )

    return run_moe


# One op for every shape: the v3 kernel derives its routing block from the operands when they are not given, so nothing here has to vary
# with the token count -- which is what keeps the token count out of the traced
# graph. The name is also what `TpuPlatform` puts in `splitting_ops`.
FUSED_MOE_EP_OP_NAME = "pallas::fused_moe_ep"

# One op for the process, built at weight-load time. The compiled forward may
# only look it up: a `jax.sharding.Mesh` holds a
# numpy object array of TpuDevice, and Dynamo tries to turn anything it sees
# into a tensor, so the mesh must never appear in a traced frame.
_OP: Any = None
# The shape and routing the op was built for; a layer whose own differ is
# not eligible (`fused_moe_ep_supported`).
_OP_CONTRACT: tuple | None = None
_EP_SIZE: int | None = None
_SKIP_PADDED_TOKENS = False
# This rank's mesh index, as a device tensor. The kernel needs it as data;
# see the note in the v3 kernel on why a value computed inside the traced
# function turns into a `partition-id` XLA will not accept.
_RANK_BUFFER: Any = None
# Which EP rank's expert block sits at each mesh index, or None for the
# identity. See `_to_mesh_expert_order`.
_EXPERT_ORDER: Any = None


def fused_moe_ep_supported(owner: Any = None) -> bool:
    """Whether `prebuild_fused_moe_ep` armed the fused path.

    Shape selection belongs to the FP8 method, which owns the EP exchange and
    selects the fused kernel, the few-token kernel or the grouped-matmul path
    for each compiled static shape.
    With no `owner` this reports whether the operation exists at all. `owner`
    is a MoE layer or its quant method: `prebuild_fused_moe_ep` marks both
    eligible when the layer's shape and routing match the operation, which is
    built once for the process from the first eligible layer, so a layer that
    was refused, or whose shape differs, is answered False and takes the
    grouped-matmul path.
    """
    if _OP is None:
        return False
    if owner is None:
        return True
    return bool(getattr(owner, "_tpu_fused_ep_eligible", False))


def _build_op(
    mesh,
    topk: int,
    renormalize: bool,
    activation: str,
    skip_padded_tokens: bool,
    *,
    hidden: int,
    inter: int,
    local_experts: int,
):
    run_moe = run_moe_v3(
        mesh,
        topk,
        renormalize,
        activation,
        hidden=hidden,
        inter=inter,
        experts_per_shard=local_experts,
    )

    # Add the valid-prefix operand only when padding is skipped.
    if skip_padded_tokens:

        def moe(
            x: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w1_scale: jax.Array,
            w2_scale: jax.Array,
            gating: jax.Array,
            valid_tokens: jax.Array,
            rank: jax.Array,
        ) -> jax.Array:
            return run_moe(x, w1, w2, w1_scale, w2_scale, gating, rank, valid_tokens)

        input_specs = (PartitionSpec(EP_AXIS_NAME),) * 8
    else:

        def moe(
            x: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w1_scale: jax.Array,
            w2_scale: jax.Array,
            gating: jax.Array,
            rank: jax.Array,
        ) -> jax.Array:
            return run_moe(x, w1, w2, w1_scale, w2_scale, gating, rank, None)

        input_specs = (PartitionSpec(EP_AXIS_NAME),) * 7

    spec = PartitionSpec(EP_AXIS_NAME)
    # Use shard-local output shapes: the stock bridge sizes outputs from
    # export avals that record shard_map results as replicated, which can
    # assign the mesh-wide row count to each rank's output.
    op = sharded_jax_op(
        FUSED_MOE_EP_OP_NAME,
        moe,
        mesh=mesh,
        # x, w1, w2, w1_scale, w2_scale, gating, rank -- every operand is
        # sharded on axis 0 over the EP axis, and so is the single output.
        input_partition_specs=input_specs,
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    global _OP, _SKIP_PADDED_TOKENS
    _OP = op
    _SKIP_PADDED_TOKENS = skip_padded_tokens
    return op


def _step_min_tokens() -> int:
    return int(envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS or 0)


def maybe_fused_moe_ep(
    layer,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int | None = None,
    renormalize: bool = True,
    activation: str = "silu",
    sp_input: bool = False,
) -> torch.Tensor | None:
    if not fused_moe_ep_supported(layer):
        return None
    num_tokens = int(x.shape[0])
    from vllm_torchtpu.kernels.fused_moe.v3 import (
        layer as v3_layer,
    )
    from vllm_torchtpu.kernels.fused_moe.v3 import (
        transport as v3_transport,
    )

    ep_size = _EP_SIZE if _EP_SIZE is not None else 8
    row_multiple = v3_layer.ROW_BLOCK * v3_transport.token_multiple(ep_size)
    tokens_per_rank = (
        num_tokens
        if sp_input
        else (num_tokens // ep_size if num_tokens % ep_size == 0 else -1)
    )
    if tokens_per_rank < _step_min_tokens() or tokens_per_rank % row_multiple != 0:
        return None
    w1 = layer.w13_weight
    w2 = layer.w2_weight
    w1_scale = layer.w13_weight_scale_inv
    w2_scale = layer.w2_weight_scale_inv
    return fused_moe_ep(x, w1, w2, w1_scale, w2_scale, router_logits)


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
    args = (
        hidden_states,
        w1,
        w2,
        _squeeze_channel_scale_checked(w1_scale),
        _squeeze_channel_scale_checked(w2_scale),
        _to_mesh_expert_order(router_logits),
    )
    if _SKIP_PADDED_TOKENS:
        from vllm_torchtpu.layers.adapter import token_padding

        valid_tokens = token_padding.get_local_valid_tokens()
        if valid_tokens is None:
            raise RuntimeError(
                "fused EP padding skipping is enabled but the "
                "runner did not install padding state"
            )
        return _OP(*args, valid_tokens, _RANK_BUFFER)
    return _OP(*args, _RANK_BUFFER)


def retain_fused_ep_graph_inputs(output: torch.Tensor) -> torch.Tensor:
    """Keep fused-EP buffers stable across shape-specific graphs."""
    graph_inputs = [_RANK_BUFFER, _EXPERT_ORDER]
    if _SKIP_PADDED_TOKENS:
        from vllm_torchtpu.layers.adapter import token_padding

        valid_tokens = token_padding.get_local_valid_tokens()
        if valid_tokens is None:
            raise RuntimeError(
                "fused EP padding skipping is enabled but the "
                "runner did not install padding state"
            )
        graph_inputs.append(valid_tokens)

    marker = None
    for graph_input in graph_inputs:
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
    if sorted(order) != list(range(ep)):
        raise ValueError(f"EP rank order {order} is not a permutation of range({ep})")
    # A tuple, not a list: `info_once` keys its cache on the arguments.
    logger.info_once(
        "Fused EP MoE: EP ranks are not in device-id order, so the expert "
        "axis is reindexed to mesh order | ep_rank_at_mesh_index=%s",
        order,
    )
    return torch.tensor(order, dtype=torch.int32, device=device)


def check_kernel_threshold(ep: int, tp: int) -> None:
    """The fused EP kernel and the few-token kernel meet at one row count, so
    no step falls to the grouped-matmul path between them. Row counts are per
    MoE call across the node: the few-token cap is in those units already (a
    step's rows replicated under tensor parallelism, an engine's rows gathered
    from every engine under data parallelism). With the few-token kernel off,
    the grouped-matmul path is the kernel below the fused one by choice."""
    if not envs.TPU_MOE_LOWTOK:
        return
    cap = int(envs.TPU_MOE_LOWTOK_MAX_TOKENS or 0)
    if cap <= 0:
        return
    gate = int(envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS or 0)
    if tp > 1:
        from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

        start = _sp.min_rows()
        # below the sequence-parallel start every rank holds the step's rows;
        # from it each rank holds at least ceil(start / tp) of them.
        # Token buckets double at power-of-2 boundaries (e.g. 256 -> 512 where
        # first_sp_bucket = 1 << (start - 1).bit_length() = 512 -> 64/rank).
        # When cap < start // 2 (e.g. cap=64 for N<=64 MTP draft steps while
        # N=96..128 target steps intentionally run fused_moe_gmm + SINGLE_SORT
        # before SP + fused_moe_ep kicks in at first_sp_bucket), allow the
        # 3-tier split as long as first_sp_bucket // tp >= gate.
        first_sp_bucket = 1 << (max(start, 1) - 1).bit_length()
        meets = gate <= -(-first_sp_bucket // tp)
        if cap * 2 >= start:
            rule = (
                f"the few-token kernel up to {cap} rows, sequence parallelism "
                f"and the fused EP kernel from {start} rows ({gate} per rank)"
            )
        else:
            rule = (
                f"the few-token kernel up to {cap} rows, routed grouped-matmul "
                f"in ({cap}, {start}), and sequence parallelism + fused EP "
                f"from {start} rows ({gate} per rank)"
            )
    else:
        # an engine's rows below the gate are gathered from all ep engines
        meets = cap >= ep * max(gate - 1, 0)
        rule = (
            f"the few-token kernel up to {cap} rows across the node, the "
            f"fused EP kernel from {gate} rows per engine "
            f"({ep * gate} across the node)"
        )
    if not meets:
        raise ValueError(
            "The MoE kernel settings leave steps between the two kernels ("
            f"{rule}). Set them from one threshold, as the Qwen3.5 recipe's "
            "MOE_KERNEL_THRESHOLD does."
        )
    logger.info_once("MoE kernels: %s.", rule)


def prebuild_fused_moe_ep(layer, topk: int, renormalize: bool, activation: str) -> None:
    """Resolve the mesh and register the fused-EP op, once per process.

    Called once per layer from `process_weights_after_loading`. Everything here
    is forbidden inside a compiled forward: mapping EP ranks to TPU device ids
    is an `all_gather_object`, and a `Mesh` cannot survive Dynamo's tracing.
    """
    global _EP_SIZE, _OP_CONTRACT
    layer._tpu_fused_ep_eligible = False
    if not envs.USE_MOE_FUSED_EP_KERNEL:
        return
    if envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS < 0:
        raise ValueError("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS must be >= 0")
    # Under tensor parallelism the kernel arms for the sequence-parallel steps,
    # where each rank holds its own slice of the batch; a replicated step
    # stays off the kernel (`_fused_kernel_takes` in the FP8 method), since
    # the op would do ep_size times the expert work for one batch of output.
    # Under prefill context parallelism, VllmMoERunner.
    # _fused_ep_owns_pcp_collectives drops vLLM's outer PCP all-gather and
    # reduce-scatter for every layer this kernel arms, so the kernel's own
    # cross-expert combine is the only reduction.

    tp = _tensor_parallel_size()
    # Read here, under the loader's config context, and kept for the traced
    # forward: the warm-up traces run outside any config context, where the
    # lookup below silently reads 1.
    global _TP_SIZE
    _TP_SIZE = tp
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    layer_is_sp = bool(getattr(layer.moe_config, "is_sequence_parallel", False))
    if tp > 1 and (not _sp.enabled() or not layer_is_sp):
        logger.warning_once(
            "Fused EP MoE not engaged on layer %s: under tensor parallelism "
            "(tp_size=%d) without sequence parallelism (layer_is_sp=%s) the "
            "batch is replicated on every rank, so the kernel would do %dx "
            "the expert work per batch. Replicated-input MoE paths serve "
            "these steps.",
            getattr(layer, "layer_name", "<unnamed>"),
            tp,
            layer_is_sp,
            tp,
        )
        return
    if tp > 1:
        # With sequence parallelism each rank passes its own rows to the kernel.
        logger.info_once(
            "Fused EP MoE under tensor parallelism (tp_size=%d) with sequence "
            "parallelism: each rank's own rows, prompt steps of at least "
            "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS rows.",
            tp,
        )

    mesh = build_ep_mesh()
    if mesh is None:
        logger.info_once(
            "Fused EP MoE requested but there is no EP group; "
            "the grouped-matmul path serves every step."
        )
        return
    ep = mesh.shape[EP_AXIS_NAME]
    _EP_SIZE = ep
    check_kernel_threshold(ep, tp)

    w1, w2 = layer.w13_weight, layer.w2_weight
    w1_scale = layer.w13_weight_scale_inv
    w2_scale = layer.w2_weight_scale_inv
    # under a copies placement the rank's tensors carry copy slots after its primaries; the prompt path routes
    # to primaries only and streams them alone (the forward passes the same views)
    from vllm_torchtpu import expert_placement as _placement

    if _placement.has_copies() and int(_placement.owned()) < int(w1.shape[0]):
        n_prim = int(_placement.owned())
        w1, w2, w1_scale, w2_scale = (
            w1[:n_prim],
            w2[:n_prim],
            w1_scale[:n_prim],
            w2_scale[:n_prim],
        )
    local_experts, hidden = w1.shape[0], w1.shape[1]
    inter = w1.shape[2] // 2

    # Build fused EP only when the largest batch justifies it.
    max_node_tokens = _max_node_tokens(ep)
    if max_node_tokens < envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS:
        logger.info_once(
            "Fused EP MoE not engaged: the largest node-wide batch is %d "
            "tokens, below MOE_FUSED_EP_KERNEL_MIN_TOKENS=%d.",
            max_node_tokens,
            envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS,
        )
        return

    reason = (
        _unsupported_layer_reason(layer, local_experts, ep)
        or _unsupported_routing_reason(layer)
        or _unsupported_reason(
            w1, w2, w1_scale, w2_scale, local_experts, hidden, inter, activation
        )
    )
    if reason is not None:
        logger.warning_once("Fused EP MoE not engaged: %s", reason)
        return

    idx = ep_mesh_index()
    if not (0 <= idx < ep):
        logger.warning_once(
            "Fused EP MoE not engaged: ep_mesh_index()=%d out of range [0, %d).",
            idx,
            ep,
        )
        return

    global _RANK_BUFFER, _EXPERT_ORDER
    contract = (
        topk,
        bool(renormalize),
        activation,
        hidden,
        inter,
        local_experts,
        bool(envs.TPU_MOE_SKIP_PADDED_TOKENS),
    )
    if _OP is None:  # built once, from the first eligible layer
        _build_op(
            mesh,
            topk,
            renormalize,
            activation,
            envs.TPU_MOE_SKIP_PADDED_TOKENS,
            hidden=hidden,
            inter=inter,
            local_experts=local_experts,
        )
        _OP_CONTRACT = contract
    elif contract != _OP_CONTRACT:
        logger.warning_once(
            "Fused EP MoE not engaged for a layer whose shape or routing %s "
            "differs from the operation's %s; that layer takes the "
            "grouped-matmul path.",
            contract,
            _OP_CONTRACT,
        )
        return
    # The kernel's index is this rank's position in the mesh, which is ordered
    # by device id, not its EP rank. Populate only after contract admission so
    # a rejected layer cannot clobber the active op's rank or expert buffers.
    _RANK_BUFFER = torch.tensor(
        [[idx]], dtype=torch.int32, device=layer.w13_weight.device
    )
    _EXPERT_ORDER = _expert_block_permutation(ep, layer.w13_weight.device)
    layer._tpu_fused_ep_eligible = True
    logger.info_once(
        "Fused EP MoE armed | kernel=%s hidden=%d inter=%d local_experts=%d "
        "ep=%d topk=%d capacity=%d skip_padded_tokens=%s",
        "v3",
        hidden,
        inter,
        local_experts,
        ep,
        topk,
        TILE_ROWS,
        envs.TPU_MOE_SKIP_PADDED_TOKENS,
    )


def _unsupported_layer_reason(layer, local_experts: int, ep: int) -> str | None:
    """Why this layer's parallel layout or biases cannot go through the
    kernel, or None."""
    if not layer.moe_config.moe_parallel_config.use_ep:
        return (
            "the experts are replicated, not expert-parallel; the kernel "
            "addresses experts by shard"
        )
    if layer.global_num_experts != local_experts * ep:
        return (
            f"{layer.global_num_experts} experts do not split evenly "
            f"over {ep} shards of {local_experts}"
        )
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "the bridge passes no expert biases"
    return None


def _unsupported_reason(
    w1, w2, w1_scale, w2_scale, local_experts, hidden, inter, activation
) -> str | None:
    """Why this layer cannot use the fused kernel, or None if it can."""
    if (
        w1.ndim != 3
        or w2.ndim != 3
        or w2.shape[0] < local_experts
        or w1.shape[0] != w2.shape[0]
    ):
        return f"weight rank/expert mismatch {tuple(w1.shape)} {tuple(w2.shape)}"
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return (
            f"the kernel's fp8 form takes float8_e4m3fn expert weights, "
            f"got {w1.dtype} / {w2.dtype}"
        )
    if w2.shape[1] != inter or w2.shape[2] != hidden:
        return f"w2 {tuple(w2.shape)} is not [E, {inter}, {hidden}]"
    if activation != "silu":
        return f"activation {activation!r} is not wired to the kernel's ACT_FNS"
    if _squeeze_channel_scale(w1_scale, local_experts, 2 * inter) is None:
        return (
            "w1_scale is not the per-output-channel form the kernel takes; "
            "a block-scaled weight is a different kernel format"
        )
    if _squeeze_channel_scale(w2_scale, local_experts, hidden) is None:
        return "w2_scale is not the per-output-channel form the kernel takes"
    return None


_TP_SIZE: int | None = None


def tensor_parallel_size() -> int:
    """This deployment's tensor-parallel width as recorded at weight loading
    (`prebuild_fused_moe_ep`), or the config's answer before that."""
    return _TP_SIZE if _TP_SIZE is not None else _tensor_parallel_size()


def _tensor_parallel_size() -> int:
    """This deployment's tensor-parallel width, or 1 if it cannot be read.

    Read off the config, not off `get_tp_group()`: the group coordinator is a
    mutable singleton that a draft model's weight load rewrites to width 1.
    """
    try:
        from vllm.config import get_current_vllm_config

        return int(get_current_vllm_config().parallel_config.tensor_parallel_size)
    except Exception:  # noqa: BLE001 - no config outside a served model
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
