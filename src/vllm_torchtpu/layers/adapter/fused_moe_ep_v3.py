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
"""Torch bridge for the fused expert-parallel MoE kernel v3.

Selected with MOE_FUSED_EP_KERNEL_IMPL=v3. Like v2 (`fused_moe_ep.py`) the
whole layer -- top-k, dispatch, both expert matmuls, combine -- runs as one
SPMD program over the EP mesh, taking and returning this rank's own tokens.

v3 serves expert parallelism over each rank's own tokens: data parallelism
(TP=1), or TP with sequence parallelism on its large steps (PCP=1), with
per-channel fp8 weights and a plain softmax top-k router. A deployment outside
that raises at weight load; a layer outside it (shape, routing, weights) is
declined on its own and keeps the grouped-matmul path.

Per step, an armed layer runs v3 only when this rank's step reaches
`step_min_tokens` rows; smaller steps take the grouped-matmul path between
vLLM's own dispatch and combine (`fused_moe_ep_v3_dispatch.apply`). The fork
needs a trace per side, and only the backbone is re-traced per bucket group,
so a layer compiled outside it -- the MTP draft's -- is declined.

The op reads only buffers the grouped-matmul path reads too -- the layer's
`experts_start`, from which it derives the shard index, and with
TPU_MOE_SKIP_PADDED_TOKENS the runner's padding mask -- so a trace that takes
one path and a trace that takes the other lift the same inputs.

Padded rows: the kernel gets a NaN logit row for every row the padding mask
marks, which its selector routes nowhere (`drop_nan_rows`); the input row is
zeroed so nothing of it travels, and the kernel returns zero for it.
"""

from collections import Counter
from collections.abc import Callable
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
)
from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe
from vllm_torchtpu.layers.adapter import (
    moe_routing,
    sequence_parallel,
    token_padding,
)
from vllm_torchtpu.layers.adapter.fused_moe_ep import (
    _pcp_size,
    _squeeze_channel_scale,
    _squeeze_channel_scale_checked,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The FFN tile height (the kernel's `tile_rows`). Smaller tiles are far slower
# with the kernel's DMA strategy, so it is not a knob.
TILE_ROWS = 128
WEIGHT_FORMAT = "fp8"

# One op per distinct closure, as in v2: a per-layer op would re-export the
# same program once per MoE layer, and two `CustomOpDef`s under one qualname
# silently share a dispatch entry. The name must not be v2's
# `pallas::fused_moe_ep`: both modules can be imported in one process.
FUSED_MOE_EP_V3_OP_NAME = "pallas::fused_moe_ep_v3"
_OPS: dict[tuple, Any] = {}

# The attribute prebuild's `ArmedOp` is recorded under, on the layer's quant
# method.
FUSED_MOE_EP_V3_OP_ATTR = "_fused_moe_ep_v3_op"

# Rows per rank from which a step runs v3; smaller steps take the grouped-matmul
# path. The value PR #989 served with, pending the kernel probe's measured
# crossover, which lands per shape in STEP_MIN_TOKENS_BY_SHAPE.
STEP_MIN_TOKENS_DEFAULT = 33
# (hidden, inter, experts_per_shard, top_k) -> rows per rank.
STEP_MIN_TOKENS_BY_SHAPE: dict[tuple[int, int, int, int], int] = {}

# Every prebuild's outcome, "armed" or the refusal, for the boot summary.
_ARMING: Counter = Counter()


class ArmedOp(NamedTuple):
    """What prebuild records for an armed layer."""

    run: Callable[..., torch.Tensor]
    step_min_tokens: int


def step_min_tokens(
    *, hidden: int, inter: int, experts_per_shard: int, top_k: int
) -> int:
    """Rows per rank from which a step of this shape runs v3."""
    key = (hidden, inter, experts_per_shard, top_k)
    return STEP_MIN_TOKENS_BY_SHAPE.get(key, STEP_MIN_TOKENS_DEFAULT)


def mask_padded_rows(x: jax.Array, gating: jax.Array, padded: jax.Array):
    """Padded rows route nowhere and carry nothing: a row whose `padded`
    entry is nonzero gets NaN logits and a zero input row. Elementwise, so
    it partitions over the token axis without a partition-id."""
    valid = (padded == 0)[:, None]
    gating = jnp.where(valid, gating, jnp.nan)
    x = jnp.where(valid, x, jnp.zeros_like(x))
    return x, gating


def served_config(*, hidden: int, inter: int, experts_per_shard: int, top_k: int):
    """The kernel's Config for this shape, logged once."""
    config = Config.for_shape(
        hidden=hidden, inter=inter, experts_per_shard=experts_per_shard, top_k=top_k
    )
    logger.info_once(
        "Fused EP MoE v3 settings for hidden=%d inter=%d experts_per_shard=%d "
        "top_k=%d: tile_rows=%d %s",
        hidden,
        inter,
        experts_per_shard,
        top_k,
        TILE_ROWS,
        config,
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
    """The `run_moe` the op wraps: global arrays sharded on the EP axis in,
    one [tokens, hidden] array sharded like x out."""
    if MESH_AXIS not in mesh.axis_names:
        raise ValueError(
            f"the v3 kernel splits the experts over the mesh axis named "
            f"{MESH_AXIS!r}; the adapter's mesh has {mesh.axis_names}"
        )
    config = served_config(
        hidden=hidden, inter=inter, experts_per_shard=experts_per_shard, top_k=topk
    )

    def run_moe(x, w1, w2, w1_scale, w2_scale, gating, experts_start, padded):
        gating = gating.astype(jnp.float32)
        drop_nan_rows = padded is not None
        if drop_nan_rows:
            x, gating = mask_padded_rows(x, gating, padded)
        # experts_start is int32 [num_shards, 1], each shard's first global
        # expert: with equal contiguous blocks it is shard * experts_per_shard.
        rank = experts_start.astype(jnp.int32) // experts_per_shard
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
            experts_per_shard=experts_per_shard,
            # The torch bridge exports the op as one jitted function: its SPMD
            # partitioner refuses the layer's own inner jit, a side-effecting
            # custom call without a sharding, and `lax.axis_index`, so the
            # shard's index goes in as data.
            shard_index=rank,
            jit=False,
            side_effects=False,
        )

    return run_moe


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
    """The op for this closure, built once per distinct closure and reused."""
    key = (
        topk,
        renormalize,
        activation,
        skip_padded_tokens,
        hidden,
        inter,
        local_experts,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached
    name = (
        FUSED_MOE_EP_V3_OP_NAME
        if not _OPS
        else f"{FUSED_MOE_EP_V3_OP_NAME}_{len(_OPS)}"
    )
    run_moe = run_moe_v3(
        mesh,
        topk,
        renormalize,
        activation,
        hidden=hidden,
        inter=inter,
        experts_per_shard=local_experts,
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
        experts_start: jax.Array,
    ) -> jax.Array:
        return run_moe(x, w1, w2, w1_scale, w2_scale, gating, experts_start, None)

    def moe_skip_padded(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        experts_start: jax.Array,
        padded: jax.Array,
    ) -> jax.Array:
        return run_moe(x, w1, w2, w1_scale, w2_scale, gating, experts_start, padded)

    fn = moe_skip_padded if skip_padded_tokens else moe
    spec = PartitionSpec(EP_AXIS_NAME)
    # x, w1, w2, w1_scale, w2_scale, gating, experts_start [, padded] -- every
    # operand and the single output are sharded on axis 0 over the EP axis.
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * (8 if skip_padded_tokens else 7),
        output_partition_specs=spec,
    )

    # The kernel returns the rows it was given, so the fake needs no export.
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    run = op
    if skip_padded_tokens:
        # Decided here, at weight load, so the forward reads no env.
        ep = mesh.shape[MESH_AXIS]

        def run(x, *args):
            if sequence_parallel.is_active():
                # This rank's rows of a sequence-parallel step: its slice of
                # the step's mask, the rank read off `experts_start` (EP rank
                # = TP rank there).
                rank = (args[-1] // local_experts).reshape(1)
                padded = token_padding.rank_padding_mask(x.shape[0], ep, rank)
            else:
                # The same mask slice the grouped path's padding zeroing reads.
                padded = token_padding.local_padding_mask(x.shape[0])
            if padded is None:
                raise RuntimeError(
                    "TPU_MOE_SKIP_PADDED_TOKENS is set but the runner did not "
                    "install padding state"
                )
            return op(x, *args, padded.to(torch.int32))

    _OPS[key] = run
    return run


def fused_moe_ep_v3_supported(owner: Any) -> bool:
    """Whether `prebuild_fused_moe_ep_v3` armed v3 for this layer.

    `owner` is the layer's quant method, where the op was recorded.
    """
    return getattr(owner, FUSED_MOE_EP_V3_OP_ATTR, None) is not None


def fused_moe_ep_v3(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
    experts_start: torch.Tensor,
) -> torch.Tensor:
    """Run one MoE layer through the v3 kernel.

    Takes this rank's own tokens and returns this rank's own tokens, so the
    caller must not all-gather the input or reduce-scatter the output. Only
    call it when `fused_moe_ep_v3_supported(owner)` said yes.
    """
    return getattr(owner, FUSED_MOE_EP_V3_OP_ATTR).run(
        hidden_states,
        w1,
        w2,
        _squeeze_channel_scale_checked(w1_scale),
        _squeeze_channel_scale_checked(w2_scale),
        router_logits,
        experts_start.reshape(1, 1),
    )


def _check_deployment(layer) -> None:
    """Raise on a deployment v3 cannot serve at all.

    The operator selected v3 explicitly, so these are errors rather than a
    silent fallback to the grouped-matmul path.
    """
    from vllm.config import get_current_vllm_config

    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )

    # Read off the config, not `get_tp_group()`: a draft model's weight load
    # rewrites the group coordinator to width 1.
    tp = get_current_vllm_config().parallel_config.tensor_parallel_size
    # A sequence-parallel layer's large steps hand each rank its own rows
    # (layers/adapter/sequence_parallel); its replicated small steps stay off
    # the kernel (`fused_moe_ep_v3_dispatch.apply`).
    if tp > 1 and not layer.moe_config.moe_parallel_config.is_sequence_parallel:
        raise NotImplementedError(
            f"MOE_FUSED_EP_KERNEL_IMPL=v3 needs each rank's own tokens; "
            f"tensor_parallel_size={tp} replicates every token on each TP rank, "
            "so the kernel would do tp_size times the expert work. Use "
            "--data-parallel-size with --tensor-parallel-size=1, "
            "TPU_SEQUENCE_PARALLEL_PREFILL=1 (Qwen3-Next/Qwen3.5 under TP + EP), "
            "or MOE_FUSED_EP_KERNEL_IMPL=v2."
        )
    pcp = _pcp_size(layer)
    if pcp > 1:
        raise NotImplementedError(
            f"MOE_FUSED_EP_KERNEL_IMPL=v3 does not support prefill context "
            f"parallelism (pcp_size={pcp}). Use MOE_FUSED_EP_KERNEL_IMPL=v2."
        )
    if not layer.moe_config.moe_parallel_config.use_ep:
        raise NotImplementedError(
            "MOE_FUSED_EP_KERNEL_IMPL=v3 addresses experts by EP shard; pass "
            "--enable-expert-parallel."
        )
    if enable_pipelined_collective_and_compute():
        raise NotImplementedError(
            "MOE_FUSED_EP_KERNEL_IMPL=v3 and MoE chunk pipelining "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE) both claim the expert-parallel "
            "exchange; enable one of them."
        )


def prebuild_fused_moe_ep_v3(
    layer, topk: int, renormalize: bool, activation: str
) -> ArmedOp | None:
    """Resolve the mesh and build the v3 op for one layer.

    Called once per MoE layer -- the MTP draft's included -- from
    `process_weights_after_loading`: mapping EP ranks to devices is a
    collective and a `Mesh` cannot survive Dynamo's tracing. Raises on a
    deployment v3 cannot serve; returns None for a layer it declines, which
    keeps the grouped-matmul path.
    """
    if _draft_layer_outside_sequence_parallelism(layer):
        armed = "an MTP draft under TP, outside sequence parallelism"
    else:
        _check_deployment(layer)
        armed = _prebuild(layer, topk, renormalize, activation)
    _ARMING["armed" if isinstance(armed, ArmedOp) else armed] += 1
    return armed if isinstance(armed, ArmedOp) else None


def _draft_layer_outside_sequence_parallelism(layer) -> bool:
    """Whether this is the MoE layer of an MTP draft next to a TP
    sequence-parallel target: the draft runs the plain TP form (its rows
    replicated), so the grouped-matmul path serves it."""
    from vllm.config import get_current_vllm_config

    from vllm_torchtpu.layers.adapter import sequence_parallel

    tp = get_current_vllm_config().parallel_config.tensor_parallel_size
    return (
        tp > 1
        and sequence_parallel.enabled()
        and not layer.moe_config.moe_parallel_config.is_sequence_parallel
    )


def _prebuild(layer, topk: int, renormalize: bool, activation: str) -> ArmedOp | str:
    """The armed op, or why the layer is declined."""
    mesh = build_ep_mesh()
    if mesh is None:
        return "no EP group"
    ep = mesh.shape[EP_AXIS_NAME]

    w1, w2 = layer.w13_weight, layer.w2_weight
    w1_scale, w2_scale = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    local_experts, hidden = w1.shape[0], w1.shape[1]
    inter = w1.shape[2] // 2

    reason = (
        _unsupported_graph_reason()
        or _unsupported_layer_reason(layer, local_experts, ep)
        or _unsupported_routing_reason(layer)
        or _unsupported_weights_reason(
            w1, w2, w1_scale, w2_scale, local_experts, hidden, inter, activation
        )
    )
    if reason is not None:
        return reason

    # The mesh is rank-ordered (google-pytorch/torch_tpu#3522), so this rank's
    # mesh index is its EP rank, its expert block needs no reordering, and its
    # first expert is its index times the block.
    if moe_routing.get_experts_start(layer) != ep_mesh_index() * local_experts:
        return "its first expert is not mesh index x local experts"
    skip_padded_tokens = bool(envs.TPU_MOE_SKIP_PADDED_TOKENS)
    run = _build_op(
        mesh,
        topk,
        bool(renormalize),
        activation,
        skip_padded_tokens,
        hidden=hidden,
        inter=inter,
        local_experts=local_experts,
    )
    min_tokens = step_min_tokens(
        hidden=hidden, inter=inter, experts_per_shard=local_experts, top_k=topk
    )
    logger.info_once(
        "Fused EP MoE v3 armed | hidden=%d inter=%d local_experts=%d ep=%d "
        "topk=%d tile_rows=%d skip_padded_tokens=%s step_min_tokens=%d",
        hidden,
        inter,
        local_experts,
        ep,
        topk,
        TILE_ROWS,
        skip_padded_tokens,
        min_tokens,
    )
    return ArmedOp(run, min_tokens)


def log_arming_summary() -> None:
    """Log once, after every MoE layer (target and draft) has loaded, how many
    run v3 and why the rest fall back."""
    if not _ARMING:
        return
    total = sum(_ARMING.values())
    declined = ", ".join(
        f"{count}x {reason}"
        for reason, count in _ARMING.most_common()
        if reason != "armed"
    )
    logger.info(
        "fused EP v3: %d/%d MoE layers armed%s",
        _ARMING["armed"],
        total,
        f"; declined: {declined}" if declined else "",
    )


def _unsupported_graph_reason() -> str | None:
    """Why the graph this layer is compiled into cannot fork at the step gate.

    The gate reads the token count, which leaves a shape guard in the trace.
    Only the backbone is re-traced per bucket group
    (`TPUModelRunner._precompile_backbone`); a module loaded under another
    model tag -- the MTP or EAGLE draft, "eagle_head" -- compiles every bucket
    from its one trace, and the buckets on the far side of the gate would
    raise `ShapeSpecializationError` at its first forward.
    """
    # Read at call time: `set_model_tag` rebinds the module global.
    from vllm.compilation import backends

    if backends.model_tag != "backbone":
        return (
            f"the {backends.model_tag!r} graph is traced once for every "
            "bucket, so the per-step gate cannot fork it"
        )
    return None


def _unsupported_layer_reason(layer, local_experts: int, ep: int) -> str | None:
    """Why this layer's expert layout or biases cannot go through v3."""
    if layer.global_num_experts != local_experts * ep:
        return (
            f"{layer.global_num_experts} experts do not split evenly over "
            f"{ep} shards of {local_experts}"
        )
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "expert biases are not wired through the v3 op"
    return None


def _unsupported_routing_reason(layer) -> str | None:
    """Why this layer's routing is not the kernel's, or None.

    The kernel scores with a plain softmax and selects top-k itself, with
    `renormalize` as its only option. Stricter than v2's guard, which also
    serves sigmoid scoring, a score-correction bias, routed scaling and hash
    routing: none of those is computed here, and taking the path anyway would
    route tokens to different experts with different weights, silently.
    """
    if layer.custom_routing_function is not None:
        return "the layer supplies a custom_routing_function"
    if layer.scoring_func != "softmax":
        return f"scoring_func {layer.scoring_func!r}; the kernel scores with softmax"
    if layer.e_score_correction_bias is not None:
        return "the layer has an e_score_correction_bias"
    if layer.routed_scaling_factor != 1.0:
        return f"routed_scaling_factor {layer.routed_scaling_factor}"
    if (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    ):
        return "grouped top-k routing"
    if layer.apply_router_weight_on_input:
        return "the router weight is applied on the input"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "the layer routes by hash table"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return (
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set and the kernel "
            "does its own routing"
        )
    return None


def _unsupported_weights_reason(
    w1, w2, w1_scale, w2_scale, local_experts, hidden, inter, activation
) -> str | None:
    """Why this layer's weights or activation cannot use v3, or None."""
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        return f"weight rank/expert mismatch {tuple(w1.shape)} {tuple(w2.shape)}"
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return (
            f"the kernel's fp8 form takes float8_e4m3fn expert weights, got "
            f"{w1.dtype} / {w2.dtype}"
        )
    if w2.shape[1] != inter or w2.shape[2] != hidden:
        return f"w2 {tuple(w2.shape)} is not [E, {inter}, {hidden}]"
    if activation != "silu":
        return f"activation {activation!r}; the adapter wires silu only"
    if _squeeze_channel_scale(w1_scale, local_experts, 2 * inter) is None:
        return "w1_scale is not the per-output-channel form the kernel takes"
    if _squeeze_channel_scale(w2_scale, local_experts, hidden) is None:
        return "w2_scale is not the per-output-channel form the kernel takes"
    return None
