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
"""Small-batch MoE with the router in the kernel, for the FP8 MoE method.

With TPU_MOE_LOWTOK=1, a layer whose routing is a plain softmax top-k with
renormalized weights hands its router scores to `moe_lowtok`, which selects
the experts itself. Other layers keep vLLM-side routing and reach the same
kernel with a routing table through `fused_moe_func`.

The scores entry forks on the row count in torch, which leaves a shape guard
in the trace. Only the backbone is re-traced per bucket group, so a layer
compiled outside it -- the MTP draft's -- keeps the ids entry, whose row test
runs per compiled shape inside the JAX op.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.moe_lowtok import (
    MOE_LOWTOK_MAX_TOKENS,
    can_use_moe_lowtok,
    moe_lowtok,
)
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)
_OPS: dict[int, torch.library.CustomOpDef] = {}


class _Verdict(NamedTuple):
    method: Any
    layer_name: str
    entry: str | None  # "scores", "ids", or None when the kernel cannot serve it
    reason: str | None


_LAYERS: list[_Verdict] = []


def _jax_dtype(dtype: torch.dtype) -> jnp.dtype:
    return jnp.dtype(str(dtype).removeprefix("torch."))


def _struct(t: torch.Tensor | None) -> jax.ShapeDtypeStruct | None:
    return None if t is None else jax.ShapeDtypeStruct(t.shape, _jax_dtype(t.dtype))


def _unsupported_reason(layer: RoutedExperts, activation: str) -> str | None:
    """Why the kernel cannot serve this layer at all, or None."""
    if not layer.moe_config.moe_parallel_config.use_ep:
        return "it needs expert parallelism"
    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    in_dtype = layer.moe_config.in_dtype
    hidden = jax.ShapeDtypeStruct(
        (MOE_LOWTOK_MAX_TOKENS, layer.hidden_size), _jax_dtype(in_dtype)
    )
    if not can_use_moe_lowtok(
        hidden,
        _struct(w13),
        _struct(w2),
        _struct(s13),
        _struct(s2),
        _struct(getattr(layer, "w13_bias", None)),
        _struct(getattr(layer, "w2_bias", None)),
        activation=activation,
        rhs_quant_dtype=None,
    ):
        return (
            f"activation={activation}, w13={tuple(w13.shape)} {w13.dtype}, "
            f"w13_scale={tuple(s13.shape)}, hidden={layer.hidden_size} "
            f"{in_dtype} is outside its envelope (silu, bf16 activations, "
            "FP8 weights with per-channel scales, no bias, at most 128 local "
            "experts, gate|up width a multiple of 512, hidden and intermediate "
            "widths whole 512-column scale blocks)"
        )
    return None


def _routing_unsupported_reason(layer: RoutedExperts) -> str | None:
    """Why the in-kernel router would not reproduce this layer's routing."""
    if layer.custom_routing_function is not None:
        return "the layer supplies a custom_routing_function"
    if layer.scoring_func != "softmax":
        return f"scoring_func {layer.scoring_func!r} is not softmax"
    if not layer.renormalize:
        return "renormalize=False"
    if layer.e_score_correction_bias is not None:
        return "e_score_correction_bias is not applied in the kernel"
    if (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    ):
        return "grouped top-k is not implemented in the kernel"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing is not implemented in the kernel"
    if layer.routed_scaling_factor != 1.0:
        return f"routed_scaling_factor {layer.routed_scaling_factor}"
    if layer.apply_router_weight_on_input:
        return "apply_router_weight_on_input"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set"
    if enable_pipelined_collective_and_compute():
        return "MoE chunk pipelining keeps the router input rank-local"
    return None


def _graph_unsupported_reason() -> str | None:
    """Why the graph this layer is compiled into cannot carry the row fork.

    Only the backbone is re-traced per bucket group
    (`TPUModelRunner._precompile_backbone`); a module loaded under another
    model tag -- the MTP or EAGLE draft, "eagle_head" -- compiles every bucket
    from its one trace, and the buckets past the row cap would raise
    `ShapeSpecializationError` at its first forward.
    """
    # Read at call time: `set_model_tag` rebinds the module global.
    from vllm.compilation import backends

    if backends.model_tag != "backbone":
        return (
            f"the {backends.model_tag!r} graph is traced once for every "
            "bucket, so the row cap cannot fork it"
        )
    return None


def _build_op(topk: int) -> torch.library.CustomOpDef:
    op = _OPS.get(topk)
    if op is not None:
        return op

    # Operands follow the grouped path's order so both sides of the row-count
    # fork lift the layer's buffers in the same order.
    def moe(
        x: jax.Array,
        scores: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        w13_scale: jax.Array,
        w2_scale: jax.Array,
        experts_start: jax.Array,
    ) -> jax.Array:
        return moe_lowtok(
            x, scores, w13, w13_scale, w2, w2_scale, experts_start, topk=topk
        )

    op = jax_op(f"pallas::moe_lowtok_scores_k{topk}", moe)

    def _fake(x, *_args):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[topk] = op
    return op


def prebuild(
    method: Any, layer: RoutedExperts, activation: str
) -> torch.library.CustomOpDef | None:
    """Record the layer's entry under TPU_MOE_LOWTOK; the scores op if routing
    moves into the kernel. log_summary() reports and refuses once all MoE
    layers, draft included, are loaded.
    """
    if not envs.TPU_MOE_LOWTOK:
        return None
    unsupported = _unsupported_reason(layer, activation)
    routing = None
    if not unsupported:
        routing = _graph_unsupported_reason() or _routing_unsupported_reason(layer)
    entry = None if unsupported else ("ids" if routing else "scores")
    _LAYERS.append(_Verdict(method, layer.layer_name, entry, unsupported or routing))
    if entry != "scores":
        return None
    return _build_op(layer.moe_config.experts_per_token)


def _owner(method: Any) -> str | None:
    """The fused path that takes the layer before either entry is reached."""
    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

    if method._use_adaptive_fused_moe:
        return "adaptive fused MoE owns the layer"
    if fused_moe_ep_supported(method):
        return "the fused EP kernel owns the layer"
    return None


def log_summary() -> None:
    """Log which entry each MoE layer armed; refuse layers nothing else owns.

    Called once after the target and draft models are loaded.
    """
    if not _LAYERS:
        return
    armed: Counter[str] = Counter()
    declined: Counter[str] = Counter()
    unservable = []
    for verdict in _LAYERS:
        owner = _owner(verdict.method)
        if owner is not None:
            declined[owner] += 1
        elif verdict.entry is None:
            unservable.append(f"{verdict.layer_name}: {verdict.reason}")
        else:
            armed[verdict.entry] += 1
            if verdict.reason is not None:
                declined[f"scores declined, ids armed: {verdict.reason}"] += 1
    total = len(_LAYERS)
    _LAYERS.clear()
    if unservable:
        raise NotImplementedError(
            "TPU_MOE_LOWTOK=1 cannot serve these MoE layers: "
            + "; ".join(unservable)
            + ". Unset TPU_MOE_LOWTOK for this model."
        )
    reasons = "".join(f"; {n} x {reason}" for reason, n in sorted(declined.items()))
    logger.info(
        "moe_lowtok: %d/%d MoE layers armed (scores %d, ids %d)%s | rows<=%d",
        armed.total(),
        total,
        armed["scores"],
        armed["ids"],
        reasons,
        MOE_LOWTOK_MAX_TOKENS,
    )


def serves(op: torch.library.CustomOpDef | None, x: torch.Tensor) -> bool:
    return op is not None and x.shape[0] <= MOE_LOWTOK_MAX_TOKENS


def apply(
    op: torch.library.CustomOpDef,
    layer: RoutedExperts,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    scores = torch.softmax(router_logits.float(), dim=-1)
    if envs.TPU_MOE_SKIP_PADDED_TOKENS:
        # A zeroed score row selects no expert in the kernel's router; the
        # helper only reads the first argument's row count.
        _, scores = token_padding.zero_routing_weights_for_padding(scores, scores)
    return op(
        x,
        scores,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        layer._experts_start,
    )
