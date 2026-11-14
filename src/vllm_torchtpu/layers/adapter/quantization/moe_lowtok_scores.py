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

On the expert-parallel MoE layers of the Qwen3.5 checkpoint, device and
expert-parallel size it was measured at (`enabled`, `max_tokens`), a layer whose
routing is a plain softmax top-k with renormalized weights hands its router
scores to `pallas::moe_lowtok_scores_k{k}` (or
`pallas::moe_lowtok_scores_copies_k{k}_r{r}_o{o}` when placed by a
`TPU_EXPERT_PLACEMENT` table with copy slots). Inside the JAX op, shapes at or
below the expert-parallel size's row cap (`max_tokens`) dispatch to
`moe_lowtok` (which selects
experts in-kernel and, for copy-enabled layers, greedily picks the less-loaded
holder per hit copied expert), while shapes above the cap run top-k +
renormalization in JAX and dispatch to `fused_moe_func`. Keeping the row-count
fork inside the JAX op avoids a PyTorch-side shape guard, so every eligible MoE
layer (both `backbone` and `eagle_head`) arms the `scores` entry without
requiring Dynamo shape-variant retraces.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.moe_lowtok import (
    MOE_LOWTOK_MAX_TOKENS,
    can_use_moe_lowtok,
    hot_onehot,
    hot_tables,
    moe_lowtok,
    pack_tables,
    slot_table,
)
from vllm_torchtpu.kernels.router_topk import router_topk
from vllm_torchtpu.layers.adapter import expert_placement, moe_routing, token_padding
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    resolve_onehot_permute_threshold,
    set_lowtok_max_tokens,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu.layers.adapter.quantization.configs import TpuRoutedExperts

logger = init_logger(__name__)
_TABLE_LANES = 512
_OPS: dict[Any, torch.library.CustomOpDef] = {}


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


# Largest step, in rows per MoE call across the node, the few-token kernel
# serves, keyed on the expert-parallel size. EP8 is the Qwen3.5 decode layout
# the kernel was measured at; at any other size it is not armed.
MAX_TOKENS_BY_EP: dict[int, int] = {8: 448}
assert max(MAX_TOKENS_BY_EP.values()) <= MOE_LOWTOK_MAX_TOKENS
# The cap this process armed the kernel with (0: not armed), for the JAX op
# body, which is traced after the weights load.
_ARMED_MAX_TOKENS = 0


def max_tokens(ep_size: int) -> int:
    """The row cap for an expert-parallel size; 0 (not armed) at a size the
    kernel was not measured at."""
    return MAX_TOKENS_BY_EP.get(int(ep_size), 0)


def enabled(vllm_config: Any) -> bool:
    """Whether the few-token kernel serves this engine's MoE layers: expert
    parallelism on the checkpoint and device the kernel was measured on
    (``qwen3_5.hooks.is_measured_target``: Qwen3.5-397B-A17B on TPU7x). Every
    other model, other Qwen3.5 checkpoints included, keeps the default MoE
    path, which is main's."""
    from vllm_torchtpu.models.qwen3_5 import hooks as qwen3_5_hooks

    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return bool(
        model_config is not None
        and getattr(parallel_config, "enable_expert_parallel", False)
        and qwen3_5_hooks.is_measured_target(model_config)
    )


def armed_max_tokens() -> int:
    """The row cap the kernel is armed with in this process, 0 if it is not."""
    return _ARMED_MAX_TOKENS


def _layer_has_placement_copies(layer: RoutedExperts) -> bool:
    """Whether `layer` holds copy slots from a copy-enabled `expert_placement`."""
    layer = cast("TpuRoutedExperts", layer)
    if not expert_placement.has_copies():
        return False
    prefix = getattr(layer, "layer_name", "") or ""
    if not expert_placement.placed(prefix):
        return False
    return (
        int(layer.w13_weight.shape[0]) == int(expert_placement.slots())
        and int(layer.global_num_experts) == int(expert_placement.global_count())
        and int(layer.global_num_experts) == _TABLE_LANES
        and int(layer.moe_config.ep_size) == int(expert_placement.shards())
    )


def _set_layer_buffer(layer: Any, name: str, tensor: torch.Tensor) -> None:
    if hasattr(layer, "register_buffer"):
        layer.register_buffer(name, tensor, persistent=False)
    else:
        setattr(layer, name, tensor)


def _install_copy_tables(layer: RoutedExperts) -> None:
    """Attach per-layer holder tables as buffers on `layer`."""
    layer = cast("TpuRoutedExperts", layer)
    layer_idx = expert_placement.layer_index(getattr(layer, "layer_name", "") or "")
    assert layer_idx is not None
    rank = int(getattr(layer.moe_config, "ep_rank", 0))
    owned = int(expert_placement.owned())
    gid, prim, copy = expert_placement.holder_tables(layer_idx, rank)
    order = expert_placement.hot_order(layer_idx)
    hot, hprim, hcopy = (np.asarray(t) for t in hot_tables(prim, copy, order))
    dev = (
        torch.device("cpu")
        if layer.w13_weight.device.type == "meta"
        else layer.w13_weight.device
    )
    layer._lowtok_owned = owned
    layer._lowtok_has_copies = True
    _set_layer_buffer(
        layer, "_lowtok_prim", torch.tensor(prim, dtype=torch.int32, device=dev)
    )
    _set_layer_buffer(
        layer, "_lowtok_copy", torch.tensor(copy, dtype=torch.int32, device=dev)
    )
    _set_layer_buffer(
        layer,
        "_lowtok_hot1h",
        torch.tensor(
            np.asarray(hot_onehot(hot)).astype(np.float32),
            dtype=torch.bfloat16,
            device=dev,
        ),
    )
    _set_layer_buffer(
        layer,
        "_lowtok_tables",
        torch.tensor(
            pack_tables(
                rank * owned,
                gid,
                hot,
                hprim,
                hcopy,
                np.asarray(slot_table(gid, len(gid))),
            ),
            dtype=torch.int32,
            device=dev,
        ),
    )


def _unsupported_reason(layer: RoutedExperts, activation: str) -> str | None:
    """Why the kernel cannot serve this layer at all, or None."""
    layer = cast("TpuRoutedExperts", layer)
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
            "experts, gate|up width a multiple of 512)"
        )
    return None


def _routing_unsupported_reason(layer: RoutedExperts) -> str | None:
    """Why the in-kernel router would not reproduce this layer's routing."""
    layer = cast("TpuRoutedExperts", layer)
    if layer.custom_routing_function is not None:
        return "the layer supplies a custom_routing_function"
    if layer.scoring_func != "softmax":
        return f"scoring_func {layer.scoring_func!r} is not softmax"
    if not layer.renormalize:
        return "renormalize=False"
    if layer.e_score_correction_bias is not None:
        return "e_score_correction_bias is not applied in the kernel"
    # vLLM's router requires both group sizes when grouped top-k is on.
    if (
        layer.use_grouped_topk
        and cast("int", layer.num_expert_group) > 1
        and cast("int", layer.topk_group) < cast("int", layer.num_expert_group)
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
    if int(layer.w13_weight.shape[0]) * int(layer.moe_config.ep_size) != int(
        layer.global_num_experts
    ) and not _layer_has_placement_copies(layer):
        return "the layer holds extra copy slots"
    return None


def _graph_unsupported_reason() -> str | None:
    """Kept for backwards compatibility; the row-count fork now runs inside
    the JAX op per compiled shape, so non-backbone graphs (e.g. `eagle_head`)
    also arm the `scores` entry without forking the PyTorch Dynamo trace.
    """
    return None


def _moe_scores_fn(
    x: jax.Array,
    scores: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    w13_scale: jax.Array,
    w2_scale: jax.Array,
    experts_start: jax.Array,
    tables: jax.Array | None = None,
    prim: jax.Array | None = None,
    copy: jax.Array | None = None,
    hot1h: jax.Array | None = None,
    *,
    topk: int,
    ranks: int = 8,
    owned: int = 64,
    has_copies: bool = False,
) -> jax.Array:
    if x.shape[0] <= _ARMED_MAX_TOKENS:
        if has_copies:
            return moe_lowtok(
                x,
                scores,
                w13,
                w13_scale,
                w2,
                w2_scale,
                experts_start,
                tables,
                prim,
                copy,
                hot1h,
                topk=topk,
                ranks=ranks,
                owned=owned,
                has_copies=True,
                from_logits=True,
            )
        return moe_lowtok(
            x,
            scores,
            w13,
            w13_scale,
            w2,
            w2_scale,
            experts_start,
            topk=topk,
            from_logits=True,
        )
    logits_f32 = scores.astype(jnp.float32)
    if envs.TPU_MOE_ROUTER_TOPK == "rowmax":
        topk_logits, topk_ids = router_topk(logits_f32, topk)
    else:
        topk_logits, topk_ids = jax.lax.top_k(logits_f32, topk)
    valid_row = topk_logits[:, :1] > -3.0e38
    topk_weights = jnp.where(valid_row, jax.nn.softmax(topk_logits, axis=-1), 0.0)
    num_primary_experts = (
        owned
        if has_copies
        else (int(expert_placement.owned()) if expert_placement.has_copies() else None)
    )
    return fused_moe_func(
        x,
        w13,
        w2,
        w13_scale,
        w2_scale,
        None,
        None,
        topk_weights.astype(x.dtype),
        topk_ids,
        experts_start,
        topk=topk,
        activation="silu",
        use_ep=True,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        rhs_quant_dtype=None,
        skip_padded_tokens=envs.TPU_MOE_SKIP_PADDED_TOKENS,
        num_primary_experts=num_primary_experts,
    )


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
        return _moe_scores_fn(
            x,
            scores,
            w13,
            w2,
            w13_scale,
            w2_scale,
            experts_start,
            topk=topk,
        )

    op = jax_op(f"pallas::moe_lowtok_scores_k{topk}", moe)

    def _fake(x, *_args):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[topk] = op
    return op


def _build_copies_op(topk: int, ranks: int, owned: int) -> torch.library.CustomOpDef:
    key = (topk, ranks, owned)
    op = _OPS.get(key)
    if op is not None:
        return op

    def moe(
        x: jax.Array,
        scores: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        w13_scale: jax.Array,
        w2_scale: jax.Array,
        experts_start: jax.Array,
        tables: jax.Array,
        prim: jax.Array,
        copy: jax.Array,
        hot1h: jax.Array,
    ) -> jax.Array:
        return _moe_scores_fn(
            x,
            scores,
            w13,
            w2,
            w13_scale,
            w2_scale,
            experts_start,
            tables,
            prim,
            copy,
            hot1h,
            topk=topk,
            ranks=ranks,
            owned=owned,
            has_copies=True,
        )

    op = jax_op(f"pallas::moe_lowtok_scores_copies_k{topk}_r{ranks}_o{owned}", moe)

    def _fake(x, *_args):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def prebuild(
    method: Any, layer: RoutedExperts, activation: str
) -> torch.library.CustomOpDef | None:
    """Record the layer's entry when the few-token kernel is enabled; the
    scores op if routing moves into the kernel. log_summary() reports once all
    MoE layers, draft included, are loaded.
    """
    from vllm.config import get_current_vllm_config_or_none

    global _ARMED_MAX_TOKENS
    # A layer built outside an engine has no current config; the rule then
    # does not apply.
    if not enabled(get_current_vllm_config_or_none()):
        return None
    cap = max_tokens(int(layer.moe_config.ep_size))
    if not cap:
        # Not measured at this expert-parallel size: the default MoE path.
        return None
    if _ARMED_MAX_TOKENS not in (0, cap):
        raise ValueError(
            f"moe_lowtok: MoE layers of one model have row caps {_ARMED_MAX_TOKENS} "
            f"and {cap}; one expert-parallel size per process is supported"
        )
    _ARMED_MAX_TOKENS = cap
    set_lowtok_max_tokens(cap)
    unsupported = _unsupported_reason(layer, activation)
    routing = None
    if not unsupported:
        routing = _graph_unsupported_reason() or _routing_unsupported_reason(layer)
    entry = None if unsupported else ("ids" if routing else "scores")
    _LAYERS.append(_Verdict(method, layer.layer_name, entry, unsupported or routing))
    if entry != "scores":
        return None
    if _layer_has_placement_copies(layer):
        _install_copy_tables(layer)
        return _build_copies_op(
            layer.moe_config.experts_per_token,
            int(layer.moe_config.ep_size),
            int(expert_placement.owned()),
        )
    cast("TpuRoutedExperts", layer)._lowtok_has_copies = False
    return _build_op(layer.moe_config.experts_per_token)


def _owner(method: Any) -> str | None:
    """The fused path that takes the layer before either entry is reached."""
    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
    from vllm_torchtpu.layers.adapter.fused_moe_ep_v3 import (
        fused_moe_ep_supported as fused_moe_ep_v3_supported,
    )

    if method._use_adaptive_fused_moe:
        return "adaptive fused MoE owns the layer"
    if fused_moe_ep_supported(method) or fused_moe_ep_v3_supported(method):
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
        # The kernel is on by rule, not by request, so a layer it cannot
        # serve keeps the default MoE path.
        logger.warning(
            "moe_lowtok cannot serve these MoE layers, which keep the default "
            "MoE path: %s",
            "; ".join(unservable),
        )
    reasons = "".join(f"; {n} x {reason}" for reason, n in sorted(declined.items()))
    logger.info(
        "moe_lowtok: %d/%d MoE layers armed (scores %d, ids %d)%s | rows<=%d",
        armed.total(),
        total,
        armed["scores"],
        armed["ids"],
        reasons,
        _ARMED_MAX_TOKENS,
    )


def serves(op: torch.library.CustomOpDef | None, x: torch.Tensor) -> bool:
    del x
    return op is not None


def apply(
    op: torch.library.CustomOpDef,
    layer: RoutedExperts,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    logits = router_logits
    if envs.TPU_MOE_SKIP_PADDED_TOKENS:
        is_padding = token_padding.padding_mask_for(logits.shape[0])
        if is_padding is not None:
            logits = torch.where(
                is_padding[:, None],
                torch.full_like(logits, float("-inf")),
                logits,
            )
    if getattr(layer, "_lowtok_has_copies", False):
        return op(
            x,
            logits,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            layer._experts_start,
            layer._lowtok_tables,
            layer._lowtok_prim,
            layer._lowtok_copy,
            layer._lowtok_hot1h,
        )
    return op(
        x,
        logits,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        layer._experts_start,
    )
