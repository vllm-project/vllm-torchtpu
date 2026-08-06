# Copyright 2025 Google LLC
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
"""Routing helpers for TPU local MoE kernels."""

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Warn once at import time rather than inside maybe_force_random_routing():
# the notice then reaches every worker's startup log without adding a Python
# call to the compiled forward pass, where it could trigger a graph break in
# the exact code path we are trying to profile.
if envs.FORCE_MOE_RANDOM_ROUTING:
    logger.warning(
        "FORCE_MOE_RANDOM_ROUTING is enabled: MoE expert routing is RANDOM. "
        "This is a test-only feature and model output is meaningless. Never "
        "enable it for serving or accuracy evaluation.")


def maybe_force_random_routing(router_logits: torch.Tensor) -> torch.Tensor:
    """Optionally replace router logits with uniform noise, for profiling.

    Motivation: the TPU profiler (xprof) can only trace a single device, but
    under expert parallelism each device owns a different slice of the experts.
    Real routing is data-dependent and typically skewed, so which experts --
    and therefore which devices -- do work depends on the input. A one-device
    trace is then not representative of the full mesh.

    With ``FORCE_MOE_RANDOM_ROUTING`` set, we route on uniform noise instead of
    the model's gate. Top-k over i.i.d. noise selects a uniformly random set of
    experts per token, so expert load is balanced across all EP shards in
    expectation and any single device's profile represents the whole mesh.

    This deliberately discards the trained gate, so model output is garbage.
    Use it only for profiling and performance benchmarking -- never for serving
    or accuracy evaluation. Disabled by default; a no-op when the flag is unset.
    """
    if not envs.FORCE_MOE_RANDOM_ROUTING:
        return router_logits
    return torch.rand_like(router_logits)


def _apply_scoring_fn(scoring_fn: str,
                      router_logits: torch.Tensor) -> torch.Tensor:
    scores = router_logits.float()
    if scoring_fn == "softmax":
        return scores.softmax(dim=-1)
    if scoring_fn == "sigmoid":
        return scores.sigmoid()
    if scoring_fn == "sqrtsoftplus":
        import torch.nn.functional as F
        return torch.sqrt(F.softplus(scores))
    raise NotImplementedError(
        f"FusedMoE does not support {scoring_fn} scoring function for TPU.")


def _hash_moe_select(
    scores: torch.Tensor,
    hash_indices_table: torch.Tensor,
    input_ids: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static token-id based MoE routing for DeepSeek-V4.

    Routes tokens to experts based on precomputed hash_indices_table lookup indexed by
    flat input_ids rather than dynamic gating logits.
    """
    flat_input_ids = input_ids.long().flatten()
    topk_ids = hash_indices_table[flat_input_ids].long()
    target_shape = list(scores.shape)
    target_shape[-1] = topk_ids.shape[-1]
    topk_ids = topk_ids.reshape(target_shape)
    topk_weights = scores.gather(-1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / torch.clamp(
            topk_weights.sum(dim=-1, keepdim=True), min=1e-20)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    layer=None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local routed expert IDs and weights for non-EP execution.

    Dispatch order:
    1. DeepSeek-V4 hash routing (if `layer.hash_indices_table` is present).
    2. Grouped top-k routing (if `layer.use_grouped_topk` is enabled, e.g. DeepSeek-V2/V3).
    3. Classic top-k routing with configurable scoring_fn (e.g. `sqrtsoftplus` for DeepSeek-V4)
       and optional `e_score_correction_bias`.
    """
    router_logits = maybe_force_random_routing(router_logits)

    hash_indices_table = getattr(layer, "hash_indices_table",
                                 None) if layer is not None else None
    if hash_indices_table is not None and input_ids is not None:
        scores = _apply_scoring_fn(scoring_fn, router_logits.float())
        routed_scaling_factor = getattr(layer, "routed_scaling_factor", 1.0)
        topk_weights, topk_ids = _hash_moe_select(
            scores,
            hash_indices_table,
            input_ids,
            renormalize,
            routed_scaling_factor,
        )
        return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)

    if layer is not None and layer.use_grouped_topk:
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import \
            grouped_topk
        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            scoring_func=scoring_fn,
            routed_scaling_factor=layer.routed_scaling_factor,
            e_score_correction_bias=layer.e_score_correction_bias,
        )
        return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)

    scores = _apply_scoring_fn(scoring_fn, router_logits.float())
    e_score_correction_bias = getattr(layer, "e_score_correction_bias",
                                      None) if layer is not None else None

    if e_score_correction_bias is not None:
        bias_shape = [1] * (scores.dim() - 1) + [-1]
        bias = e_score_correction_bias.float().view(bias_shape)
        scores_for_choice = scores + bias
        _, topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_weights = scores.gather(-1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / torch.clamp(
            topk_weights.sum(dim=-1, keepdim=True), min=1e-20)
    routed_scaling_factor = getattr(layer, "routed_scaling_factor",
                                    1.0) if layer is not None else 1.0
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)


def get_experts_start(layer) -> int | None:
    """Return the first global expert id owned by this shard, or None for non-EP.

    Mirrors the formula vLLM's ``determine_expert_map`` uses for linear
    placement (``vllm/model_executor/layers/fused_moe/layer.py``), so this is
    a pure function of ``ep_rank``, ``ep_size``, and ``global_num_experts`` --
    known at layer construction time. ``validate_linear_ep_placement`` enforces
    that the placement is linear before this value is used.
    """
    if not layer.moe_config.moe_parallel_config.use_ep:
        return None
    base = layer.global_num_experts // layer.moe_config.ep_size
    remainder = layer.global_num_experts % layer.moe_config.ep_size
    return (layer.moe_config.ep_rank * base +
            min(layer.moe_config.ep_rank, remainder))


def register_experts_start_buffer(layer, *, device: torch.device) -> None:
    """Register ``layer._experts_start`` as a persistent 0-d int32 buffer
    (``None`` for non-EP layers).

    Call once per layer -- e.g. from ``process_weights_after_loading`` --
    before the first forward call. Registering this as a buffer, rather than
    wrapping ``get_experts_start``'s Python int in a fresh tensor on every
    forward call, keeps it as stable module state that flows through
    ``fused_moe_gmm`` as real tensor data instead of a Python int. This
    matters under expert parallelism: every EP rank owns a different
    ``experts_start`` value, and binding it as a Python int into the fused
    MoE kernel's JIT closure (the previous behavior) made each rank compile a
    structurally different program under the identical custom-op name --
    confirmed (2026-07-30) to desync the in-graph EP all-to-all/gather
    collectives and halt the TPU core during warmup. Same class of bug as the
    GDN/PCP rank fix in fa8faaf5 ("Make GDN PCP weight sharding rank-uniform
    in the compiled graph"); same fix shape -- rank-derived values must reach
    the compiled graph as runtime data, not compile-time constants.
    """
    experts_start = get_experts_start(layer)
    buffer = (torch.tensor(experts_start, dtype=torch.int32, device=device)
              if experts_start is not None else None)
    layer.register_buffer("_experts_start", buffer, persistent=False)


def validate_linear_ep_placement(layer) -> None:
    """Validate that this layer uses linear EP placement.

    The kernel relies on a contiguous-block ``expert_map``, which vLLM only
    produces when ``expert_placement_strategy == "linear"``.
    """
    strategy = layer.expert_placement_strategy
    if strategy != "linear":
        raise NotImplementedError(
            "fused MoE kernel currently requires linear EP placement; got "
            f"expert_placement_strategy={strategy!r}.")
