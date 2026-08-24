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
import vllm.envs as vllm_envs

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.vllm.router_topk import rowmax_select
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Resolved once, at import, so the compiled forward never reads the
# environment.
_ROUTER_TOPK = envs.TPU_MOE_ROUTER_TOPK


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the expert axis, by the configured implementation."""
    if _ROUTER_TOPK == "rowmax":
        return rowmax_select(scores, k)
    return torch.topk(scores, k=k, dim=-1)


# Resolve the routing-simulation strategy once, at import, outside the compiled
# forward pass. Doing the validation and the (test-only) warning here keeps them
# off the traced hot path. We cache the strategy implementation and call its
# route_tokens() directly in maybe_simulate_routing() rather than going through
# RoutingSimulator.simulate_routing(): that wrapper calls logger.warning_once(),
# which torch_tpu's tracer rejects inside the compiled MoE forward ("logging
# .Logger method not supported for non-export cases"). route_tokens() itself is
# pure tensor ops and traces cleanly.
#
# Exception: "uniform_random" is served by _uniform_random_routing() below
# instead of upstream's route_tokens(), so the sampled experts keep a data
# dependency on the real activations (see that docstring for why).
_UNIFORM_RANDOM = "uniform_random"

_SIMULATION_STRATEGY = None
_strategy_name = vllm_envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
if _strategy_name:
    from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import \
        RoutingSimulator
    _available = RoutingSimulator.get_available_strategies()
    if _strategy_name not in _available:
        raise ValueError(
            f"VLLM_MOE_ROUTING_SIMULATION_STRATEGY={_strategy_name!r} is not a "
            f"known routing strategy. Available strategies: {_available}.")
    _SIMULATION_STRATEGY = RoutingSimulator._routing_strategies[_strategy_name]
    logger.warning(
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY=%s: MoE expert routing is "
        "SIMULATED. This is a test/profiling-only feature and model output is "
        "meaningless. Never enable it for serving or accuracy evaluation.",
        _strategy_name)


def _uniform_random_routing(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform-random expert selection that stays anchored to the real input.

    Upstream's ``uniform_random`` strategy samples expert ids straight from
    ``torch.randint``, reading ``router_logits`` for shape only. With no data
    dependency on the activations, XLA is free to hoist and batch every layer's
    routing together -- in an xprof trace the per-layer ``topk`` ops collapse
    into one contiguous blob instead of interleaving with the real per-layer
    work, so the profile no longer represents balanced-MoE execution, which is
    the whole reason this feature exists.

    Adding ``router_logits.mean()`` to the noise creates that missing dependency
    edge. It is numerically inert for routing: the same scalar is added to every
    logit, and ``topk`` is invariant under a uniform shift, so the selected
    experts are exactly those of the unshifted noise. The mean (rather than a
    barrier) is deliberate -- it constrains scheduling without blocking
    optimization across the boundary the way ``optimization_barrier`` would.
    """
    noise = torch.rand_like(router_logits, dtype=torch.float32)
    # Anchor the noise to the activations. Uniform shift => same argsort =>
    # identical expert choice; the point is the graph edge, not the value.
    noise = noise + router_logits.float().mean()
    _, topk_ids = torch.topk(noise, k=topk, dim=-1)
    topk_weights = torch.ones((router_logits.shape[0], topk),
                              dtype=torch.float32,
                              device=router_logits.device)
    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)


def maybe_simulate_routing(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Optionally replace routing with a simulated strategy, for profiling.

    Ports upstream vLLM's routing simulation onto the TPU MoE path. When
    ``VLLM_MOE_ROUTING_SIMULATION_STRATEGY`` names a strategy (e.g.
    ``uniform_random``), expert assignment is drawn from that strategy's
    ``RoutingStrategy`` implementation instead of the trained gate. Uniform
    routing balances expert load so a single-device profile represents the whole
    EP mesh, at the cost of meaningless model output.

    Returns ``(topk_weights, topk_ids)`` when a strategy is set, else ``None``
    so the caller runs normal routing. Test/profiling only; a no-op by default.
    """
    if _SIMULATION_STRATEGY is None:
        return None
    if _strategy_name == _UNIFORM_RANDOM:
        return _uniform_random_routing(hidden_states, router_logits, topk)
    topk_weights, topk_ids = _SIMULATION_STRATEGY.route_tokens(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=topk,
        indices_type=torch.int32,
    )
    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)


def route(
    layer,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantization-independent routing entry point for the TPU MoE methods.

    Every ``apply_monolithic`` (fp8, unquantized, nvfp4, mxfp4, and the two
    compressed-tensors MoE methods) needs the same three-way routing
    decision: routing simulation (test-only override) takes
    priority if enabled, else a model-supplied ``custom_routing_function`` if
    present, else our own ``select_experts``. That decision was previously
    duplicated in each quant method; centralizing it here means the simulation
    hook -- and any future routing-path change -- lives in exactly one place.
    """
    topk = layer.moe_config.experts_per_token
    simulated = maybe_simulate_routing(hidden_states, router_logits, topk)
    if simulated is not None:
        return simulated
    # getattr with defaults, not direct attribute access: the compressed-tensors
    # MoE methods are reached with layers that need not define either attribute,
    # and previously guarded them exactly this way.
    custom_routing_fn = getattr(layer, "custom_routing_function", None)
    if custom_routing_fn is not None:
        return custom_routing_fn(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=topk,
            renormalize=layer.renormalize,
        )
    return select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk=topk,
        renormalize=layer.renormalize,
        scoring_fn=getattr(layer, "scoring_func", "softmax"),
        layer=layer,
        input_ids=input_ids,
    )


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
    0. Routing simulation, if ``VLLM_MOE_ROUTING_SIMULATION_STRATEGY`` is set
       (test/profiling-only override; replaces the routing output wholesale).
    1. DeepSeek-V4 hash routing (if `layer.hash_indices_table` is present).
    2. Grouped top-k routing (if `layer.use_grouped_topk` is enabled AND the group
       config actually requires grouped selection).
    3. Classic top-k routing with configurable scoring_fn (e.g. `sqrtsoftplus` for DeepSeek-V4)
       and optional `e_score_correction_bias`.
    """
    # Profiling-only override; a no-op unless a routing-simulation strategy is
    # set. Replaces the whole routing output, so return early.
    simulated = maybe_simulate_routing(hidden_states, router_logits, topk)
    if simulated is not None:
        return simulated

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

    use_grouped = (layer is not None and layer.use_grouped_topk
                   and layer.num_expert_group > 1
                   and layer.topk_group < layer.num_expert_group)
    if use_grouped:
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
        _, topk_ids = _topk(scores_for_choice, topk)
        topk_weights = scores.gather(-1, topk_ids.long())
    else:
        topk_weights, topk_ids = _topk(scores, topk)

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
