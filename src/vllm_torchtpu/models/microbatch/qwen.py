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
"""Direct token-window micro-batching instrumentation for Qwen3-Next / Qwen3.5.

Instruments ``Qwen3NextSparseMoeBlock``, ``Qwen3NextDecoderLayer``, and
``Qwen3NextModel`` (and ``Qwen3_5DecoderLayer`` / ``Qwen3_5Model``, which
inherit from them) to split large-token batches into ``N`` token windows using
explicit ``jax.lax.optimization_barrier`` fences (`pallas::microbatch_fence`
and `pallas::microbatch_fence_pair`).

Supported schedules (``TPU_MICROBATCH_SCHEDULE`` / ``MicroBatchConfig.schedule``):
- ``"stagewise"`` (default when ``USE_MOE_FUSED_EP_KERNEL=0``): splits each
  decoder stage (Attention, MoE) across windows; when ``merge_moe=True``,
  concatenates the routed-expert inputs across windows so ``fused_moe_gmm`` runs
  once per layer at full batch density.
- ``"layerwise"`` (also enabled by ``TPU_MICROBATCH_LAYERWISE=1``): runs each
  window through an entire decoder layer (Attention + MoE) before advancing to
  the next window.
- ``"staggered"`` (auto-selected when ``USE_MOE_FUSED_EP_KERNEL=1``): explicitly
  interleaves Window ``w-1``'s collective sub-stage (Attention TP All-Reduce,
  MoE DP/EP All-Gather, MoE Combine/All-Reduce) with Window ``w``'s compute
  sub-stage across coupled diamond fences.
- ``"attn_only"``: splits only the Attention stage across windows and runs the
  entire MLP / MoE block once on the full concatenated token batch.

Supported fence modes (``TPU_MICROBATCH_FENCE`` / ``MicroBatchConfig.fence``):
- ``"none"``: no ``optimization_barrier`` ops emitted.
- ``"window"``: 1-tensor ``microbatch_fence`` emitted per window at stage ends.
- ``"stage"``: coupled 2-tensor ``microbatch_fence_pair`` chain across windows
  at stage boundaries, blocking cross-window horizontal fusion.
- ``"substage"`` (default): coupled 2-tensor ``microbatch_fence_pair`` diamond
  fences between compute and collective sub-stages, blocking cross-window
  horizontal fusion while leaving Window ``w-1``'s collective and Window ``w``'s
  compute as concurrent sibling branches in the XLA HLO DAG.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm import forward_context as _fc_mod
from vllm.config import VllmConfig
from vllm.distributed import parallel_state as _ps
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.runner import moe_runner as _moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.fused_moe import fused_moe_gmm
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
    VllmCompressedTensorsW4ANMxfp4MoEMethod,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.microbatch.fence import (
    FenceMode,
    fence_windows,
    microbatch_fence,
    microbatch_fence_pair,
    parse_fence_mode,
)
from vllm_torchtpu.models.microbatch.metadata import (
    bind_step_inputs_for_trace_alignment,
    slice_forward_metadata,
    slice_padding_state,
)

logger = init_logger(__name__)


class ScheduleMode(StrEnum):
    """Execution schedule across token windows within a decoder layer."""

    STAGEWISE = "stagewise"
    LAYERWISE = "layerwise"
    STAGGERED = "staggered"
    ATTN_ONLY = "attn_only"


def parse_schedule_mode(
    schedule: str | ScheduleMode | None,
    layerwise: bool = False,
) -> ScheduleMode:
    """Resolve ``schedule`` and ``layerwise`` into a ``ScheduleMode``."""
    if isinstance(schedule, ScheduleMode):
        return schedule
    if schedule is None or schedule.strip() == "":
        if layerwise:
            return ScheduleMode.LAYERWISE
        if envs.USE_MOE_FUSED_EP_KERNEL:
            return ScheduleMode.STAGGERED
        return ScheduleMode.STAGEWISE
    normalized = schedule.strip().lower()
    try:
        return ScheduleMode(normalized)
    except ValueError as exc:
        valid = [m.value for m in ScheduleMode]
        raise ValueError(
            f"Invalid microbatch schedule {schedule!r}. Valid options: {valid}"
        ) from exc


@dataclass(frozen=True)
class MicroBatchConfig:
    """Configuration for Qwen token-window micro-batching."""

    factor: int = 1
    min_tokens: int = 1024
    merge_moe: bool = True
    layerwise: bool = False
    schedule: ScheduleMode | str = ""
    fence: FenceMode | str = FenceMode.SUBSTAGE

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise ValueError(f"MicroBatchConfig.factor must be >= 1, got {self.factor}")
        if self.min_tokens < 0:
            raise ValueError(
                f"MicroBatchConfig.min_tokens must be >= 0, got {self.min_tokens}"
            )
        # Validate schedule and fence eagerly.
        _ = self.resolved_schedule
        _ = self.resolved_fence

    @property
    def resolved_schedule(self) -> ScheduleMode:
        return parse_schedule_mode(self.schedule, self.layerwise)

    @property
    def resolved_fence(self) -> FenceMode:
        return parse_fence_mode(self.fence)

    @classmethod
    def from_envs(cls) -> "MicroBatchConfig":
        return cls(
            factor=envs.TPU_MICROBATCH_FACTOR,
            min_tokens=envs.TPU_MICROBATCH_MIN_TOKENS,
            merge_moe=envs.TPU_MICROBATCH_MERGE_MOE,
            layerwise=envs.TPU_MICROBATCH_LAYERWISE,
            schedule=envs.TPU_MICROBATCH_SCHEDULE,
            fence=envs.TPU_MICROBATCH_FENCE,
        )


def _split_tokens(x: torch.Tensor, factor: int) -> list[torch.Tensor]:
    """Split ``[num_tokens, ...]`` into ``factor`` contiguous views."""
    return list(x.view(factor, -1, *x.shape[1:]).unbind(0))


def _split_positions(
    positions: torch.Tensor,
    factor: int,
) -> list[torch.Tensor]:
    """Split 1D ``[num_tokens]`` or 2D MRoPE ``[3, num_tokens]`` positions."""
    if positions.ndim == 2:
        return list(positions.view(positions.shape[0], factor, -1).unbind(1))
    return list(positions.view(factor, -1, *positions.shape[1:]).unbind(0))


@contextmanager
def _window_context(
    forward_context: Any,
    window_attn_metadata: Any,
    window_padding_state: token_padding.TokenPaddingState | None,
) -> Iterator[None]:
    """Temporarily bind ``attn_metadata`` and ``TokenPaddingState`` for one window."""
    saved_md = forward_context.attn_metadata if forward_context is not None else None
    saved_pad = token_padding.get_padding_state()
    if forward_context is not None:
        forward_context.attn_metadata = window_attn_metadata
    token_padding.set_padding_state(window_padding_state)
    try:
        yield
    finally:
        if forward_context is not None:
            forward_context.attn_metadata = saved_md
        token_padding.set_padding_state(saved_pad)


# ============================================================================
# Qwen3NextSparseMoeBlock Direct Sub-Stage Instrumentation
# ============================================================================


def _get_moe_runner(experts: Any) -> MoERunner | None:
    """Resolve the ``MoERunner`` instance from ``moe_block.experts``."""
    if isinstance(experts, MoERunner):
        return experts
    runner = getattr(experts, "runner", None)
    if isinstance(runner, MoERunner):
        return runner
    return None


def _maybe_sp_context(runner: MoERunner | None):
    """Enter ``runner._sequence_parallel_context()`` when forward context is active."""
    if (
        runner is not None
        and hasattr(runner, "_sequence_parallel_context")
        and _fc_mod.is_forward_context_available()
    ):
        return runner._sequence_parallel_context()
    return nullcontext()


def _apply_layer_scale(
    hidden_states: torch.Tensor,
    scale_param: torch.Tensor,
) -> torch.Tensor:
    """Apply ``Qwen3NextDecoderLayer`` / ``Qwen3_5DecoderLayer`` layer scale."""
    scale = scale_param.to(hidden_states.dtype)
    if scale.ndim == 3:
        if hidden_states.ndim == 2:
            return hidden_states * (scale[0] + 1)
        return hidden_states * (scale + 1)
    return hidden_states * scale


def _is_fused_ep_moe(moe_block: nn.Module | None) -> bool:
    """Return True if ``moe_block`` uses the monolithic fused EP MoE kernel."""
    if moe_block is None:
        return False
    if hasattr(moe_block, "_moe_monolithic_fn"):
        return True
    experts = getattr(moe_block, "experts", None)
    if experts is None:
        return False
    runner = _get_moe_runner(experts)
    if runner is None:
        return False
    gate = getattr(runner, "gate", None) or getattr(moe_block, "gate", None)
    if gate is None:
        return False
    moe_config = getattr(runner, "moe_config", None)
    if (
        getattr(moe_block, "is_sequence_parallel", False)
        or getattr(experts, "is_sequence_parallel", False)
        or getattr(moe_config, "is_sequence_parallel", False)
    ):
        return False
    if (
        getattr(experts, "has_ici_cp_collectives", False)
        or getattr(moe_config, "pcp_size", 1) > 1
    ):
        return False
    routed = getattr(runner, "routed_experts", getattr(experts, "routed_experts", None))
    quant_method = getattr(routed, "quant_method", None)
    if quant_method is None:
        return False
    if getattr(quant_method, "_use_fused_ep", False):
        return True
    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

    return bool(fused_moe_ep_supported(quant_method))


def can_split_moe_stages(moe_block: nn.Module | None) -> bool:
    """Return True if ``moe_block`` supports decomposed pre/kernel/post MoE stages."""
    if moe_block is None:
        return False
    if hasattr(moe_block, "_moe_kernel_fn"):
        return True
    experts = getattr(moe_block, "experts", None)
    if experts is None:
        return False
    runner = _get_moe_runner(experts)
    if runner is None:
        return False
    gate = getattr(runner, "gate", None) or getattr(moe_block, "gate", None)
    if gate is None:
        return False
    moe_config = getattr(runner, "moe_config", None)
    if (
        getattr(moe_block, "is_sequence_parallel", False)
        or getattr(experts, "is_sequence_parallel", False)
        or getattr(moe_config, "is_sequence_parallel", False)
    ):
        return False
    if (
        getattr(experts, "has_ici_cp_collectives", False)
        or getattr(moe_config, "pcp_size", 1) > 1
    ):
        return False
    routed = getattr(runner, "routed_experts", getattr(experts, "routed_experts", None))
    quant_method = getattr(routed, "quant_method", None)
    if quant_method is None:
        return False
    if getattr(quant_method, "_use_fused_ep", False):
        return False
    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

    if fused_moe_ep_supported(quant_method):
        return False
    has_weights = (hasattr(routed, "w13_weight") and hasattr(routed, "w2_weight")) or (
        hasattr(routed, "w13_weight_packed") and hasattr(routed, "w2_weight_packed")
    )
    return bool(has_weights)


def _moe_forward_pre_compute(
    self: Any,
    hidden_states: torch.Tensor,
) -> tuple[
    torch.Tensor,
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Sub-stage 2a (TensorCore compute): router gate + shared expert MLP."""
    orig_shape = hidden_states.shape
    hidden_states_2d = hidden_states.view(-1, hidden_states.shape[-1])

    experts = getattr(self, "experts", None)
    runner = _get_moe_runner(experts)
    routed = getattr(runner, "routed_experts", getattr(experts, "routed_experts", None))
    if routed is not None and hasattr(routed, "_ensure_moe_quant_config_init"):
        routed._ensure_moe_quant_config_init()

    shared_experts_input: torch.Tensor | None = hidden_states_2d
    if runner is not None and hasattr(runner, "apply_routed_input_transform"):
        hidden_states_2d, shared_experts_input = runner.apply_routed_input_transform(
            hidden_states_2d
        )
    if runner is not None and hasattr(runner, "_maybe_pad_hidden_states"):
        hidden_states_2d, _, _ = runner._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states_2d,
        )

    if runner is not None and getattr(runner, "_fse_fuse_gate", False):
        runner._maybe_fuse_gate_weights()
        combined_gate_weight = runner._combined_gate_weight
        assert combined_gate_weight is not None
        router_logits = F.linear(hidden_states_2d, combined_gate_weight)
    else:
        gate: Any = getattr(runner, "gate", None) or getattr(self, "gate", None)
        assert gate is not None
        router_logits, _ = gate(hidden_states_2d)

    # Enter _sequence_parallel_context() immediately after gate (before
    # shared_experts) to match MoERunner._forward_impl's exact FX graph
    # placeholder order (specifically dp_metadata.num_tokens_across_dp_cpu).
    with _maybe_sp_context(runner):
        inner_shared_output: torch.Tensor | None = None
        outer_shared_output: torch.Tensor | None = None
        shared_experts = getattr(
            runner, "shared_experts", getattr(experts, "shared_experts", None)
        )
        if shared_experts is not None:
            sh_in = (
                shared_experts_input
                if shared_experts_input is not None
                else hidden_states_2d
            )
            inner_layer = getattr(shared_experts, "_layer", None)
            if inner_layer is not None:
                inner_shared_output = inner_layer(sh_in)
            else:
                inner_shared_output = shared_experts(sh_in)
                if hasattr(shared_experts, "_output"):
                    shared_experts._output = None

        if (
            getattr(self, "replicate_shared_expert", False)
            and getattr(self, "shared_expert", None) is not None
        ):
            outer_shared_output = self.shared_expert(hidden_states_2d)
        elif (
            shared_experts is None
            and getattr(self, "n_shared_experts", 0) > 0
            and getattr(self, "shared_expert", None) is not None
        ):
            shared_out: torch.Tensor = self.shared_expert(hidden_states_2d)
            if (
                getattr(self, "shared_expert_gate", None) is not None
                and getattr(self.shared_expert, "expert_gate", None) is None
            ):
                shared_out = (
                    F.sigmoid(self.shared_expert_gate(hidden_states_2d)) * shared_out
                )
            outer_shared_output = shared_out

    return (
        hidden_states_2d,
        orig_shape,
        router_logits,
        inner_shared_output,
        outer_shared_output,
    )


def _moe_forward_pre_dispatch_route(
    self: Any,
    pre_compute_state: tuple[
        torch.Tensor,
        tuple[int, ...],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ],
) -> tuple[
    tuple[int, ...],
    torch.Tensor | None,
    torch.Tensor | None,
    bool,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Sub-stage 2b (DP/EP All-Gather collective + top-k routing)."""
    (
        hidden_states_2d,
        orig_shape,
        router_logits,
        inner_shared_output,
        outer_shared_output,
    ) = pre_compute_state

    experts = getattr(self, "experts", None)
    if experts is None:
        # Synthetic test MoE block without a vLLM FusedMoE ``experts`` submodule.
        topk_weights, topk_ids = self.route(hidden_states_2d, router_logits)
        return (
            orig_shape,
            inner_shared_output,
            outer_shared_output,
            False,
            None,
            hidden_states_2d,
            topk_weights,
            topk_ids,
        )

    runner = _get_moe_runner(experts)
    dispatch_owner = (
        runner if runner is not None and hasattr(runner, "_maybe_dispatch") else experts
    )
    with _maybe_sp_context(runner):
        dispatch_res = dispatch_owner._maybe_dispatch(hidden_states_2d, router_logits)
        if len(dispatch_res) == 3:
            dispatched_x, dispatched_logits, should_attempt_combine = dispatch_res
        else:
            dispatched_x, dispatched_logits = dispatch_res
            should_attempt_combine = True

        routed = getattr(
            runner, "routed_experts", getattr(experts, "routed_experts", None)
        )
        topk_weights, topk_ids = moe_routing.route(
            routed,
            dispatched_x,
            dispatched_logits,
        )

        if (
            getattr(routed, "zero_expert_num", None) is not None
            and routed.zero_expert_num > 0
            and hasattr(_moe_runner_mod, "zero_expert_oot")
        ):
            topk_ids, topk_weights, zero_expert_result = (
                _moe_runner_mod.zero_expert_oot(
                    routed,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    hidden_states=dispatched_x,
                )
            )
        else:
            zero_expert_result = None

        quant_method = getattr(routed, "quant_method", None)
        is_mxfp4 = isinstance(quant_method, VllmCompressedTensorsW4ANMxfp4MoEMethod)
        if is_mxfp4:
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = topk_weights.to(dispatched_x.dtype)

        if envs.TPU_MOE_SKIP_PADDED_TOKENS or is_mxfp4:
            is_local_tensor = dispatched_x.shape[0] == hidden_states_2d.shape[0]
            topk_ids, topk_weights = token_padding.zero_routing_weights_for_padding(
                topk_ids,
                topk_weights,
                is_local_tensor=is_local_tensor,
            )

    return (
        orig_shape,
        inner_shared_output,
        outer_shared_output,
        should_attempt_combine,
        zero_expert_result,
        dispatched_x,
        topk_weights,
        topk_ids,
    )


def _moe_forward_pre(
    self: Any,
    hidden_states: torch.Tensor,
) -> tuple[
    tuple[int, ...],
    torch.Tensor | None,
    torch.Tensor | None,
    bool,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Combined MoE pre-stage (gate + shared expert + dispatch + routing)."""
    return self.forward_moe_pre_dispatch_route(
        self.forward_moe_pre_compute(hidden_states)
    )


def _moe_forward_kernel(
    self: Any,
    dispatched_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2c: routed-expert grouped-matmul kernel (``fused_moe_gmm``)."""
    custom_kernel = getattr(self, "_moe_kernel_fn", None)
    if custom_kernel is not None:
        return custom_kernel(dispatched_x, topk_weights, topk_ids)

    experts = self.experts
    runner = _get_moe_runner(experts)
    routed: Any = getattr(
        runner, "routed_experts", getattr(experts, "routed_experts", None)
    )
    assert routed is not None
    quant_method = routed.quant_method
    activation_str = getattr(quant_method, "_tpu_activation_str", None)
    if activation_str is None:
        from vllm_torchtpu.layers.adapter.fused_moe import get_fused_moe_activation

        activation_str = get_fused_moe_activation(routed.activation, routed.moe_config)

    w1 = getattr(routed, "w13_weight", getattr(routed, "w13_weight_packed", None))
    w2 = getattr(routed, "w2_weight", getattr(routed, "w2_weight_packed", None))
    w1_scale = getattr(
        routed,
        "w13_weight_scale_inv",
        getattr(routed, "w13_weight_scale", None),
    )
    w2_scale = getattr(
        routed,
        "w2_weight_scale_inv",
        getattr(routed, "w2_weight_scale", None),
    )
    kwargs: dict[str, Any] = {
        "hidden_states": dispatched_x,
        "w1": w1,
        "w2": w2,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1_bias": getattr(routed, "w13_bias", None),
        "w2_bias": getattr(routed, "w2_bias", None),
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "experts_start": getattr(routed, "_experts_start", None),
        "topk": routed.moe_config.experts_per_token,
        "activation": activation_str,
    }
    if hasattr(routed, "_rhs_quant_dtype"):
        kwargs["rhs_quant_dtype"] = routed._rhs_quant_dtype
    if isinstance(quant_method, VllmCompressedTensorsW4ANMxfp4MoEMethod):
        kwargs["skip_padded_tokens"] = True

    return fused_moe_gmm(**kwargs)


def _moe_forward_monolithic(
    self: Any,
    hidden_states_2d: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2c (monolithic fused EP MoE): invoke ``routed_experts.forward_monolithic``."""
    custom_monolithic = getattr(self, "_moe_monolithic_fn", None)
    if custom_monolithic is not None:
        return custom_monolithic(hidden_states_2d, router_logits)

    experts = self.experts
    runner = _get_moe_runner(experts)
    routed: Any = getattr(
        runner, "routed_experts", getattr(experts, "routed_experts", None)
    )
    assert routed is not None
    if hasattr(routed, "_ensure_moe_quant_config_init"):
        routed._ensure_moe_quant_config_init()
    with _maybe_sp_context(runner):
        return routed.forward_monolithic(
            x=hidden_states_2d,
            router_logits=router_logits,
            input_ids=None,
        )


def _moe_forward_post(
    self: Any,
    pre_state: tuple[
        tuple[int, ...],
        torch.Tensor | None,
        torch.Tensor | None,
        bool,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    routed_out: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2d: DP/EP Combine + shared-expert sum + TP All-Reduce."""
    (
        orig_shape,
        inner_shared_output,
        outer_shared_output,
        should_attempt_combine,
        zero_expert_result,
        _,
        _,
        _,
    ) = pre_state

    experts = getattr(self, "experts", None)
    if experts is None:
        final_hidden_states = routed_out
        if inner_shared_output is not None:
            final_hidden_states = final_hidden_states + inner_shared_output
        if outer_shared_output is not None:
            final_hidden_states = final_hidden_states + outer_shared_output
        if getattr(self, "reduce_fn", None) is not None:
            final_hidden_states = self.reduce_fn(final_hidden_states)
        return final_hidden_states.view(orig_shape)

    runner = _get_moe_runner(experts)
    assert runner is not None
    if zero_expert_result is not None:
        experts.zero_expert_result = zero_expert_result

    combine_owner: Any = runner if hasattr(runner, "_maybe_combine") else experts
    with _maybe_sp_context(runner):
        combined: Any = combine_owner._maybe_combine(inner_shared_output, routed_out)
    if isinstance(combined, tuple):
        inner_shared_output, routed_out = combined
    else:
        routed_out = combined

    fused_output_is_reduced = bool(getattr(runner, "_fused_output_is_reduced", False))

    routed_out, fused_output_is_reduced = (
        runner._maybe_reduce_routed_output_before_transform(
            routed_out,
            fused_output_is_reduced,
        )
    )

    inner_shared_output = runner._maybe_reduce_shared_expert_output(
        inner_shared_output,
        fused_output_is_reduced,
    )

    inner_shared_output, routed_out = runner._maybe_apply_routed_scale_to_output(
        inner_shared_output,
        routed_out,
    )

    fused_output = runner.apply_routed_output_transform(routed_out)
    if inner_shared_output is not None:
        output = inner_shared_output + fused_output
    else:
        output = fused_output

    final_hidden_states = runner._maybe_reduce_final_output(
        output,
        None,
        fused_output_is_reduced,
    )

    zero_owner = runner if hasattr(runner, "_maybe_add_zero_expert_output") else experts
    if hasattr(zero_owner, "_maybe_add_zero_expert_output"):
        final_hidden_states = zero_owner._maybe_add_zero_expert_output(
            final_hidden_states
        )

    if outer_shared_output is not None:
        final_hidden_states = final_hidden_states + outer_shared_output
    return final_hidden_states.view(orig_shape)


# ============================================================================
# Qwen3NextDecoderLayer Direct Stage & Sub-Stage Instrumentation
# ============================================================================


def _get_attn_module_and_out_proj(layer: nn.Module) -> tuple[Any, Any]:
    """Return ``(attn_module, out_proj)`` for a Qwen3-Next / Qwen3.5 decoder layer."""
    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention" or hasattr(layer, "linear_attn"):
        attn_mod = getattr(layer, "linear_attn", None)
        out_proj = getattr(attn_mod, "out_proj", None)
        return attn_mod, out_proj
    if layer_type == "full_attention" or hasattr(layer, "self_attn"):
        attn_mod = getattr(layer, "self_attn", None)
        out_proj = getattr(attn_mod, "o_proj", None)
        return attn_mod, out_proj
    if hasattr(layer, "attn"):
        attn_mod = getattr(layer, "attn", None)
        out_proj = getattr(attn_mod, "out_proj", getattr(attn_mod, "o_proj", None))
        return attn_mod, out_proj
    return None, None


def _layer_forward_attn_compute(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any, bool]:
    """Sub-stage 1a: input_layernorm + QKV/GDN/RPA + local output projection."""
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    attn_mod, out_proj = _get_attn_module_and_out_proj(self)
    orig_reduce = bool(getattr(out_proj, "reduce_results", False))
    if out_proj is not None and hasattr(out_proj, "reduce_results"):
        out_proj.reduce_results = False
    try:
        layer_type = getattr(self, "layer_type", None)
        if layer_type == "linear_attention" or hasattr(self, "linear_attn"):
            attn_local = self.linear_attn(hidden_states=hidden_states)
        elif layer_type == "full_attention" or hasattr(self, "self_attn"):
            attn_local = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            attn_local = attn_mod(hidden_states, positions)
        if isinstance(attn_local, tuple):
            attn_local = attn_local[0]
    finally:
        if out_proj is not None and hasattr(out_proj, "reduce_results"):
            out_proj.reduce_results = orig_reduce

    return attn_local, residual, out_proj, orig_reduce


def _layer_forward_attn_reduce(
    self: Any,
    attn_compute_state: tuple[torch.Tensor, torch.Tensor, Any, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sub-stage 1b: Attention TP All-Reduce + optional layer_scale."""
    attn_local, residual, out_proj, orig_reduce = attn_compute_state
    if orig_reduce and out_proj is not None:
        reduce_fn = getattr(out_proj, "reduce_fn", None)
        if reduce_fn is not None:
            hidden_states = reduce_fn(attn_local)
        elif getattr(out_proj, "tp_size", 1) > 1:
            hidden_states = tensor_model_parallel_all_reduce(attn_local)
        else:
            hidden_states = attn_local
    else:
        hidden_states = attn_local

    if getattr(self, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states, self.attn_layer_scale)
    return hidden_states, residual


def _layer_forward_attn(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: full attention stage (compute + TP All-Reduce)."""
    return self.forward_attn_reduce(
        self.forward_attn_compute(hidden_states, residual, positions)
    )


def _layer_forward_moe_pre_compute(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    """Sub-stage 2a: post_attention_layernorm + gate + shared_expert compute."""
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    moe_pre_compute = self.mlp.forward_moe_pre_compute(hidden_states)
    return moe_pre_compute, residual


def _layer_forward_moe_pre_dispatch_route(
    self: Any,
    layer_pre_compute_state: tuple[Any, torch.Tensor],
) -> tuple[Any, torch.Tensor]:
    """Sub-stage 2b: MoE DP/EP All-Gather dispatch + top-k routing."""
    moe_pre_compute, residual = layer_pre_compute_state
    moe_pre_state = self.mlp.forward_moe_pre_dispatch_route(moe_pre_compute)
    return moe_pre_state, residual


def _layer_forward_moe_pre(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    """Combined MoE pre-stage on decoder layer."""
    return self.forward_moe_pre_dispatch_route(
        self.forward_moe_pre_compute(hidden_states, residual)
    )


def _layer_forward_moe_kernel(
    self: Any,
    dispatched_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2c: routed-expert grouped-matmul kernel."""
    return self.mlp.forward_moe_kernel(dispatched_x, topk_weights, topk_ids)


def _layer_forward_moe_monolithic(
    self: Any,
    hidden_states_2d: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Sub-stage 2c (monolithic fused EP MoE): invoke ``self.mlp.forward_moe_monolithic``."""
    return self.mlp.forward_moe_monolithic(hidden_states_2d, router_logits)


def _layer_forward_moe_post(
    self: Any,
    layer_pre_state: tuple[Any, torch.Tensor],
    routed_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sub-stage 2d: MoE combine + TP All-Reduce + optional ffn_layer_scale."""
    moe_pre_state, residual = layer_pre_state
    hidden_states = self.mlp.forward_moe_post(moe_pre_state, routed_out)
    if getattr(self, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states, self.ffn_layer_scale)
    return hidden_states, residual


def _layer_forward_mlp(
    self: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 2 fallback for dense MLP or non-splittable MoE blocks."""
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    if getattr(self, "layer_scale", False):
        hidden_states = _apply_layer_scale(hidden_states, self.ffn_layer_scale)
    return hidden_states, residual


# ============================================================================
# Qwen3NextModel / Qwen3_5Model Micro-Batched Forward Schedules
# ============================================================================


def _run_merged_or_split_moe_kernels(
    layer: Any,
    layer_pre_states: list[tuple[Any, torch.Tensor]],
    *,
    merge_moe: bool,
    fence: FenceMode,
) -> list[torch.Tensor]:
    """Execute ``layer.forward_moe_kernel`` either merged once or per window."""
    num_windows = len(layer_pre_states)
    dispatched_xs = [state[0][5] for state in layer_pre_states]
    topk_weights_list = [state[0][6] for state in layer_pre_states]
    topk_ids_list = [state[0][7] for state in layer_pre_states]

    if merge_moe:
        if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE) and num_windows > 1:
            dispatched_xs = fence_windows(dispatched_xs, fence)
        elif fence == FenceMode.WINDOW:
            dispatched_xs = [microbatch_fence(x) for x in dispatched_xs]
        cat_x = torch.cat(dispatched_xs, dim=0)
        cat_weights = torch.cat(topk_weights_list, dim=0)
        cat_ids = torch.cat(topk_ids_list, dim=0)
        merged_out = layer.forward_moe_kernel(cat_x, cat_weights, cat_ids)
        if fence != FenceMode.NONE:
            merged_out = microbatch_fence(merged_out)
        return list(merged_out.view(num_windows, -1, merged_out.shape[-1]).unbind(0))

    routed_outs: list[torch.Tensor] = []
    for w in range(num_windows):
        dx = dispatched_xs[w]
        if w > 0 and fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
            routed_outs[w - 1], dx = microbatch_fence_pair(routed_outs[w - 1], dx)
        out_w = layer.forward_moe_kernel(
            dx,
            topk_weights_list[w],
            topk_ids_list[w],
        )
        if fence == FenceMode.WINDOW:
            out_w = microbatch_fence(out_w)
        routed_outs.append(out_w)
    return routed_outs


def _run_attn_only_mlp(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    fc: Any,
    orig_md: Any,
    orig_padding: token_padding.TokenPaddingState | None,
    *,
    fence: FenceMode,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Concatenate windows after Attention, run ``layer.forward_mlp`` once, and split back."""
    num_windows = len(hs_windows)
    hs_windows = fence_windows(hs_windows, fence)
    full_hs = torch.cat(hs_windows, dim=0)
    res_list = [r for r in res_windows if r is not None]
    full_res = torch.cat(res_list, dim=0) if len(res_list) == num_windows else None

    with _window_context(fc, orig_md, orig_padding):
        full_hs, full_res = layer.forward_mlp(full_hs, full_res)

    if fence != FenceMode.NONE:
        full_hs = microbatch_fence(full_hs)
        if full_res is not None:
            full_res = microbatch_fence(full_res)

    hs_windows = _split_tokens(full_hs, num_windows)
    res_windows = (
        [None] * num_windows
        if full_res is None
        else [t for t in _split_tokens(full_res, num_windows)]
    )
    return hs_windows, res_windows


def _run_layer_substage_staggered(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    pos_windows: list[torch.Tensor],
    md_windows: list[Any],
    pad_windows: list[token_padding.TokenPaddingState | None],
    fc: Any,
    *,
    merge_moe: bool,
    fence: FenceMode,
    orig_md: Any = None,
    orig_padding: token_padding.TokenPaddingState | None = None,
    attn_only: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Run one decoder layer with diamond-fenced sub-stage compute/collective overlap.

    Interleaves Window ``w-1``'s collective sub-stage with Window ``w``'s
    compute sub-stage across ``microbatch_fence_pair`` barriers:
      1. ``w0`` ``attn_compute``
      2. For ``w = 1..N-1``:
         ``fence_pair(attn_local[w-1], hs[w])`` ->
         (``w-1`` ``attn_reduce`` || ``w`` ``attn_compute``)
      3. If MoE is splittable:
         ``fence_pair(attn_local[N-1], hs[0])`` ->
         (``N-1`` ``attn_reduce`` || ``w0`` ``moe_pre_compute``)
      4. For ``w = 1..N-1``:
         ``fence_pair(router_logits[w-1], hs[w])`` ->
         (``w-1`` ``moe_pre_dispatch_route`` || ``w`` ``moe_pre_compute``)
      5. ``N-1`` ``moe_pre_dispatch_route``
      6. ``moe_kernel`` (merged across windows if ``merge_moe=True``, else per window)
      7. ``moe_post`` across windows, coupling ``hs[0]`` with ``routed_outs[1]``
         so Layer ``l+1``'s ``w0`` ``attn_compute`` can overlap ``w_{1..N-1}``'s
         ``moe_post`` collectives.
    """
    num_windows = len(hs_windows)
    use_stage_pair_fence = fence in (FenceMode.STAGE, FenceMode.SUBSTAGE)
    use_substage_pair_fence = fence == FenceMode.SUBSTAGE

    # --- Sub-stage 1a (w=0): Attention compute ---
    with _window_context(fc, md_windows[0], pad_windows[0]):
        attn_state_prev = layer.forward_attn_compute(
            hs_windows[0],
            res_windows[0],
            pos_windows[0],
        )

    # --- Sub-stage 1b (w-1) || Sub-stage 1a (w) for w=1..N-1 ---
    for w in range(1, num_windows):
        attn_local_prev, res_prev, out_proj_prev, orig_reduce_prev = attn_state_prev
        if use_substage_pair_fence:
            attn_local_prev, hs_windows[w] = microbatch_fence_pair(
                attn_local_prev,
                hs_windows[w],
            )
        elif fence == FenceMode.WINDOW:
            attn_local_prev = microbatch_fence(attn_local_prev)

        with _window_context(fc, md_windows[w - 1], pad_windows[w - 1]):
            hs_windows[w - 1], res_windows[w - 1] = layer.forward_attn_reduce(
                (attn_local_prev, res_prev, out_proj_prev, orig_reduce_prev)
            )

        with _window_context(fc, md_windows[w], pad_windows[w]):
            attn_state_prev = layer.forward_attn_compute(
                hs_windows[w],
                res_windows[w],
                pos_windows[w],
            )

    mlp = getattr(layer, "mlp", None)
    is_fused_ep = _is_fused_ep_moe(mlp)
    if attn_only:
        attn_local_last, res_last, out_proj_last, orig_reduce_last = attn_state_prev
        if fence == FenceMode.WINDOW:
            attn_local_last = microbatch_fence(attn_local_last)
        with _window_context(
            fc, md_windows[num_windows - 1], pad_windows[num_windows - 1]
        ):
            hs_windows[num_windows - 1], res_windows[num_windows - 1] = (
                layer.forward_attn_reduce(
                    (
                        attn_local_last,
                        res_last,
                        out_proj_last,
                        orig_reduce_last,
                    )
                )
            )
        return _run_attn_only_mlp(
            layer,
            hs_windows,
            res_windows,
            fc,
            orig_md,
            orig_padding,
            fence=fence,
        )

    if not can_split_moe_stages(mlp) and not (is_fused_ep and merge_moe):
        # Complete Window N-1's attention reduce, then run dense MLP per window.
        attn_local_last, res_last, out_proj_last, orig_reduce_last = attn_state_prev
        if fence == FenceMode.WINDOW:
            attn_local_last = microbatch_fence(attn_local_last)
        with _window_context(
            fc, md_windows[num_windows - 1], pad_windows[num_windows - 1]
        ):
            hs_windows[num_windows - 1], res_windows[num_windows - 1] = (
                layer.forward_attn_reduce(
                    (
                        attn_local_last,
                        res_last,
                        out_proj_last,
                        orig_reduce_last,
                    )
                )
            )
        hs_windows = fence_windows(hs_windows, fence)
        for w in range(num_windows):
            if w > 0 and use_stage_pair_fence:
                hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                    hs_windows[w - 1],
                    hs_windows[w],
                )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                hs_windows[w], res_windows[w] = layer.forward_mlp(
                    hs_windows[w],
                    res_windows[w],
                )
        hs_windows = fence_windows(hs_windows, fence)
        return hs_windows, res_windows

    # --- Overlap Window N-1 attn_reduce with Window 0 moe_pre_compute ---
    attn_local_last, res_last, out_proj_last, orig_reduce_last = attn_state_prev
    if use_stage_pair_fence and num_windows > 1:
        attn_local_last, hs_windows[0] = microbatch_fence_pair(
            attn_local_last,
            hs_windows[0],
        )
    elif fence != FenceMode.NONE:
        attn_local_last = microbatch_fence(attn_local_last)

    with _window_context(fc, md_windows[num_windows - 1], pad_windows[num_windows - 1]):
        hs_windows[num_windows - 1], res_windows[num_windows - 1] = (
            layer.forward_attn_reduce(
                (
                    attn_local_last,
                    res_last,
                    out_proj_last,
                    orig_reduce_last,
                )
            )
        )

    with _window_context(fc, md_windows[0], pad_windows[0]):
        pre_compute_prev = layer.forward_moe_pre_compute(
            hs_windows[0],
            res_windows[0],
        )

    if is_fused_ep:
        pre_computes: list[tuple[Any, torch.Tensor]] = []
        for w in range(1, num_windows):
            moe_pc_prev, res_pc_prev = pre_compute_prev
            (
                hs_2d_prev,
                orig_shape_prev,
                logits_prev,
                inner_sh_prev,
                outer_sh_prev,
            ) = moe_pc_prev
            if use_substage_pair_fence:
                logits_prev, hs_windows[w] = microbatch_fence_pair(
                    logits_prev,
                    hs_windows[w],
                )
                if inner_sh_prev is not None:
                    inner_sh_prev, hs_windows[w] = microbatch_fence_pair(
                        inner_sh_prev,
                        hs_windows[w],
                    )
                if outer_sh_prev is not None:
                    outer_sh_prev, hs_windows[w] = microbatch_fence_pair(
                        outer_sh_prev,
                        hs_windows[w],
                    )
            elif fence == FenceMode.WINDOW:
                logits_prev = microbatch_fence(logits_prev)
                if inner_sh_prev is not None:
                    inner_sh_prev = microbatch_fence(inner_sh_prev)
                if outer_sh_prev is not None:
                    outer_sh_prev = microbatch_fence(outer_sh_prev)
            pre_computes.append(
                (
                    (
                        hs_2d_prev,
                        orig_shape_prev,
                        logits_prev,
                        inner_sh_prev,
                        outer_sh_prev,
                    ),
                    res_pc_prev,
                )
            )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                pre_compute_prev = layer.forward_moe_pre_compute(
                    hs_windows[w],
                    res_windows[w],
                )
        pre_computes.append(pre_compute_prev)

        hs_2d_list = [pc[0][0] for pc in pre_computes]
        logits_list = [pc[0][2] for pc in pre_computes]
        if use_stage_pair_fence and num_windows > 1:
            hs_2d_list = fence_windows(hs_2d_list, fence)
            logits_list = fence_windows(logits_list, fence)
        cat_hs_2d = torch.cat(hs_2d_list, dim=0)
        cat_logits = torch.cat(logits_list, dim=0)

        # Couple cat_logits with Window 0's inner_shared_output so Window 0's
        # shared-expert TP all-reduce on SparseCore overlaps with the 1x
        # monolithic fused EP MoE kernel on TensorCore.
        inner_sh_0 = pre_computes[0][0][3]
        if use_stage_pair_fence and inner_sh_0 is not None:
            cat_logits, inner_sh_0 = microbatch_fence_pair(cat_logits, inner_sh_0)
            pc0, res0 = pre_computes[0]
            pre_computes[0] = (
                (pc0[0], pc0[1], pc0[2], inner_sh_0, pc0[4]),
                res0,
            )

        with _window_context(fc, orig_md, orig_padding):
            merged_routed_out = layer.forward_moe_monolithic(cat_hs_2d, cat_logits)
        routed_outs = list(
            merged_routed_out.view(
                num_windows, -1, merged_routed_out.shape[-1]
            ).unbind(0)
        )
        dummy_idx = torch.empty(0, dtype=torch.int32, device=cat_hs_2d.device)
        layer_pre_states = [
            (
                (
                    pc[0][1],
                    pc[0][3],
                    pc[0][4],
                    False,
                    None,
                    pc[0][0],
                    pc[0][2],
                    dummy_idx,
                ),
                pc[1],
            )
            for pc in pre_computes
        ]
    else:
        # --- Sub-stage 2b (w-1) || Sub-stage 2a (w) for w=1..N-1 ---
        layer_pre_states = []
        for w in range(1, num_windows):
            moe_pc_prev, res_pc_prev = pre_compute_prev
            (
                hs_2d_prev,
                orig_shape_prev,
                logits_prev,
                inner_sh_prev,
                outer_sh_prev,
            ) = moe_pc_prev
            if use_substage_pair_fence:
                logits_prev, hs_windows[w] = microbatch_fence_pair(
                    logits_prev,
                    hs_windows[w],
                )
                if inner_sh_prev is not None:
                    inner_sh_prev, hs_windows[w] = microbatch_fence_pair(
                        inner_sh_prev,
                        hs_windows[w],
                    )
                if outer_sh_prev is not None:
                    outer_sh_prev, hs_windows[w] = microbatch_fence_pair(
                        outer_sh_prev,
                        hs_windows[w],
                    )
            elif fence == FenceMode.WINDOW:
                logits_prev = microbatch_fence(logits_prev)
                if inner_sh_prev is not None:
                    inner_sh_prev = microbatch_fence(inner_sh_prev)
                if outer_sh_prev is not None:
                    outer_sh_prev = microbatch_fence(outer_sh_prev)

            pre_compute_fenced = (
                (
                    hs_2d_prev,
                    orig_shape_prev,
                    logits_prev,
                    inner_sh_prev,
                    outer_sh_prev,
                ),
                res_pc_prev,
            )
            with _window_context(fc, md_windows[w - 1], pad_windows[w - 1]):
                layer_pre_states.append(
                    layer.forward_moe_pre_dispatch_route(pre_compute_fenced)
                )

            with _window_context(fc, md_windows[w], pad_windows[w]):
                pre_compute_prev = layer.forward_moe_pre_compute(
                    hs_windows[w],
                    res_windows[w],
                )

        # --- Sub-stage 2b (w=N-1): Dispatch + route last window ---
        with _window_context(
            fc, md_windows[num_windows - 1], pad_windows[num_windows - 1]
        ):
            layer_pre_states.append(
                layer.forward_moe_pre_dispatch_route(pre_compute_prev)
            )

        # --- Sub-stage 2c: Routed expert GMM kernel (merged or split) ---
        routed_outs = _run_merged_or_split_moe_kernels(
            layer,
            layer_pre_states,
            merge_moe=merge_moe,
            fence=fence,
        )

    # --- Sub-stage 2d: MoE post (Combine + TP All-Reduce) ---
    for w in range(num_windows):
        with _window_context(fc, md_windows[w], pad_windows[w]):
            hs_windows[w], res_windows[w] = layer.forward_moe_post(
                layer_pre_states[w],
                routed_outs[w],
            )
        if w + 1 < num_windows and use_stage_pair_fence:
            # Chain fence_pair across w=0..N-2 so each window's moe_post is
            # coupled with the next window's routed_out (and inner_shared_output)
            # before its moe_post runs.
            hs_windows[w], routed_outs[w + 1] = microbatch_fence_pair(
                hs_windows[w],
                routed_outs[w + 1],
            )
            next_pre, next_res = layer_pre_states[w + 1]
            next_inner_sh = next_pre[1]
            if next_inner_sh is not None:
                hs_windows[w], next_inner_sh = microbatch_fence_pair(
                    hs_windows[w],
                    next_inner_sh,
                )
                layer_pre_states[w + 1] = (
                    (
                        next_pre[0],
                        next_inner_sh,
                        next_pre[2],
                        next_pre[3],
                        next_pre[4],
                        next_pre[5],
                        next_pre[6],
                        next_pre[7],
                    ),
                    next_res,
                )
        elif fence == FenceMode.WINDOW:
            hs_windows[w] = microbatch_fence(hs_windows[w])

    return hs_windows, res_windows


def _run_layer_stagewise(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    pos_windows: list[torch.Tensor],
    md_windows: list[Any],
    pad_windows: list[token_padding.TokenPaddingState | None],
    fc: Any,
    *,
    merge_moe: bool,
    fence: FenceMode,
    orig_md: Any = None,
    orig_padding: token_padding.TokenPaddingState | None = None,
    attn_only: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Run one decoder layer in stagewise order (when fence != SUBSTAGE)."""
    num_windows = len(hs_windows)

    # --- Stage 1: Attention across windows 0..N-1 ---
    for w in range(num_windows):
        if w > 0 and fence == FenceMode.STAGE:
            hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                hs_windows[w - 1],
                hs_windows[w],
            )
        with _window_context(fc, md_windows[w], pad_windows[w]):
            hs_windows[w], res_windows[w] = layer.forward_attn(
                hs_windows[w],
                res_windows[w],
                pos_windows[w],
            )
        if fence == FenceMode.WINDOW:
            hs_windows[w] = microbatch_fence(hs_windows[w])
    if fence == FenceMode.STAGE:
        hs_windows = fence_windows(hs_windows, fence)

    if attn_only:
        return _run_attn_only_mlp(
            layer,
            hs_windows,
            res_windows,
            fc,
            orig_md,
            orig_padding,
            fence=fence,
        )

    # --- Stage 2: MoE / MLP across windows 0..N-1 ---
    mlp = getattr(layer, "mlp", None)
    is_fused_ep = _is_fused_ep_moe(mlp)
    if can_split_moe_stages(mlp):
        layer_pre_states: list[tuple[Any, torch.Tensor]] = []
        for w in range(num_windows):
            if w > 0 and fence == FenceMode.STAGE:
                hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                    hs_windows[w - 1],
                    hs_windows[w],
                )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                layer_pre_states.append(
                    layer.forward_moe_pre(hs_windows[w], res_windows[w])
                )

        routed_outs = _run_merged_or_split_moe_kernels(
            layer,
            layer_pre_states,
            merge_moe=merge_moe,
            fence=fence,
        )

        for w in range(num_windows):
            with _window_context(fc, md_windows[w], pad_windows[w]):
                hs_windows[w], res_windows[w] = layer.forward_moe_post(
                    layer_pre_states[w],
                    routed_outs[w],
                )
            if fence == FenceMode.WINDOW:
                hs_windows[w] = microbatch_fence(hs_windows[w])
        if fence == FenceMode.STAGE:
            hs_windows = fence_windows(hs_windows, fence)
    elif is_fused_ep and merge_moe:
        pre_computes: list[tuple[Any, torch.Tensor]] = []
        for w in range(num_windows):
            if w > 0 and fence == FenceMode.STAGE:
                hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                    hs_windows[w - 1],
                    hs_windows[w],
                )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                pre_computes.append(
                    layer.forward_moe_pre_compute(hs_windows[w], res_windows[w])
                )

        hs_2d_list = [pc[0][0] for pc in pre_computes]
        logits_list = [pc[0][2] for pc in pre_computes]
        if fence == FenceMode.STAGE and num_windows > 1:
            hs_2d_list = fence_windows(hs_2d_list, fence)
            logits_list = fence_windows(logits_list, fence)
        cat_hs_2d = torch.cat(hs_2d_list, dim=0)
        cat_logits = torch.cat(logits_list, dim=0)

        with _window_context(fc, orig_md, orig_padding):
            merged_routed_out = layer.forward_moe_monolithic(cat_hs_2d, cat_logits)
        routed_outs = list(
            merged_routed_out.view(
                num_windows, -1, merged_routed_out.shape[-1]
            ).unbind(0)
        )
        dummy_idx = torch.empty(0, dtype=torch.int32, device=cat_hs_2d.device)
        for w, pc in enumerate(pre_computes):
            pre_state_w = (
                (
                    pc[0][1],
                    pc[0][3],
                    pc[0][4],
                    False,
                    None,
                    pc[0][0],
                    pc[0][2],
                    dummy_idx,
                ),
                pc[1],
            )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                hs_windows[w], res_windows[w] = layer.forward_moe_post(
                    pre_state_w,
                    routed_outs[w],
                )
            if fence == FenceMode.WINDOW:
                hs_windows[w] = microbatch_fence(hs_windows[w])
        if fence == FenceMode.STAGE:
            hs_windows = fence_windows(hs_windows, fence)
    else:
        for w in range(num_windows):
            if w > 0 and fence == FenceMode.STAGE:
                hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                    hs_windows[w - 1],
                    hs_windows[w],
                )
            with _window_context(fc, md_windows[w], pad_windows[w]):
                hs_windows[w], res_windows[w] = layer.forward_mlp(
                    hs_windows[w],
                    res_windows[w],
                )
            if fence == FenceMode.WINDOW:
                hs_windows[w] = microbatch_fence(hs_windows[w])
        if fence == FenceMode.STAGE:
            hs_windows = fence_windows(hs_windows, fence)

    return hs_windows, res_windows


def _run_layer_layerwise(
    layer: Any,
    hs_windows: list[torch.Tensor],
    res_windows: list[torch.Tensor | None],
    pos_windows: list[torch.Tensor],
    md_windows: list[Any],
    pad_windows: list[token_padding.TokenPaddingState | None],
    fc: Any,
    *,
    fence: FenceMode,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Run one decoder layer in full-layer window order (Window 0, Window 1, ...)."""
    num_windows = len(hs_windows)
    for w in range(num_windows):
        if w > 0 and fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
            hs_windows[w - 1], hs_windows[w] = microbatch_fence_pair(
                hs_windows[w - 1],
                hs_windows[w],
            )
            res_prev = res_windows[w - 1]
            res_curr = res_windows[w]
            if res_prev is not None and res_curr is not None:
                res_windows[w - 1], res_windows[w] = microbatch_fence_pair(
                    res_prev,
                    res_curr,
                )
        with _window_context(fc, md_windows[w], pad_windows[w]):
            if hasattr(layer, "forward_attn") and hasattr(layer, "forward_mlp"):
                hs_w, res_w = layer.forward_attn(
                    hs_windows[w],
                    res_windows[w],
                    pos_windows[w],
                )
                if fence == FenceMode.SUBSTAGE:
                    hs_w = microbatch_fence(hs_w)
                mlp = getattr(layer, "mlp", None)
                if can_split_moe_stages(mlp):
                    pre_state = layer.forward_moe_pre(hs_w, res_w)
                    routed_out = layer.forward_moe_kernel(
                        pre_state[0][5],
                        pre_state[0][6],
                        pre_state[0][7],
                    )
                    hs_w, res_w = layer.forward_moe_post(pre_state, routed_out)
                else:
                    hs_w, res_w = layer.forward_mlp(hs_w, res_w)
                hs_windows[w], res_windows[w] = hs_w, res_w
            else:
                hs_windows[w], res_windows[w] = layer(
                    positions=pos_windows[w],
                    hidden_states=hs_windows[w],
                    residual=res_windows[w],
                )
        if fence == FenceMode.WINDOW:
            hs_windows[w] = microbatch_fence(hs_windows[w])
            res_win = res_windows[w]
            if res_win is not None:
                res_windows[w] = microbatch_fence(res_win)

    if fence in (FenceMode.STAGE, FenceMode.SUBSTAGE):
        hs_windows = fence_windows(hs_windows, fence)
    return hs_windows, res_windows


def _active_layers(model: Any) -> list[nn.Module]:
    layers = getattr(model, "layers", None)
    if layers is None:
        return []
    start_layer = getattr(model, "start_layer", 0)
    end_layer = getattr(model, "end_layer", len(layers))
    return list(layers[start_layer:end_layer])


def _forward_unsplit(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor | IntermediateTensors, tuple[torch.Tensor, ...]]:
    """Standard unsplit forward loop over ``self.layers``."""
    aux_hidden_states = []
    use_aux = getattr(self, "aux_hidden_state_layers", ())
    for layer_idx, layer in enumerate(
        _active_layers(self), start=getattr(self, "start_layer", 0)
    ):
        if layer_idx in use_aux:
            assert residual is not None
            aux_hidden_states.append(hidden_states + residual)
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

    pp_group = getattr(_ps, "_PP", None)
    is_last_rank = pp_group.is_last_rank if pp_group is not None else True

    if not is_last_rank:
        assert residual is not None
        return IntermediateTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        ), tuple(aux_hidden_states)

    if getattr(self, "norm", None) is not None:
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states, tuple(aux_hidden_states)


def _microbatch_model_forward(
    self: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    """Instrumented ``Qwen3NextModel.forward`` supporting token-window microbatching."""
    config: MicroBatchConfig | None = getattr(self, "_microbatch_config", None)
    if config is None or config.factor <= 1:
        orig_fn = getattr(self, "_orig_unpatched_forward", None)
        if orig_fn is not None:
            return orig_fn(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        orig_cls_fn = getattr(type(self), "_orig_class_forward", None)
        if orig_cls_fn is not None:
            return orig_cls_fn(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        return _orig_qwen3_next_model_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    pp_group = getattr(_ps, "_PP", None)
    is_first_rank = pp_group.is_first_rank if pp_group is not None else True

    if is_first_rank or intermediate_tensors is None:
        if inputs_embeds is not None:
            hidden_states: torch.Tensor = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual: torch.Tensor | None = None
    else:
        assert intermediate_tensors is not None
        hs_val = intermediate_tensors["hidden_states"]
        res_val = intermediate_tensors["residual"]
        assert isinstance(hs_val, torch.Tensor)
        assert isinstance(res_val, torch.Tensor)
        hidden_states = hs_val
        residual = res_val

    fc = getattr(_fc_mod, "_forward_context", None)
    orig_md = fc.attn_metadata if fc is not None else None
    orig_padding = token_padding.get_padding_state()

    # Touch step-level inputs before branching on num_tokens so small (< min_tokens)
    # and large (>= min_tokens) traces register placeholders in identical order.
    num_tokens = hidden_states.shape[0]
    aligned_pos, restore_inputs = bind_step_inputs_for_trace_alignment(
        positions,
        orig_md,
        orig_padding,
        num_tokens=num_tokens,
    )
    assert aligned_pos is not None
    positions = aligned_pos
    try:
        factor = config.factor
        if num_tokens < config.min_tokens or num_tokens % factor != 0:
            out, aux = _forward_unsplit(self, positions, hidden_states, residual)
            if getattr(self, "aux_hidden_state_layers", ()):
                assert isinstance(out, torch.Tensor)
                return out, list(aux)
            return out

        # Force aot_autograd's ShapeEnv to bind s0 = bucket_size before s0 // factor
        # is evaluated so window shapes are concrete Python ints in StaticCompiler.
        hidden_states = hidden_states.view(
            hidden_states.size(0), *hidden_states.shape[1:]
        )
        if residual is not None:
            residual = residual.view(residual.size(0), *residual.shape[1:])

        window_tokens = num_tokens // factor
        hs_windows = _split_tokens(hidden_states, factor)
        res_windows: list[torch.Tensor | None] = (
            [None] * factor
            if residual is None
            else [t for t in _split_tokens(residual, factor)]
        )
        pos_windows = _split_positions(positions, factor)
        md_windows = [
            slice_forward_metadata(orig_md, w, window_tokens) for w in range(factor)
        ]
        pad_windows = [
            slice_padding_state(orig_padding, w, window_tokens) for w in range(factor)
        ]

        schedule = config.resolved_schedule
        fence = config.resolved_fence
        merge_moe = config.merge_moe

        aux_hidden_states: list[torch.Tensor] = []
        use_aux = getattr(self, "aux_hidden_state_layers", ())
        start_layer = getattr(self, "start_layer", 0)

        for layer_idx, layer in enumerate(_active_layers(self), start=start_layer):
            if layer_idx in use_aux:
                aux_windows = [
                    h + r if r is not None else h
                    for h, r in zip(hs_windows, res_windows)
                ]
                aux_hidden_states.append(torch.cat(aux_windows, dim=0))

            if schedule == ScheduleMode.LAYERWISE:
                hs_windows, res_windows = _run_layer_layerwise(
                    layer,
                    hs_windows,
                    res_windows,
                    pos_windows,
                    md_windows,
                    pad_windows,
                    fc,
                    fence=fence,
                )
            elif (
                schedule in (ScheduleMode.STAGGERED, ScheduleMode.ATTN_ONLY)
                or fence == FenceMode.SUBSTAGE
            ):
                hs_windows, res_windows = _run_layer_substage_staggered(
                    layer,
                    hs_windows,
                    res_windows,
                    pos_windows,
                    md_windows,
                    pad_windows,
                    fc,
                    merge_moe=merge_moe,
                    fence=fence,
                    orig_md=orig_md,
                    orig_padding=orig_padding,
                    attn_only=(schedule == ScheduleMode.ATTN_ONLY),
                )
            else:
                hs_windows, res_windows = _run_layer_stagewise(
                    layer,
                    hs_windows,
                    res_windows,
                    pos_windows,
                    md_windows,
                    pad_windows,
                    fc,
                    merge_moe=merge_moe,
                    fence=fence,
                    orig_md=orig_md,
                    orig_padding=orig_padding,
                    attn_only=(schedule == ScheduleMode.ATTN_ONLY),
                )

        hidden_states = torch.cat(hs_windows, dim=0)
        if any(r is not None for r in res_windows):
            residual = torch.cat([r for r in res_windows if r is not None], dim=0)
        else:
            residual = None

        is_last_rank = pp_group.is_last_rank if pp_group is not None else True

        if not is_last_rank:
            assert residual is not None
            out: torch.Tensor | IntermediateTensors = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        elif getattr(self, "norm", None) is not None:
            if residual is None:
                out = self.norm(hidden_states)
            else:
                out, _ = self.norm(hidden_states, residual)
        else:
            out = hidden_states

        if use_aux:
            assert isinstance(out, torch.Tensor)
            return out, aux_hidden_states
        return out
    finally:
        restore_inputs()


# ============================================================================
# Class Patching & Model Installation Entry Points
# ============================================================================

_orig_qwen3_next_model_forward: Any = None
_qwen_classes_patched: bool = False


def _instrument_decoder_layer_class(layer_cls: type[nn.Module]) -> None:
    """Attach stage and sub-stage methods onto a decoder layer class if missing."""
    methods = {
        "forward_attn_compute": _layer_forward_attn_compute,
        "forward_attn_reduce": _layer_forward_attn_reduce,
        "forward_attn": _layer_forward_attn,
        "forward_moe_pre_compute": _layer_forward_moe_pre_compute,
        "forward_moe_pre_dispatch_route": _layer_forward_moe_pre_dispatch_route,
        "forward_moe_pre": _layer_forward_moe_pre,
        "forward_moe_kernel": _layer_forward_moe_kernel,
        "forward_moe_monolithic": _layer_forward_moe_monolithic,
        "forward_moe_post": _layer_forward_moe_post,
        "forward_mlp": _layer_forward_mlp,
    }
    for name, fn in methods.items():
        if not hasattr(layer_cls, name):
            setattr(layer_cls, name, fn)


def _instrument_moe_block_class(moe_cls: type[nn.Module]) -> None:
    """Attach decomposed MoE sub-stage methods onto a SparseMoeBlock class."""
    methods = {
        "forward_moe_pre_compute": _moe_forward_pre_compute,
        "forward_moe_pre_dispatch_route": _moe_forward_pre_dispatch_route,
        "forward_moe_pre": _moe_forward_pre,
        "forward_moe_kernel": _moe_forward_kernel,
        "forward_moe_monolithic": _moe_forward_monolithic,
        "forward_moe_post": _moe_forward_post,
    }
    for name, fn in methods.items():
        if not hasattr(moe_cls, name):
            setattr(moe_cls, name, fn)


def patch_qwen_microbatch() -> None:
    """Instrument ``Qwen3NextSparseMoeBlock``, layer, and model classes.

    Registered in ``patch_registry.py`` at ``_ACTIVE`` stages so that when
    ``TPU_MICROBATCH_FACTOR > 1``, ``TorchCompileWithNoGuardsWrapper`` captures
    the instrumented ``Qwen3NextModel.forward`` bytecode when the model is
    constructed. When ``TPU_MICROBATCH_FACTOR <= 1``, this function is a no-op
    so upstream Qwen3Next/Qwen3.5 class forwards remain untouched.
    """
    global _orig_qwen3_next_model_forward, _qwen_classes_patched
    if envs.TPU_MICROBATCH_FACTOR <= 1 or _qwen_classes_patched:
        return

    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextDecoderLayer,
        Qwen3NextModel,
        Qwen3NextSparseMoeBlock,
    )

    _instrument_moe_block_class(Qwen3NextSparseMoeBlock)
    _instrument_decoder_layer_class(Qwen3NextDecoderLayer)

    if _orig_qwen3_next_model_forward is None:
        _orig_qwen3_next_model_forward = Qwen3NextModel.forward
    if not hasattr(Qwen3NextModel, "_orig_class_forward"):
        Qwen3NextModel._orig_class_forward = Qwen3NextModel.forward
    Qwen3NextModel.forward = _microbatch_model_forward
    _qwen_classes_patched = True


def _find_inner_model(model: nn.Module) -> nn.Module | None:
    """Locate the inner decoder backbone (module with ``.layers``)."""
    visited: set[int] = set()
    queue: list[nn.Module] = [model]
    while queue:
        curr = queue.pop(0)
        if id(curr) in visited:
            continue
        visited.add(id(curr))
        layers: Any = getattr(curr, "layers", None)
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            return curr
        for attr in ("model", "language_model"):
            child = getattr(curr, attr, None)
            if isinstance(child, nn.Module):
                queue.append(child)
    return None


def _has_tp_collectives(
    inner: nn.Module | None,
    vllm_config: VllmConfig | None = None,
) -> bool:
    """Return True when Attention or MoE has tensor-parallel (TP > 1) collectives."""
    if vllm_config is not None:
        p_cfg = getattr(vllm_config, "parallel_config", None)
        if p_cfg is not None and getattr(p_cfg, "tensor_parallel_size", 1) > 1:
            return True
    if inner is None:
        return False
    for layer in _active_layers(inner):
        _, out_proj = _get_attn_module_and_out_proj(layer)
        if getattr(out_proj, "tp_size", 1) > 1:
            return True
        mlp = getattr(layer, "mlp", None)
        if getattr(mlp, "tp_size", 1) > 1:
            return True
        experts = getattr(mlp, "experts", None)
        runner = _get_moe_runner(experts) if experts is not None else None
        moe_config = getattr(runner, "moe_config", None)
        if getattr(moe_config, "tp_size", 1) > 1:
            return True
    return False


def unsupported_config_reasons(
    model: nn.Module,
    vllm_config: VllmConfig | None = None,
) -> list[str]:
    """Return reasons why ``model`` / ``vllm_config`` cannot use microbatching."""
    reasons: list[str] = []
    if vllm_config is not None:
        p_cfg = getattr(vllm_config, "parallel_config", None)
        if p_cfg is not None:
            if getattr(p_cfg, "prefill_context_parallel_size", 1) > 1:
                reasons.append("prefill_context_parallel_size > 1")
            if getattr(p_cfg, "decode_context_parallel_size", 1) > 1:
                reasons.append("decode_context_parallel_size > 1")

    if envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0:
        reasons.append("TPU_MOE_COLLECTION_CHUNK_SIZE > 0")

    inner = _find_inner_model(model)
    has_tp = _has_tp_collectives(inner, vllm_config)
    if envs.USE_MOE_FUSED_EP_KERNEL and not has_tp:
        reasons.append("USE_MOE_FUSED_EP_KERNEL=1")

    if inner is None:
        reasons.append("model has no decoder layers list")
        return reasons

    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

    for layer in _active_layers(inner):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            runner = _get_moe_runner(experts)
            moe_config = getattr(runner, "moe_config", None)
            if (
                getattr(mlp, "is_sequence_parallel", False)
                or getattr(experts, "is_sequence_parallel", False)
                or getattr(moe_config, "is_sequence_parallel", False)
            ):
                reasons.append("FusedMoE.is_sequence_parallel=True")
                break
            if (
                getattr(experts, "has_ici_cp_collectives", False)
                or getattr(moe_config, "pcp_size", 1) > 1
            ):
                reasons.append("FusedMoE.has_ici_cp_collectives=True")
                break
            routed = getattr(
                runner, "routed_experts", getattr(experts, "routed_experts", None)
            )
            qm = getattr(routed, "quant_method", None)
            if not has_tp and (
                getattr(qm, "_use_fused_ep", False) or fused_moe_ep_supported(qm)
            ):
                reasons.append("FusedMoE quant_method._use_fused_ep=True")
                break

    return reasons


def install_microbatch(
    model: nn.Module,
    config: MicroBatchConfig,
) -> bool:
    """Enable token-window micro-batching on ``model`` with ``config``."""
    inner = _find_inner_model(model)
    if inner is None:
        return False

    # Ensure custom decoder layers / MoE blocks (e.g. in unit tests) also have
    # the stage/sub-stage methods attached.
    uses_fused_ep = False
    for layer in _active_layers(inner):
        _instrument_decoder_layer_class(type(layer))
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, nn.Module):
            _instrument_moe_block_class(type(mlp))
            if _is_fused_ep_moe(mlp):
                uses_fused_ep = True

    if (
        (uses_fused_ep or envs.USE_MOE_FUSED_EP_KERNEL)
        and isinstance(config.schedule, str)
        and config.schedule.strip() == ""
        and not config.layerwise
    ):
        config = replace(config, schedule=ScheduleMode.STAGGERED)

    inner_any: Any = inner
    model_any: Any = model
    inner_any._microbatch_config = config
    model_any._microbatch_config = config

    inner_cls: Any = type(inner)
    if inner_cls.forward is not _microbatch_model_forward:
        if not hasattr(inner_cls, "_orig_class_forward"):
            inner_cls._orig_class_forward = inner_cls.forward
        inner_cls.forward = _microbatch_model_forward
        if hasattr(inner, "vllm_config") and hasattr(inner, "_compiled_callable"):
            from vllm.compilation.wrapper import reset_compile_wrapper
            from vllm.config import set_current_vllm_config

            inner_any.cleanup()
            with set_current_vllm_config(inner_any.vllm_config):
                reset_compile_wrapper(inner)

    return True


def uninstall_microbatch(model: nn.Module) -> None:
    """Disable token-window micro-batching on ``model`` and restore original forward."""
    inner = _find_inner_model(model)
    for target in (model, inner):
        if target is None:
            continue
        target_any: Any = target
        target_cls: Any = type(target)
        if hasattr(target, "_microbatch_config"):
            delattr(target, "_microbatch_config")
        if hasattr(target, "_orig_unpatched_forward"):
            target_any.forward = target_any._orig_unpatched_forward
            delattr(target, "_orig_unpatched_forward")
        if hasattr(target_cls, "_orig_class_forward"):
            target_cls.forward = target_cls._orig_class_forward
            delattr(target_cls, "_orig_class_forward")


def maybe_install_microbatch(
    model: nn.Module,
    vllm_config: VllmConfig | None = None,
    config: MicroBatchConfig | None = None,
) -> bool:
    """Validate config and install micro-batching on ``model`` if supported."""
    if config is None:
        config = MicroBatchConfig.from_envs()
    if config.factor <= 1:
        return False

    reasons = unsupported_config_reasons(model, vllm_config)
    if reasons:
        logger.warning(
            "Skipping TPU microbatching (factor=%d): unsupported configuration (%s).",
            config.factor,
            ", ".join(reasons),
        )
        return False

    installed = install_microbatch(model, config)
    if installed:
        logger.info(
            "Installed TPU Qwen microbatching: factor=%d, min_tokens=%d, "
            "schedule=%s, merge_moe=%s, fence=%s",
            config.factor,
            config.min_tokens,
            config.resolved_schedule.value,
            config.merge_moe,
            config.resolved_fence.value,
        )
    return installed
