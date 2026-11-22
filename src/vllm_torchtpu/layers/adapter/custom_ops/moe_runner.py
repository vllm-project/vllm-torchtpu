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
"""Out-of-tree (OOT) MoE runner: the TPU side of vLLM's MoE forward path."""

import inspect
from collections.abc import Callable

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.model_executor.layers.fused_moe.runner import moe_runner as _moe_runner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.vllm_adapter import (
    adaptive_fused_moe_supported,
)
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.fused_moe_ep import (
    FUSED_MOE_EP_RUNNER_SCALE_ATTR,
    fused_moe_ep_supported,
)


def _shared_output_scale(quant_method) -> float | None:
    """The routed scale a layer's fused EP op applies before adding the shared
    expert (it then returns that sum alone), or None."""
    return getattr(quant_method, "shared_expert_output_scale", None)


# Taken once at import, outside any trace: the MoE forward is traced by
# torch.compile, and reading the upstream property from inside the trace does
# not reliably produce upstream's answer (see VllmMoERunner.
# _fused_output_is_reduced).
_upstream_fused_output_is_reduced = MoERunner._fused_output_is_reduced.fget
assert _upstream_fused_output_is_reduced is not None


@MoERunner.register_oot
class VllmMoERunner(MoERunner):
    """vLLM builds every MoE layer's runner as ``MoERunner(...)`` unless a
    model passes its own ``runner_cls``, so registering here reaches each
    layer without patching the upstream class.

    A ``runner_cls`` subclass is not replaced (the swap is keyed by class
    name) and gets none of the behavior below. No model this plugin serves
    passes one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._take_over_moe_quant_config_init()
        # The shared expert module itself, for a fused EP op that computes it
        # (MOE_FUSED_EP_PUSH_SHARED_EXPERT). Not registered as a submodule:
        # the model already owns it, under its own parameter names.
        bound = inspect.signature(MoERunner.__init__).bind(self, *args, **kwargs)
        object.__setattr__(
            self, "_tpu_shared_mlp", bound.arguments.get("shared_experts")
        )
        # The scale this runner gives the routed output before adding the
        # shared expert's, for a fused EP op that does both
        # (MOE_FUSED_EP_PUSH_SHARED_ADD); read when the op is built. The op
        # scales in bf16, so it is handed over only for a bf16 model and a
        # scale bf16 holds exactly.
        scale = float(self.routed_scaling_factor)
        exact = torch.tensor(scale, dtype=torch.bfloat16).item() == scale
        model_config = get_current_vllm_config().model_config
        if (
            self._tpu_shared_mlp is not None
            and exact
            and self.routed_output_transform is None
            and model_config is not None
            and model_config.dtype == torch.bfloat16
        ):
            setattr(
                self.routed_experts,
                FUSED_MOE_EP_RUNNER_SCALE_ATTR,
                scale,
            )

    def _apply_quant_method(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
        shared_experts_overlapping=False,
    ):
        # A fused EP op built with the shared expert inside computes both
        # outputs; the runner then must not run the shared expert again.
        limit = getattr(self._quant_method, "shared_expert_limit", None)
        if limit is None:
            return super()._apply_quant_method(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                shared_experts_overlapping,
            )
        mlp = self._tpu_shared_mlp
        if mlp is None or shared_experts_overlapping:
            raise RuntimeError(
                "the fused EP op of this layer computes a shared expert, but "
                "the runner has none to give it or runs it on another stream"
            )
        if shared_experts_input is not hidden_states:
            raise NotImplementedError(
                "the in-op shared expert reads the routed experts' input; this "
                "layer transforms or pads one of them"
            )
        *shared, shared_limit = mlp.fused_operands()
        if shared_limit != limit:
            raise ValueError(
                f"the fused EP op was built for swiglu limit {limit}, the shared "
                f"expert has {shared_limit}"
            )
        result = self.routed_experts.quant_method.forward_monolithic_with_shared_expert(
            self.routed_experts,
            hidden_states,
            router_logits,
            input_ids,
            tuple(shared),
        )
        scale = _shared_output_scale(self.routed_experts.quant_method)
        if scale is None:
            routed, shared_out = result
            return shared_out, routed
        # The op scales in bf16 (upstream divides the shared output instead
        # for fp16) by the scale this runner recorded at init.
        if scale != self.routed_scaling_factor or hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                f"the fused EP op adds the shared expert after scaling by {scale} "
                f"in bfloat16; this runner scales by {self.routed_scaling_factor} "
                f"in {hidden_states.dtype}"
            )
        # The op returned the shared + scaled routed sum: no shared output
        # for the runner to add, and no scale left to apply.
        return None, result

    def _maybe_apply_routed_scale_to_output(self, shared_output, fused_output):
        if _shared_output_scale(self._quant_method) is not None:
            return shared_output, fused_output
        return super()._maybe_apply_routed_scale_to_output(shared_output, fused_output)

    def _take_over_moe_quant_config_init(self) -> None:
        routed_experts = self.routed_experts
        upstream_init = routed_experts._ensure_moe_quant_config_init

        def ensure_moe_quant_config_init():
            # _forward_impl calls this on every forward until the quant
            # method's moe_quant_config is set, and it calls the quant
            # method's get_fused_moe_quant_config. The TPU MoE methods
            # subclass upstream methods but skip their __init__, so the
            # inherited hook can read state that was never set. A monolithic
            # TPU method runs its own kernel and consumes no
            # FusedMoEQuantConfig, so there is nothing to build.
            if routed_experts.quant_method.is_monolithic:
                return
            upstream_init()

        routed_experts._ensure_moe_quant_config_init = ensure_moe_quant_config_init

    def _select_forward(self) -> Callable:
        # Upstream takes the direct path only when is_tpu(), which is False
        # under this OOT plugin. The torch.ops.vllm.moe_forward custom op it
        # would use instead captures weight tensors as closures and trips
        # torch_tpu's MLIR builder.
        return (
            _moe_runner._moe_forward
            if self.shared_experts is None
            else _moe_runner._moe_forward_shared
        )

    @property
    def _fused_output_is_reduced(self) -> bool:
        # `fused_moe_ep` returns this rank's tokens combined over ALL experts:
        # the kernel pushes each result row back to the chip that owns the
        # token. Upstream decides from quant_method.moe_kernel, which the TPU
        # methods never set, so without this it all-reduces the combined
        # output whenever tp_size > 1 or ep_size > 1.
        #
        # Today that changes no answer: prebuild_fused_moe_ep refuses
        # tp_size > 1, so the kernel only arms with a TP group of world size
        # one, where every all-reduce this flag gates is `return input_`.
        # This keeps that true if the refusal is lifted, when the upstream
        # answer would silently multiply every routed output by tp_size.
        #
        # Reporting True is right rather than dropping the all-reduce: the
        # SHARED expert is genuinely TP-partial
        # (`Qwen3NextMLP(..., reduce_results=False)`), and this same flag
        # routes it to _maybe_reduce_shared_expert_output.
        #
        # skip_final_all_reduce asserts the fused output is NOT pre-reduced,
        # and in that mode the model reduces externally anyway.
        #
        # Not super()._fused_output_is_reduced: under torch.compile dynamo
        # reads that as truthy, and the routed output then skips its TP
        # all-reduce.
        if not envs.USE_MOE_FUSED_EP_KERNEL or getattr(
            self.moe_config, "skip_final_all_reduce", False
        ):
            return _upstream_fused_output_is_reduced(self)
        # Asked of this layer's quant method, because arming is per layer: a
        # model whose layers do not all qualify must not have one armed layer
        # answer for the rest.
        owner = self._quant_method
        return (
            fused_moe_ep_supported(owner)
            or adaptive_fused_moe_supported(owner)
            or _upstream_fused_output_is_reduced(self)
        )

    def _fused_ep_owns_pcp_collectives(self) -> bool:
        # The fused EP kernel is armed per layer after checking that layer's
        # weights, routing and device resources, so an armed layer suppresses
        # the outer PCP pair while a refused layer in the same model keeps the
        # explicit collectives. The process-wide choice is made in
        # _patch_moe_explicit_pcp_collectives.
        if self.moe_config.pcp_size <= 1:
            return False
        owner = self._quant_method
        return fused_moe_ep_supported(owner) or adaptive_fused_moe_supported(owner)

    def _maybe_dispatch(self, hidden_states, router_logits):
        if self._fused_ep_owns_pcp_collectives():
            # An armed fused method also reports supports_internal_mk, so
            # upstream's generic DP/EP dispatch is a no-op. The only work
            # being bypassed here is its explicit PCP all-gather pair.
            return hidden_states, router_logits
        if (
            self.do_naive_dispatch_combine
            and self.moe_config.pcp_size <= 1
            and self.routed_experts.quant_method.is_monolithic
            and self.routed_experts.custom_routing_function is None
            and moe_routing._SIMULATION_STRATEGY is None
            and router_logits is not hidden_states
        ):
            group = (
                get_ep_group()
                if self.moe_config.is_sequence_parallel
                else get_dp_group()
            )
            if group.world_size > 1:
                hidden_states = group.all_gather(hidden_states, dim=0)
            return hidden_states, router_logits
        return super()._maybe_dispatch(hidden_states, router_logits)

    def _maybe_combine(self, shared_output, hidden_states):
        if _shared_output_scale(self._quant_method) is not None:
            # The fused op combined the routed rows onto their own rank and
            # added the shared expert: its one output is the layer's.
            return hidden_states
        if self._fused_ep_owns_pcp_collectives():
            # The fused program has already pushed and summed routed rows
            # back onto their token-owning PCP rank. Preserve upstream's
            # return shape for a separately-computed shared expert.
            if self.shared_experts is not None:
                return shared_output, hidden_states
            return hidden_states
        return super()._maybe_combine(shared_output, hidden_states)
