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

from collections.abc import Callable

from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner import moe_runner as _moe_runner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)

from vllm_torchtpu import decode_switches, envs
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.vllm_adapter import (
    adaptive_fused_moe_supported,
)
from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

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
        # What the quant method needs to decide this layer's owner combine at
        # weight load (decode_switches.arm_owner_combine).
        config = self.moe_config
        setattr(
            self.routed_experts,
            decode_switches.OWNER_FACTS_ATTR,
            decode_switches.OwnerCombineFacts(
                sequence_parallel=bool(config.is_sequence_parallel),
                pcp_size=config.pcp_size,
                skip_final_all_reduce=bool(
                    getattr(config, "skip_final_all_reduce", False)
                ),
                has_output_transform=self.routed_output_transform is not None,
                scaling_factor=self.routed_scaling_factor,
                zero_expert_router=isinstance(self.router, ZeroExpertRouter),
            ),
        )

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
        #
        # The ragged all-to-all owner combine returns the final sum, shared
        # expert included.
        if self._owner_combine_armed():
            return True
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
        return super()._maybe_dispatch(hidden_states, router_logits)

    def _maybe_combine(self, shared_output, hidden_states):
        if self._fused_ep_owns_pcp_collectives():
            # The fused program has already pushed and summed routed rows
            # back onto their token-owning PCP rank. Preserve upstream's
            # return shape for a separately-computed shared expert.
            if self.shared_experts is not None:
                return shared_output, hidden_states
            return hidden_states
        if self._owner_combine_armed():
            # The routed op returned the final sum, shared expert included.
            if self.shared_experts is not None:
                return None, hidden_states
            return hidden_states
        return super()._maybe_combine(shared_output, hidden_states)

    def _owner_combine_armed(self) -> bool:
        """Whether the ragged all-to-all op returns this rank's tokens summed
        over TP, shared expert included (the owner combine). Decided
        once per layer, at weight load, by ``decode_switches.arm_owner_combine``
        for the op and for this runner."""
        return getattr(self.routed_experts, decode_switches.OWNER_COMBINE_ATTR, False)

    def _apply_quant_method(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
        **kwargs,
    ):
        if not self._owner_combine_armed():
            return super()._apply_quant_method(
                hidden_states, router_logits, shared_experts_input, input_ids, **kwargs
            )
        # Shared experts first, so the routed op can add their partial before
        # its TP sum.
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )
        if (
            kwargs.get("shared_experts_overlapping")
            and self._shared_experts is not None
        ):
            self._shared_experts.wait()
        shared = (
            self._shared_experts.output if self._shared_experts is not None else None
        )
        routed = self.routed_experts
        routed._tpu_owner_shared_output = shared
        try:
            fused_out = routed.forward_monolithic(
                x=hidden_states, router_logits=router_logits, input_ids=input_ids
            )
        finally:
            routed._tpu_owner_shared_output = None
        return None, fused_out
