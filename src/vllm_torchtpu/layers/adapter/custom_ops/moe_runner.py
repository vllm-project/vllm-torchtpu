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

import torch
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.model_executor.layers.fused_moe.runner import moe_runner as _moe_runner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.vllm_adapter import (
    adaptive_fused_moe_supported,
)
from vllm_torchtpu.layers.adapter import moe_routing
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
        if self._fused_ep_owns_pcp_collectives():
            # The fused program has already pushed and summed routed rows
            # back onto their token-owning PCP rank. Preserve upstream's
            # return shape for a separately-computed shared expert.
            if self.shared_experts is not None:
                return shared_output, hidden_states
            return hidden_states
        return super()._maybe_combine(shared_output, hidden_states)

    def forward_prequantized(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
        *,
        hidden_states_q: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run explicit FP8 inputs for routed and compatible shared experts.

        Adapted from vLLM v0.31.1rc0 ``forward`` / ``_forward_impl``: their
        custom-op and quant-method seams cannot carry q/scale side operands.
        Keep those operands explicit and reuse upstream padding, dispatch,
        combine and output-reduction helpers in the same order. Shared experts
        run synchronously on TPU; CUDA overlap and fused shared gates are not
        supported here. A runner without a gate requires actual router logits;
        a runner-owned gate computes them from BF16 hidden states.
        """
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            VllmFp8MoEMethodTPU,
        )
        from vllm_torchtpu.layers.adapter.quantization.mxfp4 import (
            VllmDeepseekV4Mxfp4MoEMethod,
        )
        from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP

        method = self._quant_method
        if (hidden_states_q is None) != (hidden_states_scale is None):
            raise ValueError("routed prequantized inputs must be paired")
        if self.gate is None and router_logits is None:
            raise ValueError("prequantized inputs require router logits or a gate")
        if hidden_states_q is not None:
            if not isinstance(
                method, (VllmFp8MoEMethodTPU, VllmDeepseekV4Mxfp4MoEMethod)
            ) or not fused_moe_ep_supported(method):
                raise ValueError(
                    "prequantized inputs require an armed TPU fused EP method"
                )
            if self.do_naive_dispatch_combine:
                raise ValueError("prequantized routed inputs require internal dispatch")
        if self.routed_input_transform is not None:
            raise ValueError(
                "prequantized inputs do not support routed input transforms"
            )
        shared_layer = (
            self._shared_experts._layer if self._shared_experts is not None else None
        )
        reuse_shared_quantization = (
            hidden_states_q is not None
            and (shared_experts_input is None or shared_experts_input is hidden_states)
            and isinstance(shared_layer, DeepseekV4MLP)
            and shared_layer.supports_prequantized_input()
        )
        # Routed padding must not change the shared expert's original input width.
        shared_input_q = hidden_states_q
        if shared_experts_input is None and self._shared_experts is not None:
            shared_experts_input = hidden_states
        original_width = hidden_states.shape[-1]
        hidden_states, pre_trunc, post_trunc = self._maybe_pad_hidden_states(
            shared_experts_input, hidden_states
        )
        if hidden_states_q is not None and hidden_states.shape[-1] != original_width:
            hidden_states_q = F.pad(
                hidden_states_q, (0, hidden_states.shape[-1] - original_width)
            )
        self.routed_experts._ensure_moe_quant_config_init()
        # Preserve the upstream custom entry's forward-context layer accounting.
        _moe_runner.get_layer_from_name(
            _moe_runner._resolve_layer_name(self._encode_layer_name())
        )
        if self.gate is not None:
            if self._fse_fuse_gate:
                raise ValueError(
                    "prequantized inputs do not support fused shared gates"
                )
            router_logits, _ = self.gate(hidden_states)
        assert router_logits is not None
        with self._sequence_parallel_context():
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states, router_logits
            )
            shared_output = None
            if shared_layer is not None:
                if reuse_shared_quantization:
                    shared_output = shared_layer.forward_prequantized(
                        shared_input_q,
                        hidden_states_scale,
                        output_dtype=shared_experts_input.dtype,
                    )
                else:
                    shared_output = shared_layer(shared_experts_input)
            if hidden_states_q is not None:
                if isinstance(method, VllmDeepseekV4Mxfp4MoEMethod):
                    fused_output = method._forward_monolithic_tpu(
                        self.routed_experts,
                        hidden_states,
                        router_logits,
                        input_ids,
                        hidden_states_q=hidden_states_q,
                        hidden_states_scale=hidden_states_scale,
                    )
                else:
                    assert isinstance(method, VllmFp8MoEMethodTPU)
                    fused_output = method.apply_monolithic(
                        self.routed_experts,
                        hidden_states,
                        router_logits,
                        input_ids,
                        hidden_states_q=hidden_states_q,
                        hidden_states_scale=hidden_states_scale,
                    )
            else:
                fused_output = self.routed_experts.forward_monolithic(
                    x=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )
            result = self._maybe_combine(shared_output, fused_output)
        shared_output, fused_output = _moe_runner._unpack(result)
        assert isinstance(fused_output, torch.Tensor)
        if pre_trunc is not None:
            fused_output = fused_output[..., :pre_trunc]
        reduced = self._fused_output_is_reduced
        fused_output, reduced = self._maybe_reduce_routed_output_before_transform(
            fused_output, reduced
        )
        shared_output = self._maybe_reduce_shared_expert_output(shared_output, reduced)
        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )
        fused_output = self.apply_routed_output_transform(fused_output)
        result = (
            shared_output + fused_output if shared_output is not None else fused_output
        )
        result = self._maybe_reduce_final_output(result, post_trunc, reduced)
        return self._maybe_add_zero_expert_output(result)
