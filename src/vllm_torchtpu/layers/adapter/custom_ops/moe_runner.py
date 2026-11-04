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
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner import moe_runner as _moe_runner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu import envs
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
        # The single EP combine returns the final sum, shared expert included.
        if self._single_ep_combine_armed():
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
        if self._single_ep_combine_armed():
            return self._single_ep_combine(shared_output, hidden_states)
        return super()._maybe_combine(shared_output, hidden_states)

    def _single_ep_combine_armed(self) -> bool:
        """Whether one EP all-reduce replaces the naive combine and the final
        TP all-reduce (TPU_MOE_FUSED_COMBINE).

        Every rank of the EP group then holds a partial sum for every token
        of its DP group, so summing over the EP group gives what the DP
        reduce-scatter followed by the TP all-reduce gives. Refused where
        forward() would still touch the output between those two steps.
        """
        if not envs.TPU_MOE_FUSED_COMBINE or not self.do_naive_dispatch_combine:
            return False
        config = self.moe_config
        return (
            not config.is_sequence_parallel
            and config.pcp_size == 1
            and not getattr(config, "skip_final_all_reduce", False)
            and self.routed_output_transform is None
            and self.routed_scaling_factor == 1.0
            and not isinstance(self.router, ZeroExpertRouter)
            and get_ep_group().world_size
            == get_dp_group().world_size * get_tp_group().world_size
        )

    def _single_ep_combine(
        self, shared_output: torch.Tensor | None, hidden_states: torch.Tensor
    ):
        if isinstance(hidden_states, UnfinalizedMoEOutput):
            raise RuntimeError(
                "Naive expert-parallel combine cannot consume a deferred MoE output."
            )
        dp = get_dp_group()
        rows = hidden_states.shape[0] // dp.world_size
        start = dp.rank_in_group * rows
        if shared_output is not None:
            # Only this DP rank's TP group adds its tokens' shared expert
            # partials; the other ranks add zeros to those rows.
            hidden_states = hidden_states + F.pad(
                shared_output, (0, 0, start, hidden_states.shape[0] - start - rows)
            )
        combined = get_ep_group().all_reduce(hidden_states)[start : start + rows]
        if self.shared_experts is not None:
            # The shared output is inside the sum; forward() adds nothing more.
            return None, combined
        return combined
