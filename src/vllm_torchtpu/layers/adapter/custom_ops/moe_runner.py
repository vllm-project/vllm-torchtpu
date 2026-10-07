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

from torch.nn import functional as F
from vllm.distributed import parallel_state
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
        return super()._maybe_combine(shared_output, hidden_states)

    def _forward_impl(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
    ):
        # Small compiled token buckets can leave the fused EP kernel for the
        # TPU-issued gather/GMM/combine path; see _external_dispatch_mode. Both
        # selectors are off by default, which keeps upstream's forward.
        mode = _external_dispatch_mode(self, hidden_states.shape[0])
        if mode is None:
            # Named rather than super(): the tests drive this with a stand-in
            # runner that is not a VllmMoERunner instance.
            return MoERunner._forward_impl(
                self, hidden_states, router_logits, shared_experts_input, input_ids
            )
        return _forward_with_external_dispatch(
            self,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            gather_before_routing=mode == "gather_then_route",
        )


def _external_dispatch_mode(runner, num_tokens: int) -> str | None:
    """Which TPU-issued dispatch path serves this compiled token bucket.

    - "gather_then_route": at or below TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS,
      all-gather the hidden states and route the gathered rows, which drops
      the separate router-logits all-gather.
    - "below_fused_ep": below TPU_MOE_FUSED_EP_STEP_MIN_TOKENS with fused EP
      enabled, run vLLM's naive dispatch around the grouped-matmul path even
      though the layer's method owns the exchange for larger buckets.
    - None: upstream's forward.

    The token count is static per compiled bucket, so this is a trace-time
    choice, not a per-step branch. Neither path handles PCP's extra gather.
    """
    gather_max = envs.TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS
    if gather_max < 0:
        raise ValueError("TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS must be >= 0")
    fused_min = envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS
    if gather_max == 0 and (fused_min <= 0 or not envs.USE_MOE_FUSED_EP_KERNEL):
        return None
    config = runner.moe_config
    if config.dp_size <= 1 or getattr(config, "pcp_size", 1) > 1:
        return None
    quant_method = runner.routed_experts.quant_method
    if getattr(quant_method, "apply_monolithic_external_dispatch", None) is None:
        return None
    if (
        0 < num_tokens <= gather_max
        and runner.gate is not None
        and not config.is_sequence_parallel
    ):
        return "gather_then_route"
    if envs.USE_MOE_FUSED_EP_KERNEL and num_tokens < fused_min:
        return "below_fused_ep"
    return None


def _forward_with_external_dispatch(
    runner,
    hidden_states,
    router_logits,
    shared_experts_input,
    input_ids,
    *,
    gather_before_routing: bool,
):
    """``MoERunner._forward_impl`` with the DP/EP dispatch issued here.

    Upstream gates its dispatch and combine on ``do_naive_dispatch_combine``,
    which is off for a layer whose method owns the exchange. This issues them
    unconditionally around the method's external-dispatch entry point.
    """
    config = runner.moe_config
    runner.routed_experts._ensure_moe_quant_config_init()

    # Launch a multi-stream shared expert before routed expert dispatch.
    shared_experts_overlapping = False
    if runner._shared_experts is not None:
        shared_experts_overlapping = runner._shared_experts.maybe_forward_async(
            shared_experts_input
        )

    def route(rows):
        if runner._fse_fuse_gate:
            runner._maybe_fuse_gate_weights()
            return F.linear(rows, runner._combined_gate_weight)
        logits, _ = runner.gate(rows)
        return logits

    with runner._sequence_parallel_context():
        if gather_before_routing:
            # Match the communicator dispatch group.
            hidden_states = parallel_state.get_dp_group().all_gather(
                hidden_states, dim=0
            )
            router_logits = route(hidden_states)
        else:
            if runner.gate is not None:
                router_logits = route(hidden_states)
            result = _moe_runner.get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        runner._maybe_apply_shared_experts(
            shared_experts_input,
            _moe_runner.SharedExpertsOrder.NO_OVERLAP,
        )
        quant_method = runner.routed_experts.quant_method
        fused_output = quant_method.apply_monolithic_external_dispatch(
            layer=runner.routed_experts,
            x=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        if shared_experts_overlapping:
            assert runner._shared_experts is not None
            runner._shared_experts.wait()

        fused_output = _moe_runner.get_ep_group().combine(
            fused_output,
            False if gather_before_routing else config.is_sequence_parallel,
        )
        if runner._shared_experts is None:
            return fused_output
        return runner._shared_experts.output, fused_output
