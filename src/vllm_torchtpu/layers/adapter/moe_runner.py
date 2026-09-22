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

from collections.abc import Callable

from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner, _moe_forward, _moe_forward_shared)

from vllm_torchtpu import envs


@MoERunner.register_oot
class TpuMoERunner(MoERunner):
    """MoERunner with the TPU forward entry and fused-EP collective ownership.

    ``PluggableLayer.__new__`` builds this class wherever vLLM constructs
    ``MoERunner``. Everything not overridden here is upstream's, and every
    override falls back to ``super()`` outside its own case.

    Registration is keyed by the exact class name, so it does not reach
    subclasses of ``MoERunner``. At vLLM v0.29.0 those are
    ``TransformersMoERunner`` (vllm/model_executor/models/transformers/moe.py),
    ``LatentMoERunner`` (vllm/models/kimi_k3/nvidia/latent_moe_runner.py) and
    ``ROCmLatentMoERunner`` (vllm/models/kimi_k3/amd/latent_moe_runner.py),
    each passed to ``FusedMoE`` as ``runner_cls``. They are built unchanged and
    no longer inherit these overrides, as they did while ``MoERunner`` itself
    was mutated. This plugin registers its own Kimi models, never passes a
    ``runner_cls``, and does not use the Transformers modelling backend, so
    none of them is constructed on TPU; one that becomes reachable needs its
    own registration.
    """

    def _select_forward(self) -> Callable:
        """Always take the direct ``_moe_forward`` entry.

        Upstream picks it only when ``is_tpu()`` or ``is_cpu()``. An
        out-of-tree platform is neither, so upstream would return
        ``torch.ops.vllm.moe_forward``, the same function wrapped as an opaque
        custom op. The op takes activations and a layer name only, so the
        expert weights it reads are captured as closures instead of appearing
        as graph inputs, which trips torch_tpu's MLIR builder. Called directly,
        the routing and expert code is traced with the rest of the model.
        """
        return (_moe_forward
                if self.shared_experts is None else _moe_forward_shared)

    @property
    def _fused_output_is_reduced(self) -> bool:
        """Tell vLLM the fused EP MoE kernel already reduced its output.

        ``fused_moe_ep`` returns this rank's tokens combined over ALL experts:
        the kernel pushes each result row back to the chip that owns the token,
        so there is nothing left to reduce. Upstream cannot know that. It reads
        ``quant_method.moe_kernel.output_is_reduced()``, and the TPU fp8 method
        never sets ``moe_kernel`` (it declares itself through
        ``supports_internal_mk``), so upstream would all-reduce the combined
        output whenever ``tp_size > 1 or ep_size > 1``.

        The answer is load-bearing whenever ``tp_size > 1``. Fused EP arms
        under tensor parallelism: the kernel takes disjoint slices of the
        replicated TP input and restores the full output on every rank. An
        all-reduce over that output would silently multiply every routed MoE
        contribution by ``tp_size``. With a TP group of world size one the
        all-reduces this flag gates return their input, so nothing changes.

        Reporting True is the right signal, rather than removing the
        all-reduce, because the SHARED expert genuinely is TP-partial
        (``Qwen3NextMLP(..., reduce_results=False)``): the same flag routes it
        to ``_maybe_reduce_shared_expert_output``, which reduces it on its own.
        """
        # The kernel can only arm with this flag set: `prebuild_fused_moe_ep`
        # returns on its first statement without it. With the flag off, answer
        # exactly as upstream does and do not import the bridge. If arming ever
        # grows a non-env path, this gate has to go with it.
        #
        # `skip_final_all_reduce` asserts the fused output is NOT pre-reduced,
        # and in that mode the model reduces externally anyway, so leave it be.
        if (not envs.USE_MOE_FUSED_EP_KERNEL
                or self.moe_config.skip_final_all_reduce):
            return super()._fused_output_is_reduced
        try:
            from vllm_torchtpu.layers.adapter.fused_moe_ep import \
                fused_moe_ep_supported
        except ImportError:
            return super()._fused_output_is_reduced
        # Asked of the layer's own quant method, because arming is per layer:
        # a model whose layers do not all qualify must not have one armed layer
        # answer for the rest.
        return (fused_moe_ep_supported(self._quant_method)
                or super()._fused_output_is_reduced)

    def _fused_ep_owns_pcp_collectives(self) -> bool:
        """Whether this layer's fused EP program does the PCP dispatch/combine.

        ``FusedMoEParallelConfig.use_all2all_kernels`` (patched in
        ``_patch_moe_explicit_pcp_collectives``) picks the owner process-wide,
        which suits chunk pipelining. The fused EP kernel is armed per layer,
        after checking that layer's weights, routing and device resources, so
        an armed layer has to suppress the outer PCP pair itself while a
        refused layer in the same model keeps the explicit collectives.
        """
        if self.moe_config.pcp_size <= 1:
            return False
        from vllm_torchtpu.layers.adapter.fused_moe_ep import \
            fused_moe_ep_supported
        return fused_moe_ep_supported(self._quant_method)

    def _maybe_dispatch(self, hidden_states, router_logits):
        if self._fused_ep_owns_pcp_collectives():
            # An armed fused method also reports supports_internal_mk, so
            # upstream's generic DP/EP dispatch is a no-op.  The only work
            # being bypassed here is its explicit PCP all-gather pair.
            return hidden_states, router_logits
        return super()._maybe_dispatch(hidden_states, router_logits)

    def _maybe_combine(self, shared_output, hidden_states):
        if self._fused_ep_owns_pcp_collectives():
            # The fused program has already pushed and summed routed rows
            # back onto their token-owning PCP rank.  Preserve upstream's
            # return shape for a separately-computed shared expert.
            if self.shared_experts is not None:
                return shared_output, hidden_states
            return hidden_states
        return super()._maybe_combine(shared_output, hidden_states)
