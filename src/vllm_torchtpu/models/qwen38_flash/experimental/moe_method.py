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
"""Model-owned FP8 MoE method for the Qwen3.8-Flash MoE layers.

The shared TPU FP8 MoE method serves every model. `own_moe_layer` turns the
method of one of this model's routed-expert layers, at construction and before
any weight is loaded, into `Qwen4ExpFp8MoEMethod`: the shared weight loading
and processing stay, and on top of them

- the adaptive fused EP MoE op is built from the model's kernel copy
  (kernels/adaptive_fused_moe), which gathers the FP8 activations over a
  leading axis, plus a token-sharded variant for the sequence-parallel
  prefill backbone (`SEQUENCE_PARALLEL_OP_ATTR`);
- a layer armed for the fused MoE decode kernel (moe_decode) runs it ahead of
  the adaptive op.

Layers of any other quantization keep their method.
"""

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts

from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

from .kernels.adaptive_fused_moe import vllm_adapter as adaptive_moe

SEQUENCE_PARALLEL_OP_ATTR = adaptive_moe.FUSED_MOE_EP_SP_OP_ATTR


class Qwen4ExpFp8MoEMethod(VllmFp8MoEMethodTPU):
    """VllmFp8MoEMethodTPU with this model's adaptive MoE ops and fused decode."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if not self._use_adaptive_fused_moe:
            return
        kwargs = dict(
            topk=layer.moe_config.experts_per_token,
            renormalize=layer.renormalize,
            activation=self._tpu_activation_str,
        )
        setattr(
            self,
            adaptive_moe.FUSED_MOE_EP_OP_ATTR,
            adaptive_moe.prebuild_adaptive_fused_moe(layer, **kwargs),
        )
        adaptive_moe.prebuild_sequence_parallel_variant(self, layer, **kwargs)

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Deferred: the fused decode adapter loads its kernel copy.
        from .moe_decode import fused_moe_decode_forward

        out = fused_moe_decode_forward(self, layer, x, router_logits)
        if out is not None:
            return out
        if self._use_adaptive_fused_moe:
            return adaptive_moe.run_adaptive_fused_moe(
                self,
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                router_logits,
            )
        return super().apply_monolithic(layer, x, router_logits, input_ids)


def own_moe_layer(
    routed_experts: torch.nn.Module,
    *,
    sequence_parallel_variant: bool,
) -> bool:
    """Give one routed-expert layer the model-owned FP8 method.

    Call at construction, before weights load. Returns False, leaving the layer
    as it is, unless its method is exactly the shared FP8 MoE method.
    """
    method = getattr(routed_experts, "quant_method", None)
    if type(method) is not VllmFp8MoEMethodTPU:
        return False
    method.__class__ = Qwen4ExpFp8MoEMethod
    setattr(routed_experts, adaptive_moe.AG_LEAD_AXIS_ATTR, True)
    setattr(
        routed_experts,
        adaptive_moe.SEQUENCE_PARALLEL_VARIANT_ATTR,
        bool(sequence_parallel_variant),
    )
    return True
