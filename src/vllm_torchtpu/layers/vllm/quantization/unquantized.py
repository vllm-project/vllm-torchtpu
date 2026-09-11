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
"""
TPU-native unquantized (bfloat16/float16) support for vLLM.

This module provides TPU-compatible implementations for unquantized models
like Qwen3-Coder. It bypasses vLLM's default `torch_xla`-based TPU path
and uses our native TorchTPU + Pallas kernels.

== Supported Models ==
- Qwen3-Coder-30B-A3B (unquantized MoE)
- Any MoE model without quantization

== Key Classes ==
- VllmUnquantizedConfig: Quantization config for unquantized models
- VllmUnquantizedFusedMoEMethod: MoE forward pass using GMM kernel
- VllmUnquantizedLinearMethod: Linear layer using standard matmul

== Execution Flow ==
1. vLLM loads model with quantization=None (unquantized)
2. get_tpu_quantization_config() returns VllmUnquantizedConfig
3. For RoutedExperts layers, VllmUnquantizedFusedMoEMethod.apply() is called
4. apply() routes to our TPU Pallas local-topk MoE kernels
"""

from typing import Any, Optional

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (RoutedExperts,
                                                  UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import replace_parameter

from vllm_torchtpu.layers.common.quant_methods import (UNQUANTIZED,
                                                       get_tpu_quant_method)
from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (TpuMoEActivationMixin,
                                                 fused_moe_gmm,
                                                 prebuild_fused_moe_kernel)
from vllm_torchtpu.layers.vllm.linear_common import (KEEP_VLLM_LAYOUT_ATTR,
                                                     WEIGHT_FLIPPED_ATTR)
from vllm_torchtpu.layers.vllm.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute, pipelined_fused_moe_gmm)
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


class VllmUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Dense linear on the canonical (k, n) weight layout.

    vLLM's stock `UnquantizedLinearMethod` runs `F.linear(x, W)` with `W`
    stored `[n_out, n_in]`, i.e. `(m, k) @ (n, k).T`. The transposed
    contraction is inefficient on TPU, so the weight is transposed once after
    loading and the forward pass becomes a plain `(m, k) @ (k, n)` matmul.
    This matches the layout the FP8 dense path and the MoE GMM kernels already
    use.

    `create_weights` is inherited unchanged on purpose: vLLM's loaders slice
    `[n_out, n_in]` using the parameter's `input_dim=1` / `output_dim=0`
    attributes to shard across TP ranks, so the transpose has to happen after
    loading rather than at creation.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False):
            # A consumer reads this weight raw and expects [n_out, n_in];
            # `apply` below falls back to the stock contraction.
            return
        if getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            return  # already flipped; reload paths re-enter this hook
        weight = layer.weight.data
        if weight.dim() != 2:
            # Not a plain dense weight; leave the stock behaviour alone.
            return
        weight = weight.transpose(0, 1).contiguous()
        replace_parameter(layer, "weight",
                          torch.nn.Parameter(weight, requires_grad=False))
        setattr(layer, WEIGHT_FLIPPED_ATTR, True)
        if layer.weight.device.type == "tpu":
            synchronize_tensors([layer.weight])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False):
            # Left in vLLM's [n_out, n_in] for a consumer that reads it raw.
            return super().apply(layer, x, bias)
        # layer.weight is [n_in, n_out], so this contracts x's trailing dim
        # against axis 0 with no transpose.
        weight = layer.weight
        if bias is None:
            return torch.matmul(x, weight)
        # Fuse the bias via addmm rather than `matmul(...) + bias`: the latter
        # rounds the product to the activation dtype before adding, which
        # diverges from F.linear's single rounding and would silently change
        # outputs for every bf16 model. addmm accumulates then rounds once,
        # keeping this bit-identical to vLLM's stock path.
        orig_out_shape = (*x.shape[:-1], weight.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        return torch.addmm(bias, x_2d, weight).reshape(orig_out_shape)


@register_quantization_config(get_tpu_quant_method(UNQUANTIZED))
class VllmUnquantizedConfig(QuantizationConfig, VllmQuantConfig):
    """
    TPU-specific configuration for unquantized models.

    This config is registered under "tpu-unquantized" and provides
    TPU-compatible methods for RoutedExperts and Linear layers.
    """

    @classmethod
    def get_name(cls) -> str:
        return UNQUANTIZED

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # Always supported on TPU

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # No extra configs required

    @classmethod
    def from_config(cls, _: dict[str, Any]) -> "VllmUnquantizedConfig":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        """
        Return the appropriate quantization method for a layer.

        Args:
            layer: The layer to handle.
            prefix: The parameter prefix (e.g., "model.layers.0.mlp").

        Returns:
            A QuantizeMethodBase implementation, or None to use defaults.
        """
        if isinstance(layer, LinearBase):
            # TPU-native dense linear: canonical (k, n) weight layout so the
            # forward pass is (m, k) @ (k, n) rather than vLLM's
            # (m, k) @ (n, k).T.
            return VllmUnquantizedLinearMethod()

        if isinstance(layer, RoutedExperts):
            # For MoE layers, use our custom TPU implementation
            moe_config = self.get_moe_config(layer)
            return VllmUnquantizedFusedMoEMethod(moe_config)

        if isinstance(layer, Attention):
            return None
        return None


class VllmUnquantizedFusedMoEMethod(TpuMoEActivationMixin,
                                    UnquantizedFusedMoEMethod):
    """
    TPU-native implementation of unquantized RoutedExperts.

    Uses is_monolithic=True so vLLM's DefaultMoERunner calls
    apply_monolithic(layer, x, router_logits) directly, bypassing the
    CUDA-only router.select_experts() path.
    """

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        # Parent sets _is_monolithic=False on TPU and skips wiring
        # apply_monolithic.  Override both after super().__init__().
        self.apply_monolithic = self._select_monolithic()

    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def supports_internal_mk(self) -> bool:
        # We need to take control of collective communication (AllGather/ReduceScatter)
        # to pipeline them with MoE computation when chunking is enabled.
        return enable_pipelined_collective_and_compute()

    def _select_monolithic(self):
        return self._forward_monolithic_tpu

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, RoutedExperts)

        # Pre-compute activation string: layer.activation is a MoEActivation
        # enum in v0.17.1 but the Pallas kernel expects a plain string.
        # Resolve here, outside torch.compile.
        activation_str = self._set_tpu_activation(layer)

        w13_weight = layer.w13_weight.data
        w2_weight = layer.w2_weight.data

        w13_interleave = activation_str == "swigluoai"
        if w13_interleave:
            w1_weight = w13_weight[:, ::2, :]
            w3_weight = w13_weight[:, 1::2, :]
            w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

        w13_weight = w13_weight.transpose(1, 2).contiguous()
        w2_weight = w2_weight.transpose(1, 2).contiguous()

        # Pad each w13 half (gate, up) to a multiple of 128 so the fused GMM
        # kernel's sub-view of the up half lands on a tile boundary.
        half = w13_weight.shape[-1] // 2
        aligned_half = (half + 127) // 128 * 128
        if aligned_half != half:
            pad = w13_weight.new_zeros(
                (*w13_weight.shape[:-1], aligned_half - half))
            w13_weight = torch.cat(
                [w13_weight[..., :half], pad, w13_weight[..., half:], pad],
                dim=-1).contiguous()
            pad2 = w2_weight.new_zeros(
                (w2_weight.shape[0], aligned_half - half, w2_weight.shape[2]))
            w2_weight = torch.cat([w2_weight, pad2], dim=1).contiguous()

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        if self.moe.has_bias:
            w13_bias = layer.w13_bias.data
            w2_bias = layer.w2_bias.data

            if w13_interleave:
                w1_bias = w13_bias[:, ::2]
                w3_bias = w13_bias[:, 1::2]
                w13_bias = torch.cat([w1_bias, w3_bias], dim=1)

            if aligned_half != half:
                bpad = w13_bias.new_zeros(
                    (*w13_bias.shape[:-1], aligned_half - half))
                w13_bias = torch.cat(
                    [w13_bias[..., :half], bpad, w13_bias[..., half:], bpad],
                    dim=-1).contiguous()

            layer.w13_bias = torch.nn.Parameter(
                w13_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
            layer.w2_bias = torch.nn.Parameter(
                w2_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )

        if layer.w13_weight.device.type == "tpu":
            to_sync = [layer.w13_weight, layer.w2_weight]
            if self.moe.has_bias:
                to_sync.append(layer.w13_bias)
                to_sync.append(layer.w2_bias)
            synchronize_tensors(to_sync)

        logger.info_once("Unquantized weights transposed for GMM kernel: "
                         f"w13={list(layer.w13_weight.shape)}, "
                         f"w2={list(layer.w2_weight.shape)}")
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(
            layer, device=layer.w13_weight.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )

    def _forward_monolithic_tpu(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using TPU-native GMM kernel."""
        activation_str = self._tpu_activation_str
        assert activation_str is not None, (
            "[moe] process_weights_after_loading did not run for this layer")
        # Step 1: Routing
        # Quantization-independent routing decision (simulation override ->
        # custom_routing_function -> select_experts); shared across all TPU MoE
        # methods so the routing-simulation hook lives in exactly one place.
        topk_weights, topk_ids = moe_routing.route(layer, x, router_logits)

        kwargs = {
            "hidden_states": x,
            "w1": layer.w13_weight,
            "w2": layer.w2_weight,
            "w1_scale": None,
            "w2_scale": None,
            "w1_bias": getattr(layer, 'w13_bias', None),
            "w2_bias": getattr(layer, 'w2_bias', None),
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "experts_start": layer._experts_start,
            "topk": layer.moe_config.experts_per_token,
            "activation": activation_str,
        }
        if enable_pipelined_collective_and_compute():
            return pipelined_fused_moe_gmm(**kwargs)

        return fused_moe_gmm(**kwargs)
