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
MXFP4 Quantization Support for TPU.

This module provides TPU-compatible MXFP4 quantization for MoE models like
openai/gpt-oss-20b. MXFP4 is a 4-bit floating point format with block-wise
scaling.

== Execution Flow ==

1. MODEL LOADING:
   - vLLM detects quantization="mxfp4" in model config
   - get_tpu_quantization_config() returns VllmMxfp4Config
   - VllmMxfp4Config is registered with vLLM's quantization registry

2. LAYER CREATION:
   - For each RoutedExperts layer, vLLM calls quant_config.get_quant_method(layer)
   - VllmMxfp4Config.get_quant_method() returns VllmMxfp4MoEMethod

3. WEIGHT CREATION:
   - VllmMxfp4MoEMethod.create_weights() is called
   - Allocates w13_weight, w2_weight, scales, and biases

4. WEIGHT LOADING:
   - Weights are loaded from safetensors/checkpoint
   - VllmMxfp4MoEMethod.process_weights_after_loading() is called
   - Dequantizes MXFP4 -> FP32 -> re-quantizes for TPU kernel

5. INFERENCE:
   - VllmMxfp4MoEMethod.apply() is called with input tensor
   - Routes tokens to experts using gating output
   - Calls TPU MoE kernel (GMM or fused MoE)
   - Returns output tensor
"""

from typing import Any, Optional

import jax.numpy as jnp
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig, mxfp4_w4a16_moe_quant_config)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import \
    Mxfp4MoeBackend as Mxfp4Backend
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import (Mxfp4Config,
                                                           Mxfp4MoEMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from vllm_torchtpu import envs
from vllm_torchtpu.layers.common.quant_methods import (MXFP4,
                                                       get_tpu_quant_method)
from vllm_torchtpu.layers.common.quantization import dequantize_mxfp4_packed
from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 prebuild_fused_moe_kernel,
                                                 quantize_native_fp4_kmajor)
from vllm_torchtpu.layers.vllm.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute, pipelined_fused_moe_gmm)
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(MXFP4))
class VllmMxfp4Config(Mxfp4Config, VllmQuantConfig):
    """
    TPU-specific MXFP4 quantization configuration.

    This class is registered with vLLM's quantization system under the name
    "tpu-mxfp4" and provides TPU-compatible quantization methods for each
    layer type.
    """

    @classmethod
    def get_name(cls) -> str:
        """Return the base quantization method name."""
        return MXFP4

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        """
        Return the appropriate quantization method for a layer.

        This is called by vLLM during model construction to get the
        quantization handler for each layer.

        Args:
            layer: The layer to quantize.
            prefix: The parameter prefix (e.g., "model.layers.0.mlp").

        Returns:
            A QuantizeMethodBase implementation, or None to skip quantization.
        """
        if isinstance(layer, LinearBase):
            # MXFP4 only quantizes MoE layers, not linear layers.
            # For linear layers (attention projections, etc.), use unquantized method.
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                from vllm_torchtpu.layers.vllm.quantization.unquantized import \
                    VllmUnquantizedLinearMethod
                return VllmUnquantizedLinearMethod()
            # MXFP4 linear layer is not implemented - use unquantized
            logger.warning_once(
                "MXFP4 linear layer is not implemented on TPU - "
                "using unquantized linear method.")
            from vllm_torchtpu.layers.vllm.quantization.unquantized import \
                VllmUnquantizedLinearMethod
            return VllmUnquantizedLinearMethod()

        elif isinstance(layer, RoutedExperts):
            # RoutedExperts is the main use case for MXFP4 (MoE models like GPT-OSS)
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config)

        elif isinstance(layer, Attention):
            logger.warning_once(
                "MXFP4 attention layer is not implemented on TPU. "
                "Skipping quantization for this layer.")

        return None


def _get_activation_str(activation) -> str:
    """Convert MoEActivation enum or string to plain string."""
    return activation.value if hasattr(activation,
                                       'value') else str(activation)


class VllmMxfp4MoEMethod(Mxfp4MoEMethod):
    """
    TPU-specific implementation of MXFP4 MoE quantization.

    Uses is_monolithic=True so vLLM's DefaultMoERunner calls
    apply_monolithic(layer, x, router_logits) directly, bypassing the
    CUDA-only router.select_experts() path.

    Inherits from Mxfp4MoEMethod to reuse its create_weights() method,
    but bypasses its __init__ to avoid the GPU backend assertion.
    """
    # Upstream Mxfp4MoEMethod.__init__ defines is_k3_situ_aiter for GPU AITER kernels.
    # Since __init__ is bypassed on TPU, explicitly define it here as False.
    is_k3_situ_aiter: bool = False

    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def supports_internal_mk(self) -> bool:
        # We need to take control of collective communication (AllGather/ReduceScatter)
        # to pipeline them with MoE computation when chunking is enabled.
        return enable_pipelined_collective_and_compute()

    def __init__(self, moe: FusedMoEConfig):
        # Skip Mxfp4MoEMethod.__init__ (GPU backend assertion)
        FusedMoEMethodBase.__init__(self, moe)
        self.mxfp4_backend = Mxfp4Backend.TRITON
        self.apply_monolithic = self._select_monolithic()

    def _select_monolithic(self):
        return self._forward_monolithic_tpu

    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, RoutedExperts)
        assert layer.moe_config.has_bias, "MXFP4 quantization requires bias."

        activation_str = _get_activation_str(layer.activation)
        layer._tpu_activation_str = activation_str

        w13_weight = dequantize_mxfp4_packed(
            layer.w13_weight.data.to("cpu"),
            layer.w13_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w13_weight.device)
        w2_weight = dequantize_mxfp4_packed(
            layer.w2_weight.data.to("cpu"),
            layer.w2_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w2_weight.device)

        w13_interleave = activation_str == "swigluoai"
        if w13_interleave:
            w1_weight = w13_weight[:, ::2, :]
            w3_weight = w13_weight[:, 1::2, :]
            w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

        w13_weight = w13_weight.transpose(1, 2).contiguous()
        w2_weight = w2_weight.transpose(1, 2).contiguous()

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        layer.w13_weight_scale = None
        layer.w2_weight_scale = None

        if layer.w13_bias is not None:
            w13_bias = layer.w13_bias.data
            if w13_interleave:
                w1_bias = w13_bias[:, ::2]
                w3_bias = w13_bias[:, 1::2]
                w13_bias = torch.cat([w1_bias, w3_bias], dim=1)

            layer.w13_bias = torch.nn.Parameter(
                w13_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
        if layer.w2_bias is not None:
            layer.w2_bias = torch.nn.Parameter(
                layer.w2_bias.data.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )

        if layer.w13_weight.device.type == "tpu":
            to_sync = [layer.w13_weight, layer.w2_weight]
            if layer.w13_bias is not None:
                to_sync.append(layer.w13_bias)
            if layer.w2_bias is not None:
                to_sync.append(layer.w2_bias)
            synchronize_tensors(to_sync)

        logger.info_once(
            "MXFP4 weights dequantized to bfloat16 and transposed for GMM kernel."
        )
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(
            layer, device=layer.w13_weight.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> Optional[FusedMoEQuantConfig]:
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    rhs_quant_dtype: Any = None

    def _forward_monolithic_tpu(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using GMM kernel."""
        activation_str = layer._tpu_activation_str

        # Step 1: Routing
        # Quantization-independent routing decision (simulation override ->
        # custom_routing_function -> select_experts); shared across all TPU MoE
        # methods so the routing-simulation hook lives in exactly one place.
        # input_ids is forwarded for DeepSeek-V4 hash routing; mxfp4 is the only
        # method that supplies it, matching the pre-consolidation behavior.
        topk_weights, topk_ids = moe_routing.route(layer,
                                                   x,
                                                   router_logits,
                                                   input_ids=input_ids)

        kwargs = {
            "hidden_states": x,
            "w1": layer.w13_weight,
            "w2": layer.w2_weight,
            "w1_scale": layer.w13_weight_scale,
            "w2_scale": layer.w2_weight_scale,
            "w1_bias": getattr(layer, "w13_bias", None),
            "w2_bias": getattr(layer, "w2_bias", None),
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "experts_start": layer._experts_start,
            "topk": layer.moe_config.experts_per_token,
            "activation": activation_str,
            "rhs_quant_dtype": self.rhs_quant_dtype,
        }
        if enable_pipelined_collective_and_compute():
            return pipelined_fused_moe_gmm(**kwargs)

        return fused_moe_gmm(**kwargs)


REQUANTIZED_BLOCK_SIZE = (int(envs.MOE_REQUANTIZE_BLOCK_SIZE)
                          if envs.MOE_REQUANTIZE_BLOCK_SIZE else 512)


# TODO(upstream): Replace this subclass with upstream vLLM's MXFP4 method once
# vLLM natively supports DeepSeek-V4 split w1/w3/w2 checkpoint loading.
class VllmDeepseekV4Mxfp4MoEMethod(VllmMxfp4MoEMethod):
    """DeepSeek-V4 MXFP4 MoE quantization and weight loading method."""
    rhs_quant_dtype = jnp.float4_e2m1fn

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        Mxfp4MoEMethod.create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def _dequantize_and_pad(
        self,
        layer: RoutedExperts,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
               | None, int]:
        """Dequantize staged shards on CPU, concatenate w13, and pad to strided block boundaries."""
        w1_scale = layer.w13_weight_scale.data.to(
            "cpu")[:, :self.intermediate_size, :]
        w3_scale = layer.w13_weight_scale.data.to(
            "cpu")[:, self.intermediate_size:, :]
        w2_scale = layer.w2_weight_scale.data.to("cpu")

        # w13 packs w1|w3 as contiguous halves along dim 1, matching how the
        # scales are sliced above.
        w13_packed = layer.w13_weight.data.to("cpu")
        w2_packed = layer.w2_weight.data.to("cpu")
        w1_packed = w13_packed[:, :self.intermediate_size, :]
        w3_packed = w13_packed[:, self.intermediate_size:, :]

        w1_dequant = dequantize_mxfp4_packed(w1_packed,
                                             w1_scale,
                                             axis=2,
                                             out_dtype=torch.bfloat16)
        w3_dequant = dequantize_mxfp4_packed(w3_packed,
                                             w3_scale,
                                             axis=2,
                                             out_dtype=torch.bfloat16)
        w2_dequant = dequantize_mxfp4_packed(w2_packed,
                                             w2_scale,
                                             axis=2,
                                             out_dtype=torch.bfloat16)
        _w13_bias = getattr(layer, "w13_bias", None)
        _w2_bias = getattr(layer, "w2_bias", None)
        w13_bias = _w13_bias.data.to("cpu") if _w13_bias is not None else None
        w2_bias = _w2_bias.data.to("cpu") if _w2_bias is not None else None

        w13_weight = torch.cat([w1_dequant, w3_dequant],
                               dim=1).transpose(1, 2).contiguous()
        w2_weight = w2_dequant.transpose(1, 2).contiguous()

        orig_hidden_size = w2_weight.shape[2]
        orig_intermediate_size = w2_weight.shape[1]
        w13_block_size = REQUANTIZED_BLOCK_SIZE

        intermediate_padded = -(-orig_intermediate_size // 128) * 128
        hidden_padded = -(-orig_hidden_size // w13_block_size) * w13_block_size

        w1_weight = w13_weight[:, :, :orig_intermediate_size]
        w3_weight = w13_weight[:, :, orig_intermediate_size:]
        pad_intermediate = intermediate_padded - orig_intermediate_size
        pad_hidden = hidden_padded - orig_hidden_size
        pad_spec_w13 = (0, pad_intermediate, 0, pad_hidden)

        w13_weight_padded = torch.cat([
            torch.nn.functional.pad(w1_weight, pad_spec_w13),
            torch.nn.functional.pad(w3_weight, pad_spec_w13),
        ],
                                      dim=2)
        w2_weight_padded = torch.nn.functional.pad(
            w2_weight, (0, pad_hidden, 0, pad_intermediate))

        w13_bias_padded = w2_bias_padded = None
        if w13_bias is not None:
            b1, b3 = (w13_bias[:, :orig_intermediate_size],
                      w13_bias[:, orig_intermediate_size:])
            w13_bias_padded = torch.cat([
                torch.nn.functional.pad(b1, (0, pad_intermediate)),
                torch.nn.functional.pad(b3, (0, pad_intermediate)),
            ],
                                        dim=1).unsqueeze(1)
        if w2_bias is not None:
            w2_bias_padded = torch.nn.functional.pad(
                w2_bias, (0, pad_hidden)).unsqueeze(1)

        return (w13_weight_padded, w2_weight_padded, w13_bias_padded,
                w2_bias_padded, orig_intermediate_size)

    def _requantize_native_fp4(
        self,
        w13_weight_padded: torch.Tensor,
        w2_weight_padded: torch.Tensor,
        orig_intermediate_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize padded tensors to native FP4 K-major layout on TPU."""
        w13_block_size = REQUANTIZED_BLOCK_SIZE
        w2_block_size = min(REQUANTIZED_BLOCK_SIZE, orig_intermediate_size)

        w13_weight_processed, w13_weight_scale = quantize_native_fp4_kmajor(
            w13_weight_padded.to(device), block=w13_block_size)
        w2_weight_processed, w2_weight_scale = quantize_native_fp4_kmajor(
            w2_weight_padded.to(device), block=w2_block_size)

        return (w13_weight_processed, w2_weight_processed, w13_weight_scale,
                w2_weight_scale)

    def _release_source_weights(self, layer: RoutedExperts):
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale, layer.w2_weight_scale
        if hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias"):
            del layer.w13_bias, layer.w2_bias

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # TODO(upstream): Extract generic on-disk quantization cache to a
        # standalone utility PR.
        assert isinstance(layer, RoutedExperts)
        device = layer.w13_weight.device

        activation_str = _get_activation_str(layer.activation)
        if activation_str.lower() in ("silu", "silu_and_mul", "swiglu"):
            activation_str = "silu_and_mul_with_clamp"
        layer._tpu_activation_str = activation_str

        (w13_weight_padded, w2_weight_padded, w13_bias_padded, w2_bias_padded,
         orig_intermediate) = self._dequantize_and_pad(layer)
        (w13_weight_processed, w2_weight_processed, w13_weight_scale,
         w2_weight_scale) = self._requantize_native_fp4(
             w13_weight_padded, w2_weight_padded, orig_intermediate, device)
        self._release_source_weights(layer)

        layer.w13_weight = torch.nn.Parameter(w13_weight_processed,
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight_processed,
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale,
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                   requires_grad=False)
        if w13_bias_padded is not None and w2_bias_padded is not None:
            layer.w13_bias = torch.nn.Parameter(w13_bias_padded.to(device),
                                                requires_grad=False)
            layer.w2_bias = torch.nn.Parameter(w2_bias_padded.to(device),
                                               requires_grad=False)

        to_sync = [layer.w13_weight, layer.w2_weight]
        if hasattr(layer, "w13_bias") and layer.w13_bias is not None:
            to_sync.append(layer.w13_bias)
        if hasattr(layer, "w2_bias") and layer.w2_bias is not None:
            to_sync.append(layer.w2_bias)
        synchronize_tensors(to_sync)

        logger.info(
            "DeepSeek-V4 MXFP4 weights processed and stored as native FP4 "
            "for GMM kernel.")
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(layer, device=device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )
