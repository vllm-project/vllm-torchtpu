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
"""TPU W4A16 MoE implementation for Compressed Tensors.

This class supprts W4A16 and W4A8 quantization scheme where weights are quantized in int4 format.
"""

import ctypes

import jax.numpy as jnp
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import \
    CompressedTensorsWNA16MoEMethod

from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 get_fused_moe_activation,
                                                 prebuild_fused_moe_kernel)
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.utils import (
    get_cpu_weight_loader_hook, release_memory_to_os)


class VllmCompressedTensorsW4A16MoEMethod(CompressedTensorsWNA16MoEMethod):
    """
    TODO [rsinghc]: Implement MOE_REQUANTIZE_BLOCK_SIZE for enabling quantizedFP8 acitvations.
    TODO [rsinghc]: optimize weight loading process to see if we can get rid of the cpu weight loader hook.
    TPU compressed-tensors packed-weight W4 MoE implementation.
    Supports symmetric signed INT4 weights packed as eight values per INT32
    carrier. Other bit widths, FP4 formats, asymmetric quantization, and other
    carrier dtypes are unsupported.

    W4A8 vs W4A16: The call to gmm_v2 is made with maybe_quantize_lhs=True.
    - If weight group_size < mxu_width (e.g. 16/32 in typical w4 checkpoints),
      the kernel automatically falls back to dequantizing weights before matmul,
      executing as W4A16 (BF16 activations) to respect MXU tile alignment.
    - If group_size >= mxu_widht (or channel-wise), the kernel executes as W4A8 by
      dynamically quantizing activations on the fly.
    """

    _PACKED_WEIGHT_NAMES = (
        "w13_weight_packed",
        "w2_weight_packed",
    )

    _INT4_SIGN_XOR = ctypes.c_int32(0x88888888).value

    @property
    def is_monolithic(self) -> bool:
        return True

    def _validate_w4a16_scheme(self) -> None:
        scheme = self.weight_quant

        num_bits = int(scheme.num_bits)
        quant_type = getattr(scheme, "type", None)
        if quant_type is None or "int" not in str(
                quant_type).lower() or num_bits != 4:
            raise NotImplementedError(
                "TPU W4A16 MoE supports integer INT4 weights only; "
                f"received quantization type {quant_type!r}. num_bits {num_bits}"
            )

        if not bool(getattr(scheme, "symmetric", False)):
            raise NotImplementedError(
                "TPU W4A16 MoE currently supports symmetric INT4 only.")

    def _validate_int32_weight_carriers(self, layer: torch.nn.Module) -> None:
        for param_name in self._PACKED_WEIGHT_NAMES:
            if not hasattr(layer, param_name):
                raise ValueError(
                    f"Missing required W4A16 parameter: {param_name}.")

            param = getattr(layer, param_name)

            if param.dtype != torch.int32:
                raise NotImplementedError(
                    "TPU W4A16 MoE requires INT4 weights packed in INT32; "
                    f"{param_name} has dtype {param.dtype}.")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._validate_w4a16_scheme()

        orig_loader = extra_weight_attrs.get("weight_loader")

        # Inject our optimized TPU CPU weight loader hook
        extra_weight_attrs["weight_loader"] = get_cpu_weight_loader_hook(
            layer, orig_loader, self.moe.tp_size, self.moe.tp_rank)

        # Parent handles creating w13_weight_packed, scales, and auxiliary shape parameters
        super().create_weights(layer, num_experts, hidden_size,
                               intermediate_size_per_partition, params_dtype,
                               **extra_weight_attrs)

        self._validate_int32_weight_carriers(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process W4 MoE weights after loading.
        Loads weight and transposes it to match the layout required by gmm_v2.
        [E,N,K/8] -> [E,K/8,N]
        It also performs bitwise xor to flip the sign bit.
        unsigned -> signed int4 required by
        Copies weight from cpu to tpu
        """
        self._validate_w4a16_scheme()
        self._validate_int32_weight_carriers(layer)

        # Override parent completely to prevent uint8 conversion

        # Perform CPU-side transposition and XOR flip on CPU, then copy to TPU
        for param_name in [
                "w13_weight_packed", "w2_weight_packed", "w13_weight_scale",
                "w2_weight_scale"
        ]:
            if hasattr(layer, param_name):
                param = getattr(layer, param_name)
                if hasattr(param, "_cpu_scratch"):
                    cpu_scratch = param._cpu_scratch

                    # Apply operation only on packed weights
                    if param_name in self._PACKED_WEIGHT_NAMES:
                        cpu_scratch.data.bitwise_xor_(self._INT4_SIGN_XOR)

                    transposed = cpu_scratch.data.transpose(-2,
                                                            -1).contiguous()
                    param.data.copy_(transposed)
                    delattr(param, "_cpu_scratch")

        if hasattr(layer, "w13_weight_packed") and hasattr(layer, "weight"):
            layer.weight = layer.w13_weight_packed
        layer.w13_bias = None
        layer.w2_bias = None

        release_memory_to_os()

        activation_str = get_fused_moe_activation(layer.activation,
                                                  layer.moe_config)
        moe_routing.register_experts_start_buffer(
            layer, device=layer.w13_weight_packed.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
            rhs_quant_dtype=jnp.int4,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activation_str = get_fused_moe_activation(layer.activation,
                                                  layer.moe_config)

        # Handle custom routing function if present
        custom_routing_fn = getattr(layer, "custom_routing_function", None)
        if custom_routing_fn is not None:
            topk_weights, topk_ids = custom_routing_fn(
                hidden_states=x,
                gating_output=router_logits,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
            )
        else:
            topk_weights, topk_ids = moe_routing.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
                scoring_fn=getattr(layer, "scoring_func", "softmax"),
                layer=layer,
            )

        res = fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=layer._experts_start,
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            rhs_quant_dtype=jnp.int4,
        )
        return res
