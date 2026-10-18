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

from math import lcm
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend as Mxfp4Backend,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.fused_moe import (
    TpuMoEActivationMixin,
    fused_moe_gmm,
    prebuild_fused_moe_kernel,
    quantize_native_fp4_kmajor,
)
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute,
    pipelined_fused_moe_gmm,
)
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.core.quant_methods import MXFP4, get_tpu_quant_method
from vllm_torchtpu.layers.core.quantization import (
    dequantize_mxfp4_packed,
    dequantize_tensor,
    e8m0_to_fp32_bits,
    materialize_static_scale,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.configs import TpuRoutedExperts

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
    def get_name(cls) -> QuantizationMethods:
        """Return the base quantization method name."""
        return MXFP4

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
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
                from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                    VllmUnquantizedLinearMethod,
                )

                return VllmUnquantizedLinearMethod()
            # MXFP4 linear layer is not implemented - use unquantized
            logger.warning_once(
                "MXFP4 linear layer is not implemented on TPU - "
                "using unquantized linear method."
            )
            from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                VllmUnquantizedLinearMethod,
            )

            return VllmUnquantizedLinearMethod()

        elif isinstance(layer, RoutedExperts):
            # RoutedExperts is the main use case for MXFP4 (MoE models like GPT-OSS)
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config)

        elif isinstance(layer, Attention):
            logger.warning_once(
                "MXFP4 attention layer is not implemented on TPU. "
                "Skipping quantization for this layer."
            )

        return None


def _dequantize_mxfp4_packed_on_device(
    weight_packed: torch.Tensor,
    scale_u8: torch.Tensor,
    axis: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """`dequantize_mxfp4_packed`, bit for bit, without leaving the device.

    Scales are decoded by constructing their float32 bits on the device.
    The fp4 nibbles are decoded with selects rather than a LUT gather.
    Every product is an fp4 value times a power of two, so it is exact in
    fp32 and in bf16.
    """
    scale = e8m0_to_fp32_bits(scale_u8)

    def decode(nibble: torch.Tensor) -> torch.Tensor:
        exponent = (nibble >> 1) & 3
        mantissa = (nibble & 1).to(torch.float32)
        # e2m1: 0, 0.5 | 1, 1.5 | 2, 3 | 4, 6
        magnitude = torch.where(
            exponent == 0,
            mantissa * 0.5,
            torch.where(
                exponent == 1,
                1.0 + mantissa * 0.5,
                torch.where(exponent == 2, 2.0 + mantissa, 4.0 + mantissa * 2.0),
            ),
        )
        return torch.where((nibble & 8) != 0, -magnitude, magnitude)

    packed = weight_packed.to(torch.int32)
    unpacked = torch.stack([decode(packed & 0x0F), decode(packed >> 4)], dim=-1)
    unpacked = unpacked.reshape(*weight_packed.shape[:-1], -1)
    return dequantize_tensor(unpacked, scale, axis, out_dtype)


def _get_activation_str(activation) -> str:
    """Convert MoEActivation enum or string to plain string."""
    return activation.value if hasattr(activation, "value") else str(activation)


class VllmMxfp4MoEMethod(TpuMoEActivationMixin, Mxfp4MoEMethod):
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

    def _resolve_tpu_activation(self, layer) -> str:
        # Deliberately not get_fused_moe_activation: that also encodes the
        # situ beta parameters, which this kernel path does not take.
        return _get_activation_str(layer.activation)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, RoutedExperts)
        layer = cast("TpuRoutedExperts", layer)
        assert layer.moe_config.has_bias, "MXFP4 quantization requires bias."

        activation_str = self._set_tpu_activation(layer)

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

        # Same as `= None`, which torch's stubs reject.
        layer.register_parameter("w13_weight_scale", None)
        layer.register_parameter("w2_weight_scale", None)

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
        moe_routing.register_experts_start_buffer(layer, device=layer.w13_weight.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )

    rhs_quant_dtype: Any = None

    def _forward_monolithic_tpu(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using GMM kernel."""
        layer = cast("TpuRoutedExperts", layer)
        activation_str = self._tpu_activation_str
        assert activation_str is not None, (
            "[moe] process_weights_after_loading did not run for this layer"
        )

        # Step 1: Routing
        # Quantization-independent routing decision (simulation override ->
        # custom_routing_function -> select_experts); shared across all TPU MoE
        # methods so the routing-simulation hook lives in exactly one place.
        # input_ids is forwarded for DeepSeek-V4 hash routing; mxfp4 is the only
        # method that supplies it, matching the pre-consolidation behavior.
        topk_weights, topk_ids = moe_routing.route(
            layer, x, router_logits, input_ids=input_ids
        )

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


REQUANTIZED_BLOCK_SIZE = (
    int(envs.MOE_REQUANTIZE_BLOCK_SIZE) if envs.MOE_REQUANTIZE_BLOCK_SIZE else 512
)


def _row_padded_flat_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """[E, K/block, 1, N] -> the kernel's row-padded flat [E * rows, N].

    rows is K/block rounded up to the f32 sublane tile (fp4_scale_rows); the
    padding rows are zero and the kernel never reads them. Materialize the
    logical shape once at load time so the runtime does not copy the view
    before every compiled step.
    """
    from vllm_torchtpu.kernels.fused_moe.v2.host import fp4_scale_rows

    experts, blocks, _, n = scale.shape
    rows = fp4_scale_rows(blocks, 1, padded=True)
    padded = torch.nn.functional.pad(scale.squeeze(2), (0, 0, 0, rows - blocks))
    flat = padded.reshape(experts * rows, n).clone()
    return materialize_static_scale(flat)


# TODO(upstream): Replace this subclass with upstream vLLM's MXFP4 method once
# vLLM natively supports DeepSeek-V4 split w1/w3/w2 checkpoint loading.
class VllmDeepseekV4Mxfp4MoEMethod(VllmMxfp4MoEMethod):
    """DeepSeek-V4 MXFP4 MoE quantization and weight loading method."""

    rhs_quant_dtype = jnp.float4_e2m1fn

    @property
    def supports_internal_mk(self) -> bool:
        # An armed layer owns its own dispatch and combine, so vLLM must not
        # bracket it with an all-gather/reduce-scatter of its own.
        from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

        return enable_pipelined_collective_and_compute() or fused_moe_ep_supported(self)

    def _prebuild_w4a8(
        self,
        layer: RoutedExperts,
        activation: str,
        w13_weight_padded: torch.Tensor,
        w2_weight_padded: torch.Tensor,
        orig_intermediate_size: int,
    ) -> Any | None:
        """Admit the fused EP layout before the weights are quantized to it.

        Runs on the padded but still-dequantized tensors: the requantized
        layout is a function of their shapes and the block size, so admission
        needs nothing lossy to have happened yet, and the meta tensors below
        describe exactly what `_requantize_native_fp4` goes on to produce.
        Shapes are read off those tensors rather than recomputed so the two
        cannot drift apart. Returns the op, or None to leave this layer on GMM.
        """
        from vllm_torchtpu.layers.adapter.fused_moe_ep import prebuild_fused_moe_ep

        if not (
            envs.USE_MOE_FUSED_EP_KERNEL
            and envs.MOE_FUSED_EP_ENABLE_W4A8
            and layer.moe_config.moe_parallel_config.use_ep
        ):
            return None
        block = REQUANTIZED_BLOCK_SIZE
        if min(block, orig_intermediate_size) != block:
            # w2 would be quantized at a smaller block than w13, and the op
            # closes over one `rhs_qb` that both matmuls read.
            logger.info_once(
                "DeepSeek-V4 MXFP4 fused EP not engaged: w2's block is "
                "min(%d, intermediate=%d), which differs from w13's %d.",
                block,
                orig_intermediate_size,
                block,
            )
            return None
        experts, hidden, w13_channels = w13_weight_padded.shape
        inter = w2_weight_padded.shape[1]
        # Torch stores two FP4 values per element of a `float4_e2m1fn_x2`
        # tensor, so every shape here has its LAST axis halved against the
        # logical one; the scales are unpacked f32 and keep theirs.
        weights = (
            torch.empty(
                (experts, hidden, w13_channels // 2),
                dtype=torch.float4_e2m1fn_x2,
                device="meta",
            ),
            torch.empty(
                (experts, inter, hidden // 2),
                dtype=torch.float4_e2m1fn_x2,
                device="meta",
            ),
            torch.empty(
                (experts, hidden // block, 1, w13_channels),
                dtype=torch.float32,
                device="meta",
            ),
            torch.empty(
                (experts, inter // block, 1, hidden), dtype=torch.float32, device="meta"
            ),
        )
        return prebuild_fused_moe_ep(
            layer,
            topk=layer.moe_config.experts_per_token,
            renormalize=layer.renormalize,
            activation=activation,
            weight_format="fp4",
            rhs_qb=block,
            weights=weights,
            supplies_hash_ids=True,
        )

    def _resolve_tpu_activation(self, layer) -> str:
        # This variant runs the clamped kernel, so the string is not
        # recoverable from layer.activation alone -- resolving it at apply
        # time instead of storing it here would silently drop the clamp.
        activation_str = super()._resolve_tpu_activation(layer)
        if activation_str.lower() in ("silu", "silu_and_mul", "swiglu"):
            return "silu_and_mul_with_clamp"
        return activation_str

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
        layer: "TpuRoutedExperts",
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, int
    ]:
        """Dequantize staged shards, concatenate w13, and pad to strided block
        boundaries: on the device the shards sit on when
        VLLM_TPU_MXFP4_DEVICE_DEQUANT is set, else on CPU."""
        assert isinstance(layer.w13_weight, torch.Tensor)
        assert isinstance(layer.w13_weight_scale, torch.Tensor)
        assert isinstance(layer.w2_weight, torch.Tensor)
        assert isinstance(layer.w2_weight_scale, torch.Tensor)
        if envs.VLLM_TPU_MXFP4_DEVICE_DEQUANT:
            work_device = layer.w13_weight.device
            dequantize = _dequantize_mxfp4_packed_on_device
        else:
            work_device = torch.device("cpu")
            dequantize = dequantize_mxfp4_packed
        w1_scale = layer.w13_weight_scale.data.to(work_device)[
            :, : self.intermediate_size, :
        ]
        w3_scale = layer.w13_weight_scale.data.to(work_device)[
            :, self.intermediate_size :, :
        ]
        w2_scale = layer.w2_weight_scale.data.to(work_device)

        # w13 packs w1|w3 as contiguous halves along dim 1, matching how the
        # scales are sliced above.
        w13_packed = layer.w13_weight.data.to(work_device)
        w2_packed = layer.w2_weight.data.to(work_device)
        w1_packed = w13_packed[:, : self.intermediate_size, :]
        w3_packed = w13_packed[:, self.intermediate_size :, :]

        w1_dequant = dequantize(w1_packed, w1_scale, axis=2, out_dtype=torch.bfloat16)
        w3_dequant = dequantize(w3_packed, w3_scale, axis=2, out_dtype=torch.bfloat16)
        w2_dequant = dequantize(w2_packed, w2_scale, axis=2, out_dtype=torch.bfloat16)
        _w13_bias = getattr(layer, "w13_bias", None)
        _w2_bias = getattr(layer, "w2_bias", None)
        w13_bias = _w13_bias.data.to(work_device) if _w13_bias is not None else None
        w2_bias = _w2_bias.data.to(work_device) if _w2_bias is not None else None

        w13_weight = (
            torch.cat([w1_dequant, w3_dequant], dim=1).transpose(1, 2).contiguous()
        )
        w2_weight = w2_dequant.transpose(1, 2).contiguous()

        orig_hidden_size = w2_weight.shape[2]
        orig_intermediate_size = w2_weight.shape[1]
        w13_block_size = REQUANTIZED_BLOCK_SIZE

        # W13's output needs the GMM tile alignment; W2 contracts over the
        # same dimension and must also contain complete requantization blocks.
        # V4.1's 2304 channels satisfy the former but not a block of 512.
        w2_block_size = min(w13_block_size, orig_intermediate_size)
        intermediate_alignment = lcm(128, w2_block_size)
        intermediate_padded = (
            -(-orig_intermediate_size // intermediate_alignment)
            * intermediate_alignment
        )
        hidden_padded = -(-orig_hidden_size // w13_block_size) * w13_block_size

        w1_weight = w13_weight[:, :, :orig_intermediate_size]
        w3_weight = w13_weight[:, :, orig_intermediate_size:]
        pad_intermediate = intermediate_padded - orig_intermediate_size
        pad_hidden = hidden_padded - orig_hidden_size
        pad_spec_w13 = (0, pad_intermediate, 0, pad_hidden)

        w13_weight_padded = torch.cat(
            [
                torch.nn.functional.pad(w1_weight, pad_spec_w13),
                torch.nn.functional.pad(w3_weight, pad_spec_w13),
            ],
            dim=2,
        )
        w2_weight_padded = torch.nn.functional.pad(
            w2_weight, (0, pad_hidden, 0, pad_intermediate)
        )

        w13_bias_padded = w2_bias_padded = None
        if w13_bias is not None:
            b1, b3 = (
                w13_bias[:, :orig_intermediate_size],
                w13_bias[:, orig_intermediate_size:],
            )
            w13_bias_padded = torch.cat(
                [
                    torch.nn.functional.pad(b1, (0, pad_intermediate)),
                    torch.nn.functional.pad(b3, (0, pad_intermediate)),
                ],
                dim=1,
            ).unsqueeze(1)
        if w2_bias is not None:
            w2_bias_padded = torch.nn.functional.pad(
                w2_bias, (0, pad_hidden)
            ).unsqueeze(1)

        return (
            w13_weight_padded,
            w2_weight_padded,
            w13_bias_padded,
            w2_bias_padded,
            orig_intermediate_size,
        )

    def _requantize_native_fp4(
        self,
        w13_weight_padded: torch.Tensor,
        w2_weight_padded: torch.Tensor,
        orig_intermediate_size: int,
        device: torch.device,
        pack: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize padded tensors to native FP4 K-major layout on TPU.

        `pack=False` leaves the fp4 pairs unpacked as `float4_e2m1fn_x2`,
        which is what the fused EP kernel reads; the GMM path wants the uint8
        packing its `should_unpack` step undoes. Same values either way.
        """
        w13_block_size = REQUANTIZED_BLOCK_SIZE
        w2_block_size = min(REQUANTIZED_BLOCK_SIZE, orig_intermediate_size)

        w13_weight_processed, w13_weight_scale = quantize_native_fp4_kmajor(
            w13_weight_padded.to(device), block=w13_block_size, pack=pack
        )
        w2_weight_processed, w2_weight_scale = quantize_native_fp4_kmajor(
            w2_weight_padded.to(device), block=w2_block_size, pack=pack
        )

        return (
            w13_weight_processed,
            w2_weight_processed,
            w13_weight_scale,
            w2_weight_scale,
        )

    def _release_source_weights(self, layer: RoutedExperts):
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale, layer.w2_weight_scale
        if hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias"):
            del layer.w13_bias, layer.w2_bias

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # TODO(upstream): Extract generic on-disk quantization cache to a
        # standalone utility PR.
        assert isinstance(layer, RoutedExperts)
        layer = cast("TpuRoutedExperts", layer)
        device = layer.w13_weight.device

        activation_str = self._set_tpu_activation(layer)

        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            FUSED_MOE_EP_OP_ATTR,
            fused_moe_ep_unsupported_reason,
            register_score_bias_buffer,
        )

        (
            w13_weight_padded,
            w2_weight_padded,
            w13_bias_padded,
            w2_bias_padded,
            orig_intermediate,
        ) = self._dequantize_and_pad(layer)
        # Decided before the quantization, not after: the two paths differ
        # only in whether the fp4 pairs cross the bridge packed, and that is
        # an argument to the quantizer.
        op = self._prebuild_w4a8(
            layer,
            activation_str,
            w13_weight_padded,
            w2_weight_padded,
            orig_intermediate,
        )
        (
            w13_weight_processed,
            w2_weight_processed,
            w13_weight_scale,
            w2_weight_scale,
        ) = self._requantize_native_fp4(
            w13_weight_padded,
            w2_weight_padded,
            orig_intermediate,
            device,
            pack=op is None,
        )
        self._release_source_weights(layer)

        layer.w13_weight = torch.nn.Parameter(w13_weight_processed, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight_processed, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        if w13_bias_padded is not None and w2_bias_padded is not None:
            layer.w13_bias = torch.nn.Parameter(
                w13_bias_padded.to(device), requires_grad=False
            )
            layer.w2_bias = torch.nn.Parameter(
                w2_bias_padded.to(device), requires_grad=False
            )

        to_sync = [layer.w13_weight, layer.w2_weight]
        if hasattr(layer, "w13_bias") and layer.w13_bias is not None:
            to_sync.append(layer.w13_bias)
        if hasattr(layer, "w2_bias") and layer.w2_bias is not None:
            to_sync.append(layer.w2_bias)
        synchronize_tensors(to_sync)

        logger.info(
            "DeepSeek-V4 MXFP4 weights processed and stored as native FP4 "
            "for GMM kernel."
        )
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(layer, device=device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )

        setattr(self, FUSED_MOE_EP_OP_ATTR, op)
        if op is not None:
            reason = fused_moe_ep_unsupported_reason(
                layer,
                (
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                ),
                activation_str,
                weight_format="fp4",
                rhs_qb=REQUANTIZED_BLOCK_SIZE,
                supplies_hash_ids=True,
            )
            # Admission already accepted this exact layout on meta tensors, so
            # a mismatch here is a preparation bug rather than a fallback case.
            assert reason is None, (
                "DeepSeek-V4 MXFP4 fused EP prepared weights violate the "
                f"admitted layout: {reason}"
            )
            register_score_bias_buffer(layer)
            # The kernel's row-padded flat scales avoid per-step relayouts.
            # Clone alone retains higher-rank storage in the pinned runtime;
            # materialize the static scale buffers before registration.
            row_padded = envs.TPU_DEEPSEEK_V41_MOE_SCALE_ROW_PADDED
            if row_padded:
                w13_scale = _row_padded_flat_block_scale(layer.w13_weight_scale)
                w2_scale = _row_padded_flat_block_scale(layer.w2_weight_scale)
            else:
                w13_scale = materialize_static_scale(
                    layer.w13_weight_scale.squeeze(2).clone()
                )
                w2_scale = materialize_static_scale(
                    layer.w2_weight_scale.squeeze(2).clone()
                )
            layer.register_buffer(
                "_tpu_fused_w13_scale",
                w13_scale,
                persistent=False,
            )
            layer.register_buffer(
                "_tpu_fused_w2_scale",
                w2_scale,
                persistent=False,
            )
            logger.info_once(
                "DeepSeek-V4 MXFP4 fused EP W4A8 enabled: block-%d%s",
                REQUANTIZED_BLOCK_SIZE,
                ", row-padded flat scales" if row_padded else "",
            )

    def _forward_monolithic_tpu(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            fused_moe_ep,
            fused_moe_ep_supported,
            hash_topk_ids_operand,
            score_bias_operand,
        )

        layer = cast("TpuRoutedExperts", layer)
        if fused_moe_ep_supported(self):
            # This rank's own tokens in and out: the kernel scores, selects,
            # dispatches, runs both matmuls and combines inside one program,
            # so `input_ids` has no reader here -- hash-routed layers are
            # refused at admission and take the branch below instead.
            return fused_moe_ep(
                self,
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer._tpu_fused_w13_scale,
                layer._tpu_fused_w2_scale,
                router_logits,
                score_bias_operand(layer),
                hash_topk_ids_operand(layer, input_ids),
            )
        return super()._forward_monolithic_tpu(layer, x, router_logits, input_ids)
