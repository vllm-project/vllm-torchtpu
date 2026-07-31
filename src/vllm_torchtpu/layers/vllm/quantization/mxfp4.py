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
   - For each FusedMoE layer, vLLM calls quant_config.get_quant_method(layer)
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

import os
import sys

# Disable JAX preallocation so it doesn't preallocate TPU HBM during weight loading.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import os
from enum import Enum
from typing import Optional

import torch
from torch_tpu._internal import sync
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig, mxfp4_w4a16_moe_quant_config)
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import \
    Mxfp4MoeBackend as Mxfp4Backend
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
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
from vllm_torchtpu.layers.common.quantization import (pack_fp4_indices,
                                                      quantize_tensor_to_fp4)
from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 prebuild_fused_moe_kernel)
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Apply local patch to FusedMoE.weight_loader for MXFP4 redirection to CPU
_orig_weight_loader = FusedMoE.weight_loader


def patched_fused_moe_weight_loader(self,
                                    param,
                                    loaded_weight,
                                    weight_name,
                                    shard_id,
                                    expert_id,
                                    return_success=False):
    print(
        f"[DEBUG_LOADER_CALL] weight_name={weight_name} shard_id={shard_id} expert_id={expert_id} param_shape={list(param.shape)} loaded_shape={list(loaded_weight.shape)}",
        flush=True)
    orig_tp_rank = self.moe_parallel_config.tp_rank
    self.moe_parallel_config.tp_rank = 0
    try:
        if hasattr(self, "w1_temp") and shard_id in ("w1", "w3", "w2"):
            if "scale" in weight_name:
                return _orig_weight_loader(self, param, loaded_weight,
                                           weight_name, shard_id, expert_id,
                                           return_success)
            if shard_id == "w1":
                temp_param = self.w1_temp
                shard_dim = 0
            elif shard_id == "w3":
                temp_param = self.w3_temp
                shard_dim = 0
            else:
                temp_param = self.w2_temp
                shard_dim = 1

            num_local_experts = temp_param.shape[0]
            experts_start = getattr(self, "expert_start", None)
            if experts_start is None:
                experts_start = moe_routing.get_experts_start(self)
            if experts_start is None:
                experts_start = 0

            local_expert_id = expert_id - experts_start
            if local_expert_id < 0 or local_expert_id >= num_local_experts:
                return True if return_success else None

            expert_data = temp_param.data[local_expert_id]

            if loaded_weight.ndim > 0:
                shard_size = expert_data.shape[shard_dim]
                start_offset = shard_size * 0  # Force tp_rank=0 for EP experts sharding
                narrow_size = min(
                    shard_size, loaded_weight.shape[shard_dim] - start_offset)
                if narrow_size > 0:
                    loaded_weight = loaded_weight.narrow(
                        shard_dim, start_offset, narrow_size)
                    if expert_data.shape[shard_dim] > loaded_weight.shape[
                            shard_dim]:
                        expert_data = expert_data.narrow(
                            shard_dim, 0, loaded_weight.shape[shard_dim])
                    print(
                        f"[DEBUG_LOAD] shard_id={shard_id} expert_id={expert_id} | expert_shape={list(expert_data.shape)} loaded_shape={list(loaded_weight.shape)} shard_dim={shard_dim}",
                        flush=True)
                    expert_data.copy_(loaded_weight)
            return True if return_success else None

        return _orig_weight_loader(self, param, loaded_weight, weight_name,
                                   shard_id, expert_id, return_success)
    finally:
        self.moe_parallel_config.tp_rank = orig_tp_rank


FusedMoE.weight_loader = patched_fused_moe_weight_loader
logger.info("FusedMoE.weight_loader patched locally in VllmMxfp4MoEMethod.")


class MoEBackend(Enum):
    FUSED_MOE = "fused_moe"
    GMM_TP = "gmm_tp"
    GMM_EP = "gmm_ep"


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
                return UnquantizedLinearMethod()
            # MXFP4 linear layer is not implemented - use unquantized
            logger.warning_once(
                "MXFP4 linear layer is not implemented on TPU - "
                "using unquantized linear method.")
            return UnquantizedLinearMethod()

        elif isinstance(layer, FusedMoE):
            # FusedMoE is the main use case for MXFP4 (MoE models like GPT-OSS)
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config)

        elif isinstance(layer, Attention):
            logger.warning_once(
                "MXFP4 attention layer is not implemented on TPU. "
                "Skipping quantization for this layer.")

        return None


def _get_activation_str(activation) -> str:
    """Convert MoEActivation enum or string to plain string, mapping silu to silu_and_mul_with_clamp for DeepSeek-V4."""
    act_str = activation.value if hasattr(activation,
                                          'value') else str(activation)
    if act_str.lower() in ("silu", "silu_and_mul", "swiglu"):
        return "silu_and_mul_with_clamp"
    return act_str


REQUANTIZED_BLOCK_SIZE = (int(envs.MOE_REQUANTIZE_BLOCK_SIZE)
                          if envs.MOE_REQUANTIZE_BLOCK_SIZE else 32)


class VllmMxfp4MoEMethod(Mxfp4MoEMethod, CustomOp):
    """
    TPU-specific implementation of MXFP4 MoE quantization.

    Uses is_monolithic=True so vLLM's DefaultMoERunner calls
    apply_monolithic(layer, x, router_logits) directly, bypassing the
    CUDA-only router.select_experts() path.

    Inherits from Mxfp4MoEMethod to reuse its create_weights() method,
    but bypasses its __init__ to avoid the GPU backend assertion.
    """

    name = "vllm_mxfp4_moe_method"

    @property
    def is_monolithic(self) -> bool:
        return True

    def __init__(self, moe: FusedMoEConfig):
        # Skip Mxfp4MoEMethod.__init__ (GPU backend assertion)
        FusedMoEMethodBase.__init__(self, moe)
        self.mxfp4_backend = Mxfp4Backend.NONE

    def create_weights(
        self,
        layer: FusedMoE,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # 1. Call base Mxfp4MoEMethod.create_weights to allocate final packed parameters
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

        # 2. Allocate temporary PACKED uint8 tensors for w1, w3, w2 to receive checkpoint weights
        layer.w1_temp = torch.zeros(num_experts,
                                    intermediate_size_per_partition,
                                    hidden_size // 2,
                                    dtype=torch.uint8,
                                    device="cpu")
        layer.w3_temp = torch.zeros(num_experts,
                                    intermediate_size_per_partition,
                                    hidden_size // 2,
                                    dtype=torch.uint8,
                                    device="cpu")
        layer.w2_temp = torch.zeros(num_experts,
                                    hidden_size,
                                    intermediate_size_per_partition // 2,
                                    dtype=torch.uint8,
                                    device="cpu")

    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, FusedMoE)
        has_bias = layer.moe_config.has_bias
        device = layer.w13_weight.device

        # 1. Force use_ep = True if EP is enabled on the platform
        if layer.ep_size > 1:
            layer.moe_config.moe_parallel_config.use_ep = True

        activation_str = _get_activation_str(layer.activation)
        layer._tpu_activation_str = activation_str

        # Determine the MoE backend and slice ranges
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_backend = MoEBackend.GMM_EP
            ep_size = layer.ep_size
            ep_rank = layer.ep_rank
            base = layer.global_num_experts // ep_size
            remainder = layer.global_num_experts % ep_size
            start_idx = ep_rank * base + min(ep_rank, remainder)
            end_idx = start_idx + base + (1 if ep_rank < remainder else 0)
        else:
            moe_backend = MoEBackend.GMM_TP
            start_idx = 0
            end_idx = layer.global_num_experts

        # Get TP size for reordering
        tp_size = get_tensor_model_parallel_world_size()

        import os

        import torch.distributed as dist
        layer_idx = getattr(layer, "layer_idx", None)
        rank = dist.get_rank() if dist.is_initialized() else 0
        cache_dir = os.environ.get("VLLM_FP4_CACHE_DIR", "")
        cache_path = None
        w13_shape_str = f"{layer.w13_weight.shape[1]}x{layer.w13_weight.shape[2]}" if hasattr(
            layer, "w13_weight") and layer.w13_weight.ndim >= 3 else "default"
        w2_shape_str = f"{layer.w2_weight.shape[1]}x{layer.w2_weight.shape[2]}" if hasattr(
            layer, "w2_weight") and layer.w2_weight.ndim >= 3 else "default"
        if cache_dir and layer_idx is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(
                cache_dir,
                f"rank_{rank}_layer_{layer_idx}_w13_{w13_shape_str}_w2_{w2_shape_str}.pt"
            )

        loaded_from_cache = False
        if cache_path and os.path.exists(cache_path):
            try:
                print(
                    f"[PATCH] Loading pre-quantized parameters from cache: {cache_path}",
                    file=sys.stderr,
                    flush=True)
                logger.warning(
                    "Loading pre-quantized parameters from cache: %s",
                    cache_path)
                checkpoint = torch.load(cache_path, map_location="cpu")
                w13_weight_packed = checkpoint["w13_weight_packed"]
                w2_weight_packed = checkpoint["w2_weight_packed"]
                w13_weight_scale = checkpoint["w13_weight_scale"]
                w2_weight_scale = checkpoint["w2_weight_scale"]
                w13_bias_padded = checkpoint.get("w13_bias_padded", None)
                w2_bias_padded = checkpoint.get("w2_bias_padded", None)
                loaded_from_cache = True

                # Delete original TPU parameters safely if they exist to free up HBM
                if hasattr(layer, "w13_weight"):
                    del layer.w13_weight
                if hasattr(layer, "w2_weight"):
                    del layer.w2_weight
                if hasattr(layer, "w13_weight_scale"):
                    del layer.w13_weight_scale
                if hasattr(layer, "w2_weight_scale"):
                    del layer.w2_weight_scale
                if has_bias:
                    if hasattr(layer, "w13_bias"):
                        del layer.w13_bias
                    if hasattr(layer, "w2_bias"):
                        del layer.w2_bias
                import gc
                gc.collect()
            except Exception as e:
                logger.warning("Failed to load cached parameters from %s: %s",
                               cache_path, e)

        if not loaded_from_cache:
            # 1. Load packed temp weights into CPU memory
            orig_local_intermediate_size = self.intermediate_size
            w1_packed = layer.w1_temp.data.to("cpu")
            w3_packed = layer.w3_temp.data.to("cpu")
            w2_packed = layer.w2_temp.data.to("cpu")
            w13_bias = layer.w13_bias.data.to(
                "cpu") if has_bias and layer.w13_bias is not None else None
            w2_bias = layer.w2_bias.data.to(
                "cpu") if has_bias and layer.w2_bias is not None else None

            # 2. Load scales (which were populated by default_weight_loader into w13_weight_scale / w2_weight_scale)
            w1_scale = layer.w13_weight_scale.data.to(
                "cpu")[:, :orig_local_intermediate_size, :]
            w3_scale = layer.w13_weight_scale.data.to(
                "cpu")[:, orig_local_intermediate_size:, :]
            w2_scale = layer.w2_weight_scale.data.to("cpu")

            # 3. Dequantize packed weights to BF16 (dequantize along Axis 2, which is hidden_size // 2 for w1/w3, and hidden_size for w2)
            from vllm_torchtpu.layers.common.quantization import \
                dequantize_mxfp4_packed
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

            # Delete the temp parameters and scales to free HBM/RAM immediately
            del layer.w1_temp
            del layer.w3_temp
            del layer.w2_temp
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            if has_bias:
                del layer.w13_bias
                del layer.w2_bias
            import gc
            gc.collect()

            # 4. Concatenate w1 and w3 to form w13
            w13_weight = torch.cat([w1_dequant, w3_dequant], dim=1)
            w2_weight = w2_dequant

            if layer_idx == 0:
                w13_mean = w13_weight.mean().item()
                w13_max = w13_weight.max().item()
                w13_min = w13_weight.min().item()
                w13_nan = torch.isnan(w13_weight).any().item()
                print(
                    f"[DEBUG_DEQUANTIZED_WEIGHTS] layer_0 w13_weight: Mean={w13_mean:.6f} Max={w13_max:.6f} Min={w13_min:.6f} HasNaN={w13_nan}",
                    flush=True)

                w2_mean = w2_weight.mean().item()
                w2_max = w2_weight.max().item()
                w2_min = w2_weight.min().item()
                w2_nan = torch.isnan(w2_weight).any().item()
                print(
                    f"[DEBUG_DEQUANTIZED_WEIGHTS] layer_0 w2_weight: Mean={w2_mean:.6f} Max={w2_max:.6f} Min={w2_min:.6f} HasNaN={w2_nan}",
                    flush=True)

            # 3. Transpose non-contracting dim to right-most dim:
            # w13_weight: (num_experts, 2 * intermediate_size, hidden_size) -> (num_experts, hidden_size, 2 * intermediate_size)
            # w2_weight: (num_experts, hidden_size, intermediate_size) -> (num_experts, intermediate_size, hidden_size)
            w13_weight = w13_weight.transpose(1, 2).contiguous()
            w2_weight = w2_weight.transpose(1, 2).contiguous()

            orig_hidden_size = w2_weight.shape[2]
            orig_local_intermediate_size = w2_weight.shape[1]
            logger.info(
                f"[DEBUG_INPUT_WEIGHT_SHAPES] layer_{layer_idx} w13: {w13_weight.shape}, w2: {w2_weight.shape}"
            )

            w13_block_size = REQUANTIZED_BLOCK_SIZE
            w2_block_size = REQUANTIZED_BLOCK_SIZE
            w2_block_size = min(w2_block_size, orig_local_intermediate_size)

            # Align local intermediate dimension to 128 for GMM alignment
            local_intermediate_size_padded = (orig_local_intermediate_size +
                                              128 - 1) // 128 * 128
            hidden_size_padded = (orig_hidden_size + w13_block_size -
                                  1) // w13_block_size * w13_block_size

            # 4. Pad local tensors
            # w13_weight has shape [num_experts, orig_hidden_size, 2 * orig_local_intermediate_size]
            # We split it into w1 and w3 to pad the intermediate dimension individually
            w1_weight = w13_weight[:, :, :orig_local_intermediate_size]
            w3_weight = w13_weight[:, :, orig_local_intermediate_size:]

            pad_w13_intermediate = local_intermediate_size_padded - orig_local_intermediate_size
            pad_w13_hidden = hidden_size_padded - orig_hidden_size

            # PyTorch pad spec starts from the last dimension: (pad_last, pad_second_last, ...)
            pad_spec_w13 = (0, pad_w13_intermediate, 0, pad_w13_hidden)
            w1_weight_padded = torch.nn.functional.pad(w1_weight, pad_spec_w13)
            w3_weight_padded = torch.nn.functional.pad(w3_weight, pad_spec_w13)
            w13_weight_padded = torch.cat([w1_weight_padded, w3_weight_padded],
                                          dim=2)

            if w13_bias is not None:
                b1_bias = w13_bias[:, :orig_local_intermediate_size]
                b3_bias = w13_bias[:, orig_local_intermediate_size:]
                b1_bias_padded = torch.nn.functional.pad(
                    b1_bias, (0, pad_w13_intermediate))
                b3_bias_padded = torch.nn.functional.pad(
                    b3_bias, (0, pad_w13_intermediate))
                w13_bias_padded = torch.cat([b1_bias_padded, b3_bias_padded],
                                            dim=1)
            else:
                w13_bias_padded = None

            # w2_weight has shape [num_experts, orig_local_intermediate_size, orig_hidden_size]
            pad_spec_w2 = (0, pad_w13_hidden, 0, pad_w13_intermediate)
            w2_weight_padded = torch.nn.functional.pad(w2_weight, pad_spec_w2)

            if w2_bias is not None:
                w2_bias_padded = torch.nn.functional.pad(
                    w2_bias, (0, pad_w13_hidden))
            else:
                w2_bias_padded = None

            # 5. Quantize to FP8 (block-32) on TPU v6e, or FP4 (block-512) on other devices
            is_tpu_v6e = False
            try:
                from vllm_torchtpu.tpu_info import get_tpu_type
                tpu_type = get_tpu_type()
                if tpu_type and tpu_type.startswith(("v5lite", "v6e")):
                    is_tpu_v6e = True
            except Exception as e:
                logger.warning(
                    "Failed to detect TPU type, defaulting to native FP4 path: %s",
                    e)

            if is_tpu_v6e:
                logger.info(
                    "TPU v6e detected. Re-quantizing expert weights to FP8 float8_e4m3fn (block_size=32)"
                )
                from vllm_torchtpu.layers.common.quantization import \
                    quantize_tensor
                w13_weight_fp8, w13_weight_scale = quantize_tensor(
                    w13_weight_padded,
                    quant_dtype=torch.float8_e4m3fn,
                    axis=2,
                    block_size=32)
                w2_weight_fp8, w2_weight_scale = quantize_tensor(
                    w2_weight_padded,
                    quant_dtype=torch.float8_e4m3fn,
                    axis=1,
                    block_size=32)

                w13_weight_processed = w13_weight_fp8.transpose(
                    1, 2).contiguous()
                w2_weight_processed = w2_weight_fp8.transpose(1,
                                                              2).contiguous()

                s13 = w13_weight_scale.transpose(1, 2).contiguous().to(
                    torch.float32)
                w13_weight_scale = torch.empty(s13.shape[0],
                                               s13.shape[1],
                                               1,
                                               s13.shape[2],
                                               dtype=s13.dtype,
                                               device=s13.device).copy_(
                                                   s13.unsqueeze(2))
                s2 = w2_weight_scale.contiguous().to(torch.float32)
                w2_weight_scale = torch.empty(s2.shape[0],
                                              s2.shape[1],
                                              1,
                                              s2.shape[2],
                                              dtype=s2.dtype,
                                              device=s2.device).copy_(
                                                  s2.unsqueeze(2))
            else:
                # Quantization is along contracting dimensions: axis 1 (hidden_size) for w13, axis 1 (intermediate_size) for w2
                w13_indices, w13_weight_scale = quantize_tensor_to_fp4(
                    w13_weight_padded, axis=1, block_size=w13_block_size)
                w2_indices, w2_weight_scale = quantize_tensor_to_fp4(
                    w2_weight_padded, axis=1, block_size=w2_block_size)

                # Keep Case B layout (unpacked contracting dim, packed non-contracting dim) by calling pack on raw indices
                w13_weight_processed = pack_fp4_indices(w13_indices, axis=1)
                w2_weight_processed = pack_fp4_indices(w2_indices, axis=1)
                logger.info(
                    f"[DEBUG_MXFP4_SHAPES] layer_{layer_idx} w13_processed: {w13_weight_processed.shape}, w2_processed: {w2_weight_processed.shape}"
                )
                w13_weight_processed = w13_weight_processed.contiguous()
                w2_weight_processed = w2_weight_processed.contiguous()

                s13 = w13_weight_scale.contiguous()
                w13_weight_scale = torch.empty(s13.shape[0],
                                               s13.shape[1],
                                               1,
                                               s13.shape[2],
                                               dtype=s13.dtype,
                                               device=s13.device).copy_(
                                                   s13.unsqueeze(2))
                s2 = w2_weight_scale.contiguous()
                w2_weight_scale = torch.empty(s2.shape[0],
                                              s2.shape[1],
                                              1,
                                              s2.shape[2],
                                              dtype=s2.dtype,
                                              device=s2.device).copy_(
                                                  s2.unsqueeze(2))

            if w13_bias_padded is not None:
                w13_bias_padded = w13_bias_padded.unsqueeze(1)
            if w2_bias_padded is not None:
                w2_bias_padded = w2_bias_padded.unsqueeze(1)

            if cache_path:
                try:
                    logger.warning(
                        "Saving pre-quantized parameters to cache: %s",
                        cache_path)
                    checkpoint_dict = {
                        "w13_weight_packed": w13_weight_processed.to("cpu"),
                        "w2_weight_packed": w2_weight_processed.to("cpu"),
                        "w13_weight_scale": w13_weight_scale.to("cpu"),
                        "w2_weight_scale": w2_weight_scale.to("cpu"),
                    }
                    if w13_bias_padded is not None:
                        checkpoint_dict[
                            "w13_bias_padded"] = w13_bias_padded.to("cpu")
                    if w2_bias_padded is not None:
                        checkpoint_dict["w2_bias_padded"] = w2_bias_padded.to(
                            "cpu")
                    torch.save(checkpoint_dict, cache_path)
                except Exception as e:
                    logger.warning(
                        "Failed to save cached parameters to %s: %s",
                        cache_path, e)
        else:
            w13_weight_processed = w13_weight_packed
            w2_weight_processed = w2_weight_packed

        # Convert back to PyTorch-XLA parameters (retaining float/uint8 dtypes)
        from torch.nn.parameter import Parameter

        def to_xla_parameter(tensor: torch.Tensor) -> Parameter:
            if tensor is None:
                return None
            xla_tensor = tensor.to(device=device)
            return Parameter(xla_tensor, requires_grad=False)

        if loaded_from_cache:
            w13_weight_processed = w13_weight_packed
            w2_weight_processed = w2_weight_packed

        layer.w13_weight = to_xla_parameter(w13_weight_processed)
        layer.w2_weight = to_xla_parameter(w2_weight_processed)

        layer.w13_weight_scale = to_xla_parameter(w13_weight_scale)
        layer.w2_weight_scale = to_xla_parameter(w2_weight_scale)

        if has_bias:
            layer.w13_bias = to_xla_parameter(w13_bias_padded)
            layer.w2_bias = to_xla_parameter(w2_bias_padded)

        if device.type == "tpu":
            sync.synchronize(layer.w13_weight, wait=True)
            sync.synchronize(layer.w2_weight, wait=True)
            if has_bias:
                sync.synchronize(layer.w13_bias, wait=True)
                sync.synchronize(layer.w2_bias, wait=True)

        if "w13_weight_packed" in locals():
            del w13_weight_packed, w2_weight_packed, w13_weight_scale, w2_weight_scale
        if "w13_weight" in locals():
            del w13_weight, w2_weight
        if "w13_weight_padded" in locals():
            del w13_weight_padded, w2_weight_padded, w1_weight, w3_weight
            del w1_weight_padded, w3_weight_padded
        if "w13_indices" in locals():
            del w13_indices, w2_indices
        import gc
        gc.collect()

        print(
            "[PATCH] MXFP4 weights processed and stored as native FP4 for GMM kernel.",
            file=sys.stderr,
            flush=True)
        logger.warning(
            "MXFP4 weights processed and stored as native FP4 for GMM kernel.")
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)

        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            experts_start=moe_routing.get_experts_start(layer),
            has_bias=has_bias,
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

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using GMM kernel."""
        activation_str = getattr(layer, "_tpu_activation_str", "swiglu")

        # Step 1: Routing
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
                input_ids=input_ids,
            )

        res = fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=moe_routing.get_experts_start(layer),
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
        )
        layer_idx = getattr(layer, "layer_idx", -1)
        if os.getenv("DSV4_TRACE_ACTIVATIONS") == "1":
            x_flt = x.to(torch.float32)
            rl_flt = router_logits.to(torch.float32)
            res_flt = res.to(torch.float32)
            x_mean = x_flt.mean().item()
            x_std = (x_flt - x_mean).pow(2).mean().sqrt().item()
            rl_mean = rl_flt.mean().item()
            rl_std = (rl_flt - rl_mean).pow(2).mean().sqrt().item()
            res_mean = res_flt.mean().item()
            res_std = (res_flt - res_mean).pow(2).mean().sqrt().item()
            print(
                f"[ACT_TRACE] [layers.{layer_idx}.moe] input_x: shape={list(x.shape)} mean={x_mean:.6f} std={x_std:.6f} min={x_flt.min().item():.6f} max={x_flt.max().item():.6f}",
                flush=True)
            print(
                f"[ACT_TRACE] [layers.{layer_idx}.moe] router_logits: shape={list(router_logits.shape)} mean={rl_mean:.6f} std={rl_std:.6f} min={rl_flt.min().item():.6f} max={rl_flt.max().item():.6f}",
                flush=True)
            print(
                f"[ACT_TRACE] [layers.{layer_idx}.moe] output_res: shape={list(res.shape)} mean={res_mean:.6f} std={res_std:.6f} min={res_flt.min().item():.6f} max={res_flt.max().item():.6f}",
                flush=True)
        return res
