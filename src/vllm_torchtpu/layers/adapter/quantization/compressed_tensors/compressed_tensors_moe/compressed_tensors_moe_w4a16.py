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
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.fused_moe import (FusedMoEConfig,
                                                  FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import \
    CompressedTensorsWNA16MoEMethod
from vllm.model_executor.utils import set_weight_attrs

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.fused_moe import (fused_moe_gmm,
                                                    get_fused_moe_activation,
                                                    prebuild_fused_moe_kernel)
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute, pipelined_fused_moe_gmm)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.utils import (
    get_cpu_weight_loader_hook, release_memory_to_os,
    requantize_int4_to_fp4_weights, requantize_int4_weights)


class VllmCompressedTensorsW4A16MoEMethod(CompressedTensorsWNA16MoEMethod):
    """TPU compressed-tensors packed-weight W4 MoE implementation."""

    _PACKED_WEIGHT_NAMES = (
        "w13_weight_packed",
        "w2_weight_packed",
    )

    _INT4_SIGN_XOR = ctypes.c_int32(0x88888888).value

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):

        # Skip CompressedTensorsWNA16MoEMethod.__init__ (GPU backend assertion)
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.num_bits = int(getattr(weight_quant, "num_bits", 4))
        self.packed_factor = 32 // self.num_bits
        self.strategy = getattr(weight_quant, "strategy", "group")
        self.group_size = getattr(weight_quant, "group_size", None)
        self.is_transposed = True

    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def supports_internal_mk(self) -> bool:
        # We need to take control of collective communication (AllGather/ReduceScatter)
        # to pipeline them with MoE computation when chunking is enabled.
        return enable_pipelined_collective_and_compute()

    @staticmethod
    def _loaded_data(parameter: torch.nn.Parameter) -> torch.Tensor:
        """Return checkpoint scratch data or an already-materialized weight."""
        scratch = getattr(parameter, "_cpu_scratch", None)
        return parameter.data if scratch is None else scratch.data

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
        # Bypass upstream create weights method as well
        # to prevent upfront TPU HBM pre-allocation during weight creation,
        # which leads to OOM during weight loading from GCS Streamer/HF without EP
        self._validate_w4a16_scheme()

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
        else:
            if self.group_size is None or self.group_size <= 0:
                raise ValueError(
                    f"Invalid group_size for W4A16 MoE: {self.group_size}")
            if hidden_size % self.group_size != 0 or intermediate_size_per_partition % self.group_size != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) and intermediate_size_per_partition "
                    f"({intermediate_size_per_partition}) must be divisible by group_size ({self.group_size})"
                )
            num_groups_w13 = hidden_size // self.group_size
            num_groups_w2 = intermediate_size_per_partition // self.group_size

        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        orig_loader = extra_weight_attrs.get("weight_loader")

        # Inject our optimized TPU CPU weight loader hook
        extra_weight_attrs["weight_loader"] = get_cpu_weight_loader_hook(
            layer, orig_loader, self.moe.tp_size, self.moe.tp_rank)

        extra_weight_attrs.update({
            "is_transposed": self.is_transposed,
            "quant_method": self.strategy
        })

        w13_shards = 2 if getattr(self.moe, "is_act_and_mul", True) else 1
        w13_shape = (num_experts, hidden_size // self.packed_factor,
                     w13_shards * intermediate_size_per_partition)
        w2_shape = (num_experts,
                    intermediate_size_per_partition // self.packed_factor,
                    hidden_size)
        w13_scale_shape = (num_experts, num_groups_w13,
                           w13_shards * intermediate_size_per_partition)
        w2_scale_shape = (num_experts, num_groups_w2, hidden_size)

        for name, shape, dtype in [
            ("w13_weight_packed", w13_shape, torch.int32),
            ("w2_weight_packed", w2_shape, torch.int32),
            ("w13_weight_scale", w13_scale_shape, params_dtype),
            ("w2_weight_scale", w2_scale_shape, params_dtype),
        ]:
            param = torch.nn.Parameter(torch.empty(0, dtype=dtype),
                                       requires_grad=False)
            param._orig_shape = shape
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)

        self._validate_int32_weight_carriers(layer)

    def _process_single_weight_pair(
        self,
        packed_param: torch.nn.Parameter,
        scale_param: torch.nn.Parameter,
        requant_block: int | None,
        in_group_size: int | None,
        target_dtype: jnp.dtype,
    ) -> None:
        target_device = packed_param.device
        w_raw = self._loaded_data(packed_param).to(target_device)
        s_raw = self._loaded_data(scale_param).to(target_device)

        if hasattr(packed_param, "_cpu_scratch"):
            delattr(packed_param, "_cpu_scratch")
        if hasattr(scale_param, "_cpu_scratch"):
            delattr(scale_param, "_cpu_scratch")

        if target_dtype == jnp.float4_e2m1fn:
            block = requant_block or in_group_size or 64
            w_raw, s_raw = requantize_int4_to_fp4_weights(
                w_raw,
                s_raw,
                block,
                in_group_size=in_group_size or block,
            )
            w_out = w_raw.transpose(
                -2, -1).contiguous() if w_raw.dim() >= 2 else w_raw
            s_out = s_raw.transpose(
                -2, -1).contiguous() if s_raw.dim() >= 2 else s_raw
        else:
            # 1. Optional Block-Wise Requantization (100% on TPU)
            if requant_block is not None and in_group_size is not None:
                w_raw, s_raw = requantize_int4_weights(
                    w_raw,
                    s_raw,
                    requant_block,
                    in_group_size=in_group_size,
                )

            # 2. Standard W4A16 Layout & Sign Preparation (Common to All Paths)
            w_raw.bitwise_xor_(self._INT4_SIGN_XOR)
            w_out = w_raw.transpose(
                -2, -1).contiguous() if w_raw.dim() >= 2 else w_raw
            s_out = s_raw.transpose(
                -2, -1).contiguous() if s_raw.dim() >= 2 else s_raw

        packed_param.data = w_out
        scale_param.data = s_out

        del w_raw, s_raw, w_out, s_out

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process W4 MoE weights after loading.
        Loads weight and transposes it to match the layout required by gmm_v2.
        [E,N,K/8] -> [E,K/8,N] for INT4 or [E,K/2,N] for FP4.
        For INT4: performs bitwise xor to flip the sign bit.
        Copies weight from cpu to tpu
        """
        self._validate_w4a16_scheme()
        self._validate_int32_weight_carriers(layer)

        in_group_size = getattr(self.weight_quant, "group_size", None)
        in_group_size = int(
            in_group_size) if in_group_size is not None else None

        requant_block = envs.MOE_REQUANTIZE_BLOCK_SIZE
        requant_block = int(requant_block) if (
            requant_block is not None and in_group_size is not None) else None

        target_dtype_str = (envs.MOE_REQUANTIZE_WEIGHT_DTYPE or "").lower()
        if target_dtype_str in ("fp4", "float4_e2m1fn", "mxfp4", "nvfp4",
                                "float4"):
            layer._rhs_quant_dtype = jnp.float4_e2m1fn
        else:
            layer._rhs_quant_dtype = jnp.int4

        # 1. Process both projection pairs (w13 gate/up and w2 down)
        for packed, scale in [
            (layer.w13_weight_packed, layer.w13_weight_scale),
            (layer.w2_weight_packed, layer.w2_weight_scale),
        ]:
            self._process_single_weight_pair(
                packed,
                scale,
                requant_block,
                in_group_size,
                layer._rhs_quant_dtype,
            )

        # 2. Alias primary weight if the layer defines a .weight attribute
        if hasattr(layer, "weight"):
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
            rhs_quant_dtype=layer._rhs_quant_dtype,
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

        # Quantization-independent routing decision (simulation override ->
        # custom_routing_function -> select_experts); shared across all TPU MoE
        # methods so the routing-simulation hook lives in exactly one place.
        topk_weights, topk_ids = moe_routing.route(layer, x, router_logits)

        kwargs = {
            "hidden_states": x,
            "w1": layer.w13_weight_packed,
            "w2": layer.w2_weight_packed,
            "w1_scale": layer.w13_weight_scale,
            "w2_scale": layer.w2_weight_scale,
            "w1_bias": None,
            "w2_bias": None,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "experts_start": layer._experts_start,
            "topk": layer.moe_config.experts_per_token,
            "activation": activation_str,
            "rhs_quant_dtype": getattr(layer, "_rhs_quant_dtype", jnp.int4),
        }
        if enable_pipelined_collective_and_compute():
            return pipelined_fused_moe_gmm(**kwargs)

        return fused_moe_gmm(**kwargs)
