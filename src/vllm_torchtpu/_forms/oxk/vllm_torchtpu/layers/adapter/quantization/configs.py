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
Base configuration classes for TPU quantization.

These classes provide the foundation for all TPU-specific quantization configs.
"""

from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    import torch
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsScheme,
    )

    from vllm_torchtpu.layers.adapter.quantization.online_fp8 import OnlineFp8Policy

    class TpuRoutedExperts(RoutedExperts):
        """Typing-only view of a RoutedExperts layer with the parameters and
        buffers that the TPU MoE quantization methods register on it."""

        w13_weight: torch.nn.Parameter
        w2_weight: torch.nn.Parameter
        w13_weight_scale: torch.nn.Parameter
        w2_weight_scale: torch.nn.Parameter
        w13_weight_scale_inv: torch.nn.Parameter
        w2_weight_scale_inv: torch.nn.Parameter
        w13_weight_scale_2: torch.nn.Parameter
        w2_weight_scale_2: torch.nn.Parameter
        w13_weight_packed: torch.nn.Parameter
        w2_weight_packed: torch.nn.Parameter
        w13_bias: torch.nn.Parameter | None
        w2_bias: torch.nn.Parameter | None
        num_groups_w13: int
        num_groups_w2: int
        _experts_start: torch.Tensor | None
        _tpu_fused_w13_scale: torch.Tensor
        _tpu_fused_w2_scale: torch.Tensor
        _rhs_quant_dtype: Any

    class TpuLinear(LinearBase):
        """Typing-only view of a LinearBase layer with the parameters and
        attributes that the TPU linear quantization methods set on it."""

        weight: torch.nn.Parameter
        weight_scale: torch.nn.Parameter
        weight_scale_inv: torch.nn.Parameter
        weight_scale_2: torch.nn.Parameter
        logical_widths: list[int]
        weight_block_size: tuple[int, int] | int
        scheme: CompressedTensorsScheme
        _tpu_online_fp8_processed: bool


logger = init_logger(__name__)


def should_quantize_on_load(prefix: str) -> bool:
    """Return True if prefix matches any dot-bounded pattern in
    QUANTIZE_ON_LOAD_PREFIXES."""
    padded = f".{prefix}."
    return any(f".{p.strip('.')}." in padded for p in envs.QUANTIZE_ON_LOAD_PREFIXES)


class VllmQuantLinearConfig:
    """
    Configuration for quantized linear layers on TPU.

    This class holds sharding and optimization settings for linear layers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig | None,
        layer: LinearBase,
    ):
        assert isinstance(layer, LinearBase)
        self.vllm_config = vllm_config
        self.output_sizes = [layer.output_size]
        self.enable_quantized_matmul_kernel = envs.ENABLE_QUANTIZED_MATMUL_KERNEL
        self.requant_block_size = envs.REQUANTIZE_BLOCK_SIZE
        self.requant_weight_dtype = envs.REQUANTIZE_WEIGHT_DTYPE

        # TODO: Implement proper sharding configuration for TPU
        # This should determine weight/bias sharding based on layer type
        # (RowParallel, ColumnParallel, QKVParallel, etc.)


class VllmQuantConfig:
    """
    Base class for all TPU quantization configurations.

    All TPU-specific quantization configs should inherit from this class.
    It provides class-level storage for vllm_config which is shared
    across all instances.

    Subclasses must implement:
        - get_name(): Return the quantization method name
        - get_quant_method(): Return the appropriate quantization method for a layer
    """

    # Optional per-model policy; enabled configs assign their own instance.
    online_fp8_policy: "OnlineFp8Policy | None" = None

    # Class-level config storage (set once, shared by all instances)
    vllm_config: VllmConfig | None = None

    # Set by vLLM's QuantizationConfig, which every concrete subclass extends.
    packed_modules_mapping: dict[str, list[str]]

    @classmethod
    def set_configs(cls, vllm_config: VllmConfig):
        """
        Set the shared configuration for this quantization class.

        Called once during model initialization before any layers are created.

        Args:
            vllm_config: The vLLM configuration.
        """
        cls.vllm_config = vllm_config

    @classmethod
    def get_name(cls) -> str:
        """
        Return the name of this quantization method.

        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_name()")

    def get_linear_config(self, layer: LinearBase) -> VllmQuantLinearConfig:
        """
        Get configuration for a linear layer.

        Args:
            layer: The linear layer to configure.

        Returns:
            VllmQuantLinearConfig with sharding/optimization settings.
        """
        assert isinstance(layer, LinearBase)
        return VllmQuantLinearConfig(self.vllm_config, layer)

    def get_moe_config(self, layer: RoutedExperts) -> FusedMoEConfig:
        """
        Get configuration for a MoE layer.

        Args:
            layer: The RoutedExperts layer to configure.

        Returns:
            FusedMoEConfig with parallelism settings.
        """
        assert isinstance(layer, RoutedExperts)
        moe_config = layer.moe_config
        assert self.vllm_config is not None
        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        moe_config.moe_parallel_config.use_ep = use_ep
        return moe_config

    @property
    def mesh(self):
        # Provides a default fallback for JAX/TPU layers that inspect .mesh on
        # config objects so they do not raise an AttributeError when a device mesh
        # is not explicitly bound on the quantization config.
        return None
