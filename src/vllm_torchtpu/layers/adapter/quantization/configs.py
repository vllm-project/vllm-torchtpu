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

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class VllmQuantLinearConfig:
    """
    Configuration for quantized linear layers on TPU.

    This class holds sharding and optimization settings for linear layers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer: LinearBase,
    ):
        assert isinstance(layer, LinearBase)
        self.vllm_config = vllm_config
        self.output_sizes = [layer.output_size]
        self.enable_quantized_matmul_kernel = (
            envs.ENABLE_QUANTIZED_MATMUL_KERNEL)
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

    # Class-level config storage (set once, shared by all instances)
    vllm_config: VllmConfig = None

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
        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        moe_config.moe_parallel_config.use_ep = use_ep
        return moe_config

    @property
    def mesh(self):
        # Provides a default fallback for JAX/TPU layers that inspect .mesh on
        # config objects so they do not raise an AttributeError when a device mesh
        # is not explicitly bound on the quantization config.
        return None
