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
TPU Quantization Module for vLLM.

This module provides TPU-specific quantization configurations and methods,
allowing vLLM to use custom quantization backends on TPU hardware.

Flow:
1. vLLM loads a model with quantization config (e.g., "mxfp4")
2. get_tpu_quantization_config() is called to get a TPU-specific config
3. The config returns TPU-specific quant methods for each layer type
4. During model loading, these methods handle weight processing
5. During inference, the apply() method runs TPU-optimized kernels
"""

import copy

from vllm.config import VllmConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
)

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    VllmCompressedTensorsConfig,
)
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import VllmMxfp4Config
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4Config
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import attach_online_fp8
from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.layers.core import quant_methods
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _apply_online_fp8_scope(
    config: VllmFp8Config, args: QuantizationConfigArgs | None
) -> None:
    """Translate upstream `--quantization-config` onto the TPU FP8 config.

    `args` is the QuantizationConfigArgs already resolved by EngineArgs
    (`--quantization` shorthand defaults merged with `--quantization-config`).
    TPU load-time FP8 has a single scheme (per-channel weight, dynamic
    activation), so the only thing taken from `args` is *which* layer kinds
    to quantize; any other weight scheme is rejected rather than silently
    substituted.
    """
    if args is None:
        return
    if config.is_checkpoint_fp8_serialized:
        # A quantized checkpoint fixes which layers are FP8 (via its own
        # ignored_layers); dropping a kind here would feed FP8 tensors to an
        # unquantized method.
        raise NotImplementedError(
            "--quantization-config layer scoping only applies to load-time "
            "FP8 of BF16 checkpoints, not to FP8-serialized checkpoints"
        )
    if args.targets:
        raise NotImplementedError(
            "quantization_config.targets is not supported on TPU; use "
            "quantization_config.linear / .moe / .ignore"
        )
    for kind in ("linear", "moe"):
        spec = getattr(args, kind)
        if spec is None:
            continue
        if spec.weight not in (None, kFp8StaticChannelSym):
            raise NotImplementedError(
                f"TPU online FP8 only supports fp8_per_channel_static weights "
                f"for quantization_config.{kind}, got {spec}"
            )
        if spec.activation is not None:
            raise NotImplementedError(
                "activation overrides are not supported for TPU online FP8 "
                f"(quantization_config.{kind}.activation={spec.activation})"
            )
    config.quantize_linear = args.linear is not None and args.linear.weight is not None
    config.quantize_moe = args.moe is not None and args.moe.weight is not None
    if args.ignore:
        config.ignored_layers = list(args.ignore)


def get_tpu_quantization_config(
    vllm_config: VllmConfig,
) -> QuantizationConfig:
    """
    Get TPU-specific quantization configuration.

    This function is called during model initialization to replace the default
    quantization config with a TPU-compatible version.

    Args:
        vllm_config: The vLLM configuration containing model and quant settings.

    Returns:
        A TPU-compatible QuantizationConfig.

    Raises:
        NotImplementedError: If the quantization method is not supported on TPU.
    """
    model_config = copy.deepcopy(vllm_config.model_config)

    # Map from quant method name to TPU config class
    method_to_config: dict[str | None, type[VllmQuantConfig]] = {
        None: VllmUnquantizedConfig,
        quant_methods.FP8: VllmFp8Config,
        quant_methods.FP8_PER_CHANNEL: VllmFp8Config,
        quant_methods.ONLINE: VllmFp8Config,
        quant_methods.MXFP4: VllmMxfp4Config,
        quant_methods.NVFP4: VllmNvfp4Config,
        quant_methods.COMPRESSED_TENSORS: VllmCompressedTensorsConfig,
        quant_methods.DEEPSEEK_V4_FP8: VllmDeepseekV4Fp8Config,
        # TODO: Add more quantization methods as needed
        # quant_methods.AWQ: VllmAWQConfig,
    }

    if model_config.quantization not in method_to_config:
        raise NotImplementedError(
            f"{model_config.quantization} quantization method not supported on TPU. "
            f"Supported methods are: {list(method_to_config.keys())}"
        )

    quant_config_cls = method_to_config[model_config.quantization]
    assert issubclass(quant_config_cls, VllmQuantConfig)

    # Resolved by EngineArgs: `--quantization` shorthand defaults merged with
    # `--quantization-config`. Consumed below for FP8; cleared from
    # model_config so upstream does not attach its CUDA-only
    # OnlineQuantizationConfig on top of the TPU config.
    online_args = model_config.quantization_config
    if online_args is not None and not isinstance(online_args, QuantizationConfigArgs):
        online_args = QuantizationConfigArgs(**online_args)
    if model_config.quantization == quant_methods.ONLINE and online_args is None:
        raise ValueError(
            "--quantization online requires --quantization-config, e.g. "
            '\'{"moe": "fp8_per_channel_static"}\''
        )
    model_config.quantization_config = None

    # Set global config for this quantization class
    quant_config_cls.set_configs(vllm_config)

    # Register the TPU quant method name so vLLM uses our custom config
    model_config.quantization = quant_methods.get_tpu_quant_method(
        quant_config_cls.get_name()
    )
    config = VllmConfig.get_quantization_config(model_config, vllm_config.load_config)
    assert config is not None
    if isinstance(config, VllmFp8Config):
        _apply_online_fp8_scope(config, online_args)
    elif online_args is not None:
        logger.warning(
            "--quantization-config is ignored for %s on TPU; it is only "
            "honoured for fp8 / fp8_per_channel / online.",
            config.get_name(),
        )
    return attach_online_fp8(config, vllm_config)
