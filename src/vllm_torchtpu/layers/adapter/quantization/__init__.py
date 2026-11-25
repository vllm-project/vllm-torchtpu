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
from vllm.model_executor.layers import quantization as vllm_quantization
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    VllmCompressedTensorsConfig,
)
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.adapter.quantization.modelopt_mixed import (
    VllmModelOptMixedConfig,
)
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import VllmMxfp4Config
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4Config
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import attach_online_fp8
from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.layers.core import quant_methods
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Built-in map from checkpoint quant method to TPU config class.
_BUILTIN_QUANT_CONFIGS: dict[str | None, type[VllmQuantConfig]] = {
    None: VllmUnquantizedConfig,
    quant_methods.FP8: VllmFp8Config,
    quant_methods.MXFP4: VllmMxfp4Config,
    quant_methods.NVFP4: VllmNvfp4Config,
    quant_methods.MODELOPT_MIXED: VllmModelOptMixedConfig,
    quant_methods.COMPRESSED_TENSORS: VllmCompressedTensorsConfig,
    quant_methods.DEEPSEEK_V4_FP8: VllmDeepseekV4Fp8Config,
    # TODO: Add more quantization methods as needed
    # quant_methods.AWQ: VllmAWQConfig,
}

# Checkpoint quant methods that model directories register for themselves.
_MODEL_QUANT_CONFIGS: dict[str | None, type[VllmQuantConfig]] = {}


def register_tpu_quant_config(
    quant_method: str, config_cls: type[VllmQuantConfig]
) -> None:
    """Serve checkpoints whose resolved quant method is `quant_method`.

    `quant_method` is the name vLLM resolves for the checkpoint
    (`ModelConfig.quantization`, the dispatch key), not the checkpoint's own
    `quant_method` string. vLLM can rename it with
    `override_quantization_method`: a ModelOpt NVFP4 checkpoint declares
    `modelopt` but resolves to `modelopt_fp4`, so a class registered under
    `modelopt` never serves it.

    Adds the method to the TPU dispatch table and to the current platform's
    `supported_quantization`, and registers `config_cls` with vLLM under
    `tpu-<quant_method>`, the name `get_tpu_quantization_config` builds it by.
    When vLLM's `QUANTIZATION_METHODS` does not list `quant_method`,
    `config_cls` is also registered under the plain name, so the checkpoint
    passes vLLM's name check (`ModelConfig._verify_quantization`). A name vLLM
    already lists (e.g. `mxfp8`) keeps vLLM's own class for that check.

    The table is keyed by method name only: `config_cls` serves every
    checkpoint in the process that resolves to `quant_method`, whatever its
    architecture. Names a built-in TPU config serves (the table keys and the
    names those classes return from `get_name()`, e.g. `unquantized`) cannot
    be registered.

    Call it at plugin load so it runs in every process, including the one that
    validates the quant method before the model config is built.

    Raises:
        ValueError: `config_cls.get_name()` is not `quant_method`, or another
            class (built-in or registered) already serves `quant_method`.
    """
    if config_cls.get_name() != quant_method:
        raise ValueError(
            f"{config_cls.__name__}.get_name() returns "
            f"{config_cls.get_name()!r}, not {quant_method!r}."
        )
    served = _builtin_config_for(quant_method) or _MODEL_QUANT_CONFIGS.get(quant_method)
    if served is not None and served is not config_cls:
        raise ValueError(
            f"TPU quant method {quant_method!r} is already served by {served.__name__}."
        )
    if served is None:
        _MODEL_QUANT_CONFIGS[quant_method] = config_cls
        if quant_method not in vllm_quantization.QUANTIZATION_METHODS:
            register_quantization_config(quant_method)(config_cls)
        register_quantization_config(quant_methods.get_tpu_quant_method(quant_method))(
            config_cls
        )
    # Same list vLLM's register_quantization_config extends; a name vLLM
    # already lists is not added there.
    supported = current_platform.supported_quantization
    if supported and quant_method not in supported:
        supported.append(quant_method)


def _builtin_config_for(quant_method: str) -> type[VllmQuantConfig] | None:
    """Built-in TPU config serving `quant_method` by table key or `get_name()`."""
    for method, config_cls in _BUILTIN_QUANT_CONFIGS.items():
        if quant_method in (method, config_cls.get_name()):
            return config_cls
    return None


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

    method_to_config = {**_BUILTIN_QUANT_CONFIGS, **_MODEL_QUANT_CONFIGS}

    if model_config.quantization not in method_to_config:
        raise NotImplementedError(
            f"{model_config.quantization} quantization method not supported on TPU. "
            f"Supported methods are: {list(method_to_config.keys())}"
        )

    quant_config_cls = method_to_config[model_config.quantization]
    assert issubclass(quant_config_cls, VllmQuantConfig)

    # Set global config for this quantization class
    quant_config_cls.set_configs(vllm_config)

    # Some ModelOpt checkpoints keep the per-layer map only in hf_quant_config.json.
    # vLLM falls back to that file only for the method name "modelopt_mixed", not
    # for the "tpu-" name. Removing a quantization_config that lacks
    # quantized_layers makes vLLM read the file instead.
    if quant_config_cls is VllmModelOptMixedConfig:
        for cfg_holder in (
            model_config.hf_config,
            getattr(model_config.hf_config, "text_config", None),
        ):
            qcfg = getattr(cfg_holder, "quantization_config", None)
            if not isinstance(qcfg, dict):
                continue
            nested = qcfg.get("quantization")
            has_layers = "quantized_layers" in qcfg or (
                isinstance(nested, dict) and "quantized_layers" in nested
            )
            if not has_layers:
                delattr(cfg_holder, "quantization_config")

    # Register the TPU quant method name so vLLM uses our custom config
    model_config.quantization = quant_methods.get_tpu_quant_method(
        quant_config_cls.get_name()
    )
    config = VllmConfig.get_quantization_config(model_config, vllm_config.load_config)
    assert config is not None
    return attach_online_fp8(config, vllm_config)
