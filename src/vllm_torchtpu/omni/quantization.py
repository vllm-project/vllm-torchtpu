# Copyright 2026 Google LLC
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
"""TPU FP8 quantization support for vLLM-Omni diffusion transformers.

Hooks into ``vllm_omni.quantization.factory.register_quantization_override``
for ``"fp8"`` so ``Omni(..., quantization_config="fp8")`` (or a per-component /
dict ``quantization_config`` with ``ignored_layers``) quantizes eligible DiT
``LinearBase`` projections on TPU via :class:`VllmFp8LinearMethodTPU`
(K-major ``float8_e4m3fn`` weights + per-channel ``float32`` scales + dynamic
FP8 activations in ``xla_quantized_matmul``) while keeping precision-sensitive
modulation, conditioning, refiner, boundary, and text-encoder layers in
BF16/FP32.
"""

from __future__ import annotations

from functools import cache as run_once
from typing import Any

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)

from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    VllmFp8MoEMethodTPU,
    resolve_online_fp8,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Precision-sensitive diffusion / multimodal submodules kept unquantized (BF16)
# by default during online load-time FP8 quantization unless explicitly disabled
# with ``skip_modulation_layers=False``:
#   - AdaLN / modulation projections (zero-init shift/scale/gate, M <= 4 rows)
#   - Prompt conditioning & text-refiner blocks (executed once per request)
#   - Patch / timestep / final output heads & standalone text encoders
DEFAULT_DIFFUSION_FP8_SKIP_PATTERNS: tuple[str, ...] = (
    "*adaln_proj*",
    "*adaLN_modulation*",
    "*img_mod*",
    "*txt_mod*",
    "*scale_shift_table*",
    "condition_proj*",
    "*condition_proj*",
    "token_refiner*",
    "*token_refiner*",
    "time_embed*",
    "*time_embed*",
    "final_layer*",
    "*final_layer*",
    "*patch_proj*",
    "*patch_embed*",
    "text_encoder*",
)


class TpuOmniFp8Config(VllmFp8Config):
    """TPU FP8 quantization config for vLLM-Omni diffusion and omni models."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
        skip_modulation_layers: bool = True,
    ) -> None:
        effective_ignored = list(ignored_layers or [])
        if not is_checkpoint_fp8_serialized and skip_modulation_layers:
            for pat in DEFAULT_DIFFUSION_FP8_SKIP_PATTERNS:
                if pat not in effective_ignored:
                    effective_ignored.append(pat)
        super().__init__(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=effective_ignored,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )
        self.skip_modulation_layers = skip_modulation_layers

    def _is_skipped(self, prefix: str) -> bool:
        if not self.ignored_layers:
            return False
        return should_ignore_layer(
            prefix,
            ignore=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            use_fnmatch=True,
        )

    def _get_checkpoint_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            moe_config = self.get_moe_config(layer)
            if self._is_skipped(prefix):
                return VllmUnquantizedFusedMoEMethod(moe_config)
            return VllmFp8MoEMethodTPU(self, moe_config)

        if isinstance(layer, LinearBase):
            if self._is_skipped(prefix):
                return VllmUnquantizedLinearMethod()
            return VllmFp8LinearMethodTPU(
                self, self.get_linear_config(layer), prefix=prefix
            )

        if isinstance(layer, Attention):
            return None
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        base_method = self._get_checkpoint_quant_method(layer, prefix)
        return resolve_online_fp8(self, layer, prefix, base_method)


def build_tpu_omni_fp8_config(**kw: Any) -> QuantizationConfig:
    """Builder registered with ``vllm_omni.quantization.factory`` for ``fp8``."""
    if kw.get("weight_scheme") == "per_channel":
        from vllm_omni.quantization.fp8_config import (
            SerializedPerChannelFp8Config,
        )

        return SerializedPerChannelFp8Config.from_config(kw)
    ignored = kw.get("ignored_layers")
    if not ignored and "modules_to_not_convert" in kw:
        ignored = kw["modules_to_not_convert"]
    return TpuOmniFp8Config(
        is_checkpoint_fp8_serialized=bool(
            kw.get("is_checkpoint_fp8_serialized", False)
        ),
        activation_scheme=str(kw.get("activation_scheme", "dynamic")),
        ignored_layers=list(ignored) if ignored is not None else None,
        weight_block_size=kw.get("weight_block_size"),
        store_dtype=kw.get("store_dtype"),
        skip_modulation_layers=bool(kw.get("skip_modulation_layers", True)),
    )


@run_once
def register_omni_tpu_quantization() -> None:
    """Register TPU-native ``fp8`` quantization builder in vLLM-Omni."""
    try:
        from vllm_omni.quantization.factory import (
            register_quantization_override,
        )
    except ImportError:
        return
    register_quantization_override("fp8", build_tpu_omni_fp8_config)
    logger.info("Registered TPU online/serialized FP8 builder for vLLM-Omni.")
