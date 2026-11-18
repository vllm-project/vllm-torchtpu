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
"""ModelOpt MIXED_PRECISION quantization support for TPU.

A ModelOpt mixed-precision checkpoint (``quant_algo: MIXED_PRECISION``) names a
quantization algorithm per layer in ``quantized_layers``. vLLM resolves it to
the ``modelopt_mixed`` method.

This config reuses upstream's checkpoint parsing and dispatches each layer to
the TPU method that serves that algorithm in a uniform checkpoint:

  - ``FP8`` linear layers: ``VllmFp8LinearMethodTPU`` (per-tensor weight
    scale). The checkpoint's static ``input_scale`` is loaded and then freed,
    because the TPU matmul quantizes activations dynamically.
  - ``NVFP4`` / ``W4A16_NVFP4`` linear layers: ``VllmNvfp4LinearMethod`` with
    the layer's own ``group_size``.
  - ``NVFP4`` / ``W4A16_NVFP4`` routed experts: ``VllmNvfp4MoEMethod`` with the
    expert group's own ``group_size``.
  - Layers that are excluded, marked ``"NONE"``, or absent from
    ``quantized_layers``: the unquantized TPU methods (supporting
    ``QUANTIZE_ON_LOAD_PREFIXES`` via ``VllmUnquantizedConfig`` and
    ``tpu_online_fp8`` via ``resolve_online_fp8``).

TPU serves ``NVFP4`` and ``W4A16_NVFP4`` identically (W4A16), so both resolve to
one ``("NVFP4", group_size)`` spec and may be mixed within a fused projection or
an expert group. An NVFP4 entry without ``group_size`` uses ModelOpt's default
of 16.

Any other layer/algorithm pair raises ``NotImplementedError``, and partially
quantized fused projections raise ``ValueError``, so a checkpoint never loads
through a path that has not been written for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)

from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    resolve_online_fp8,
)
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4Config,
    VllmNvfp4LinearMethod,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import map_online_fp8
from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.layers.core.quant_methods import (
    MODELOPT_MIXED,
    get_tpu_quant_method,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

# (quant_algo, group_size); (None, None) means unquantized.
LayerSpec = tuple[str | None, int | None]

_FALLBACK_FUSED_SHARDS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

_EXPERT_PROJ_NAMES = frozenset({"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"})

_NVFP4_ALGOS = ("NVFP4", "W4A16_NVFP4")
# ModelOpt's NVFP4 block size when a layer entry omits group_size.
_NVFP4_DEFAULT_GROUP_SIZE = 16


def _normalize_layer_spec(entry: dict[str, Any]) -> LayerSpec:
    raw_algo = entry.get("quant_algo")
    if raw_algo is None:
        return None, None
    algo = str(raw_algo).upper()
    if algo == "NONE":
        return None, None
    raw_group_size = entry.get("group_size")
    group_size = int(raw_group_size) if raw_group_size is not None else None
    if algo in _NVFP4_ALGOS:
        if group_size is None:
            group_size = _NVFP4_DEFAULT_GROUP_SIZE
        return "NVFP4", group_size
    return algo, group_size


# vLLM picks a layer's weight loader by the method's class name, so a
# subclass must register again or fused layers lose per-shard scale loading.
@register_weight_loader_v2_supported_method
class _ModelOptFp8LinearMethodTPU(VllmFp8LinearMethodTPU):
    """FP8 linear method that frees the checkpoint's unused static input scale.

    ``VllmFp8LinearMethodTPU`` quantizes activations dynamically and never reads
    ``input_scale``, which is registered only so the checkpoint tensor loads.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if hasattr(layer, "input_scale"):
            delattr(layer, "input_scale")


@register_quantization_config(get_tpu_quant_method(MODELOPT_MIXED))
class VllmModelOptMixedConfig(ModelOptMixedPrecisionConfig, VllmQuantConfig):
    """ModelOpt MIXED_PRECISION config for TPU."""

    def __init__(
        self,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
        w4a16_nvfp4_config: ModelOptNvFp4Config,
        mxfp8_config: ModelOptMxFp8Config,
    ) -> None:
        super().__init__(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
            w4a16_nvfp4_config=w4a16_nvfp4_config,
            mxfp8_config=mxfp8_config,
        )
        # ModelOpt FP8 linears store a per-tensor weight scale and a static
        # input scale; "static" makes the method register both for loading.
        # _ModelOptFp8LinearMethodTPU frees the input scale after loading.
        self.tpu_fp8_config = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
        )
        # Upstream ModelOptMixedPrecisionConfig._from_config collapses NVFP4
        # group_size onto a single value from the first NVFP4 entry. Cache TPU
        # NVFP4 configs keyed by each layer's own resolved group_size instead.
        self._tpu_nvfp4_configs: dict[int, VllmNvfp4Config] = {}
        self._child_specs, self._expert_proj_specs = self._build_spec_index()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return MODELOPT_MIXED

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        # The mapper renames the quantized_layers keys the index is built from.
        self._child_specs, self._expert_proj_specs = self._build_spec_index()
        map_online_fp8(self, hf_to_vllm_mapper)

    def _build_spec_index(
        self,
    ) -> tuple[dict[str, set[LayerSpec]], dict[str, set[LayerSpec]]]:
        """Index the quantized_layers specs by ancestor module prefix.

        Returns the specs of every entry under each module prefix, and the
        specs of expert projections listed directly under each module (e.g.
        ``...mlp.gate_proj`` indexed under ``...mlp``).
        """
        child_specs: dict[str, set[LayerSpec]] = {}
        expert_proj_specs: dict[str, set[LayerSpec]] = {}
        for key, info in self.quantized_layers.items():
            spec = _normalize_layer_spec(info)
            parts = key.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                child_specs.setdefault(parent, set()).add(spec)
                if parts[i] in _EXPERT_PROJ_NAMES:
                    expert_proj_specs.setdefault(parent, set()).add(spec)
        return child_specs, expert_proj_specs

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        base_method = self._get_checkpoint_quant_method(layer, prefix)
        return resolve_online_fp8(self, layer, prefix, base_method)

    def _get_tpu_nvfp4_config(self, group_size: int) -> VllmNvfp4Config:
        cfg = self._tpu_nvfp4_configs.get(group_size)
        if cfg is None:
            cfg = VllmNvfp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=self.kv_cache_quant_method,
                exclude_modules=[],
                group_size=group_size,
            )
            self._tpu_nvfp4_configs[group_size] = cfg
        return cfg

    def _resolve_fused_shards(
        self, prefix: str, base_candidates: Sequence[str], shard_names: Sequence[str]
    ) -> tuple[bool, LayerSpec]:
        for base_candidate in base_candidates:
            shard_prefixes = [f"{base_candidate}.{s}" for s in shard_names]
            if not any(sp in self.quantized_layers for sp in shard_prefixes):
                continue
            specs = [
                _normalize_layer_spec(self.quantized_layers[sp])
                if sp in self.quantized_layers
                else (None, None)
                for sp in shard_prefixes
            ]
            unique_specs = set(specs)
            if len(unique_specs) > 1:
                detail = dict(zip(shard_names, specs))
                raise ValueError(
                    f"Inconsistent quantization across shards of fused layer "
                    f"{prefix!r}: {detail}. All shards must share the same "
                    "quant_algo and group_size."
                )
            return True, specs[0]
        return False, (None, None)

    def _resolve_layer_quant_spec(self, prefix: str) -> LayerSpec:
        # 1. Direct lookup.
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            if candidate in self.quantized_layers:
                return _normalize_layer_spec(self.quantized_layers[candidate])

        # 2. Packed / fused layer lookup via packed_modules_mapping.
        if "." in prefix:
            base, proj_name = prefix.rsplit(".", 1)
            base_candidates = self._quantized_layer_prefix_candidates(base)
            if self.packed_modules_mapping and proj_name in self.packed_modules_mapping:
                matched, spec = self._resolve_fused_shards(
                    prefix, base_candidates, self.packed_modules_mapping[proj_name]
                )
                if matched:
                    return spec
        else:
            proj_name = prefix

        # 3. Prefix-based lookup for RoutedExperts / parent modules.
        child_specs: set[LayerSpec] = set()
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            child_specs |= self._child_specs.get(candidate, set())
        if len(child_specs) > 1:
            raise ValueError(
                f"Mixed quantization specs under module {prefix!r}: {child_specs}."
            )
        if len(child_specs) == 1:
            return next(iter(child_specs))

        # Checkpoints that list expert projections directly under the MoE block
        # (e.g. "...moe.gate_proj") while vLLM names RoutedExperts "...moe.experts".
        if prefix.endswith(".experts"):
            for candidate in self._quantized_layer_prefix_candidates(prefix):
                parent = candidate.rsplit(".experts", 1)[0]
                sibling_specs = self._expert_proj_specs.get(parent, set())
                if len(sibling_specs) > 1:
                    raise ValueError(
                        f"Mixed quantization specs for experts {prefix!r}: "
                        f"{sibling_specs}."
                    )
                if len(sibling_specs) == 1:
                    return next(iter(sibling_specs))

        # 4. Fallback shard names when packed_modules_mapping is not populated.
        fallback_shards = _FALLBACK_FUSED_SHARDS.get(proj_name)
        if fallback_shards is not None and "." in prefix:
            base = prefix.rsplit(".", 1)[0]
            matched, spec = self._resolve_fused_shards(
                prefix,
                self._quantized_layer_prefix_candidates(base),
                fallback_shards,
            )
            if matched:
                return spec

        return None, None

    def _get_checkpoint_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, (Attention, MLAAttention)):
            # KV-cache quantization on TPU follows --kv-cache-dtype.
            return None
        if self.is_layer_excluded(prefix):
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)
        quant_algo, group_size = self._resolve_layer_quant_spec(prefix)
        if quant_algo is None:
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)

        nvfp4_cfg = None
        if quant_algo == "NVFP4":
            assert group_size is not None
            nvfp4_cfg = self._get_tpu_nvfp4_config(group_size)
        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return _ModelOptFp8LinearMethodTPU(
                    self.tpu_fp8_config,
                    self.get_linear_config(layer),
                    prefix=prefix,
                )
            if nvfp4_cfg is not None:
                return VllmNvfp4LinearMethod(nvfp4_cfg, self.get_linear_config(layer))
        elif isinstance(layer, RoutedExperts):
            if nvfp4_cfg is not None:
                return VllmNvfp4MoEMethod(nvfp4_cfg, self.get_moe_config(layer))
        raise NotImplementedError(
            f"ModelOpt MIXED_PRECISION on TPU does not support quant_algo "
            f"{quant_algo!r} for {type(layer).__name__} at {prefix!r}. TPU serves "
            "FP8, NVFP4, and W4A16_NVFP4 linear layers and NVFP4 or W4A16_NVFP4 "
            "routed experts; requantize this layer to one of those or list it in "
            "the checkpoint's exclude_modules."
        )
