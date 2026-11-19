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

A ModelOpt ``MIXED_PRECISION`` checkpoint names a quantization algorithm for
each layer in ``quantized_layers``. vLLM detects these checkpoints as the
``modelopt_mixed`` method. This config reuses upstream's checkpoint parsing.
Each layer gets the TPU method that a single-algorithm checkpoint of the same
algorithm would use.

TPU supports ``FP8`` on linear layers, and ``NVFP4`` or ``W4A16_NVFP4`` on linear
layers and routed experts. Layers that are excluded, marked ``"NONE"``, or
absent from ``quantized_layers`` load unquantized. Any other layer and algorithm
pair raises ``NotImplementedError``. A fused projection or expert group whose
entries disagree raises ``ValueError``.

By default, TPU runs both ``NVFP4`` and ``W4A16_NVFP4`` layers as W4A16, with
FP4 weights and bf16 activations. The two therefore share one ``NVFP4`` spec and
may be mixed within a fused projection or an expert group. An NVFP4 entry
without ``group_size`` uses the checkpoint's top-level ``group_size``, or
ModelOpt's default of 16.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

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


class LayerSpec(NamedTuple):
    """How the checkpoint quantizes one layer."""

    # None when the layer is unquantized.
    quant_algo: str | None
    # None for algorithms without block scales, such as FP8.
    group_size: int | None


_UNQUANTIZED = LayerSpec(None, None)

_FALLBACK_FUSED_SHARDS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

_EXPERT_PROJ_NAMES = frozenset({"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"})

_NVFP4_ALGOS = ("NVFP4", "W4A16_NVFP4")
# ModelOpt's NVFP4 block size when a layer entry omits group_size.
_NVFP4_DEFAULT_GROUP_SIZE = 16


def _normalize_layer_spec(
    entry: dict[str, Any], nvfp4_default_group_size: int
) -> LayerSpec:
    raw_algo = entry.get("quant_algo")
    if raw_algo is None:
        return _UNQUANTIZED
    algo = str(raw_algo).upper()
    if algo == "NONE":
        return _UNQUANTIZED
    raw_group_size = entry.get("group_size")
    group_size = int(raw_group_size) if raw_group_size is not None else None
    if algo in _NVFP4_ALGOS:
        if group_size is None:
            group_size = nvfp4_default_group_size
        return LayerSpec("NVFP4", group_size)
    return LayerSpec(algo, group_size)


# vLLM enables weight_loader_v2 for a method only if its class name is
# registered. Without it, fused layers would not load their per-shard scales.
@register_weight_loader_v2_supported_method
class _ModelOptFp8LinearMethodTPU(VllmFp8LinearMethodTPU):
    """FP8 linear method that frees the checkpoint's unused static input scale.

    ``VllmFp8LinearMethodTPU`` quantizes activations dynamically and never reads
    ``input_scale``. The parameter exists only so the checkpoint tensor has
    somewhere to load.
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
        # input scale. The "static" activation scheme makes the method register
        # an input_scale parameter for that tensor to load into.
        self.tpu_fp8_config = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
        )
        # Upstream builds one NVFP4 sub-config with a single group_size. Layers
        # can use different group sizes, so this config keeps one TPU NVFP4
        # config per group_size.
        self._tpu_nvfp4_configs: dict[int, VllmNvfp4Config] = {}
        self._nvfp4_default_group_size = _NVFP4_DEFAULT_GROUP_SIZE
        self._child_specs, self._expert_proj_specs = self._build_spec_index()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return MODELOPT_MIXED

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> VllmModelOptMixedConfig:
        # vLLM rejects an hf_quant_config.json not produced by ModelOpt, but
        # only for the method name "modelopt_mixed". The "tpu-" prefix skips
        # that check, so it is repeated here. A config with a nested
        # "quantization" section and no "quant_method" key came from that file.
        if "quantization" in config and "quant_method" not in config:
            producer = config.get("producer", {}).get("name")
            if producer != "modelopt":
                raise ValueError(
                    "ModelOpt MIXED_PRECISION expects hf_quant_config.json "
                    f"produced by modelopt, got producer {producer!r}."
                )
        cfg = super().from_config(config)
        assert isinstance(cfg, VllmModelOptMixedConfig)
        return cfg

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> VllmModelOptMixedConfig:
        cfg = super()._from_config(
            quant_method=quant_method,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            original_config=original_config,
            group_size=group_size,
            **kwargs,
        )
        assert isinstance(cfg, VllmModelOptMixedConfig)
        # group_size is the checkpoint's top-level value, or None if it has
        # none. Upstream fills None from the first NVFP4 entry, but an entry
        # without its own group_size must use the top-level value or 16.
        if group_size is not None:
            cfg._nvfp4_default_group_size = group_size
            cfg._child_specs, cfg._expert_proj_specs = cfg._build_spec_index()
        return cfg

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        # The mapper renames the quantized_layers keys the index is built from.
        self._child_specs, self._expert_proj_specs = self._build_spec_index()
        map_online_fp8(self, hf_to_vllm_mapper)

    def _build_spec_index(
        self,
    ) -> tuple[dict[str, set[LayerSpec]], dict[str, set[LayerSpec]]]:
        """Index the quantized_layers specs by ancestor module prefix.

        Returns two maps from module prefix to specs. The first holds every
        entry under the module. The second holds only the expert projections
        directly under the module, so ``...mlp.gate_proj`` is indexed under
        ``...mlp``.
        """
        child_specs: dict[str, set[LayerSpec]] = {}
        expert_proj_specs: dict[str, set[LayerSpec]] = {}
        for key, info in self.quantized_layers.items():
            spec = _normalize_layer_spec(info, self._nvfp4_default_group_size)
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
                _normalize_layer_spec(
                    self.quantized_layers[sp], self._nvfp4_default_group_size
                )
                if sp in self.quantized_layers
                else _UNQUANTIZED
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
        return False, _UNQUANTIZED

    def _resolve_layer_quant_spec(self, prefix: str) -> LayerSpec:
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            if candidate in self.quantized_layers:
                return _normalize_layer_spec(
                    self.quantized_layers[candidate], self._nvfp4_default_group_size
                )

        # quantized_layers keys a fused layer by its checkpoint shard names.
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

        # A module without its own entry, such as RoutedExperts, takes the one
        # spec shared by all entries under it.
        child_specs: set[LayerSpec] = set()
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            child_specs |= self._child_specs.get(candidate, set())
        if len(child_specs) > 1:
            raise ValueError(
                f"Mixed quantization specs under module {prefix!r}: {child_specs}."
            )
        if len(child_specs) == 1:
            return next(iter(child_specs))

        # Some checkpoints list expert projections directly under the MoE
        # block, such as "...moe.gate_proj". vLLM names the RoutedExperts
        # module "...moe.experts", so its specs come from the parent block.
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

        # This covers qkv_proj and gate_up_proj when packed_modules_mapping is
        # empty or does not list them.
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

        return _UNQUANTIZED

    def _get_checkpoint_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, (Attention, MLAAttention)):
            # KV-cache quantization on TPU follows --kv-cache-dtype.
            return None
        if self.is_layer_excluded(prefix):
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)
        spec = self._resolve_layer_quant_spec(prefix)
        if spec.quant_algo is None:
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)

        if spec.quant_algo == "FP8" and isinstance(layer, LinearBase):
            return _ModelOptFp8LinearMethodTPU(
                self.tpu_fp8_config, self.get_linear_config(layer), prefix=prefix
            )
        if spec.quant_algo == "NVFP4" and spec.group_size is not None:
            nvfp4_cfg = self._get_tpu_nvfp4_config(spec.group_size)
            if isinstance(layer, LinearBase):
                return VllmNvfp4LinearMethod(nvfp4_cfg, self.get_linear_config(layer))
            if isinstance(layer, RoutedExperts):
                return VllmNvfp4MoEMethod(nvfp4_cfg, self.get_moe_config(layer))
        raise NotImplementedError(
            f"ModelOpt MIXED_PRECISION on TPU does not support quant_algo "
            f"{spec.quant_algo!r} for {type(layer).__name__} at {prefix!r}. TPU "
            "supports FP8 on linear layers, and NVFP4 or W4A16_NVFP4 (run as "
            "W4A16) on linear layers and routed experts. Embeddings and lm_head "
            "must be unquantized. Requantize this layer to one of those or list "
            "it in the checkpoint's exclude_modules."
        )
