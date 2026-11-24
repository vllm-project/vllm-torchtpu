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
"""TPU support for ModelOpt MIXED_PRECISION checkpoints.

These checkpoints name a quantization algorithm for each layer in
``quantized_layers``. vLLM selects the ``modelopt_mixed`` method for them, and
each layer is served by the existing TPU method for its algorithm.

``NVFP4`` and ``W4A16_NVFP4`` resolve to one spec. By default TPU runs both as
W4A16, with FP4 weights and bf16 activations, so the two may be mixed within a
fused projection or an expert group.
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
    ModelOptMixedPrecisionConfig,
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


def _summarize_names(names: Sequence[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f" and {len(names) - limit} more"
    return shown


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Upstream's _from_config also passes GPU sub-configs that TPU does not
        # use. Forwarding every argument keeps this class compatible when
        # upstream adds another one.
        super().__init__(*args, **kwargs)
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
            producer = (config.get("producer") or {}).get("name")
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
        return cfg

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        map_online_fp8(self, hf_to_vllm_mapper)

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

    def _spec(self, key: str) -> LayerSpec:
        entry = self.quantized_layers.get(key)
        if entry is None:
            return _UNQUANTIZED
        return _normalize_layer_spec(entry, self._nvfp4_default_group_size)

    def _resolve_shards(
        self, prefix: str, base_candidates: Sequence[str], shard_names: Sequence[str]
    ) -> LayerSpec | None:
        """Return the one spec shared by a layer's checkpoint shards.

        A shard missing from quantized_layers is unquantized, so a layer with
        only some shards listed raises. Returns None if no shard is listed
        under any of ``base_candidates``.
        """
        for base_candidate in base_candidates:
            keys = [f"{base_candidate}.{s}" for s in shard_names]
            if not any(k in self.quantized_layers for k in keys):
                continue
            shards_by_spec: dict[LayerSpec, list[str]] = {}
            for shard_name, key in zip(shard_names, keys):
                shards_by_spec.setdefault(self._spec(key), []).append(shard_name)
            if len(shards_by_spec) > 1:
                detail = "; ".join(
                    f"{spec}: {_summarize_names(names)}"
                    for spec, names in shards_by_spec.items()
                )
                raise ValueError(
                    f"Inconsistent quantization across the checkpoint shards of "
                    f"{prefix!r}: {detail}. All shards must share the same "
                    "quant_algo and group_size."
                )
            return next(iter(shards_by_spec))
        return None

    def _resolve_layer_quant_spec(self, prefix: str) -> LayerSpec:
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            if candidate in self.quantized_layers:
                return self._spec(candidate)
        if "." not in prefix:
            return _UNQUANTIZED

        # quantized_layers keys a fused layer by its checkpoint shard names.
        # The standard qkv_proj and gate_up_proj shard names cover models whose
        # packed_modules_mapping is empty or does not list the layer.
        base, proj_name = prefix.rsplit(".", 1)
        base_candidates = self._quantized_layer_prefix_candidates(base)
        for shard_names in (
            (self.packed_modules_mapping or {}).get(proj_name),
            _FALLBACK_FUSED_SHARDS.get(proj_name),
        ):
            if shard_names:
                spec = self._resolve_shards(prefix, base_candidates, shard_names)
                if spec is not None:
                    return spec
        return _UNQUANTIZED

    def _resolve_experts_quant_spec(
        self, layer: RoutedExperts, prefix: str
    ) -> LayerSpec:
        """Return the one spec shared by every expert weight of ``layer``.

        Checkpoints key routed experts in one of three ways: one entry for the
        RoutedExperts module, one entry per expert and projection
        (``...experts.{e}.gate_proj``), or one entry per projection directly
        under the MoE block (``...mlp.gate_proj``). In the last two layouts, a
        missing projection or expert is unquantized, so partial coverage
        raises instead of quantizing the missing weights.
        """
        candidates = self._quantized_layer_prefix_candidates(prefix)
        for candidate in candidates:
            if candidate in self.quantized_layers:
                return self._spec(candidate)

        proj_names = [
            name
            for name in (
                layer.ckpt_gate_proj_name,
                layer.ckpt_up_proj_name,
                layer.ckpt_down_proj_name,
            )
            if name is not None
        ]
        num_experts = layer.moe_config.num_logical_experts
        per_expert = [f"{e}.{p}" for e in range(num_experts) for p in proj_names]
        spec = self._resolve_shards(prefix, candidates, per_expert)
        if spec is not None:
            return spec

        # vLLM names the module "...mlp.experts" while these checkpoints list
        # the projections under "...mlp".
        if prefix.endswith(".experts"):
            parent = prefix.removesuffix(".experts")
            spec = self._resolve_shards(
                prefix, self._quantized_layer_prefix_candidates(parent), proj_names
            )
            if spec is not None:
                return spec

        # Entries under the module that match none of the layouts above would
        # otherwise be ignored and the experts loaded unquantized.
        unmatched = [
            key
            for key in self.quantized_layers
            for candidate in candidates
            if key.startswith(f"{candidate}.")
        ]
        if unmatched:
            raise ValueError(
                f"quantized_layers has entries under {prefix!r} that do not "
                f"match its expert projections {proj_names} for "
                f"{num_experts} experts: {_summarize_names(unmatched)}."
            )
        return _UNQUANTIZED

    def _get_checkpoint_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, (Attention, MLAAttention)):
            # KV-cache quantization on TPU follows --kv-cache-dtype.
            return None
        if self.is_layer_excluded(prefix):
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)
        if isinstance(layer, RoutedExperts):
            spec = self._resolve_experts_quant_spec(layer, prefix)
        else:
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
