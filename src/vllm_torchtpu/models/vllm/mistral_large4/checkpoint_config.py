# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Native ML4 configuration on the standard vLLM pin."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from transformers import AutoConfig, PretrainedConfig
from vllm.transformers_utils.config import MistralConfigParser, register_config_parser
from vllm.transformers_utils.configs.mistral import adapt_config_dict
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict


class MistralLarge4TextConfig(PretrainedConfig):
    model_type = "mistral_large_4"


class MistralLarge4PixtralConfig(PretrainedConfig):
    model_type = "mistral_large4_pixtral"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.text_config = (
            MistralLarge4TextConfig(**text_config)
            if isinstance(text_config, dict)
            else text_config
        )
        self.vision_config = (
            PretrainedConfig(**vision_config)
            if isinstance(vision_config, dict)
            else vision_config
        )


class MistralLarge4ArchConfigConvertor(ModelArchConfigConvertorBase):
    def is_deepseek_mla(self) -> bool:
        return True


AutoConfig.register(MistralLarge4TextConfig.model_type, MistralLarge4TextConfig)
AutoConfig.register(MistralLarge4PixtralConfig.model_type, MistralLarge4PixtralConfig)
cast(
    dict[str, type[ModelArchConfigConvertorBase]], MODEL_ARCH_CONFIG_CONVERTORS
).update(
    {
        MistralLarge4TextConfig.model_type: MistralLarge4ArchConfigConvertor,
        MistralLarge4PixtralConfig.model_type: MistralLarge4ArchConfigConvertor,
    }
)


@dataclass(frozen=True)
class MistralLayerSpec:
    layer_idx: int
    use_rope: bool
    kv_source_layer: int
    use_sparse: bool
    has_indexer: bool
    indexer_use_rope: bool
    is_moe: bool

    @property
    def shares_kv(self) -> bool:
        return self.kv_source_layer != self.layer_idx


def get_layer_specs(config: PretrainedConfig) -> list[MistralLayerSpec]:
    fields = config.to_dict()
    count = config.num_hidden_layers
    nope = fields.get("nope_layer_flags") or [False] * count
    sources = fields.get("kv_source_layers") or list(range(count))
    sharing = fields.get("index_sharing") or ["full"] * count
    indexer_nope = fields.get("indexer_nope_layer_flags") or [False] * count
    sparse = fields.get("index_topk") is not None
    if any(len(pattern) != count for pattern in (nope, sources, sharing, indexer_nope)):
        raise ValueError("ML4 layer patterns must match num_hidden_layers.")
    return [
        MistralLayerSpec(
            i,
            not nope[i],
            sources[i],
            sparse,
            sparse and sharing[i] == "full",
            not indexer_nope[i],
            bool(fields.get("n_routed_experts"))
            and i >= fields.get("first_k_dense_replace", 0)
            and i % fields.get("moe_layer_freq", 1) == 0,
        )
        for i in range(count)
    ]


def _expand_pattern(pattern: list[Any], count: int) -> list[Any]:
    if not pattern:
        raise ValueError("ML4 layer patterns must not be empty.")
    return [pattern[i % len(pattern)] for i in range(count)]


def adapt_mistral_large4_config(native: dict) -> PretrainedConfig:
    """Reuse Mistral's general/YaRN conversion, adding ML4 layout and FP8 blocks.

    The layout conversion follows the native ML4 specification. Standard vLLM
    interprets MLA plus shared experts as ML3, so those fields are supplied
    after its general conversion instead.
    """
    fields = deepcopy(native)
    moe = fields.pop("moe")
    nope_dim = fields.pop("qk_nope_head_dim")
    dsa = fields.pop("dsa")
    nope = fields.pop("nope", "rope")
    sharing = fields.pop("kv_sharing", {})
    vision = fields.pop("vision_encoder", None)
    mtp = fields.pop("mtp", None)
    quant = fields.get("quantization", {})
    if quant.get("qscheme_weight") == "BLOCK":
        if quant["qformat_weight"] != "fp8_e4m3" or quant["qscheme_act"] != "BLOCK":
            raise ValueError(
                "ML4 block quantization requires FP8 weights and activations."
            )
        fields.pop("quantization")
        fields["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [quant["block_size_out"], quant["block_size_in"]],
        }
    converted = adapt_config_dict(fields, defaults={}).to_dict()
    converted.pop("model_type")
    text = MistralLarge4TextConfig(**converted)
    text.architectures = ["MistralLarge4ForCausalLM"]
    text.qk_nope_head_dim = nope_dim
    text.mtp_config = mtp
    count = text.num_hidden_layers
    text.nope_layer_flags = _expand_pattern(
        [v == "nope" for v in nope.split(",")], count
    )
    if set(sharing) - {"source_layers_str"}:
        raise ValueError(f"Unsupported ML4 KV sharing: {sharing}")
    text.kv_source_layers = (
        [int(v) for v in sharing["source_layers_str"].split(",")]
        if sharing.get("source_layers_str")
        else list(range(count))
    )
    moe_keys = {
        "route_every_n": "moe_layer_freq",
        "first_k_dense_replace": "first_k_dense_replace",
        "num_experts_per_tok": "num_experts_per_tok",
        "num_experts": "n_routed_experts",
        "expert_hidden_dim": "moe_intermediate_size",
        "routed_scale": "routed_scaling_factor",
        "num_shared_experts": "n_shared_experts",
        "num_expert_groups": "n_group",
        "num_expert_groups_per_tok": "topk_group",
    }
    for old, new in moe_keys.items():
        setattr(text, new, moe[old])
    text.topk_method = None
    text.norm_topk_prob = True
    text.scoring_func = "softmax"
    aliases = {
        "topk": "index_topk",
        "n_heads": "index_n_heads",
        "head_dim": "index_head_dim",
        "rope_head_dim": "index_rope_head_dim",
        "nope": "index_nope",
        "match_nope": "indexer_match_nope",
        "ssmax_scaling": "indexer_ssmax_scaling",
        "norm_eps": "indexer_norm_eps",
        "sharing": "index_sharing",
    }
    for old, new in aliases.items():
        if old in dsa:
            value = dsa.pop(old)
            if new in dsa and dsa[new] != value:
                raise ValueError(f"Conflicting DSA fields: {old} and {new}")
            dsa[new] = value
    for key in (
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        "indexer_norm_eps",
        "indexer_ssmax_scaling",
    ):
        setattr(text, key, dsa.pop(key))
    if dsa.pop("index_rope_head_dim", None) not in (None, -1, text.qk_rope_head_dim):
        raise ValueError("ML4 indexer RoPE dimension must match attention RoPE.")
    if dsa.pop("index_nope", False) or not dsa.pop("fp8_indexer", True):
        raise NotImplementedError(
            "ML4 requires the FP8 indexer and per-layer NoPE flags."
        )
    text.indexer_nope_layer_flags = (
        list(text.nope_layer_flags)
        if dsa.pop("indexer_match_nope", False)
        else [False] * count
    )
    text.fp8_kv_cache = dsa.pop("fp8_kv_cache", False)
    text.index_sharing = _expand_pattern(
        dsa.pop("index_sharing", "full").split(","), count
    )
    if set(text.index_sharing) - {"full", "shared"} or text.index_sharing[0] != "full":
        raise ValueError("ML4 index sharing must start with full and use full/shared.")
    if dsa:
        raise ValueError(f"Unparsed ML4 DSA configuration: {dsa}")
    specs = get_layer_specs(text)
    for spec in specs:
        source = spec.kv_source_layer
        if not 0 <= source <= spec.layer_idx or specs[source].shares_kv:
            raise ValueError("ML4 KV sources must own their cache and precede users.")
        if spec.use_rope != specs[source].use_rope:
            raise ValueError("ML4 shared KV layers must use the source's RoPE mode.")
    if vision is None:
        return text
    result = MistralLarge4PixtralConfig(
        architectures=["MistralLarge4ForConditionalGeneration"],
        text_config=text,
        vision_config=PretrainedConfig(**vision),
    )
    if "quantization_config" in text.to_dict():
        result.quantization_config = text.quantization_config
    return result


@register_config_parser("mistral")
class MistralLarge4ConfigParser(MistralConfigParser):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        native = get_hf_file_to_dict("params.json", model, revision)
        if native is None:
            raise ValueError(f"ML4 params.json was not found in {model}.")
        if not native.get("dsa"):
            return super().parse(
                model, trust_remote_code, revision, code_revision, **kwargs
            )
        return native, adapt_mistral_large4_config(native)
