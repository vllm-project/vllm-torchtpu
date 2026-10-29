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
"""Declarative `ModelDef` for Qwen3.8-2.4T MoE (`L=92, H=8192`) at TP x DCP."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from ..geometry import FromHF, ModelGeometry
from ..model_def import LMHead, ModelDef, register_model
from ..modules import GDN, GQA, Embedding, MoE, Norm
from ..schedule import FromHFLayerTypes, Schedule
from ..spec import ModelSpec, Module
from .qwen38_moe_profile import (
    DEFAULT_HF_CONFIG,
    PROFILE,
    QWEN38_MOE_MODEL_NAME,
    TP32_DCP8_PROFILE,
)


@dataclass(frozen=True, init=False, eq=False)
class Qwen38MoeGeometry(ModelGeometry):
    """Architecture geometry for Qwen3.8-2.4T MoE."""

    def __init__(self, **overrides: Any) -> None:
        base: dict[str, Any] = (
            dataclasses.asdict(FULL_EP) if "FULL_EP" in globals() else {}
        )
        super().__init__(**(base | overrides))

    def to_model_spec(self, R: int = 32) -> ModelSpec:
        """Convert this geometry into its declarative `ModelSpec`."""
        return qwen38_moe_spec(self, R=R)


@register_model(
    name=QWEN38_MOE_MODEL_NAME,
    hf_architectures=(
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
    ),
    hf_model_type=("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text"),
)
class Qwen38Moe(ModelDef):
    """Declarative model definition for Qwen3.8-2.4T MoE."""

    display_name = "Qwen3.8"
    geometry_cls = Qwen38MoeGeometry
    layer_type_name = "qwen38_moe_layer"
    supported_tp_sizes = (32,)
    default_hf_config = DEFAULT_HF_CONFIG
    geometry = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        vocab="vocab_size",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 0.25),
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
        qk_norm=True,
        output_gate=True,
    )
    linear = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    moe = MoE.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
        moe_tp=False,
    )
    schedule = Schedule(
        mixer=FromHFLayerTypes(
            "layer_types", {"full_attention": attn, "linear_attention": linear}
        ),
        ffn=moe,
        pre_norm=Norm.bind("input_layernorm", zero_centered=True),
        post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = Embedding("model.embed_tokens")
    final_norm = Norm.bind("model.norm", zero_centered=True)
    lm_head = LMHead("lm_head", tied_to=None)
    profiles = {
        "default": PROFILE,
        "tp32-dcp8": TP32_DCP8_PROFILE,
    }


Qwen38MoeGeometry._model_def = Qwen38Moe
FULL = Qwen38Moe.derive_geometry(DEFAULT_HF_CONFIG, moe_tp=True)
FULL_EP = Qwen38Moe.derive_geometry(DEFAULT_HF_CONFIG)
TINY = dataclasses.replace(
    Qwen38Moe.shrink(DEFAULT_HF_CONFIG, periods=1, R=8), moe_tp=True
)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)
reference_step = Qwen38Moe.reference_step


def qwen38_moe_spec(
    g: ModelGeometry = FULL_EP,
    gdn_module: Module | None = None,
    *,
    R: int = 32,
) -> ModelSpec:
    """Construct the declarative `ModelSpec` for Qwen3.8-2.4T."""
    return Qwen38Moe.to_model_spec(g, gdn_module=gdn_module, R=R)
