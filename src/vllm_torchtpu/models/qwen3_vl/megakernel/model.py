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
"""Declarative `ModelDef` of the Qwen3-VL text decoder (dense, bf16)."""

from __future__ import annotations

from typing import Any, cast

from vllm_torchtpu.megakernel.core.geometry import FromHF, LMHead
from vllm_torchtpu.megakernel.core.model_def import (
    ModelDef,
    find_model,
    register_model,
)
from vllm_torchtpu.megakernel.core.modules import GQA, DenseMLP, Embedding, Norm
from vllm_torchtpu.megakernel.core.schedule import Schedule

from . import profiles

# The `text_config` of nvidia/Cosmos3-Super, the geometry this folder serves.
COSMOS3_SUPER_TEXT_CONFIG: dict[str, Any] = {
    "architectures": ["Cosmos3ForConditionalGeneration"],
    "model_type": "qwen3_vl_text",
    "vocab_size": 151936,
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000,
    "tie_word_embeddings": False,
}

_BF16_ATTN = {"att_in": "bf16", "att_out": "bf16"}
_BF16_MLP = {"gate_up": "bf16", "down": "bf16"}


class Qwen3VLText(ModelDef):
    """The Qwen3-VL text decoder: GQA with q/k RMSNorm and no output gate, a
    dense SwiGLU FFN, bf16 weights and bf16 KV pages.

    RoPE comes in per row (`rope_table`): vLLM's M-RoPE cos / sin at each
    row's position, so the kernel never derives angles from the KV index.
    Importing this module registers nothing (`register()` does).
    """

    name = profiles.MODEL_NAME
    display_name = "Qwen3-VL"
    hf_architectures = ("Cosmos3ForConditionalGeneration",)
    hf_model_type = ("cosmos3_omni",)
    supported_tp_sizes = (8,)
    default_hf_config = COSMOS3_SUPER_TEXT_CONFIG
    geometry = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        vocab="vocab_size",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
        output_gate=False,
        rope_table=True,
        kv_dtype="bf16",
        weight_dtypes=_BF16_ATTN,
    )
    mlp = DenseMLP.bind(
        prefix="mlp", inter="intermediate_size", weight_dtypes=_BF16_MLP
    )
    schedule = Schedule(
        mixer=attn,
        ffn=mlp,
        pre_norm=Norm.bind("input_layernorm"),
        post_norm=Norm.bind("post_attention_layernorm"),
    )
    embed = Embedding("model.embed_tokens")
    final_norm = Norm.bind("model.norm")
    lm_head = LMHead("lm_head")
    profiles = {"default": profiles.PROFILE}


def register() -> type[Qwen3VLText]:
    """Register `Qwen3VLText` and its profile with the library registries.
    Idempotent."""
    if find_model(Qwen3VLText.name) is Qwen3VLText:
        return Qwen3VLText
    registered = register_model(
        name=Qwen3VLText.name,
        hf_architectures=Qwen3VLText.hf_architectures,
        hf_model_type=Qwen3VLText.hf_model_type,
    )(Qwen3VLText)
    return cast("type[Qwen3VLText]", registered)
