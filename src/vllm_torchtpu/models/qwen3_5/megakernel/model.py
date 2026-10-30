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
"""Declarative `ModelDef` for Qwen3.5-397B-A17B."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.megakernel import cache, kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.geometry import FromHF, ModelGeometry
from vllm_torchtpu.megakernel.model_def import (
    LMHead,
    ModelDef,
    find_model,
    register_model,
)
from vllm_torchtpu.megakernel.modules import (
    GDN,
    GQA,
    Embedding,
    GDNModule,
    GQAModule,
    MoE,
    MoEModule,
    Norm,
)
from vllm_torchtpu.megakernel.schedule import FromHFLayerTypes, Schedule
from vllm_torchtpu.megakernel.spec import ModelSpec, Module

from . import hf_oracle, pool
from .profiles import baseline, candidate30
from .profiles.baseline import DEFAULT_HF_CONFIG, QWEN35_MODEL_NAME

_GEOM_FIELDS = tuple(f.name for f in dataclasses.fields(ModelGeometry))


def accept_states(g: Any, ex: ExecSpec, st: dict[str, Any], acc: Any) -> dict[str, Any]:
    """Delegate verify-mode GDN state selection to `GDNModule.accept`."""
    taps = GDNModule(geometry=g).cache_decl(g).conv_taps
    return GDNModule.accept(st, st, acc, T=ex.T, vh=g.vh(ex.R), taps=taps)


_REEXPORTS = (
    GDNModule,
    GQAModule,
    MoEModule,
    pool,
    accept_states,
)


@dataclass(frozen=True, init=False, eq=False)
class Qwen35Geometry(ModelGeometry):
    """Architecture geometry for Qwen3.5-397B-A17B."""

    def __init__(self, **overrides: Any) -> None:
        if not set(_GEOM_FIELDS).issubset(overrides):
            overrides = dataclasses.asdict(Qwen35.derive_geometry()) | overrides
        super().__init__(**overrides)

    def to_model_spec(self) -> ModelSpec:
        """Convert this geometry into its declarative `ModelSpec`."""
        return qwen35_spec(self)


class Qwen35(ModelDef):
    """Declarative model definition for Qwen3.5-397B-A17B.

    Importing this module registers nothing: `register()` adds the model and
    its profiles to the library registries (the template's `create` calls it).
    """

    name = QWEN35_MODEL_NAME
    hf_architectures = (
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForCausalLM",
    )
    hf_model_type = ("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text")
    display_name = "Qwen3.5"
    supported_tp_sizes = (8,)
    geometry_cls = Qwen35Geometry
    layer_type_name = "qwen35_layer"
    default_hf_config = DEFAULT_HF_CONFIG
    hf_oracle = staticmethod(hf_oracle.qwen35_hf_oracle)
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
        "default": baseline.PROFILE,
        "candidate30": candidate30.PROFILE,
        "candidate30-fused": candidate30.FUSED_PROFILE,
    }


Qwen35Geometry._model_def = Qwen35


def register() -> type[Qwen35]:
    """Register `Qwen35` and its profiles with the library registries, for
    the library's by-name lookups. Idempotent: once registered, the registries
    are left as they are (a profile may have been replaced since)."""
    if find_model(Qwen35.name) is Qwen35:
        return Qwen35
    return register_model(
        name=Qwen35.name,
        hf_architectures=Qwen35.hf_architectures,
        hf_model_type=Qwen35.hf_model_type,
    )(Qwen35)


MKConfig = Qwen35Geometry
FULL = Qwen35.derive_geometry(DEFAULT_HF_CONFIG, moe_tp=True)
FULL_EP = dataclasses.replace(FULL, moe_tp=False)
TINY = dataclasses.replace(Qwen35.shrink(DEFAULT_HF_CONFIG, periods=1), moe_tp=True)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)
reference_step = Qwen35.reference_step


def qwen35_spec(
    g: Qwen35Geometry = FULL, gdn_module: Module | None = None
) -> ModelSpec:
    """Construct the declarative `ModelSpec` for Qwen3.5."""
    return Qwen35.to_model_spec(g, gdn_module=gdn_module)


def build(g: Qwen35Geometry, ex: ExecSpec, *, gdn_module: Module | None = None):
    """Build Qwen3.5 megakernel `(call, info)` via the generic `kernel.build` engine."""
    return kernel.build(qwen35_spec(g, gdn_module=gdn_module), ex)


def make_step(
    mesh,
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    gdn_module: Module | None = None,
    donate_states: bool = False,
):
    """Build Qwen3.5 `jax.jit` + `shard_map` step `(step, info)`."""
    spec = qwen35_spec(g, gdn_module=gdn_module)
    return kernel.make_step(mesh, spec, ex, donate_states=donate_states)


def make_pooled_step(
    mesh,
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    gdn_module: Module | None = None,
    donate_pools: bool = False,
    fused_pool_io: bool | str | None = None,
    max_b_low: int = 24,
    min_b_hi: int = 56,
):
    """Build Qwen3.5 paged-pool `shard_map` step (`PallasWriter`)."""
    spec = qwen35_spec(g, gdn_module=gdn_module)
    return cache.make_pooled_step(
        mesh,
        spec,
        ex,
        donate_pools=donate_pools,
        fused_pool_io=fused_pool_io,
        max_b_low=max_b_low,
        min_b_hi=min_b_hi,
    )
