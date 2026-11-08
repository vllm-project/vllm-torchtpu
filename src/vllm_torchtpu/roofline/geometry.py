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
"""Model geometry and 10-family layer registry from HuggingFace config.json."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

LAYER_FAMILIES: tuple[str, ...] = (
    "gqa",
    "mla",
    "dsa",
    "swa",
    "gdn",
    "kda",
    "dense_mlp",
    "moe",
    "mtp",
    "ple",
)

_VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "full_attention",
        "attention",
        "gqa",
        "mla",
        "mla_attention",
        "dsa",
        "dsa_attention",
        "sparse_mla",
        "sliding_attention",
        "swa",
        "sliding_window_attention",
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
        "dense_mlp",
        "mlp",
        "moe",
        "mtp",
        "ple",
    }
)

_LINEAR_ATTENTION_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
    }
)


@dataclass(frozen=True)
class ModelGeometry:
    """Architecture dimensions shared across attention, recurrence, and FFN families."""

    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    attn_output_gate: bool
    linear_k_heads: int
    linear_v_heads: int
    linear_k_dim: int
    linear_v_dim: int
    conv_kernel: int
    num_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    vocab: int
    mtp_layers: int
    dense_inter: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    sliding_window: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    ple_dim: int = 0

    @staticmethod
    def from_hf_config(cfg: Mapping[str, Any]) -> ModelGeometry:
        """Build ModelGeometry from a raw or nested HF config mapping."""
        return from_hf_config(cfg)


def _resolve_layer_types(
    tc: Mapping[str, Any],
    num_layers: int,
    kv_lora_rank: int,
    index_topk: int,
) -> tuple[str, ...]:
    """Resolve sequence of layer type strings for each transformer layer."""
    raw_types = tc.get("layer_types")
    if raw_types is not None and type(raw_types) in (list, tuple):
        if len(raw_types) != num_layers:
            msg = (
                f"layer_types length {len(raw_types)} != num_hidden_layers {num_layers}"
            )
            raise ValueError(msg)
        layer_types = tuple(str(x) for x in raw_types)
        for token in layer_types:
            if token not in _VALID_LAYER_TOKENS:
                supported = sorted(_VALID_LAYER_TOKENS)
                msg = f"Unknown layer_type {token!r}; supported tokens: {supported}"
                raise ValueError(msg)
        return layer_types

    interval = int(tc.get("full_attention_interval", 0) or 0)
    if interval > 0:
        att_tok = (
            "dsa"
            if (kv_lora_rank > 0 and index_topk > 0)
            else ("mla" if kv_lora_rank > 0 else "full_attention")
        )
        lin_tok = (
            "kda"
            if str(tc.get("model_type", "")).lower().startswith("kimi")
            else "linear_attention"
        )
        return tuple(
            att_tok if (i + 1) % interval == 0 else lin_tok for i in range(num_layers)
        )

    if kv_lora_rank > 0:
        default_mla = "dsa" if index_topk > 0 else "mla"
        return (default_mla,) * num_layers

    return ("full_attention",) * num_layers


def _resolve_moe_dimensions(
    tc: Mapping[str, Any],
) -> tuple[int, int, int, int, int]:
    """Resolve (num_experts, top_k, moe_inter, num_shared_experts, shared_inter)."""
    num_experts = int(
        tc.get("num_experts")
        or tc.get("n_routed_experts")
        or tc.get("num_local_experts")
        or 0
    )
    dense_inter = int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size") or 0
    )
    moe_inter = int(tc.get("moe_intermediate_size") or tc.get("latent_moe_dim") or 0)
    if num_experts > 0 and moe_inter == 0 and dense_inter > 0:
        moe_inter = dense_inter

    num_shared = int(tc.get("n_shared_experts") or tc.get("num_shared_experts") or 0)
    shared_inter = int(tc.get("shared_expert_intermediate_size", 0) or 0)
    if shared_inter == 0 and num_shared > 0 and moe_inter > 0:
        shared_inter = num_shared * moe_inter

    top_k = int(tc.get("num_experts_per_tok") or tc.get("num_selected_experts") or 0)
    return num_experts, top_k, moe_inter, num_shared, shared_inter


def _resolve_linear_attention_dimensions(
    tc: Mapping[str, Any],
    layer_types: tuple[str, ...],
) -> tuple[int, int, int, int, int]:
    """Resolve (lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel)."""
    lin_cfg_raw = tc.get("linear_attn_config")
    lin_cfg: Mapping[str, Any] = (
        lin_cfg_raw if lin_cfg_raw is not None and type(lin_cfg_raw) is dict else {}
    )
    lin_k_heads = int(
        tc.get("linear_num_key_heads") or lin_cfg.get("num_key_heads") or 0
    )
    lin_v_heads = int(
        tc.get("linear_num_value_heads") or lin_cfg.get("num_value_heads") or 0
    )
    lin_k_dim = int(tc.get("linear_key_head_dim") or lin_cfg.get("key_head_dim") or 0)
    lin_v_dim = int(
        tc.get("linear_value_head_dim") or lin_cfg.get("value_head_dim") or 0
    )

    has_linear = any(t in _LINEAR_ATTENTION_LAYER_TOKENS for t in layer_types)
    raw_conv = tc.get("linear_conv_kernel_dim")
    if has_linear:
        if raw_conv is None:
            msg = (
                "Model with linear-attention layers missing required field: "
                "'linear_conv_kernel_dim'"
            )
            raise ValueError(msg)
        conv_kernel = int(raw_conv)
    else:
        conv_kernel = 0
    return lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel


def _resolve_mtp_layers(tc: Mapping[str, Any], cfg: Mapping[str, Any]) -> int:
    """Resolve number of Multi-Token Prediction (MTP) layers."""
    mtp_raw = tc.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = cfg.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = tc.get(
            "num_nextn_predict_layers", cfg.get("num_nextn_predict_layers", 0)
        )
    return int(mtp_raw or 0)


def from_hf_config(cfg: Mapping[str, Any]) -> ModelGeometry:
    """Build ModelGeometry from a raw or nested HF config mapping."""
    sub = cfg.get("text_config")
    tc: Mapping[str, Any] = sub if sub is not None and type(sub) is dict else cfg
    hidden = int(tc["hidden_size"])
    num_layers = int(tc["num_hidden_layers"])
    num_heads = int(tc["num_attention_heads"])
    num_kv_heads = int(tc.get("num_key_value_heads", num_heads))
    head_dim = int(tc.get("head_dim") or (hidden // num_heads))

    kv_lora_rank = int(tc.get("kv_lora_rank", 0) or 0)
    q_lora_rank = int(tc.get("q_lora_rank", 0) or 0)
    qk_nope_head_dim = int(tc.get("qk_nope_head_dim", 0) or 0)
    qk_rope_head_dim = int(tc.get("qk_rope_head_dim", 0) or 0)
    v_head_dim = int(tc.get("v_head_dim", 0) or 0)
    index_n_heads = int(tc.get("index_n_heads", 0) or 0)
    index_head_dim = int(tc.get("index_head_dim", 0) or 0)
    index_topk = int(tc.get("index_topk", 0) or 0)
    sliding_window = int(tc.get("sliding_window", 0) or 0)

    layer_types = _resolve_layer_types(tc, num_layers, kv_lora_rank, index_topk)
    dense_inter = int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size") or 0
    )
    num_experts, top_k, moe_inter, num_shared, shared_inter = _resolve_moe_dimensions(
        tc
    )
    lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel = (
        _resolve_linear_attention_dimensions(tc, layer_types)
    )
    mtp_layers = _resolve_mtp_layers(tc, cfg)
    ple_dim = int(tc.get("per_layer_input_dim") or tc.get("ple_dim") or 0)

    return ModelGeometry(
        hidden=hidden,
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=bool(tc.get("attn_output_gate", False)),
        linear_k_heads=lin_k_heads,
        linear_v_heads=lin_v_heads,
        linear_k_dim=lin_k_dim,
        linear_v_dim=lin_v_dim,
        conv_kernel=conv_kernel,
        num_experts=num_experts,
        top_k=top_k,
        moe_inter=moe_inter,
        shared_inter=shared_inter,
        vocab=int(tc.get("vocab_size", cfg.get("vocab_size", 0)) or 0),
        mtp_layers=mtp_layers,
        dense_inter=dense_inter,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        sliding_window=sliding_window,
        num_shared_experts=num_shared,
        first_k_dense_replace=int(tc.get("first_k_dense_replace", 0) or 0),
        ple_dim=ple_dim,
    )
