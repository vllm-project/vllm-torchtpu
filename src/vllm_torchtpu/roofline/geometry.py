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
"""Model geometry extracted from HuggingFace config.json."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelGeometry:
    """Architecture dimensions shared with megakernel block declarations."""

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


def from_hf_config(cfg: Mapping[str, Any]) -> ModelGeometry:
    """Build ModelGeometry from a raw or nested HF config mapping."""
    sub = cfg.get("text_config")
    tc: Mapping[str, Any] = sub if isinstance(sub, Mapping) else cfg
    hidden = int(tc["hidden_size"])
    num_layers = int(tc["num_hidden_layers"])
    num_heads = int(tc["num_attention_heads"])
    num_kv_heads = int(tc.get("num_key_value_heads", num_heads))
    head_dim = int(tc.get("head_dim") or (hidden // num_heads))

    raw_types = tc.get("layer_types")
    if isinstance(raw_types, (list, tuple)) and len(raw_types) == num_layers:
        layer_types = tuple(str(x) for x in raw_types)
    else:
        interval = int(tc.get("full_attention_interval", 0) or 0)
        if interval > 0:
            layer_types = tuple(
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            )
        else:
            layer_types = ("full_attention",) * num_layers

    mtp_raw = tc.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = cfg.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = tc.get(
            "num_nextn_predict_layers", cfg.get("num_nextn_predict_layers", 0)
        )

    return ModelGeometry(
        hidden=hidden,
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=bool(tc.get("attn_output_gate", False)),
        linear_k_heads=int(tc.get("linear_num_key_heads", 0) or 0),
        linear_v_heads=int(tc.get("linear_num_value_heads", 0) or 0),
        linear_k_dim=int(tc.get("linear_key_head_dim", 0) or 0),
        linear_v_dim=int(tc.get("linear_value_head_dim", 0) or 0),
        conv_kernel=int(tc.get("linear_conv_kernel_dim", 4) or 4),
        num_experts=int(tc.get("num_experts", 0) or 0),
        top_k=int(tc.get("num_experts_per_tok", 0) or 0),
        moe_inter=int(tc.get("moe_intermediate_size", 0) or 0),
        shared_inter=int(tc.get("shared_expert_intermediate_size", 0) or 0),
        vocab=int(tc.get("vocab_size", cfg.get("vocab_size", 0)) or 0),
        mtp_layers=int(mtp_raw or 0),
        dense_inter=int(tc.get("intermediate_size", 0) or 0),
    )
