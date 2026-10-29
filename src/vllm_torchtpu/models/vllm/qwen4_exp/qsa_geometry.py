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
"""Shared QSA projection and compression geometry checks."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig


def validate_qsa_geometry(
    config: "Qwen4ExpTextConfig", *, tp_size: int, cache_dtype: str
) -> None:
    if cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise ValueError("QSA requires BF16 or per-tensor FP8 E4M3 cache dtype")
    if tp_size <= 0:
        raise ValueError(f"QSA requires positive tp_size, got {tp_size}")
    if config.hidden_size <= 0:
        raise ValueError("QSA hidden_size must be positive")
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    if heads <= 0 or heads % tp_size:
        raise ValueError(
            "QSA num_attention_heads must be positive and divisible by tp_size"
        )
    if kv_heads <= 0 or (kv_heads % tp_size and tp_size % kv_heads):
        raise ValueError(
            "QSA num_key_value_heads must shard or replicate evenly across tp_size"
        )
    if heads % kv_heads:
        raise ValueError(
            "QSA num_attention_heads must be divisible by num_key_value_heads"
        )
    if config.indexer_kv_heads != 1:
        raise ValueError(
            "QSA indexer_kv_heads must be 1 for the replicated MQA indexer"
        )
    head_dim, index_dim, index_heads = (
        config.head_dim,
        config.indexer_head_dim,
        config.indexer_n_heads,
    )
    if head_dim is None or index_dim is None or index_heads is None:
        raise ValueError("QSA requires head_dim, indexer_head_dim and indexer_n_heads")
    if min(head_dim, index_dim, index_heads) <= 0:
        raise ValueError(
            "QSA head_dim, indexer_head_dim and indexer_n_heads must be positive"
        )
    ratio = config.indexer_compress_ratio
    if ratio is None or ratio <= 0:
        raise ValueError("QSA indexer_compress_ratio must be positive")
    budget = config.indexer_budget
    if budget is None or budget <= 0 or budget % ratio:
        raise ValueError(
            "QSA indexer_budget must be positive and divisible by "
            "indexer_compress_ratio"
        )
    rotary_dim = int(head_dim * config.partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > min(head_dim, index_dim):
        raise ValueError(
            "QSA rotary_dim must be positive, even and no larger than "
            "either head dimension"
        )
    rope = config.rope_parameters
    if rope is None:
        raise ValueError("QSA requires RoPE parameters")
    theta = rope["rope_theta"]
    epsilon = config.rms_norm_eps
    if (
        not isinstance(theta, (int, float))
        or not math.isfinite(theta)
        or theta <= 0
        or epsilon is None
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("QSA RoPE theta and norm epsilon must be finite and positive")
    if cache_dtype in ("fp8", "fp8_e4m3") and head_dim % 2:
        raise ValueError("QSA FP8 cache requires an even head_dim")
