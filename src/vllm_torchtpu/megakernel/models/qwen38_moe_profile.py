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
"""`qwen38-baseline` and `qwen38-tp32-dcp8`: execution profiles for Qwen3.8-2.4T."""

from __future__ import annotations

import dataclasses
from typing import Any

from ..exec_spec import Tiles
from ..profile import Profile, register_profile
from .qwen35_profile import (
    DECODE_TILES,
    TINY_TILES,
    VERIFY_TILES,
    WIDE_TILES,
    baseline_tiles,
)

QWEN38_MOE_MODEL_NAME = "qwen3_8_moe"
DEFAULT_HF_CONFIG: dict[str, Any] = {
    "model_type": "qwen3_5_moe_text",
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "vocab_size": 248320,
    "hidden_size": 8192,
    "num_hidden_layers": 92,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 128,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 2048,
    "shared_expert_intermediate_size": 2048,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
}


def qwen38_baseline_tiles(B: int, *, moe_tp: bool = False, R: int = 32) -> Tiles:
    """Qwen3.8 baseline tiles: `ebank=1152 KiB`, bounded ring depths for `H=8192`."""
    t = baseline_tiles(B, moe_tp=moe_tp, R=R)
    return dataclasses.replace(
        t, nbd=min(t.nbd, 8), nbe=min(t.nbe, 6), ebank=1152 << 10
    )


def tp32_dcp8_tiles(B: int, *, moe_tp: bool = False, R: int = 32) -> Tiles:
    """Qwen3.8 TP32xDCP8 tiles with recursive-doubling (`hier=3`) all-reduce
    at `R=32`.
    """
    t = qwen38_baseline_tiles(B, moe_tp=moe_tp, R=R)
    if R == 32 and B <= 16:
        return dataclasses.replace(t, hier=3, rsag=False)
    return t


PROFILE = register_profile(
    Profile(
        name="qwen38-baseline",
        model=QWEN38_MOE_MODEL_NAME,
        numerics="exact",
        model_default=True,
        tiles=qwen38_baseline_tiles,
        description=(
            "Qwen3.8 default reference profile with flat-ring all-reduce at R=32 "
            "and all opt-in approximation flags off."
        ),
    )
)

TP32_DCP8_PROFILE = register_profile(
    Profile(
        name="qwen38-tp32-dcp8",
        model=QWEN38_MOE_MODEL_NAME,
        numerics="approx",
        model_default=False,
        tiles=tp32_dcp8_tiles,
        buckets=(4, 8, 16, 32, 64),
        verify_buckets=(),
        description=(
            "Qwen3.8 TP32xDCP8 multi-host decode profile using recursive-doubling "
            "all-reduce (hier=3) for B <= 16."
        ),
    )
)

__all__ = [
    "DECODE_TILES",
    "DEFAULT_HF_CONFIG",
    "PROFILE",
    "QWEN38_MOE_MODEL_NAME",
    "TINY_TILES",
    "TP32_DCP8_PROFILE",
    "VERIFY_TILES",
    "WIDE_TILES",
    "baseline_tiles",
    "tp32_dcp8_tiles",
]
