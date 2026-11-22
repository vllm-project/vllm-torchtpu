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
"""Architecture configuration for TPU-portable Inkling 2 / Inkling 2 Loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Inkling2LoopConfig:
    """Text-backbone configuration for Inkling 2 and Inkling 2 Loop models.

    Defaults match ``thinkingmachines/Inkling-2-Mini`` (24 physical layers
    unrolled to 48 logical layer visits via 3 passes over core layers 6..17).
    Setting ``num_passes=1`` represents the non-looped Inkling 2 schedule.
    """

    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    local_layer_ids: tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
    )
    sconv_kernel_size: int = 4
    sliding_window_size: int = 1024
    n_layers_per_attn_res_block: int = 6
    attn_res_normalize_values: bool = True
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    intermediate_size: int = 1152
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    final_logit_softcapping: float = 30.0
    core_start: int = 6
    core_end: int = 18
    num_passes: int = 3
    gate_rank: int = 64

    @property
    def local_layer_set(self) -> frozenset[int]:
        """Set of physical layer indices using sliding-window attention."""
        return frozenset(self.local_layer_ids)

    @property
    def visits(self) -> tuple[tuple[int, int], ...]:
        """Ordered ``(physical_layer_id, pass_idx)`` execution schedule."""
        if self.num_passes <= 1 or self.core_start >= self.core_end:
            return tuple((layer_id, 0) for layer_id in range(self.num_hidden_layers))
        prelude = [(layer_id, 0) for layer_id in range(self.core_start)]
        core = [
            (layer_id, pass_idx)
            for pass_idx in range(self.num_passes)
            for layer_id in range(self.core_start, self.core_end)
        ]
        coda = [
            (layer_id, 0) for layer_id in range(self.core_end, self.num_hidden_layers)
        ]
        return tuple(prelude + core + coda)

    @property
    def num_logical_layers(self) -> int:
        """Total logical layer applications per token."""
        return len(self.visits)

    @classmethod
    def from_hf_config(cls, data: dict[str, Any]) -> Inkling2LoopConfig:
        """Parse either a multimodal HF config (with ``text_config``) or flat dict."""
        tc = data.get("text_config", data)
        lc = tc.get("loop_config", {})
        return cls(
            vocab_size=tc["vocab_size"],
            unpadded_vocab_size=tc.get("unpadded_vocab_size", tc["vocab_size"]),
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            d_rel=tc["d_rel"],
            rel_extent=tc["rel_extent"],
            log_scaling_n_floor=tc.get("log_scaling_n_floor", 8192),
            log_scaling_alpha=tc.get("log_scaling_alpha", 0.1),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            qk_l2_clip_threshold=tc["qk_l2_clip_threshold"],
            rel_l2_clip_threshold=tc["rel_l2_clip_threshold"],
            local_layer_ids=tuple(tc["local_layer_ids"]),
            sconv_kernel_size=tc.get("sconv_kernel_size", 4),
            sliding_window_size=tc.get("sliding_window_size", 1024),
            n_layers_per_attn_res_block=tc.get("n_layers_per_attn_res_block", 6),
            attn_res_normalize_values=tc.get("attn_res_normalize_values", True),
            n_routed_experts=tc["n_routed_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            intermediate_size=tc["intermediate_size"],
            mlp_gate_softcap=tc.get("mlp_gate_softcap", 4.0),
            mlp_up_softcap=tc.get("mlp_up_softcap", 25.0),
            route_scale=tc.get("route_scale", 8.0),
            final_logit_softcapping=tc.get("final_logit_softcapping", 30.0),
            core_start=lc.get("core_start", 6),
            core_end=lc.get("core_end", 18),
            num_passes=lc.get("num_passes", 1 if not lc else 3),
            gate_rank=lc.get("gate_rank", 64),
        )
