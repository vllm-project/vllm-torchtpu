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
"""Shared model fixtures and Pareto point builders for roofline unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from vllm_torchtpu.roofline import (
    ParetoPoint,
    hbm_breakdown_per_core_bytes,
    sweep_prefix_hit_crossovers,
)
from vllm_torchtpu.roofline.constants import UNIT_MILLIS_PER_SECOND

if TYPE_CHECKING:
    from vllm_torchtpu.roofline import (
        ChipSpec,
        Dtypes,
        ModelGeometry,
        PoolMeshPlan,
        ServingPlan,
    )

QWEN35_CFG: dict[str, object] = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "dtype": "bfloat16",
    }
}

SMALL_HYBRID_MOE_CFG: dict[str, object] = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "dtype": "bfloat16",
    }
}

NEMOTRON_CFG: dict[str, object] = {
    "model_type": "nemotron_h",
    "hidden_size": 4096,
    "num_hidden_layers": 6,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "conv_kernel": 4,
    "hybrid_override_pattern": "M-M*ME",
    "n_routed_experts": 16,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 2048,
    "intermediate_size": 8192,
    "vocab_size": 131072,
}


def cold_crossover_sweep(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    plan_obj: ServingPlan,
    *,
    repeats: int = 3,
) -> tuple[tuple[float, float] | None, float | None]:
    """Run prefix-hit crossover sweep after clearing the HBM breakdown cache."""
    last_result: tuple[tuple[float, float] | None, float | None] = (None, None)
    for _ in range(repeats):
        hbm_breakdown_per_core_bytes.cache_clear()
        last_result = sweep_prefix_hit_crossovers(geom, dtypes, chip, plan_obj)
    return last_result


def make_pareto_point(
    decode_pool: PoolMeshPlan,
    concurrency: int,
    token_throughput_per_chip: float,
    interactivity_tok_s: float,
    *,
    prefill_pool: PoolMeshPlan | None = None,
) -> ParetoPoint:
    """Build a synthetic SLO-meeting ParetoPoint from pool plans and rates."""
    mode: Literal["mixed", "disagg"] = "disagg" if prefill_pool is not None else "mixed"
    prefill_chips = (
        prefill_pool.instances * prefill_pool.chips_per_instance
        if prefill_pool is not None
        else 0
    )
    total_chips = prefill_chips + decode_pool.instances * decode_pool.chips_per_instance
    return ParetoPoint(
        mode=mode,
        concurrency=concurrency,
        total_chips=total_chips,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        est_ttft_ms=100.0,
        est_tpot_ms=UNIT_MILLIS_PER_SECOND / interactivity_tok_s,
        est_interactivity_tok_s=interactivity_tok_s,
        output_throughput_per_chip=token_throughput_per_chip * 0.1,
        token_throughput_per_chip=token_throughput_per_chip,
        p_mixed=0.0 if prefill_pool is not None else 0.05,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
    )
