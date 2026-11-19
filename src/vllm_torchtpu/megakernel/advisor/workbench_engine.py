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
"""Client-side ES2022 roofline and on-chip VMEM physics evaluator."""

from __future__ import annotations

import dataclasses
import json

from . import constants
from .assets import load_asset
from .models import DEPLOYMENT_PRESETS, RECOGNIZED_MODEL_TYPES
from .specs import (
    CORE_MODULE_FAMILIES,
    CORE_SUPPORTED_DTYPES,
    STRUCTURAL_LAYER_TAGS,
    SUPPORTED_ONCHIP_FAMILIES,
    WorkloadSpec,
)
from .support import CORE_LIMITS


def _string_list(raw: object) -> list[str]:
    """Return a JSON list or tuple as a list of strings, or an empty list."""
    match raw:
        case list() | tuple():
            return [str(item) for item in raw]
        case _:
            return []


def build_engine_seed_defaults() -> dict[str, object]:
    """Return Python-authoritative CORE_LIMITS and WorkloadSpec defaults for JS."""
    declared: dict[str, list[dict[str, object]]] = {}
    for model_name, dep in DEPLOYMENT_PRESETS.items():
        entries: list[dict[str, object]] = []
        all_layouts = (
            dep.decode,
            dep.prefill,
            dep.saturation,
            *dep.megakernel_layouts,
        )
        for layout in all_layouts:
            if layout is None or layout.megakernel is None:
                continue
            mk = layout.megakernel
            if str(mk.get("source", "")).strip():
                entries.append(
                    {
                        "tp": layout.tp,
                        "ep": layout.ep,
                        "dp": layout.dp,
                        "pp": layout.pp,
                        "dcp": layout.dcp,
                        "pcp": layout.pcp,
                        "scopes": _string_list(mk.get("scopes")),
                        "max_rows": int(str(mk.get("max_rows", 0))),
                        "max_total_rows": int(str(mk.get("max_total_rows", 0))),
                        "ranks": int(str(mk.get("ranks", 0))),
                        "phases": _string_list(mk.get("phases")),
                        "attention_dp_in_call": bool(
                            mk.get("attention_dp_in_call", False)
                        ),
                        "moe_layout": str(mk.get("moe_layout", "")),
                        "expert_owner_groups": int(
                            str(mk.get("expert_owner_groups", 0))
                        ),
                        "collectives": mk.get("collectives"),
                        "expert_mode_by_rows": mk.get("expert_mode_by_rows"),
                        "source": str(mk.get("source", "")),
                    }
                )
        if entries:
            declared[model_name.lower()] = entries
    wl_default = WorkloadSpec()
    core_dict: dict[str, object] = {
        **dataclasses.asdict(CORE_LIMITS),
        "verified_ranks": list(CORE_LIMITS.verified_ranks),
        "chip_kinds": list(CORE_LIMITS.chip_kinds),
        "supported_onchip_families": sorted(SUPPORTED_ONCHIP_FAMILIES),
        "core_module_families": sorted(CORE_MODULE_FAMILIES),
        "core_supported_dtypes": sorted(CORE_SUPPORTED_DTYPES),
        "structural_layer_tags": sorted(STRUCTURAL_LAYER_TAGS),
        "recognized_model_types": sorted(RECOGNIZED_MODEL_TYPES),
        "declared_megakernels": declared,
        "default_declared_phases": list(constants.DEFAULT_DECLARED_PHASES),
    }
    return {
        "core_limits": core_dict,
        "dtype_bytes": dict(constants.DTYPE_BYTES),
        "subchannel_block_k_threshold": constants.SUBCHANNEL_BLOCK_K_THRESHOLD,
        "subchannel_vec_ops_per_block": constants.SUBCHANNEL_VPU_OPS_PER_BLOCK,
        "vector_max_elements_per_word": constants.VECTOR_MAX_ELEMENTS_PER_WORD,
        "vector_sub_byte_elements_per_word": (
            constants.VECTOR_SUB_BYTE_ELEMENTS_PER_WORD
        ),
        "indexer_scale_block_size": constants.INDEXER_SCALE_BLOCK_SIZE,
        "word_align_bytes": constants.WORD_ALIGN_BYTES,
        "hbm_cacheline_bytes": constants.HBM_CACHELINE_BYTES,
        "indexer_radix_passes": constants.INDEXER_RADIX_PASSES,
        "indexer_radix_buckets": constants.INDEXER_RADIX_BUCKETS,
        "indexer_head_reduce_vec_ops": constants.INDEXER_HEAD_REDUCE_VEC_OPS,
        "indexer_scale_vec_ops": constants.INDEXER_SCALE_VEC_OPS,
        "indexer_radix_pass_vec_ops": constants.INDEXER_RADIX_PASS_VEC_OPS,
        "dsa_unpack_mask_vec_ops": constants.DSA_UNPACK_MASK_VEC_OPS,
        "dsa_softmax_vec_ops": constants.DSA_SOFTMAX_VEC_OPS,
        "dsa_output_rescale_vec_ops": constants.DSA_OUTPUT_RESCALE_VEC_OPS,
        "dcp_lse_merge_vec_ops": constants.DCP_LSE_MERGE_VEC_OPS,
        "per_op_indexer_score_roundtrip_bytes": (
            constants.PER_OP_INDEXER_SCORE_ROUNDTRIP_BYTES
        ),
        "per_op_dsa_kv_roundtrip_factor": (constants.PER_OP_DSA_KV_ROUNDTRIP_FACTOR),
        "hbm_utilization_cap": constants.HBM_UTILIZATION_CAP,
        "compiled_xla_gap_p50_us": constants.COMPILED_XLA_GAP_P50_US,
        "compiled_xla_gap_p90_us": constants.COMPILED_XLA_GAP_P90_US,
        "unfused_eager_gap_us": constants.UNFUSED_EAGER_GAP_US,
        "default_launch_gap_upper_us": constants.DEFAULT_LAUNCH_GAP_UPPER_US,
        "min_efficiency_floor": constants.MIN_EFFICIENCY_FLOOR,
        "boundary_sync_hop_factor": constants.BOUNDARY_SYNC_HOP_FACTOR,
        "per_op_mixer_kernels_per_layer": constants.PER_OP_MIXER_KERNELS_PER_LAYER,
        "per_op_base_ffn_kernels_per_layer": (
            constants.PER_OP_BASE_FFN_KERNELS_PER_LAYER
        ),
        "per_op_shared_expert_kernels_per_layer": (
            constants.PER_OP_SHARED_EXPERT_KERNELS_PER_LAYER
        ),
        "attn_tile_tokens": constants.ATTN_TILE_TOKENS,
        "dense_ffn_tile_cap": constants.DENSE_FFN_TILE_CAP,
        "bf16_bytes": constants.BF16_BYTES,
        "fp32_bytes": constants.FP32_BYTES,
        "lse_scalars_per_head": constants.LSE_SCALARS_PER_HEAD,
        "default_declared_calls_per_layer": (
            constants.DEFAULT_DECLARED_CALLS_PER_LAYER
        ),
        "per_op_collective_call_factor": (constants.PER_OP_COLLECTIVE_CALL_FACTOR),
        "workload_defaults": {
            "batch": wl_default.batch,
            "isl": wl_default.isl,
            "osl": wl_default.osl,
            "tp": wl_default.tp,
            "pp": wl_default.pp,
            "dp": wl_default.dp,
            "ep": wl_default.ep,
            "dcp": wl_default.dcp,
            "pcp": wl_default.pcp,
            "kv_split": wl_default.kv_split,
        },
    }


def render_workbench_engine_js() -> str:
    """Return dependency-free ES2022 evaluator script hydrated from #advisor-data."""
    seed_json = json.dumps(build_engine_seed_defaults(), sort_keys=True)
    return f"globalThis.__MK_ENGINE_DEFAULTS = {seed_json};\n" + load_asset(
        "workbench_engine.js"
    )
