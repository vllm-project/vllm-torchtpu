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
from .specs import SUPPORTED_ONCHIP_FAMILIES, WorkloadSpec
from .support import CORE_LIMITS


def build_engine_seed_defaults() -> dict[str, object]:
    """Return Python-authoritative CORE_LIMITS and WorkloadSpec defaults for JS."""
    declared: dict[str, list[dict[str, object]]] = {}
    for model_name, dep in DEPLOYMENT_PRESETS.items():
        entries: list[dict[str, object]] = []
        for layout in (dep.decode, dep.prefill, dep.saturation):
            mk = layout.megakernel if layout is not None else None
            if mk and str(mk.get("source", "")):
                scopes_raw = mk.get("scopes")
                scopes = (
                    [str(s) for s in scopes_raw] if type(scopes_raw) is list else []
                )
                phases_raw = mk.get("phases")
                entries.append(
                    {
                        "scopes": scopes,
                        "max_rows": int(str(mk.get("max_rows", 0))),
                        "max_total_rows": int(str(mk.get("max_total_rows", 0))),
                        "ranks": int(str(mk.get("ranks", 0))),
                        "phases": (
                            [str(phase) for phase in phases_raw]
                            if type(phases_raw) is list
                            else []
                        ),
                        "attention_dp_in_call": bool(
                            mk.get("attention_dp_in_call", False)
                        ),
                        "moe_layout": str(mk.get("moe_layout", "")),
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
        "hbm_utilization_cap": constants.HBM_UTILIZATION_CAP,
        "compiled_xla_gap_p50_us": constants.COMPILED_XLA_GAP_P50_US,
        "compiled_xla_gap_p90_us": constants.COMPILED_XLA_GAP_P90_US,
        "unfused_eager_gap_us": constants.UNFUSED_EAGER_GAP_US,
        "default_launch_gap_upper_us": constants.DEFAULT_LAUNCH_GAP_UPPER_US,
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
