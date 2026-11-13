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
"""2D phase map, local FileReader calibration dropzone, and bucket export UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .assets import load_asset
from .constants import PHASE_PREFILL_TOKEN_GRID, PHASE_S_GRID, PHASE_T_GRID
from .support import _find_deployment_preset

if TYPE_CHECKING:
    from .models import ParallelLayout


def render_phase_stage_tab_button() -> str:
    """Return the stage tab button for the 2D Tokens x Context Phase Map."""
    return (
        '<button class="wb-tab-btn" id="btn-stage-phase" type="button" '
        "onclick=\"wbSetStageTab('phase')\">"
        "2D Tokens &times; Context Phase Map &amp; Bucket Schedule</button>"
    )


def _role_seed(layout: ParallelLayout) -> dict[str, object]:
    """Return one deployment role's layout in the fields a phase map cell uses."""
    return {
        "tp": layout.tp,
        "ep": layout.ep,
        "dp": layout.dp,
        "pp": layout.pp,
        "dcp": layout.dcp,
        "pcp": layout.pcp,
        "layer_groups": list(layout.layer_groups),
        "label": layout.label(),
    }


def build_phase_grid_seed(model_name: str) -> dict[str, object]:
    """Return the phase map seed: its grids and the model's deployment role layouts.

    Decode columns run on the decode role and prefill columns on the prefill role.
    A model without a deployment entry gets no roles and no prefill columns; its
    decode columns run on the workbench's current layout.
    """
    deployment = _find_deployment_preset(model_name)
    roles: dict[str, dict[str, object]] = {}
    if deployment is not None:
        roles = {
            "decode": _role_seed(deployment.decode),
            "prefill": _role_seed(deployment.prefill),
        }
    return {
        "t_grid": list(PHASE_T_GRID),
        "s_grid": list(PHASE_S_GRID),
        "roles": roles,
        "prefill_token_grid": list(PHASE_PREFILL_TOKEN_GRID) if roles else [],
    }


def render_phase_extension_js(model_name: str) -> str:
    """Return ES2022 logic for the 2D phase map, calibration dropzone, and export.

    The script starts with the phase map seed of `model_name`.
    """
    phase_grid_json = json.dumps(build_phase_grid_seed(model_name), sort_keys=True)
    return f"globalThis.__MK_PHASE_GRID = {phase_grid_json};\n" + load_asset(
        "workbench_phase.js"
    )
