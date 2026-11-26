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
"""Resolve MiniMax M3's model plan: {slot: variant name} for one vLLM config.

The only switch is the optional
``--additional-config '{"model_plan": {"profile": "<name>"}}'`` (default
`profiles.DEFAULT_PROFILE`). Each slot takes the first variant of the
profile's list whose ``eligible(vllm_config)`` holds. The model and its
hooks ask the plan (`resolve_plan`, `selected`) instead of reading
environment variables or other ``additional_config`` keys.
"""

from typing import Any

from vllm_torchtpu.models.minimax_m3.experimental.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
)
from vllm_torchtpu.models.minimax_m3.experimental.variants import VARIANTS

PLAN_KEY = "model_plan"
_PLAN_FIELDS = frozenset({"profile"})


def profile_name(vllm_config: Any) -> str:
    """The profile ``additional_config["model_plan"]`` names, else the default."""
    additional = getattr(vllm_config, "additional_config", None)
    plan = additional.get(PLAN_KEY) if isinstance(additional, dict) else None
    if plan is None:
        return DEFAULT_PROFILE
    if not isinstance(plan, dict) or not set(plan) <= _PLAN_FIELDS:
        raise ValueError(
            f'additional_config.{PLAN_KEY} must look like {{"profile": "<name>"}}, '
            f"got {plan!r}"
        )
    name = plan.get("profile", DEFAULT_PROFILE)
    if name not in PROFILES:
        raise ValueError(
            f"Unknown MiniMax M3 profile {name!r}; known: {sorted(PROFILES)}"
        )
    return name


def _pick(vllm_config: Any, name: str, slot: str) -> str:
    """Profile `name`'s first eligible variant for `slot`."""
    choices = PROFILES[name][slot]
    pick = next((c for c in choices if VARIANTS[slot][c].eligible(vllm_config)), None)
    if pick is None:
        raise ValueError(
            f"MiniMax M3 profile {name!r}: no eligible {slot} variant among "
            f"{list(choices)} for this config."
        )
    return pick


def resolve_plan(vllm_config: Any) -> dict[str, str]:
    """{slot: variant name}: per slot, the profile's first eligible variant."""
    name = profile_name(vllm_config)
    return {slot: _pick(vllm_config, name, slot) for slot in PROFILES[name]}


def selected(vllm_config: Any, slot: str) -> Any:
    """The implementation (import or value) the plan picks for `slot`."""
    return VARIANTS[slot][_pick(vllm_config, profile_name(vllm_config), slot)].load()
