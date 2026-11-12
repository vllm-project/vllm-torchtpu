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
"""Adapter constants for lazy module exports, plugin hooks, and suppression."""

from __future__ import annotations

DRIVER_EXPORTS: frozenset[str] = frozenset({"ProgramDriver"})
JAX_BRIDGE_EXPORTS: frozenset[str] = frozenset(
    {"build_jax_step_op", "make_typed_step_wrapper", "sharded_jax_callable"}
)
SUPPRESS_ATTR: str = "_megakernel_suppress_depth"

HOOK_LAYOUT_REQUEST: str = "layout_request"
HOOK_LOAD_TENSOR: str = "load_tensor"
PLUGIN_HOOK_NAMES: frozenset[str] = frozenset({HOOK_LAYOUT_REQUEST, HOOK_LOAD_TENSOR})

__all__ = [
    "DRIVER_EXPORTS",
    "HOOK_LAYOUT_REQUEST",
    "HOOK_LOAD_TENSOR",
    "JAX_BRIDGE_EXPORTS",
    "PLUGIN_HOOK_NAMES",
    "SUPPRESS_ATTR",
]
