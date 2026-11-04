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
"""GDN gated-norm activation validation and dispatch helpers."""

from __future__ import annotations

from typing import Any

import jax

GDN_GATE_ACTS: frozenset[str] = frozenset({"silu", "sigmoid"})


def validate_gdn_gate(gate: Any, err_cls: type[Exception] = ValueError) -> str:
    """Validate GDN gated-norm activation name (`'silu' | 'sigmoid'`)."""
    if not isinstance(gate, str) or gate not in GDN_GATE_ACTS:
        raise err_cls(f"GDNModule gate must be 'silu' or 'sigmoid', got {gate!r}")
    return gate


def gdn_gate_act(gate: str, err_cls: type[Exception] = ValueError):
    """Return the activation callable (`jax.nn.silu` or `jax.nn.sigmoid`) for `gate`."""
    validate_gdn_gate(gate, err_cls)
    return jax.nn.silu if gate == "silu" else jax.nn.sigmoid
