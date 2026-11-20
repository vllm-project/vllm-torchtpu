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
"""EPLB redundant-expert placement and `ecopy` descriptor encoding helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .. import planner
from ..ops import F32


def default_eplb_cores(g: Any, R: int) -> list[list[list[int]]]:
    """Build a deterministic per-layer redundant expert placement `[L][R][EW]`."""
    er, ew = g.ER(R), g.EW(R)
    extra = ew - er
    layers_cores: list[list[list[int]]] = []
    for lyr in range(g.L):
        cores_l: list[list[int]] = []
        for r1 in range(R):
            primary = [r1 * er + s0 for s0 in range(er)]
            redundant: list[int] = []
            if not g.moe_tp and extra > 0 and R > 1:
                for k in range(extra):
                    r0 = (r1 + 1 + (k % (R - 1))) % R
                    s0 = (lyr + k) % er
                    redundant.append(r0 * er + s0)
            else:
                redundant = [-1] * max(0, extra)
            cores_l.append(primary + redundant)
        layers_cores.append(cores_l)
    return layers_cores


def encode_eplb_ecopy(
    g: Any,
    R: int,
    e_cores: list[list[list[int]]] | None = None,
) -> jax.Array:
    """Encode `[L][R][EW]` expert placement table into `[L, E]` `f32` `ecopy` rows."""
    if e_cores is None:
        e_cores = default_eplb_cores(g, R)
    er, ew = g.ER(R), g.EW(R)
    rows = [
        planner.encode_ecopy(g.E, er, ew, g.moe_tp, e_cores[lyr]) for lyr in range(g.L)
    ]
    return jnp.asarray(rows, dtype=F32)


def resolve_mapping_ecopy(
    mapping: Any, lyr: int | None, g: Any, R: int
) -> jax.Array | None:
    """Resolve `[L, E]` or `[E]` (`lyr` != None) `ecopy` from `mapping`."""
    if "ecopy" in mapping:
        ec = mapping["ecopy"]
        return ec if lyr is None else ec[lyr]
    if "e_cores" in mapping:
        ec = encode_eplb_ecopy(g, R, mapping["e_cores"])
        return ec if lyr is None else ec[lyr]
    return None


__all__ = ["default_eplb_cores", "encode_eplb_ecopy", "resolve_mapping_ecopy"]
