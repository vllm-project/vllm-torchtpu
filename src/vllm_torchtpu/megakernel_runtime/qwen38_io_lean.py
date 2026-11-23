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
"""VLLM_TPU_MK_IO_LEAN: which operand-prep levers of the whole-step call are on.

Every lever computes the same values as the code it replaces (integer indexing and data
movement only); it only changes how many XLA ops the call spends around the kernel.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs

LEVERS = ("ring", "cktab")


def io_lean_levers(raw: str | None = None) -> frozenset[str]:
    """The levers named by VLLM_TPU_MK_IO_LEAN (`raw` overrides the env, for tests)."""
    raw = (envs.VLLM_TPU_MK_IO_LEAN if raw is None else raw) or ""
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if parts & {"1", "all", "true"}:
        return frozenset(LEVERS)
    parts -= {"0", "false", "off"}
    unknown = parts - set(LEVERS)
    assert not unknown, f"VLLM_TPU_MK_IO_LEAN: unknown levers {sorted(unknown)}; known {LEVERS}"
    return frozenset(parts)


def io_lean(lever: str) -> bool:
    assert lever in LEVERS, lever
    return lever in io_lean_levers()


def ring_columns(tables: Any, n_rows: int, per_layer: int = 3) -> list:
    """VLLM_TPU_MK_IO_LEAN=ring: every QSA layer's ring-id table (flat `[n_rows * W]`, the
    runner's `[n_rows, W]` block table) replaced by its column 0, the only one read
    (`r.reshape(n, -1)[:, 0]` in the inputs and the cache writer), as rows of one array.

    One strided read of every layer's table instead of a `[n_rows, W]` relayout or column
    slice per layer and reader (xprof j-3d2506a9: 12 x ~0.7 us in each of the two readers).
    The barrier keeps XLA from pushing the readers' slices back into the 12 parameters.
    The other two tables of each layer pass through unchanged.
    """
    tables = list(tables)
    rings = [tables[o] for o in range(0, len(tables), per_layer)]
    cols = jnp.stack([r.reshape(-1)[:: r.size // n_rows] for r in rings])
    cols = jax.lax.optimization_barrier(cols)
    for k, o in enumerate(range(0, len(tables), per_layer)):
        tables[o] = cols[k]
    return tables
