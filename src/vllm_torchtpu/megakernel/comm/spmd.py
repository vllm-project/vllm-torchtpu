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
"""SPMD-safe mesh axis index helper avoiding host HLO `partition-id` ops."""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import jax
from jax import lax
from jax._src.pallas import core as pallas_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..ops import I32

_STOCK_AXIS_INDEX = lax.axis_index
_AXIS_INDEX_LOCK = threading.Lock()
_AXIS_INDEX_DEPTH = 0
_AXIS_INDEX_INTERPRET: Any = False


def _in_pallas_context() -> bool:
    return (
        pallas_core._pallas_tracing_env.grid_context is not None
        or pallas_core.current_grid_env() is not None
    )


def mesh_axis_index(axis_name: str, *, interpret: Any = False) -> jax.Array:
    """Read `axis_index(axis_name)` via a 1-scalar SMEM Pallas kernel outside Pallas."""
    if _in_pallas_context():
        return _STOCK_AXIS_INDEX(axis_name)

    def _kernel(out_ref):
        out_ref[0] = _STOCK_AXIS_INDEX(axis_name).astype(I32)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((1,), I32),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        interpret=interpret,
        name="spmd_safe_axis_index",
    )()[0]


@contextlib.contextmanager
def spmd_safe_axis_index(*, interpret: Any = False):
    """Route outer `jax.lax.axis_index` calls through `mesh_axis_index`."""

    def _patched_axis_index(axis_name):
        return mesh_axis_index(axis_name, interpret=_AXIS_INDEX_INTERPRET)

    global _AXIS_INDEX_DEPTH, _AXIS_INDEX_INTERPRET
    with _AXIS_INDEX_LOCK:
        prev_interpret = _AXIS_INDEX_INTERPRET
        _AXIS_INDEX_INTERPRET = interpret
        if _AXIS_INDEX_DEPTH == 0:
            jax.lax.axis_index = _patched_axis_index
        _AXIS_INDEX_DEPTH += 1
    try:
        yield
    finally:
        with _AXIS_INDEX_LOCK:
            _AXIS_INDEX_DEPTH -= 1
            _AXIS_INDEX_INTERPRET = prev_interpret
            if _AXIS_INDEX_DEPTH == 0:
                jax.lax.axis_index = _STOCK_AXIS_INDEX
