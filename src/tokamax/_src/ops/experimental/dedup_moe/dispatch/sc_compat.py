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
# ==============================================================================
"""Backports of two JAX 0.12.0 Pallas fixes the SparseCore dispatch needs.

1. `pl.kernel` discharge keeps output memory spaces. Without it, outputs such
   as `SemaphoreType.DMA(...) @ mesh` come back as plain device arrays once
   refs are discharged.
2. A TensorCore `pallas_call` accepts a `SEMAPHORE @ VectorSubcoreMesh`
   operand (used by the SC return pushes and the SC gateway combine).

Each backport is feature-detected and installed at most once, and only when
`ensure_sc_compat()` is called; on a JAX that has the fixes it does nothing.
"""

import dataclasses
import importlib
import inspect
import threading
from typing import Any

_LOCK = threading.Lock()
_APPLIED: dict[str, bool] | None = None

# Lowering functions that pick window params from the TPU memory space.
# `_memory_space_to_mosaic_attribute` is not one: it needs the wrapped space.
_WINDOW_PARAM_CALLERS = frozenset({
    "_check_block_mappings",
    "lower_jaxpr_into_pipelined_module",
    "lower_jaxpr_to_pipelined_module",
    "lower_jaxpr_into_unpipelined_module",
})


def _import(name: str) -> Any:
  try:
    return importlib.import_module(name)
  except ImportError:
    return None


def _backport_discharge_out_avals() -> bool:
  """Installs backport 1. Returns whether it was installed."""
  mpmd = _import("jax._src.pallas.mpmd")
  discharge = _import("jax._src.state.discharge")
  if mpmd is None or discharge is None:
    return False
  rule = getattr(mpmd, "_mpmd_map_discharge_rule", None)
  try:
    if rule is None or "out_avals" in inspect.signature(rule).parameters:
      return False  # Fixed: the rule reads the primitive's own out_avals.
  except (TypeError, ValueError):
    return False
  prim = getattr(mpmd, "mpmd_map_p", None)
  rules = getattr(discharge, "_discharge_rules", None)
  discharged_aval: Any = getattr(discharge, "discharged_aval", None)
  if prim is None or rules is None or discharged_aval is None:
    return False
  orig = rules.get(prim)
  if orig is None or getattr(orig, "_tokamax_sc_compat", False):
    return False

  def patched(ctx, *args, input_output_aliases, out_avals, **params):
    outin = {o: i for i, o in dict(input_output_aliases).items()}
    fixed = tuple(
        discharged_aval(
            ctx.in_avals[outin[i]] if i in outin else a,
            discharge=True,
            strip_memory_space=ctx.strip_memory_space,
        )
        for i, a in enumerate(out_avals)
    )
    # The unfixed rule takes its output avals from ctx.out_avals.
    return orig(
        dataclasses.replace(ctx, out_avals=fixed),
        *args,
        input_output_aliases=input_output_aliases,
        out_avals=out_avals,
        **params,
    )

  setattr(patched, "_tokamax_sc_compat", True)
  rules[prim] = patched
  return True


def _backport_cross_core_semaphore_windows() -> bool:
  """Installs backport 2. Returns whether it was installed."""
  tpu_core = _import("jax._src.pallas.mosaic.core")
  pallas_core = _import("jax._src.pallas.core")
  if tpu_core is None or pallas_core is None:
    return False
  if hasattr(tpu_core, "unwrap_core_memory_space"):
    return False  # Fixed: lowering unwraps cross-core memory spaces itself.
  core_ms_cls = getattr(pallas_core, "CoreMemorySpace", None)
  orig = getattr(tpu_core, "memory_space_to_tpu_memory_space", None)
  if core_ms_cls is None or orig is None:
    return False
  if getattr(orig, "_tokamax_sc_compat", False):
    return False
  unwrap = (tpu_core.MemorySpace.SEMAPHORE, tpu_core.MemorySpace.HBM)

  def patched(memory_space, *args, **kwargs):
    out = orig(memory_space, *args, **kwargs)
    if isinstance(out, core_ms_cls) and out.memory_space in unwrap:
      frame = inspect.currentframe()
      caller = frame.f_back if frame else None
      try:
        if caller and caller.f_code.co_name in _WINDOW_PARAM_CALLERS:
          return out.memory_space
      finally:
        del frame, caller
    return out

  setattr(patched, "_tokamax_sc_compat", True)
  tpu_core.memory_space_to_tpu_memory_space = patched
  return True


def ensure_sc_compat() -> dict[str, bool]:
  """Installs the backports this JAX needs, once per process.

  Returns:
    `{"discharge_out_avals": bool, "cross_core_semaphore_windows": bool}`,
    True for each backport installed in this process.
  """
  global _APPLIED
  with _LOCK:
    if _APPLIED is None:
      _APPLIED = {
          "discharge_out_avals": _backport_discharge_out_avals(),
          "cross_core_semaphore_windows": (
              _backport_cross_core_semaphore_windows()
          ),
      }
    return dict(_APPLIED)
