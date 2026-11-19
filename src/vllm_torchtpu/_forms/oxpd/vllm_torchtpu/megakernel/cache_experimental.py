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
"""Experimental opt-in 1-launch fused pool I/O pipeline (`cache_experimental`).

Contract & Safety Model:
------------------------
Mosaic TPU rejects `input_output_aliases` on large HBM pool operands inside a
communicating (`collective_id != None`) `pallas_call` by inserting a full-pool
HBM copy before the kernel. To avoid that 2.30 GiB/pool copy in 1-launch mode
(`CacheExec.FUSED`):

1. **On TPU (`ex.interpret == False`)**:
   - The communicating `decode_megakernel` declares `ST[POOL_STATE_NAME]` with
     `access="r"` and sets **no** `pallas_call` `input_output_aliases` on the
     pools.
   - `GQABlock` and `GDNBlock` (via `experimental/fused_pool_io.py`, reached
     through `fused_io_hook`) issue async HBM DMA reads and writes
     directly against `ST[POOL_STATE_NAME]`.
   - The outer `jax.jit` (`donate_argnums` on `raw_pools`) returns the donated
     input `raw_pools` references directly as outputs, relying on XLA buffer
     donation (`jax.buffer_donor = true` / `HloModule input_output_alias`) to
     alias the outer input and output buffers in-place.
   - **Important**: Because `pallas_call` does not declare a write or alias on
     `ST[POOL_STATE_NAME]` on TPU, correctness relies entirely on outer buffer
     donation being honored with zero intermediate full-pool copies. Keep this
     path **opt-in** (`fused_pool_io=False` by default everywhere) and verify
     both `check_fused_pooled_step_hlo` and outer `torch_tpu` donation IR on
     every compiled bucket when enabled. `make_pooled_step` /
     `make_flat_pooled_step` raise `ValueError` for a fused build with
     `donate_pools=False`, and the vLLM driver IR-checks every fused bucket
     during warmup (eager mode included).

2. **On CPU (`ex.interpret != False`)**:
   - `GQABlock.outputs` declares `O[POOL_STATE_NAME]` (`access="w"`), and
     `kernel.build` sets `input_output_aliases` from `ST[POOL_STATE_NAME]` to
     `O[POOL_STATE_NAME]` so CPU `interpret=True` mutates the aliased output
     copy and returns it through `out_map[POOL_STATE_NAME]`.
"""

from __future__ import annotations

from typing import Any

from .cache_common import require_pool_backend, resolve_model_spec
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .fused_io_hook import fused_io_available
from .pool import POOL_KERNEL_NAMES
from .pool.config import cache_exec


def should_fuse_pool_io(
    model_spec: Any,
    ex: ExecSpec,
    *,
    fused_pool_io: bool | str | None = False,
    max_b_low: int = 24,
    min_b_hi: int = 56,
) -> bool:
    """Resolve whether `(model_spec, ex)` should use opt-in 1-launch `fused_pool_io`.

    Args:
      model_spec: `ModelSpec` or geometry config implementing `.to_model_spec()`.
      ex: Execution specification (`ex.B`, `cache_exec(ex).fused_pool_io`).
      fused_pool_io:
        - `None`: defer to `bool(cache_exec(ex).fused_pool_io)` (default `False`).
        - `False` (default) / `"false"` / `"0"` / `"off"` / `"split"` / `"3stage"`:
          use the default split pool I/O pipeline.
        - `True` / `"true"` / `"1"` / `"on"` / `"fused"`: force 1-launch fused I/O.
        - `"auto"`: opt-in selective heuristic that fuses when `ex.B <= max_b_low`
          (`24`) or `ex.B >= min_b_hi` (`56`) and uses split I/O for `32..48`.
      max_b_low: Upper batch threshold for low-`B` fusion in `"auto"` mode.
      min_b_hi: Lower batch threshold for high-`B` fusion in `"auto"` mode.

    Without a registered fused implementation (`fused_io_available()`), or when
    the pool backend doesn't support fused I/O, `"auto"` resolves to split pool
    I/O and an explicit `True` raises `MegakernelConfigError`.
    """
    ms = resolve_model_spec(model_spec)
    pb = require_pool_backend(ms)
    if fused_pool_io is None:
        want_fused = bool(cache_exec(ex).fused_pool_io)
        explicit_true = want_fused
    elif isinstance(fused_pool_io, bool):
        want_fused = fused_pool_io
        explicit_true = fused_pool_io
    elif isinstance(fused_pool_io, str):
        s = fused_pool_io.strip().lower()
        if s in ("1", "true", "yes", "on", "fused"):
            want_fused = True
            explicit_true = True
        elif s in ("", "0", "false", "no", "off", "none", "split", "3stage"):
            want_fused = False
            explicit_true = False
        elif s == "auto":
            want_fused = bool(int(max_b_low) >= ex.B or int(min_b_hi) <= ex.B)
            explicit_true = False
        else:
            raise ValueError(
                f"Unsupported fused_pool_io={fused_pool_io!r}; "
                "expected bool, 'auto', 'true', or 'false'"
            )
    else:
        raise TypeError(
            f"Unsupported fused_pool_io type {type(fused_pool_io).__name__}"
        )

    if not pb.supports_fused_io:
        if explicit_true:
            raise MegakernelConfigError(
                f"ModelSpec {ms.name!r} cache_backend does not support fused_pool_io"
            )
        return False
    if want_fused and not fused_io_available():
        if explicit_true:
            raise MegakernelConfigError(
                "fused_pool_io=True needs a registered fused pool I/O "
                "implementation (the megakernel experimental package is not loaded)"
            )
        return False
    return want_fused


def check_fused_pooled_step_hlo(
    hlo_text: str,
    *,
    kernel_names: tuple[str, str] | None = None,
) -> tuple[bool, str]:
    """Verify that a 1-launch `fused_pool_io` HLO has no separate gather/scatter."""
    gather_name, scatter_name = (
        POOL_KERNEL_NAMES if kernel_names is None else kernel_names
    )
    if (gather_name and gather_name in hlo_text) or (
        scatter_name and scatter_name in hlo_text
    ):
        return (
            False,
            "unexpected separate gather/scatter kernel in fused_pool_io HLO",
        )
    return True, "ok"


def make_flat_fused_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = True,
    fused_pool_io: bool | str = True,
    max_b_low: int = 24,
    min_b_hi: int = 56,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]:
    """Build an experimental flat-signature `shard_map` step with `fused_pool_io`."""
    from . import cache

    return cache.make_flat_pooled_step(
        mesh,
        model_spec,
        ex,
        donate_pools=donate_pools,
        fused_pool_io=fused_pool_io,
        max_b_low=max_b_low,
        min_b_hi=min_b_hi,
    )


def make_fused_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = True,
    fused_pool_io: bool | str = True,
    max_b_low: int = 24,
    min_b_hi: int = 56,
) -> Any:
    """Build an experimental `jax.jit` `shard_map` step with `fused_pool_io`.

    `donate_pools` must stay True for a fused build (enforced by `make_pooled_step`).
    """
    from . import cache

    return cache.make_pooled_step(
        mesh,
        model_spec,
        ex,
        donate_pools=donate_pools,
        fused_pool_io=fused_pool_io,
        max_b_low=max_b_low,
        min_b_hi=min_b_hi,
    )


__all__ = [
    "check_fused_pooled_step_hlo",
    "make_flat_fused_pooled_step",
    "make_fused_pooled_step",
    "should_fuse_pool_io",
]
