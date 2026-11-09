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
   - `GQAModule` and `GDNModule` (via `experimental/fused_pool_io.py`, reached
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
   - `GQAModule.outputs` declares `O[POOL_STATE_NAME]` (`access="w"`), and
     `kernel.build` sets `input_output_aliases` from `ST[POOL_STATE_NAME]` to
     `O[POOL_STATE_NAME]` so CPU `interpret=True` mutates the aliased output
     copy and returns it through `out_map[POOL_STATE_NAME]`.
"""

from __future__ import annotations

from typing import Any

from .cache_common import should_fuse_pool_io
from .cache_hlo import check_fused_pooled_step_hlo
from .exec_spec import ExecSpec


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
