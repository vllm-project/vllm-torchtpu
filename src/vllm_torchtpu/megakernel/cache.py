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
"""Model-agnostic cache pool pipeline (`CachePipe`, `DirectHBM`, `PallasWriter`).

Architecture (Default Split Pool I/O vs. Opt-In Experimental `fused_pool_io`):
------------------------------------------------------------------------------
`PallasWriter` (`make_pooled_step` / `make_flat_pooled_step`) operates over
unified `[num_pages, 2 * kvh, hd // 4, 4, page_size] fp8` block pools:

1. **Default split pool I/O (`fused_pool_io=False`)**:
   - **3-stage split (or 2-stage with `GPR` in-kernel GDN gather)**:
     - Non-communicating gather (`cache_backend.gather_states`, or in-kernel
       read-only `GPR` gather on wide `T > 1` verify) reads per-sequence
       recurrent/conv checkpoints from the unified block pools.
     - Communicating megakernel (`kernel.build(..., paged_kv=True)`) takes the
       unified block pools as **read-only** `HBM` inputs (`access="r"`) with no
       Pallas pool aliasing, performs all layer compute and cross-chip
       collectives, and emits small per-step state outputs (`knew`, `vnew`,
       `conv`/`conv_ckpt`, `rec`/`rec_ckpt`).
     - Non-communicating scatter (`cache_backend.scatter_states`) runs with
       `collective_id=None` and Pallas `input_output_aliases` to scatter new KV
       and recurrent/conv states in-place into the unified block pools.

2. **Experimental opt-in 1-launch fused pool I/O (`fused_pool_io=True`)**:
   - Isolated in `cache_experimental.py` and `experimental/fused_pool_io.py`
     (disabled by default everywhere: `fused_pool_io=False`).
   - On TPU (`ex.interpret=False`), `ST[POOL_STATE_NAME]` is declared
     `access="r"` with no Pallas-level alias (avoiding Mosaic's full-pool copy
     on communicating kernels), `GQABlock` and `GDNBlock` issue async HBM DMA
     reads/writes directly against `ST[POOL_STATE_NAME]`, and the outer
     `jax.jit` (`donate_argnums`) returns the donated input pool references
     directly as outputs (relying on outer XLA `input_output_alias` /
     `jax.buffer_donor = true`). On CPU (`ex.interpret=True`), `O[POOL_STATE_NAME]`
     is aliased to `ST[POOL_STATE_NAME]` via `pallas_call(input_output_aliases=...)`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from . import kernel
from .cache_common import resolve_model_spec, should_use_pool_gather
from .cache_experimental import should_fuse_pool_io
from .cache_flat import make_flat_pooled_step
from .cache_hlo import check_pooled_step_hlo
from .cache_pooled import make_pooled_step
from .exec_spec import ExecSpec
from .pool.config import with_cache


# ---------------------------------------------------------------------------
# CachePipe Hierarchy (`DirectHBM` & `PallasWriter`)
# ---------------------------------------------------------------------------
class CachePipe(abc.ABC):
    """Abstract base class for megakernel cache I/O pipelines."""

    @abc.abstractmethod
    def make_step(
        self,
        mesh: Any,
        g_or_spec: Any,
        ex: ExecSpec,
        *,
        donate: bool = False,
    ) -> Any:
        """Build a step callable `(step_fn, info)` for `(mesh, g_or_spec, ex)`."""


@dataclass(frozen=True)
class DirectHBM(CachePipe):
    """Direct-HBM state cache pipe (`kc`/`vc`, `conv`/`rec` in HBM buffers)."""

    def make_step(
        self,
        mesh: Any,
        g_or_spec: Any,
        ex: ExecSpec,
        *,
        donate: bool = False,
        donate_states: bool | None = None,
    ) -> Any:
        use_donate = donate if donate_states is None else bool(donate_states)
        ms = resolve_model_spec(g_or_spec)
        ex_direct = with_cache(ex, paged_kv=False, fused_pool_io=False)
        return kernel.make_step(mesh, ms, ex_direct, donate_states=use_donate)


@dataclass(frozen=True)
class PallasWriter(CachePipe):
    """Pooled cache pipe (default split 3-stage; opt-in GPR or `fused_pool_io`)."""

    fused_pool_io: bool | str | None = None
    max_b_low: int = 24
    min_b_hi: int = 56

    def make_step(
        self,
        mesh: Any,
        g_or_spec: Any,
        ex: ExecSpec,
        *,
        donate: bool = False,
        donate_pools: bool | None = None,
        fused_pool_io: bool | str | None = None,
    ) -> Any:
        use_donate = donate if donate_pools is None else bool(donate_pools)
        f_mode = self.fused_pool_io if fused_pool_io is None else fused_pool_io
        ms = resolve_model_spec(g_or_spec)
        return make_pooled_step(
            mesh,
            ms,
            ex,
            donate_pools=use_donate,
            fused_pool_io=f_mode,
            max_b_low=self.max_b_low,
            min_b_hi=self.min_b_hi,
        )


__all__ = [
    "CachePipe",
    "DirectHBM",
    "PallasWriter",
    "check_pooled_step_hlo",
    "make_flat_pooled_step",
    "make_pooled_step",
    "should_fuse_pool_io",
    "should_use_pool_gather",
]
