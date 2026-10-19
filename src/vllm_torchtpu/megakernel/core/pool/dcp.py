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
"""Decode context parallelism (DCP): which rank holds which KV page.

With DCP size `d`, the `d` ranks of one DCP group share a KV head and each
holds `1 / d` of every sequence's KV pages: global kernel page `p` of a
sequence sits on DCP rank `p % d`, at column `p // d` of that rank's page
table. That is vLLM's DCP layout when `cp_kv_cache_interleave_size` equals
the kernel page size, and the rule the batched RPA kernel and the PD
connector's decode reshard use. vLLM builds DCP groups as contiguous runs of
`d` TP ranks, so a kernel's DCP rank is `tp_rank % d`. At `d = 1` every rank
of a KV-head group holds every page (the pool is replicated).

The traced helpers run inside Pallas kernels and `jax.jit`; the `host_`
helpers are their plain-Python twins for metadata checks and tests.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax

# `PoolLayoutFacts.dcp_page_owner` values: every page on every rank of the
# group (DCP 1), or global page `p` on DCP rank `p % d` at column `p // d`.
DCP_OWNER_REPLICATED = "replicated"
DCP_OWNER_PAGE_MOD = "page_mod_dcp"


def dcp_rank(me: Any, dcp: int) -> Any:
    """This kernel's DCP rank from its TP rank `me` (`lax.axis_index("tp")`)."""
    return lax.rem(me, jnp.int32(dcp)) if dcp > 1 else jnp.int32(0)


def local_kv_len(plen: Any, page_size: int, dcp: int, rank: Any) -> Any:
    """Cached tokens of a `plen`-token sequence held by DCP rank `rank`.

    All of them at `dcp == 1`; else those of the kernel pages `p` with
    `p % dcp == rank` (the batched RPA kernel's `cp_local_cache_len`). A
    negative `plen` (a padded row) holds none.
    """
    if dcp == 1:
        return plen
    span = dcp * page_size
    full = lax.div(plen, span) * page_size
    rem = jnp.clip(lax.rem(plen, span) - rank * page_size, 0, page_size)
    return full + rem


def page_owner_col(lp: Any, dcp: int, rank: Any) -> tuple[Any, Any]:
    """`(rank holds it, page-table column)` of global kernel page `lp`."""
    if dcp == 1:
        return jnp.bool_(True), lp
    return lax.rem(lp, dcp) == rank, lax.div(lp, dcp)


def host_local_kv_len(plen: int, page_size: int, dcp: int, rank: int) -> int:
    """`local_kv_len` on host ints."""
    if dcp == 1:
        return int(plen)
    if plen <= 0:
        return 0
    span = dcp * page_size
    rem = min(max(plen % span - rank * page_size, 0), page_size)
    return (plen // span) * page_size + rem


def host_page_owner_col(lp: int, dcp: int) -> tuple[int, int]:
    """`(owner DCP rank, page-table column)` of global kernel page `lp`."""
    return (lp % dcp, lp // dcp) if dcp > 1 else (0, lp)


__all__ = [
    "DCP_OWNER_PAGE_MOD",
    "DCP_OWNER_REPLICATED",
    "dcp_rank",
    "host_local_kv_len",
    "host_page_owner_col",
    "local_kv_len",
    "page_owner_col",
]
