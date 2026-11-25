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
"""Qwen3.5 raw-pool layout: `PoolLayout` / `PoolBackend` from `Qwen35Geometry`.

Each of the `g.NATT` attention layers owns one raw pool holding its KV pages and
the checkpoint slots of the `att_every - 1` GDN layers preceding it (see
`pool/layout.py`). The pool kernels and packers live in the generic `pool/`
package; the `g`-taking wrappers below keep the public Qwen3.5 API.
"""

from __future__ import annotations

from typing import Any

import jax

from .. import pool
from ..blocks.gdn import GDNBlock
from ..blocks.gqa import GQABlock
from ..decls import KVCacheDecl, RecurrentCacheDecl
from ..exec_spec import ExecSpec
from ..pool import (
    PoolLayout,
    conv_to_raw_page,
    kv_to_raw_page,
    raw_page_to_conv,
    raw_page_to_kv,
    raw_pages_to_ssm,
    ssm_to_raw_pages,
)
from ..pool.config import PoolBackend


def qwen35_layer_cache_decls(
    g: Any,
) -> list[KVCacheDecl | RecurrentCacheDecl]:
    """Build the per-layer cache declaration sequence for `Qwen35Geometry`."""
    kv_decl = GQABlock(geometry=g).cache_decl(g)
    if g.att_every == 1:
        return [kv_decl] * g.NATT
    rec_decl = GDNBlock(geometry=g).cache_decl(g)
    group: list[KVCacheDecl | RecurrentCacheDecl] = [rec_decl] * (g.att_every - 1) + [
        kv_decl
    ]
    return group * g.NATT


def qwen35_pool_layout(g: Any, *, R: int = 8) -> PoolLayout:
    """`PoolLayout` of `Qwen35Geometry`, derived from per-layer block cache decls."""
    return pool.derive_pool_layout(qwen35_layer_cache_decls(g), R=R)


def qwen35_pool_backend(g: Any) -> PoolBackend:
    """Create the model-injected `PoolBackend` for `Qwen35Geometry`."""
    return pool.make_pool_backend(
        qwen35_pool_layout(g),
        check=getattr(g, "check", None),
        rec_chunk=GDNBlock.chunk_rows,
    )


def default_pool_layout(g: Any, ex: ExecSpec) -> tuple[Any, ...]:
    """`pool.allocate_test_layout` for `g`: `(page_table, ckpt, roff, min_pages)`."""
    return pool.allocate_test_layout(qwen35_pool_layout(g), ex)


def states_to_pool(
    g: Any, ex: ExecSpec, *args: Any, **kw: Any
) -> tuple[jax.Array, ...]:
    return pool.states_to_pool(qwen35_pool_layout(g), ex, *args, **kw)


def pool_to_states(g: Any, ex: ExecSpec, *args: Any, **kw: Any) -> dict[str, jax.Array]:
    return pool.pool_to_states(qwen35_pool_layout(g), ex, *args, **kw)


def gather_gdn_states(g: Any, ex: ExecSpec, *args: Any, **kw: Any) -> Any:
    if hasattr(g, "check"):
        g.check(ex)
    return pool.gather_gdn_states(qwen35_pool_layout(g), ex, *args, **kw)


def scatter_pool_states(g: Any, ex: ExecSpec, *args: Any, **kw: Any) -> Any:
    if hasattr(g, "check"):
        g.check(ex)
    return pool.scatter_pool_states(qwen35_pool_layout(g), ex, *args, **kw)


__all__ = [
    "conv_to_raw_page",
    "default_pool_layout",
    "gather_gdn_states",
    "kv_to_raw_page",
    "pool_to_states",
    "qwen35_layer_cache_decls",
    "qwen35_pool_backend",
    "qwen35_pool_layout",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "scatter_pool_states",
    "ssm_to_raw_pages",
    "states_to_pool",
]
