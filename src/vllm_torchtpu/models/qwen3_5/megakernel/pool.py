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
"""Qwen3.5 raw-pool layout: `PoolLayout` / `PoolBackend` from `Qwen35`.

Each of the `g.NATT` attention layers owns one raw pool holding its KV pages and
the checkpoint slots of the `att_every - 1` GDN layers preceding it (see
`pool/layout.py`). The pool kernels and packers live in the generic `pool/`
package; the `g`-taking wrappers below keep the public Qwen3.5 API.
"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.megakernel.cache import (
    default_pool_layout,
    gather_gdn_states,
    pool_to_states,
    scatter_pool_states,
    states_to_pool,
)
from vllm_torchtpu.megakernel.decls import KVCacheDecl, RecurrentCacheDecl
from vllm_torchtpu.megakernel.pool import (
    PoolLayout,
    conv_to_raw_page,
    kv_to_raw_page,
    raw_page_to_conv,
    raw_page_to_kv,
    raw_pages_to_ssm,
    ssm_to_raw_pages,
)
from vllm_torchtpu.megakernel.pool.config import PoolBackend


def qwen35_layer_cache_decls(
    g: Any,
) -> list[KVCacheDecl | RecurrentCacheDecl]:
    """Build the per-layer cache declaration sequence for `Qwen35`."""
    from .model import Qwen35

    return Qwen35.layer_cache_decls(g)


def qwen35_pool_layout(g: Any, *, R: int = 8) -> PoolLayout:
    """`PoolLayout` of `Qwen35Geometry`, derived from `Qwen35` schedule cache decls."""
    from .model import Qwen35

    return Qwen35.pool_layout(g, R=R)


def qwen35_pool_backend(g: Any, *, R: int = 8) -> PoolBackend:
    """Create the model-injected `PoolBackend` for `Qwen35Geometry`."""
    from .model import Qwen35

    return Qwen35.pool_backend(g, R=R)


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
