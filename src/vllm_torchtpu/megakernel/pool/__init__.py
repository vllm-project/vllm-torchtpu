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
"""Generic raw-pool layout, paged cache configuration, and host/test packing.

Framework-free (jax + numpy only) and model-agnostic: everything is driven by a
`PoolLayout` (see `pool/layout.py`). Paged execution settings are the
`CacheExec` extension of `ExecSpec` (`pool/config.py`).
"""

from __future__ import annotations

from .backend import check_ckpt_bounds
from .config import (
    ACT_MODE_VALUES,
    DEFAULT_CACHE_EXEC,
    POOL_GATHER_VALUES,
    POOL_IO_NUMERICS,
    POOL_STATE_NAME,
    CacheExec,
    PoolBackend,
    cache_exec,
    resolved_max_pages_per_seq,
    with_cache,
)
from .layout import PoolLayout
from .pack import (
    allocate_test_layout,
    conv_to_raw_page,
    kv_to_raw_page,
    pool_to_states,
    raw_page_to_conv,
    raw_page_to_kv,
    raw_pages_to_ssm,
    ssm_to_raw_pages,
    states_to_pool,
)

__all__ = [
    "ACT_MODE_VALUES",
    "DEFAULT_CACHE_EXEC",
    "POOL_GATHER_VALUES",
    "POOL_IO_NUMERICS",
    "POOL_STATE_NAME",
    "CacheExec",
    "PoolBackend",
    "cache_exec",
    "resolved_max_pages_per_seq",
    "with_cache",
    "PoolLayout",
    "allocate_test_layout",
    "check_ckpt_bounds",
    "conv_to_raw_page",
    "kv_to_raw_page",
    "pool_to_states",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "ssm_to_raw_pages",
    "states_to_pool",
]
