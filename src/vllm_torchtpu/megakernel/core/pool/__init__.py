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
"""Generic raw-pool layout, packing, and Pallas gather / scatter kernels.

Framework-free (jax + numpy only) and model-agnostic: everything is driven by a
`PoolLayout` (see `pool/layout.py`). Models build their layout and wrap it in a
`PoolBackend` with `make_pool_backend(layout)`. Paged execution settings are
the `CacheExec` extension of `ExecSpec` (`pool/config.py`).
"""

from __future__ import annotations

from .backend import (
    GATHER_KERNEL_NAME,
    POOL_KERNEL_NAMES,
    SCATTER_KERNEL_NAME,
    can_pool_gather,
    check_ckpt_bounds,
    make_pool_backend,
)
from .checks import check_step_pool_metadata, safe_kv_page, safe_page_base
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
from .derive import derive_pool_layout
from .gather import gather_gdn_states
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
from .scatter import scatter_pool_states

__all__ = [
    "ACT_MODE_VALUES",
    "DEFAULT_CACHE_EXEC",
    "GATHER_KERNEL_NAME",
    "POOL_GATHER_VALUES",
    "POOL_IO_NUMERICS",
    "POOL_STATE_NAME",
    "CacheExec",
    "PoolBackend",
    "cache_exec",
    "resolved_max_pages_per_seq",
    "with_cache",
    "POOL_KERNEL_NAMES",
    "SCATTER_KERNEL_NAME",
    "PoolLayout",
    "allocate_test_layout",
    "can_pool_gather",
    "check_ckpt_bounds",
    "check_step_pool_metadata",
    "conv_to_raw_page",
    "derive_pool_layout",
    "gather_gdn_states",
    "kv_to_raw_page",
    "make_pool_backend",
    "pool_to_states",
    "raw_page_to_conv",
    "raw_page_to_kv",
    "raw_pages_to_ssm",
    "safe_kv_page",
    "safe_page_base",
    "scatter_pool_states",
    "ssm_to_raw_pages",
    "states_to_pool",
]
