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
"""Measured decode tiles for the packed StreamIndex KV cache."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import get_device_name

logger = init_logger(__name__)


class TunedKey(NamedTuple):
    device: str
    query_shape: tuple[int, ...]
    cache_page_shape: tuple[int, ...]
    pages_per_sequence: int
    query_dtype: np.dtype
    topk: int
    compression_ratio: int


# 64K decode, H64/D128 FP8, HEAD_ALONG_SUBLANE, 256 positions per page.
# 24 pages reduce scoring from 22 pipeline iterations to 3, including decode
# growth beyond the initial 64K prompt. Larger tiles did not improve latency.
TUNED_DECODE_KV_PAGES = {
    TunedKey(
        "TPU v7", (1, 64, 128), (64, 4, 256), 65, np.dtype(jnp.float8_e4m3fn), 512, 4
    ): 24,
}

# Persistent SAL cache with the same 64K context and H64/D128 query geometry.
TUNED_DECODE_KV_PAGES.update(
    {
        TunedKey(
            "TPU v7",
            (batch, 64, 128),
            (33, 4, 256),
            65,
            np.dtype(jnp.float8_e4m3fn),
            512,
            4,
        ): pages
        for batch, pages in ((1, 24), (4, 12), (64, 6))
    }
)


def get_decode_kv_pages_per_block(
    q: jax.Array,
    cache_kv: jax.Array,
    page_indices: jax.Array,
    *,
    k: int,
    compression_ratio: int,
) -> int:
    """Select a measured single-query tile; retain the existing default otherwise.

    The lookup uses cache capacity, not the dynamic sequence length. The entry
    targets 64K decode; shorter contexts using that capacity may not benefit.
    """
    key = TunedKey(
        get_device_name(),
        tuple(q.shape),
        tuple(cache_kv.shape[1:]),
        page_indices.shape[-1],
        q.dtype,
        k,
        compression_ratio,
    )
    if key in TUNED_DECODE_KV_PAGES:
        return TUNED_DECODE_KV_PAGES[key]
    logger.warning_once(
        "No tuned StreamIndex decode tile for %s; using 3 KV pages", key
    )
    return 3


# Joint page/group tuning, keyed by the same complete workload signature.
# Entries are populated only from matched decode experiments.
TUNED_DECODE_GROUP_SIZES: dict[TunedKey, int] = {}


def get_decode_group_size(
    q: jax.Array,
    cache_kv: jax.Array,
    page_indices: jax.Array,
    *,
    k: int,
    compression_ratio: int,
) -> int:
    """Choose a measured request group size, preserving existing defaults."""
    key = TunedKey(
        get_device_name(),
        tuple(q.shape),
        tuple(cache_kv.shape[1:]),
        page_indices.shape[-1],
        q.dtype,
        k,
        compression_ratio,
    )
    default = (16 if cache_kv.shape[1] == 33 else 32) if q.shape[0] == 64 else 4
    return TUNED_DECODE_GROUP_SIZES.get(key, default)


# DECODE_NEXT_MEASURED_ENTRIES
# TPU7 SAL decode, 256 request metadata slots. Page tiles amortize DMA and
# grouped requests reduce pipeline launches; larger tiles/groups regressed.
for _batch, _capacity, _pages, _group in (
    (8, 9, 6, 8),
    (8, 33, 12, 8),
    (8, 65, 12, 8),
    (8, 129, 12, 8),
    (16, 9, 6, 8),
    (16, 33, 6, 16),
    (16, 65, 12, 16),
    (16, 129, 6, 16),
    (32, 9, 6, 16),
    (32, 33, 6, 32),
    (32, 65, 6, 32),
    (32, 129, 6, 32),
):
    _key = TunedKey(
        "TPU v7",
        (_batch, 64, 128),
        (33, 4, 256),
        _capacity,
        np.dtype(jnp.float8_e4m3fn),
        512,
        4,
    )
    TUNED_DECODE_KV_PAGES[_key] = _pages
    TUNED_DECODE_GROUP_SIZES[_key] = _group


# Single-row CSA decode: 16 cooperative subcores spend more time reducing
# per-subcore histograms and gathering selected indices than four do. The
# target 18,432-score row fits comfortably as four 4,608-word slices. This
# entry changes neither score arithmetic nor the exact selection algorithm.
TUNED_TOPK_PARTITIONS = {
    ("TPU v7", 1, 18432, 512): 4,
}


def get_topk_partition(batch: int, width: int, k: int, *, default: int) -> int:
    key = (get_device_name(), batch, width, k)
    if key in TUNED_TOPK_PARTITIONS:
        return TUNED_TOPK_PARTITIONS[key]
    logger.warning_once(
        "No tuned SparseCore TopK partition for %s; using %d", key, default
    )
    return default
