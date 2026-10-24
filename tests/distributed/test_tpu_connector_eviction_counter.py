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
"""Only cached blocks reused as new blocks count as evictions.

See `TPUConnector.bind_gpu_block_pool`.
"""

from types import SimpleNamespace

import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnector

BLOCK_SIZE = 16


def _manager(enable_caching: bool = True) -> tuple[KVCacheManager, SimpleNamespace]:
    """Returns a 3-usable-block cache manager and a connector bound to it."""
    init_none_hash(sha256)
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=4,  # 3 usable blocks; block 0 is the null block.
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    manager = KVCacheManager(
        config,
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=enable_caching,
    )
    connector = SimpleNamespace()
    TPUConnector.bind_gpu_block_pool(connector, manager.block_pool)
    return manager, connector


def _allocate(manager: KVCacheManager, request_id: str, start: int, num_blocks: int):
    prompt = list(range(start, start + num_blocks * BLOCK_SIZE))
    request = Request(
        request_id,
        prompt,
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    blocks, num_hit_tokens = manager.get_computed_blocks(request)[:2]
    assert num_hit_tokens == 0
    assert manager.allocate_slots(request, len(prompt), 0, blocks) is not None
    return request


def _drain(connector) -> list[int] | None:
    stats = TPUConnector._take_scheduler_stats(connector)
    return None if stats is None else stats.data["num_evicted_blocks"]


def test_counts_cached_blocks_reused_for_new_blocks():
    manager, connector = _manager()
    manager.free(_allocate(manager, "cached", start=0, num_blocks=2))
    assert _drain(connector) is None

    # 1 empty block + 2 cached blocks are free; taking all 3 evicts both.
    _allocate(manager, "new", start=1000, num_blocks=3)
    assert _drain(connector) == [2]
    # Draining resets the count.
    assert _drain(connector) is None


def test_invalidation_is_not_an_eviction():
    manager, connector = _manager()
    request = _allocate(manager, "cached", start=0, num_blocks=2)
    block_ids = manager.get_block_ids(request.request_id)[0]
    manager.free(request)

    # As after a failed KV load: the hash is dropped, the block is not reused.
    manager.evict_blocks({block_ids[0]})
    assert _drain(connector) is None

    # Only the block that still holds a hash is evicted when reused.
    _allocate(manager, "new", start=1000, num_blocks=3)
    assert _drain(connector) == [1]


def test_no_evictions_without_prefix_caching():
    manager, connector = _manager(enable_caching=False)
    manager.free(_allocate(manager, "first", start=0, num_blocks=2))
    _allocate(manager, "second", start=0, num_blocks=3)
    assert _drain(connector) is None


def test_binding_twice_wraps_once():
    manager, connector = _manager()
    pool = manager.block_pool
    wrapped = (pool.get_new_blocks, pool._maybe_evict_cached_block)

    TPUConnector.bind_gpu_block_pool(SimpleNamespace(), pool)

    assert (pool.get_new_blocks, pool._maybe_evict_cached_block) == wrapped
    manager.free(_allocate(manager, "cached", start=0, num_blocks=2))
    _allocate(manager, "new", start=1000, num_blocks=3)
    assert _drain(connector) == [2]


def test_without_block_pool():
    assert TPUConnector._take_scheduler_stats(SimpleNamespace()) is None
