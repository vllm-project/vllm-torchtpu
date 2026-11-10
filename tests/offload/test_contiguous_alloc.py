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
"""contiguous_alloc on vLLM's real BlockPool: runs of consecutive ids from the
oldest free blocks, with the pool's own bookkeeping."""

from vllm.v1.core.block_pool import BlockPool

from vllm_torchtpu.offload import contiguous_alloc


def _pool(num_blocks=64):
    pool = BlockPool(num_blocks, enable_caching=True, hash_block_size=16)
    contiguous_alloc.install(pool)
    return pool


def _scatter(pool, ids):
    """Leave exactly `ids` free, oldest first in that order; every other block
    is held by someone."""
    queue = pool.free_block_queue
    for b in queue.get_all_free_blocks():
        queue.remove(b)
        b.ref_cnt = 1
    head = [pool.blocks[i] for i in ids]
    for b in head:
        b.ref_cnt = 0
    queue.append_n(head)


def _ids(blocks):
    return [b.block_id for b in blocks]


def test_takes_longest_runs_among_oldest_in_id_order():
    pool = _pool()
    _scatter(pool, [9, 30, 31, 32, 5, 33, 6, 40, 7, 41])
    blocks = pool.get_new_blocks(4)
    assert _ids(blocks) == [30, 31, 32, 33]
    assert all(b.ref_cnt == 1 for b in blocks)
    assert pool.get_num_free_blocks() == 10 - 4


def test_fills_with_next_longest_runs():
    pool = _pool()
    _scatter(pool, [9, 30, 31, 5, 6, 7, 40, 50, 51, 52])
    assert _ids(pool.get_new_blocks(5)) == [5, 6, 7, 50, 51]


def test_single_block_keeps_lru_order():
    pool = _pool()
    _scatter(pool, [9, 30, 31])
    assert _ids(pool.get_new_blocks(1)) == [9]


def test_evicts_cached_hash_of_reused_block():
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    pool = _pool()
    _scatter(pool, [20, 21, 3])
    key = make_block_hash_with_group_id(b"h", 0)
    pool._insert_block_hash(key, pool.blocks[21], num_tokens=None)
    assert _ids(pool.get_new_blocks(2)) == [20, 21]
    assert pool.blocks[21].block_hash is None
    assert pool.get_cached_block(b"h", [0]) is None


def _cache(pool, ids):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    for i in ids:
        key = make_block_hash_with_group_id(b"h%d" % i, 0)
        pool._insert_block_hash(key, pool.blocks[i], num_tokens=None)


def test_takes_no_more_cached_blocks_than_the_oldest_would():
    pool = _pool()
    # Empty blocks 9, 5, 7 first (as free_blocks prepends them), then a
    # cached run 30-33: the oldest three hold no cached block, so neither may
    # the choice.
    _scatter(pool, [9, 5, 7, 30, 31, 32, 33])
    _cache(pool, [30, 31, 32, 33])
    assert _ids(pool.get_new_blocks(3)) == [5, 7, 9]
    assert all(pool.blocks[i].block_hash is not None for i in (30, 31, 32, 33))


def test_cached_budget_follows_the_oldest():
    pool = _pool()
    _scatter(pool, [9, 30, 5, 31, 32, 33])
    _cache(pool, [30, 31, 32, 33])
    # The oldest three (9, 30, 5) hold one cached block: one may go.
    assert _ids(pool.get_new_blocks(3)) == [5, 9, 30]


def test_scan_cap():
    pool = BlockPool(64, enable_caching=True, hash_block_size=16)
    contiguous_alloc.install(pool, max_scan=3)
    _scatter(pool, [9, 5, 7, 30, 31])
    assert _ids(pool.get_new_blocks(2)) == [5, 7]  # 30-31 lie past the cap


def test_whole_pool():
    pool = _pool(8)
    blocks = pool.get_new_blocks(7)
    assert _ids(blocks) == list(range(1, 8))
    assert pool.get_num_free_blocks() == 0
