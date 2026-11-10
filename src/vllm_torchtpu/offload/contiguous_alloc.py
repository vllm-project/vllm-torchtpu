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
"""Hand out fresh KV blocks as runs of consecutive ids for the offload tier.

Raiden moves a run of blocks consecutive on host and device as one copy per
KV array (`raiden_store._dma_order`). vLLM's block pool hands out the oldest
free blocks, which after some churn are scattered ids, so a load into them or
a later store of them pays a copy every few blocks and parks the TPU's
in-order queue that much longer. `install` makes the pool pick, among the
oldest few multiples of the request, the longest runs of consecutive ids:
eviction stays among the oldest free blocks, only their order changes.
(The LIFO free-block patch in `vllm_torchtpu/__init__.py`, which keeps block
reuse to a small hot set, applies only with prefix caching off; this widens
the set at most to the oldest few multiples of a request.)
"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def oldest_runs(queue: Any, num_blocks: int, scan: int) -> list | None:
    """`num_blocks` free blocks from the first `scan` of `queue` (oldest
    first), taken as the longest runs of consecutive block ids, in id order.

    Evicts no more prefix-cache entries than taking the oldest `num_blocks`
    would: the pool puts empty blocks first on purpose, so cached blocks are
    taken only up to the count among the oldest `num_blocks`. None when the
    scan finds no more than `num_blocks` blocks.
    """
    candidates = []
    block = queue.fake_free_list_head.next_free_block
    tail = queue.fake_free_list_tail
    while block is not None and block is not tail and len(candidates) < scan:
        candidates.append(block)
        block = block.next_free_block
    if len(candidates) <= num_blocks:
        return None
    cached_budget = sum(b.block_hash is not None for b in candidates[:num_blocks])
    by_id = sorted(candidates, key=lambda b: b.block_id)
    runs: list[list] = []
    for b in by_id:
        if runs and b.block_id == runs[-1][-1].block_id + 1:
            runs[-1].append(b)
        else:
            runs.append([b])
    # Longest first; a stable sort keeps lower ids first among equals.
    runs.sort(key=len, reverse=True)
    chosen: list = []
    taken: set[int] = set()

    def take(b) -> bool:
        nonlocal cached_budget
        if b.block_hash is not None:
            if cached_budget == 0:
                return False
            cached_budget -= 1
        chosen.append(b)
        taken.add(b.block_id)
        return True

    for run in runs:
        for b in run:
            if len(chosen) == num_blocks or not take(b):
                break
    # Top up in LRU order: the oldest `num_blocks` alone fit the budget.
    for b in candidates:
        if len(chosen) == num_blocks:
            break
        if b.block_id not in taken:
            take(b)
    chosen.sort(key=lambda b: b.block_id)
    return chosen


def install(block_pool: Any, scan_factor: int = 3, max_scan: int = 1024) -> None:
    """Route `block_pool.get_new_blocks` through `oldest_runs`.

    The chosen blocks are moved to the head of the free queue and the pool's
    own `get_new_blocks` pops them, so its bookkeeping (cache eviction,
    reuse watchers, ref counts, metrics) is unchanged. A request of n blocks
    looks at the oldest min(scan_factor * n, max(n, max_scan)) free blocks;
    `scan_factor` <= 1 leaves the pool alone.
    """
    if scan_factor <= 1 or getattr(block_pool, "_tpu_contiguous_alloc", False):
        return
    queue = block_pool.free_block_queue
    needed = ("fake_free_list_head", "fake_free_list_tail", "remove", "prepend_n")
    if not all(hasattr(queue, name) for name in needed):
        logger.warning(
            "contiguous_alloc: vLLM's free block queue lacks %s; leaving the "
            "block pool's allocation order alone",
            [name for name in needed if not hasattr(queue, name)],
        )
        return
    pool_get_new_blocks = block_pool.get_new_blocks

    def get_new_blocks(num_blocks: int) -> list:
        if 1 < num_blocks <= block_pool.get_num_free_blocks():
            scan = min(scan_factor * num_blocks, max(num_blocks, max_scan))
            chosen = oldest_runs(queue, num_blocks, scan)
            if chosen is not None:
                for b in chosen:
                    queue.remove(b)
                queue.prepend_n(chosen)
        return pool_get_new_blocks(num_blocks)

    block_pool.get_new_blocks = get_new_blocks
    block_pool._tpu_contiguous_alloc = True
    logger.info(
        "contiguous_alloc: fresh KV blocks as consecutive runs from the oldest "
        "min(%d x n, max(n, %d)) free blocks",
        scan_factor,
        max_scan,
    )
