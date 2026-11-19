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
"""The LIFO free-block patch keeps every block in the pool that owns it."""

from vllm.v1.core.block_pool import BlockPool

from vllm_torchtpu import _patch_vllm_block_pool_lifo_free


def test_mixed_pool_free_returns_each_block_to_its_owner():
    _patch_vllm_block_pool_lifo_free()
    shared = BlockPool(num_gpu_blocks=20, enable_caching=True, hash_block_size=16)
    private = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=16)
    shared_blocks = shared.get_new_blocks(5)
    private_blocks = private.get_new_blocks(5)
    # A deferred request free hands the shared pool blocks of every group.
    shared.free_blocks(list(reversed(shared_blocks + private_blocks)))

    assert shared.get_num_free_blocks() == 19
    assert private.get_num_free_blocks() == 7
    reused = shared.get_new_blocks(19)
    ids = [block.block_id for block in reused]
    assert len(ids) == len(set(ids))
    assert all(block.pool is shared for block in reused)


def test_freed_hashless_blocks_are_reused_first():
    _patch_vllm_block_pool_lifo_free()
    pool = BlockPool(num_gpu_blocks=10, enable_caching=False, hash_block_size=16)
    blocks = pool.get_new_blocks(3)
    # Freed tail-first, so the tail block is handed out again first.
    pool.free_blocks(reversed(blocks))
    reused = pool.get_new_blocks(3)
    assert [b.block_id for b in reused] == [b.block_id for b in reversed(blocks)]
