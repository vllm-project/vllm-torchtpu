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
"""No prefix-cache hit on a block the current step has not written yet.

The TPU attention kernel writes each sequence's KV as it walks the batch, so
a request that hits a block another request fills in the same step can read
it before it is written. See `_patch_vllm_same_step_prefix_hits`.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu import _patch_vllm_same_step_prefix_hits

BLOCK_SIZE = 16
# One full block plus a partial one: the most a hit can cover is one block.
PROMPT = list(range(BLOCK_SIZE + 8))


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    """Patch unpatched methods around a stub `Scheduler.schedule`.

    The plugin may already have patched these in this process; unwrap to
    upstream first so the test patches exactly once, and restore afterwards.
    The stub keeps the one thing the real `schedule()` does that matters here:
    it calls `kv_cache_manager.new_step_starts()` once per step, before it
    looks anything up or allocates. So the `[scheduler]` cases check the wiring
    through `Scheduler.schedule`, not vLLM's scheduler itself; that the real
    `schedule()` clears the set is covered in tests/test_abort_on_preemption.py.
    """
    for name in ("_insert_block_hash", "get_cached_block"):
        monkeypatch.setattr(BlockPool, name, inspect.unwrap(getattr(BlockPool, name)))
    monkeypatch.setattr(
        KVCacheManager,
        "new_step_starts",
        inspect.unwrap(KVCacheManager.new_step_starts),
    )
    monkeypatch.setattr(
        Scheduler,
        "schedule",
        lambda self, *args, **kwargs: self.kv_cache_manager.new_step_starts(),
    )
    _patch_vllm_same_step_prefix_hits()


@pytest.fixture
def manager():
    init_none_hash(sha256)
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
    )


def _request(request_id: str) -> Request:
    return Request(
        request_id,
        PROMPT,
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _admit(manager: KVCacheManager, request: Request) -> int:
    """Schedule a waiting request the way the scheduler does; return its hit."""
    blocks, num_hit_tokens = manager.get_computed_blocks(request)[:2]
    assert (
        manager.allocate_slots(
            request, request.num_tokens - num_hit_tokens, num_hit_tokens, blocks
        )
        is not None
    )
    return num_hit_tokens


def _scheduler_step(manager: KVCacheManager) -> None:
    # `vllm serve`: the step begins in `Scheduler.schedule`.
    Scheduler.schedule(SimpleNamespace(kv_cache_manager=manager))


def _kv_cache_manager_step(manager: KVCacheManager) -> None:
    # An engine that drives `KVCacheManager` without vLLM's scheduler.
    manager.new_step_starts()


@pytest.fixture(
    params=[_scheduler_step, _kv_cache_manager_step],
    ids=["scheduler", "kv_cache_manager"],
)
def start_step(request):
    return request.param


def test_same_step_request_recomputes_unwritten_prefix(manager, start_step):
    start_step(manager)
    assert _admit(manager, _request("filler")) == 0
    # Upstream hands this request the block "filler" is still filling.
    assert _admit(manager, _request("same_step")) == 0


def test_later_step_hits_written_prefix(manager, start_step):
    start_step(manager)
    _admit(manager, _request("filler"))
    _admit(manager, _request("same_step"))

    start_step(manager)
    assert _admit(manager, _request("next_step")) == BLOCK_SIZE


def test_patch_is_idempotent():
    patched_methods = (
        BlockPool._insert_block_hash,
        BlockPool.get_cached_block,
        KVCacheManager.new_step_starts,
    )

    _patch_vllm_same_step_prefix_hits()

    assert (
        BlockPool._insert_block_hash,
        BlockPool.get_cached_block,
        KVCacheManager.new_step_starts,
    ) == patched_methods
