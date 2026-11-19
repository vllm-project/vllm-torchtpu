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

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputSchedulerConnector,
)
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

from tests.platforms.same_step_prefix_hits_test_utils import apply_patch
from vllm_torchtpu import _patch_vllm_same_step_prefix_hits

BLOCK_SIZE = 16
# One full block plus a partial one: the most a hit can cover is one block.
PROMPT = list(range(BLOCK_SIZE + 8))


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    """Patch upstream's methods around a stub `Scheduler.schedule`.

    The stub keeps the one thing the real `schedule()` does that matters here:
    it calls `kv_cache_manager.new_step_starts()` once per step, before it
    looks anything up or allocates. So the `[scheduler]` cases check the wiring
    through `Scheduler.schedule`, not vLLM's scheduler itself; that the real
    `schedule()` clears the set is covered in tests/test_abort_on_preemption.py.
    """
    monkeypatch.setattr(
        Scheduler,
        "schedule",
        lambda self, *args, **kwargs: self.kv_cache_manager.new_step_starts(),
    )
    apply_patch(monkeypatch)


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


def test_connector_sees_unwritten_blocks_and_admission_does_not(manager):
    """vLLM's routed-experts connector asks whether a block is still cached,
    so it sees the block a request filled this step. Admission still does not
    hit that block."""
    manager.new_step_starts()
    filler = _request("filler")
    _admit(manager, filler)
    connector = AuxOutputSchedulerConnector(manager)

    assert connector._gpu_cached(filler.block_hashes[0]) is not None
    assert _admit(manager, _request("same_step")) == 0


def test_connector_asks_the_pools_own_lookup():
    """The connector's lookup still goes through its block pool's own
    `get_cached_block`, which a stand-in pool in a test may replace."""
    cached = object()
    pool = SimpleNamespace(
        get_cached_block=lambda block_hash, group_ids: [cached], hash_block_size=4
    )
    connector = AuxOutputSchedulerConnector(
        SimpleNamespace(
            coordinator=SimpleNamespace(
                single_type_managers=[SimpleNamespace(block_size=4, block_pool=pool)]
            ),
            block_pool=pool,
            kv_cache_config=SimpleNamespace(num_blocks=16),
        )
    )

    assert connector._gpu_cached(b"block") == [cached]


def test_patch_is_idempotent():
    patched_methods = (
        BlockPool._insert_block_hash,
        BlockPool.get_cached_block,
        KVCacheManager.new_step_starts,
        AuxOutputSchedulerConnector._gpu_cached,
    )

    _patch_vllm_same_step_prefix_hits()

    assert (
        BlockPool._insert_block_hash,
        BlockPool.get_cached_block,
        KVCacheManager.new_step_starts,
        AuxOutputSchedulerConnector._gpu_cached,
    ) == patched_methods
