# SPDX-License-Identifier: Apache-2.0
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

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
# One full block plus a partial one: the most a hit can cover is one block.
PROMPT = list(range(BLOCK_SIZE + 8))


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    """Patch unpatched BlockPool methods around a stub `Scheduler.schedule`.

    The plugin may already have patched these in this process; unwrap to
    upstream first so the test patches exactly once, and restore afterwards.
    """
    for name in ("_insert_block_hash", "get_cached_block"):
        monkeypatch.setattr(BlockPool, name, inspect.unwrap(getattr(BlockPool, name)))
    monkeypatch.setattr(Scheduler, "schedule", lambda self, *args, **kwargs: None)
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


def _start_step(manager: KVCacheManager) -> None:
    Scheduler.schedule(SimpleNamespace(kv_cache_manager=manager))


def test_same_step_request_recomputes_unwritten_prefix(manager):
    _start_step(manager)
    assert _admit(manager, _request("filler")) == 0
    # Upstream hands this request the block "filler" is still filling.
    assert _admit(manager, _request("same_step")) == 0


def test_later_step_hits_written_prefix(manager):
    _start_step(manager)
    _admit(manager, _request("filler"))
    _admit(manager, _request("same_step"))

    _start_step(manager)
    assert _admit(manager, _request("next_step")) == BLOCK_SIZE


def test_patch_is_idempotent():
    patched_methods = (
        BlockPool._insert_block_hash,
        BlockPool.get_cached_block,
        Scheduler.schedule,
    )

    _patch_vllm_same_step_prefix_hits()

    assert (
        BlockPool._insert_block_hash,
        BlockPool.get_cached_block,
        Scheduler.schedule,
    ) == patched_methods
