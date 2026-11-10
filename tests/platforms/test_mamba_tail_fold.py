# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MAMBA_TAIL_FOLD: the scheduler folds the tail, the manager
reserves the boundary block, and the worker writes exactly that block.

See `vllm_torchtpu.core.mamba_tail_fold`. The worker never sees the
manager's decision, so it re-derives it (`prefill_checkpoints`). These tests
pin that mirror against vLLM's own rule and against a real `MambaManager`
driven the way the scheduler drives it: a request whose block the manager
hashes at the boundary must be one whose block the worker fills.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import mamba_tail_fold
from vllm_torchtpu.core.mamba_tail_fold import (
    prefill_checkpoint_targets,
    prefill_checkpoints,
    with_prefill_checkpoint,
)

B = 16


@pytest.fixture(autouse=True)
def _restore_split():
    original = Scheduler._mamba_block_aligned_split
    try:
        yield
    finally:
        Scheduler._mamba_block_aligned_split = original


def _mirror(start: int, scheduled: int) -> tuple[int, int]:
    offsets, columns = prefill_checkpoints(np.array([start]), np.array([scheduled]), B)
    return int(offsets[0]), int(columns[0])


def test_worker_mirror_matches_vllm_rule():
    for start in range(0, 8 * B, 4):
        for scheduled in range(0, 8 * B):
            end = start + scheduled
            position = get_mamba_prefill_checkpoint_position(end, B, False)
            valid = scheduled > 0 and is_mamba_prefill_checkpoint_valid(
                query_start=start,
                query_end=end,
                checkpoint_position=position,
                hash_block_size=B,
                mamba_block_size=B,
                checkpoint_alignment=B,
            )
            want = (position - start, cdiv(end, B) - 2) if valid else (0, 0)
            assert _mirror(start, scheduled) == want, (start, scheduled)


def test_worker_mirror_typical_tail():
    # A 1000-token prompt with 512-token blocks: one step of 1000 tokens,
    # state at 512 into column 0 (the running state sits in column 1).
    offsets, columns = prefill_checkpoints(np.array([0]), np.array([1000]), 512)
    assert (offsets[0], columns[0]) == (512, 0)
    # A decode row, or a chunk too short to hold a block plus a tail: none.
    offsets, _ = prefill_checkpoints(
        np.array([4096, 512, 0]), np.array([4, 500, 512]), 512
    )
    assert offsets.tolist() == [0, 0, 0]


def _manager(num_speculative_blocks: int) -> KVCacheManager:
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=B, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    state = with_prefill_checkpoint(
        MambaSpec(
            block_size=B,
            shapes=((1, 64),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            page_size_padded=attention.page_size_bytes,
            num_speculative_blocks=num_speculative_blocks,
        )
    )
    assert state.num_prefill_checkpoint_blocks == 1
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["state"], state),
            KVCacheGroupSpec(["attention"], attention),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=B,
        hash_block_size=B,
        enable_caching=True,
    )


def _request(request_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id,
        tokens,
        SamplingParams(max_tokens=16),
        None,
        block_hasher=get_request_block_hasher(B, sha256),
    )


@pytest.mark.parametrize("num_speculative_blocks", [0, 3])
@pytest.mark.parametrize("prompt_len", [40, 48, 70, 33])
@pytest.mark.parametrize("hit_first", [False, True])
def test_manager_reserves_the_block_the_worker_writes(
    num_speculative_blocks, prompt_len, hit_first
):
    manager = _manager(num_speculative_blocks)
    state_manager = manager.coordinator.single_type_managers[0]
    assert isinstance(state_manager, MambaManager)
    prompt = list(range(prompt_len))
    if hit_first:
        # A previous request leaves the first block cached (prefix hit of B).
        warm = _request("warm", prompt[:B] + [9999])
        assert manager.allocate_slots(warm, B + 1) is not None
        warm.num_computed_tokens = B + 1
        manager.free(warm)
    request = _request("r", prompt)
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(request)[:2]
    assert hit == (B if hit_first else 0)
    worker_table = [b.block_id for b in blocks.blocks[0]]
    scheduled = prompt_len - hit
    new = manager.allocate_slots(request, scheduled, hit, blocks)
    assert new is not None
    worker_table += new.get_block_ids()[0]

    offset, column = _mirror(hit, scheduled)
    # Every case here holds a block plus a tail past its start.
    reserved = state_manager._checkpoints.get("r")
    assert reserved is not None and offset > 0
    position, index = reserved
    # The worker writes the reserved block, at the reserved position.
    assert (hit + offset, column) == (position, index)
    block = state_manager.req_to_blocks["r"][index]
    assert not block.is_null and worker_table[column] == block.block_id
    # ... and that block is what a later prefix hit at `position` resumes from.
    request.num_computed_tokens = prompt_len
    cached = manager.block_pool.get_cached_block(
        request.block_hashes[position // B - 1], [0]
    )
    assert cached is not None and cached[0] is block


def test_prefill_checkpoint_targets_reads_each_groups_table():
    ckpt = torch.tensor([[512, 0, 1024], [0, 3, 2]], dtype=torch.int32)
    table = torch.arange(1, 13, dtype=torch.int32).reshape(3, 4)
    offsets, blocks = prefill_checkpoint_targets(ckpt, table.reshape(-1), 3)
    assert offsets.tolist() == [512, 0, 1024]
    # Row 1 has no checkpoint: block 0 whatever its column holds.
    assert blocks.tolist() == [1, 0, 11]


def _split_request(prompt_len: int, computed: int = 0):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        shared_prefix_boundary=0,
    )


def _split_scheduler(checkpoint_blocks: bool):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        block_size=1024,
        use_eagle=True,
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=checkpoint_blocks,
        mamba_prefill_checkpoint_alignment=512,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=512,
        # Geometry already verified (see test_geometry_guard_*).
        _tpu_tail_fold_ok=checkpoint_blocks,
    )


def test_split_folds_the_tail_and_counts(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD", "1")
    Scheduler._mamba_block_aligned_split = inspect.unwrap(
        Scheduler._mamba_block_aligned_split
    )
    counter = mamba_tail_fold.SCHEDULER_COUNTER
    monkeypatch.setattr(counter, "hits", 0)
    monkeypatch.setattr(counter, "total", 0)
    upstream = Scheduler._mamba_block_aligned_split
    # Upstream without checkpoint blocks: the 488-token tail runs alone.
    assert upstream(_split_scheduler(False), _split_request(1000), 1000) == 512
    mamba_tail_fold.patch_mamba_tail_fold()
    split = Scheduler._mamba_block_aligned_split
    assert split(_split_scheduler(True), _split_request(1000), 1000) == 1000
    assert (counter.hits, counter.total) == (1, 1)
    # Continuation chunk starting at a block: folds too.
    assert split(_split_scheduler(True), _split_request(5000, 4096), 904) == 904
    # The budget ends the chunk before the prefill end: aligned, not counted.
    assert split(_split_scheduler(True), _split_request(9000), 8192) == 8192
    # The remainder starts mid-block: the tail split stays (and is counted).
    assert split(_split_scheduler(True), _split_request(1000, 100), 900) == 412
    assert (counter.hits, counter.total) == (2, 3)
    # Without the internal checkpoint the wrapper is transparent.
    assert split(_split_scheduler(False), _split_request(1000), 1000) == 512
    assert (counter.hits, counter.total) == (2, 3)


def _geometry_scheduler(hash_block: int, eagle_drop: bool = False):
    managers = []
    for _ in range(2):
        m = object.__new__(MambaManager)
        m.block_size = 512
        m.has_prefill_checkpoint_blocks = True
        m.drop_eagle_checkpoint_block = False
        m.block_pool = SimpleNamespace(hash_block_size=hash_block)
        m.cache_hit_alignment_tokens = hash_block
        managers.append(m)
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        block_size=1024,
        use_eagle_block_drop=eagle_drop,
        hash_block_size=hash_block,
        mamba_partial_cache_hit=hash_block < 1024,
        mamba_has_prefill_checkpoint_blocks=True,
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=managers)
        ),
    )
    return scheduler, managers


def test_geometry_guard_accepts_the_served_geometry():
    scheduler, managers = _geometry_scheduler(512)
    assert mamba_tail_fold._verify(scheduler)
    assert scheduler.mamba_has_prefill_checkpoint_blocks
    assert all(m.has_prefill_checkpoint_blocks for m in managers)


@pytest.mark.parametrize("hash_block,eagle_drop", [(1024, False), (512, True)])
def test_geometry_guard_turns_the_fold_off(hash_block, eagle_drop):
    scheduler, managers = _geometry_scheduler(hash_block, eagle_drop)
    assert not mamba_tail_fold._verify(scheduler)
    assert not scheduler.mamba_has_prefill_checkpoint_blocks
    assert not any(m.has_prefill_checkpoint_blocks for m in managers)
