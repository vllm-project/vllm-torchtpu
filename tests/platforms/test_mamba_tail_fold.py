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


# --- VLLM_TPU_MAMBA_TAIL_FOLD_UNIT: checkpoint and hit at a finer unit -------

U, MB = 256, 512


def test_fold_unit_defaults_and_validation(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", raising=False)
    assert mamba_tail_fold.fold_unit(MB) == MB
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", "512")
    assert mamba_tail_fold.fold_unit(MB) == MB
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", "256")
    monkeypatch.setenv("TPU_GDN_CHUNK_PAIR", "1")
    assert mamba_tail_fold.fold_unit(MB) == U
    # 128 is not a whole chunk-pair tile (256 rows).
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", "128")
    with pytest.raises(ValueError, match="GDN prefill tile"):
        mamba_tail_fold.fold_unit(MB)
    monkeypatch.setenv("TPU_GDN_CHUNK_PAIR", "0")
    assert mamba_tail_fold.fold_unit(MB) == 128
    # Must divide the Mamba block.
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", "384")
    with pytest.raises(ValueError, match="divide"):
        mamba_tail_fold.fold_unit(MB)


def test_worker_mirror_matches_vllm_rule_fine_unit():
    b, u = 16, 8
    for start in range(0, 8 * b, 4):
        for scheduled in range(0, 8 * b):
            end = start + scheduled
            position = get_mamba_prefill_checkpoint_position(end, u, False)
            valid = scheduled > 0 and is_mamba_prefill_checkpoint_valid(
                query_start=start,
                query_end=end,
                checkpoint_position=position,
                hash_block_size=u,
                mamba_block_size=b,
                checkpoint_alignment=u,
            )
            want = (position - start, cdiv(end, b) - 2) if valid else (0, 0)
            got = prefill_checkpoints(np.array([start]), np.array([scheduled]), b, u)
            assert (int(got[0][0]), int(got[1][0])) == want, (start, scheduled)


def test_worker_mirror_fine_unit_typical_tail():
    # 1000-token prompt: the checkpoint moves from 512 to 768 (unit 256),
    # still into column 0 (the running state sits in column 1).
    offsets, columns = prefill_checkpoints(np.array([0]), np.array([1000]), MB, U)
    assert (offsets[0], columns[0]) == (768, 0)
    # A turn resuming at a 256-aligned hit (768) of a 6000-token prompt.
    offsets, columns = prefill_checkpoints(np.array([768]), np.array([5232]), MB, U)
    assert (offsets[0] + 768, columns[0]) == (5888, 10)


def _fine_manager(monkeypatch) -> KVCacheManager:
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", str(U))
    monkeypatch.setenv("TPU_GDN_CHUNK_PAIR", "1")
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=MB, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    state = with_prefill_checkpoint(
        MambaSpec(
            block_size=MB,
            shapes=((1, 64),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            page_size_padded=attention.page_size_bytes,
        )
    )
    assert state.prefill_checkpoint_alignment == U
    config = KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["state"], state),
            KVCacheGroupSpec(["attention"], attention),
        ],
        prefix_cache_retention_interval=0,
    )
    manager = KVCacheManager(
        config,
        max_model_len=16384,
        scheduler_block_size=MB,
        hash_block_size=U,
        enable_caching=True,
    )
    assert manager.coordinator.enable_partial_hash_hits
    return manager


def _fine_request(request_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id,
        tokens,
        SamplingParams(max_tokens=16),
        None,
        block_hasher=get_request_block_hasher(U, sha256),
    )


@pytest.mark.parametrize("prompt_len", [1000, 1300, 2047, 3000])
def test_fine_unit_next_turn_hits_the_last_unit_of_the_prior_prompt(
    monkeypatch, prompt_len
):
    """Turn 1 folds its tail and checkpoints at the last 256 boundary; turn 2
    (turn 1's prompt + new tokens) then hits that boundary, not the last 512.
    The block the manager hashes there is the one the worker writes."""
    manager = _fine_manager(monkeypatch)
    state_manager = manager.coordinator.single_type_managers[0]
    prompt = list(range(prompt_len))
    first = _fine_request("t1", prompt)
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(first)[:2]
    assert hit == 0
    new = manager.allocate_slots(first, prompt_len, 0, blocks)
    assert new is not None
    worker_table = new.get_block_ids()[0]
    offset, column = prefill_checkpoints(np.array([0]), np.array([prompt_len]), MB, U)
    position, index = state_manager._checkpoints["t1"]
    assert (int(offset[0]), int(column[0])) == (position, index)
    assert position == (prompt_len - 1) // U * U
    assert worker_table[index] == state_manager.req_to_blocks["t1"][index].block_id
    first.num_computed_tokens = prompt_len
    manager.free(first)

    second = _fine_request("t2", prompt + list(range(50000, 50000 + 777)))
    manager.new_step_starts()
    _, hit = manager.get_computed_blocks(second)[:2]
    assert hit == position
    assert prompt_len - hit < U


def test_geometry_guard_fine_unit(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", str(U))
    monkeypatch.setenv("TPU_GDN_CHUNK_PAIR", "1")
    scheduler, managers = _geometry_scheduler(U)
    assert mamba_tail_fold._verify(scheduler)
    # The unit is set but prompts still hash at 512: the worker would write
    # positions the manager never reserves, so the fold must turn off.
    scheduler, managers = _geometry_scheduler(512)
    assert not mamba_tail_fold._verify(scheduler)
    assert not any(m.has_prefill_checkpoint_blocks for m in managers)


def test_config_sets_prefix_match_unit(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.config import _apply_tail_fold_unit

    def cache(unit=None):
        return SimpleNamespace(
            enable_prefix_caching=True, mamba_cache_mode="align", prefix_match_unit=unit
        )

    monkeypatch.setenv("TPU_GDN_CHUNK_PAIR", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", str(U))
    monkeypatch.delenv("VLLM_TPU_MAMBA_TAIL_FOLD", raising=False)
    c = cache()
    _apply_tail_fold_unit(c, MB)
    assert c.prefix_match_unit is None  # fold off: untouched
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD", "1")
    _apply_tail_fold_unit(c, MB)
    assert c.prefix_match_unit == U
    c = cache(64)
    _apply_tail_fold_unit(c, MB)
    assert c.prefix_match_unit == 64  # explicit value kept (guard decides)
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", "0")
    c = cache()
    _apply_tail_fold_unit(c, MB)
    assert c.prefix_match_unit is None


def test_fine_unit_turn_resuming_mid_block_folds_and_chains(monkeypatch):
    """Turn 2 resumes at an odd-256 hit (768) and folds a long chunk; its
    checkpoint is the block the worker writes, and turn 3 hits it."""
    manager = _fine_manager(monkeypatch)
    state_manager = manager.coordinator.single_type_managers[0]
    turn1 = list(range(1000))
    first = _fine_request("t1", turn1)
    manager.new_step_starts()
    blocks, _ = manager.get_computed_blocks(first)[:2]
    assert manager.allocate_slots(first, 1000, 0, blocks) is not None
    first.num_computed_tokens = 1000
    manager.free(first)

    turn2 = turn1 + list(range(60000, 65000))
    second = _fine_request("t2", turn2)
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(second)[:2]
    assert hit == 768
    scheduled = len(turn2) - hit
    assert manager.allocate_slots(second, scheduled, hit, blocks) is not None
    offset, column = prefill_checkpoints(np.array([hit]), np.array([scheduled]), MB, U)
    position, index = state_manager._checkpoints["t2"]
    assert (hit + int(offset[0]), int(column[0])) == (position, index)
    assert position == (len(turn2) - 1) // U * U
    # A sub-block Mamba hit copies the hit state into a fresh block (CoW), so
    # the request's table (what a new request's block ids carry) is the
    # manager's, not the computed blocks plus the new ones.
    assert not state_manager.req_to_blocks["t2"][index].is_null
    second.num_computed_tokens = len(turn2)
    manager.free(second)

    third = _fine_request("t3", turn2 + [7] * 300)
    manager.new_step_starts()
    _, hit = manager.get_computed_blocks(third)[:2]
    assert hit == position


# --- VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY: a last chunk with no block boundary --


@pytest.fixture
def _restore_manager():
    names = (
        "_needs_internal_checkpoint",
        "get_num_blocks_to_allocate",
        "cache_blocks",
        "new_step_starts",
    )
    saved = {name: getattr(MambaManager, name) for name in names}
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(MambaManager, name, fn)


def _nobound_env(monkeypatch, on: bool):
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", str(U))
    monkeypatch.setenv("TPU_GDN_CHUNK_PAIR", "1")
    if on:
        monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY", raising=False)


def test_nobound_mirror(monkeypatch):
    """Cached 1024 (a block boundary), prompt 1497: no 512 boundary inside
    (1024, 1497), so upstream has no column and splits 256 + 217. The fold
    puts the checkpoint at 1280 in column cdiv(1497, 512) = 3, the first
    speculative scratch block."""

    def mirror(start, scheduled, nsb):
        o, c = prefill_checkpoints(
            np.array([start]), np.array([scheduled]), MB, U, num_speculative_blocks=nsb
        )
        return int(o[0]), int(c[0])

    _nobound_env(monkeypatch, on=False)
    assert mirror(1024, 473, 3) == (0, 0)
    _nobound_env(monkeypatch, on=True)
    assert mirror(1024, 473, 3) == (256, 3)
    assert mirror(1024, 473, 0) == (0, 0)  # no scratch block: off
    assert mirror(0, 473, 3) == (256, 1)  # a fresh short prompt
    assert mirror(1024, 200, 3) == (0, 0)  # no unit inside
    assert mirror(1280, 217, 3) == (0, 0)  # starts at the checkpoint
    # A chunk with a boundary keeps upstream's column (unchanged).
    assert mirror(1024, 1000, 3) == mirror(1024, 1000, 0)
    assert mirror(1024, 1000, 3)[1] == cdiv(2024, MB) - 2
    # Exhaustive: the no-boundary case and upstream's cases never overlap,
    # and the new case is exactly nobound_valid at column cdiv(end, B).
    for start in range(0, 3 * MB, U // 2):
        for n in range(1, 2 * MB, 37):
            end = start + n
            position = max((end - 1) // U * U, 0)
            up = (
                is_mamba_prefill_checkpoint_valid(
                    query_start=start,
                    query_end=end,
                    checkpoint_position=position,
                    hash_block_size=U,
                    mamba_block_size=MB,
                    checkpoint_alignment=U,
                )
                and cdiv(end, MB) - 2 >= 0
            )
            nb = mamba_tail_fold.nobound_valid(start, end, position, U, MB, U)
            assert not (up and nb)
            got = mirror(start, n, 3)
            if nb:
                assert got == (position - start, cdiv(end, MB))
            elif up:
                assert got == (position - start, cdiv(end, MB) - 2)
            else:
                assert got == (0, 0)


def _nobound_scheduler(spec_blocks: int):
    scheduler = _split_scheduler(True)
    scheduler.hash_block_size = U
    scheduler.mamba_prefill_checkpoint_alignment = U
    scheduler._tpu_tail_fold_spec_blocks = spec_blocks
    return scheduler


def test_nobound_split_decision(monkeypatch):
    _nobound_env(monkeypatch, on=False)
    mamba_tail_fold.patch_mamba_tail_fold()
    split = Scheduler._mamba_block_aligned_split
    # Upstream (and the flag off): split at the 256 checkpoint.
    assert split(_nobound_scheduler(3), _split_request(1497, 1024), 473) == 256
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY", "1")
    assert split(_nobound_scheduler(3), _split_request(1497, 1024), 473) == 473
    assert split(_nobound_scheduler(3), _split_request(473), 473) == 473
    # No speculative scratch block: keep the split.
    assert split(_nobound_scheduler(0), _split_request(1497, 1024), 473) == 256
    # The budget stops the chunk before the prompt end: unchanged.
    monkeypatch.delenv("VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY")
    cut = split(_nobound_scheduler(3), _split_request(1497, 1024), 300)
    monkeypatch.setenv("VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY", "1")
    assert split(_nobound_scheduler(3), _split_request(1497, 1024), 300) == cut
    # A shared-prefix junction inside the chunk keeps its stop.
    junction = _split_request(1497, 1024)
    junction.shared_prefix_boundary = 1100
    sched = _nobound_scheduler(3)
    sched.mamba_shared_prefix_checkpoint = True
    assert split(sched, junction, 473) == 76
    # The rest after the 256 piece (upstream's second step) is unchanged.
    assert split(_nobound_scheduler(3), _split_request(1497, 1280), 217) == 217


def _nobound_manager(monkeypatch) -> KVCacheManager:
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=MB, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    state = with_prefill_checkpoint(
        MambaSpec(
            block_size=MB,
            shapes=((1, 64),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            page_size_padded=attention.page_size_bytes,
            num_speculative_blocks=3,
        )
    )
    config = KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["state"], state),
            KVCacheGroupSpec(["attention"], attention),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=16384,
        scheduler_block_size=MB,
        hash_block_size=U,
        enable_caching=True,
    )


def _turn(manager, rid, tokens, chunks=None):
    """Schedule `tokens` from its prefix hit to the end, in `chunks` (default
    one chunk), as the scheduler does; returns (hit, worker table, request)."""
    request = _fine_request(rid, tokens)
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(request)[:2]
    table = [b.block_id for b in blocks.blocks[0]]
    rest = len(tokens) - hit
    chunks = chunks or [rest]
    assert sum(chunks) == rest
    computed = hit
    for i, n in enumerate(chunks):
        if i:
            manager.new_step_starts()
        new = (
            manager.allocate_slots(request, n, hit, blocks)
            if i == 0
            else manager.allocate_slots(request, n)
        )
        assert new is not None
        table += new.get_block_ids()[0]
        computed += n
        request.num_computed_tokens = computed
    return hit, table, request


@pytest.mark.parametrize("extra", [473, 300, 511])
def test_nobound_manager_reserves_moves_and_next_turn_hits(
    monkeypatch, _restore_manager, extra
):
    """Turn 2 hits turn 1's 1024 and folds its `extra`-token rest whole: the
    manager reserves column 3 (the first scratch block) at the last 256, the
    worker writes that block, the next step moves the hash to a fresh block
    filled by a queued copy of it, and turn 3 hits there. Flag off, turn 2
    splits 256 + rest and turn 3 hits the same position."""
    # Turn 1 checkpoints at 1024 (its last 256 boundary); turn 2 diverges there.
    turn1 = list(range(1100))
    turn2 = turn1[:1024] + list(range(50000, 50000 + extra))
    turn3 = turn2 + [7] * 300
    position = (len(turn2) - 1) // U * U

    _nobound_env(monkeypatch, on=False)
    base = _nobound_manager(monkeypatch)
    _, _, r1 = _turn(base, "t1", turn1)
    base.free(r1)
    hit, _, r2 = _turn(base, "t2", turn2, [position - 1024, len(turn2) - position])
    assert hit == 1024
    base.free(r2)
    base_hit = _turn(base, "t3", turn3)[0]
    assert base_hit == position

    _nobound_env(monkeypatch, on=True)
    mamba_tail_fold._patch_nobound_manager()
    manager = _nobound_manager(monkeypatch)
    state = manager.coordinator.single_type_managers[0]
    _, _, r1 = _turn(manager, "t1", turn1)
    manager.free(r1)
    hit, table, r2 = _turn(manager, "t2", turn2)
    assert hit == 1024
    offsets, columns = prefill_checkpoints(
        np.array([hit]), np.array([extra]), MB, U, num_speculative_blocks=3
    )
    column = cdiv(len(turn2), MB)
    assert (int(offsets[0]), int(columns[0])) == (position - hit, column)
    assert state._checkpoints["t2"] == (position, column)
    written = state.req_to_blocks["t2"][column]
    assert table[column] == written.block_id
    assert written.block_hash_num_tokens == position
    # Positional columns hold the hit state (1) and the running state (2).
    assert len(state.req_to_blocks["t2"]) == column + 3

    # Next step: before any allocation, the hash leaves the scratch block.
    manager.new_step_starts()
    copies, retained = manager.take_kv_cache_block_copies()
    assert written.block_hash is None
    assert len(copies) == 1 and copies[0].src_block_id == written.block_id
    cow = next(b for b in retained if b.block_id == copies[0].dst_block_id)
    assert cow.block_hash_num_tokens == position
    assert "t2" not in state._checkpoints
    for block in retained:  # the scheduler's release after the copy ran
        manager.block_pool.free_blocks([block])
    manager.free(r2)

    probe = _fine_request("probe", turn3)
    cached = manager.block_pool.get_cached_block(
        probe.block_hashes[position // U - 1], [0]
    )
    assert cached is not None and cached[0] is cow
    hit3 = _turn(manager, "t3", turn3)[0]
    assert hit3 == base_hit == position
