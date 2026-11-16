# SPDX-License-Identifier: Apache-2.0
"""Mamba "align" checkpoints outlive their KV in the prefix cache.

See `vllm_torchtpu.core.mamba_checkpoint_retention`. A hybrid hit needs the
attention KV of the prefix and the state block at the hit boundary; these
tests drive the scheduler-side KV cache manager through prefill and decode
the way the scheduler does and check both halves stay usable together.
"""

import inspect

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import mamba_checkpoint_retention

B = 16
PROMPT = list(range(40))
_PATCHED = (
    (MambaManager, "cache_blocks"),
    (MambaManager, "_remove_blocks_in_range"),
    (MambaManager, "remove_skipped_blocks"),
    (MambaManager, "pop_blocks_for_free"),
    (KVCacheCoordinator, "free"),
    (KVCacheCoordinator, "pop_blocks_for_free"),
)


def _unpatch(monkeypatch):
    # Start from upstream. The same-step hit patch, if another test in this
    # process installed it, would hide every hit since no `schedule()` runs.
    for owner, name in (
        *_PATCHED,
        (BlockPool, "get_cached_block"),
        (BlockPool, "_insert_block_hash"),
    ):
        monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))


@pytest.fixture
def patched(monkeypatch):
    _unpatch(monkeypatch)
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "1")
    mamba_checkpoint_retention.patch_mamba_checkpoint_retention()


def _manager(num_speculative_blocks: int = 0) -> KVCacheManager:
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=B, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    state = MambaSpec(
        block_size=B,
        shapes=((1, 64),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        page_size_padded=attention.page_size_bytes,
        num_speculative_blocks=num_speculative_blocks,
    )
    config = KVCacheConfig(
        num_blocks=96,
        kv_cache_tensors=[],
        # State group first, as in Qwen4Exp (GDN groups precede QSA).
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
        SamplingParams(max_tokens=200),
        None,
        block_hasher=get_request_block_hasher(B, sha256),
    )


def _run(manager: KVCacheManager, request: Request, num_decode: int) -> int:
    """Admit, prefill in block-aligned chunks, then decode one token a step."""
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(request)[:2]
    prompt = request.num_tokens
    start, first = hit, True
    while start < prompt:
        # The align split stops at the last block boundary below the end.
        end = prompt if prompt // B * B <= start else prompt // B * B
        manager.new_step_starts()
        assert (
            manager.allocate_slots(
                request,
                end - start,
                hit if first else 0,
                blocks if first else None,
            )
            is not None
        )
        first = False
        request.num_computed_tokens = start = end
    request.append_output_token_ids(10_000)
    for i in range(num_decode):
        manager.new_step_starts()
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
        request.append_output_token_ids(10_001 + i)
    return hit


def _state(manager: KVCacheManager) -> MambaManager:
    state = manager.coordinator.single_type_managers[0]
    assert isinstance(state, MambaManager)
    return state


def _cached_state_block(manager: KVCacheManager, request: Request, tokens: int):
    """The state block cached at token boundary `tokens`, if any."""
    blocks = manager.block_pool.get_cached_block(
        request.block_hashes[tokens // B - 1], [0]
    )
    return None if blocks is None else blocks[0]


def _free_queue_ids(manager: KVCacheManager) -> list[int]:
    return [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ]


def test_prompt_checkpoint_stays_pinned_through_decode(patched):
    manager = _manager()
    request = _request("r", PROMPT)
    _run(manager, request, num_decode=5)

    # Replay boundary of a 40-token prompt: (40 - 1) // 16 * 16 = 32.
    checkpoint = _cached_state_block(manager, request, 32)
    assert checkpoint is not None
    # Upstream frees it at decode start (ref_cnt 0, evictable).
    assert checkpoint.ref_cnt == 1
    assert checkpoint.block_id not in _free_queue_ids(manager)


def test_checkpoint_is_evicted_after_its_kv(patched):
    manager = _manager()
    request = _request("r", PROMPT)
    _run(manager, request, num_decode=5)
    checkpoint = _cached_state_block(manager, request, 32)
    kv_ids = [
        b.block_id
        for b in manager.coordinator.single_type_managers[1].req_to_blocks["r"]
        if b.block_hash is not None
    ]

    manager.free(request)

    queue = _free_queue_ids(manager)
    assert checkpoint.block_hash is not None
    assert all(queue.index(checkpoint.block_id) > queue.index(k) for k in kv_ids)
    # Tail first: the prompt's last cached KV block goes before its first.
    assert queue.index(kv_ids[-1]) < queue.index(kv_ids[0])


def test_pop_for_free_matches_free_order(patched):
    freed, popped = _manager(), _manager()
    for manager in (freed, popped):
        _run(manager, _request("r", PROMPT), num_decode=5)
    freed.free(_request("r", PROMPT))
    popped.block_pool.free_blocks(reversed(popped.coordinator.pop_blocks_for_free("r")))
    assert _free_queue_ids(freed) == _free_queue_ids(popped)


def test_next_turn_resumes_past_previous_output(patched):
    manager = _manager()
    first = _request("turn1", PROMPT)
    # 40 prompt + 31 output tokens; 70 computed: decode crossed 48 and 64.
    _run(manager, first, num_decode=30)
    assert _cached_state_block(manager, first, 64) is not None
    # Only the newest decode checkpoint is kept; the one at 48 is retired.
    assert _cached_state_block(manager, first, 48) is None
    manager.free(first)

    second = _request("turn2", list(first.all_token_ids) + [7, 8, 9])
    # Upstream resumes at the previous prompt end (32) and recomputes the
    # whole previous output.
    assert _run(manager, second, num_decode=1) == 64


def test_retry_of_previous_prompt_still_hits_prompt_checkpoint(patched):
    manager = _manager()
    first = _request("turn1", PROMPT)
    _run(manager, first, num_decode=30)
    manager.free(first)

    retry = _request("retry", PROMPT + [5])
    assert _run(manager, retry, num_decode=1) == 32


def test_no_decode_checkpoint_with_speculative_blocks(patched):
    """With MTP the last write into a column is not the column's last token."""
    manager = _manager(num_speculative_blocks=2)
    first = _request("turn1", PROMPT)
    _run(manager, first, num_decode=30)
    assert _cached_state_block(manager, first, 64) is None
    checkpoint = _cached_state_block(manager, first, 32)
    assert checkpoint is not None and checkpoint.ref_cnt == 1
    manager.free(first)

    second = _request("turn2", list(first.all_token_ids) + [7, 8, 9])
    assert _run(manager, second, num_decode=1) == 32


def test_decode_checkpoint_knob(monkeypatch):
    _unpatch(monkeypatch)
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "0")
    mamba_checkpoint_retention.patch_mamba_checkpoint_retention()
    manager = _manager()
    first = _request("turn1", PROMPT)
    _run(manager, first, num_decode=30)
    assert _cached_state_block(manager, first, 64) is None
    assert _cached_state_block(manager, first, 32).ref_cnt == 1


def test_patch_is_idempotent(patched):
    methods = [getattr(owner, name) for owner, name in _PATCHED]
    mamba_checkpoint_retention.patch_mamba_checkpoint_retention()
    assert [getattr(owner, name) for owner, name in _PATCHED] == methods


# -- VLLM_TPU_MAMBA_PIN_HIT_CHECKPOINT: the checkpoint a request resumed from ----


@pytest.fixture
def pinned_hit(monkeypatch):
    _unpatch(monkeypatch)
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_PIN_HIT_CHECKPOINT", "1")
    mamba_checkpoint_retention.patch_mamba_checkpoint_retention()
    yield
    # The flag is read at install time; leave the module as the default sets it.
    monkeypatch.setattr(mamba_checkpoint_retention, "_pin_hit_checkpoint", False)


# A producer turn: prefill only. The next turn ends in the same 16-token block,
# so its replay boundary (42 // 16 * 16 = 32) is the one it resumes from and it
# publishes no checkpoint of its own.
NEXT_IN_BLOCK = PROMPT + [1, 2, 3]


def _finish(manager: KVCacheManager, request: Request) -> None:
    """Producer finish: `Scheduler._connector_finished` drops the skipped
    blocks before the connector takes the table, then the blocks are freed."""
    manager.remove_skipped_blocks(
        request.request_id, request.num_computed_tokens, request.num_prompt_tokens
    )
    manager.free(request)


def _previous_turn(manager: KVCacheManager):
    first = _request("turn1", PROMPT)
    _run(manager, first, num_decode=0)
    _finish(manager, first)
    checkpoint = _cached_state_block(manager, first, 32)
    assert checkpoint is not None and checkpoint.ref_cnt == 0
    return checkpoint


def _attention_ids(manager: KVCacheManager, request_id: str) -> list[int]:
    blocks = manager.coordinator.single_type_managers[1].req_to_blocks[request_id]
    return [b.block_id for b in blocks if b.block_hash is not None]


def _resumed_order(manager: KVCacheManager):
    checkpoint = _previous_turn(manager)
    turn = _request("turn2", NEXT_IN_BLOCK)
    assert _run(manager, turn, num_decode=0) == 32
    kv = _attention_ids(manager, "turn2")
    manager.remove_skipped_blocks("turn2", turn.num_computed_tokens, turn.num_prompt_tokens)
    held_after_finish = checkpoint.block_id not in _free_queue_ids(manager)
    manager.free(turn)
    queue = _free_queue_ids(manager)
    return checkpoint, kv, queue, held_after_finish


def test_resumed_checkpoint_leaves_after_the_turns_kv(pinned_hit):
    checkpoint, kv, queue, held = _resumed_order(_manager())
    assert held
    assert all(queue.index(checkpoint.block_id) > queue.index(k) for k in kv)
    assert checkpoint.ref_cnt == 0 and checkpoint.block_hash is not None


def test_flag_off_keeps_the_resumed_checkpoint_ahead_of_the_kv(patched):
    """Without the flag the order is the patch's: the resumed checkpoint is
    freed at finish, before the KV, which waits for the free."""
    checkpoint, kv, queue, held = _resumed_order(_manager())
    assert not held
    assert all(queue.index(checkpoint.block_id) < queue.index(k) for k in kv)


def test_flag_changes_nothing_for_a_request_without_a_resumed_checkpoint(monkeypatch):
    queues = []
    for flag in ("0", "1"):
        _unpatch(monkeypatch)
        monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
        monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", "1")
        monkeypatch.setenv("VLLM_TPU_MAMBA_PIN_HIT_CHECKPOINT", flag)
        mamba_checkpoint_retention.patch_mamba_checkpoint_retention()
        manager = _manager()
        request = _request("r", PROMPT)
        _run(manager, request, num_decode=30)
        _finish(manager, request)
        queues.append(_free_queue_ids(manager))
    monkeypatch.setattr(mamba_checkpoint_retention, "_pin_hit_checkpoint", False)
    assert queues[0] == queues[1]


def test_resumed_checkpoint_reference_after_abort_while_waiting(pinned_hit):
    manager = _manager()
    checkpoint = _previous_turn(manager)
    turn = _request("turn2", NEXT_IN_BLOCK)
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(turn)[:2]
    assert hit == 32
    assert manager.allocate_slots(turn, turn.num_tokens - hit, hit, blocks) is not None
    assert checkpoint.ref_cnt == 1
    manager.free(turn)  # aborted before its first forward
    assert checkpoint.ref_cnt == 0 and checkpoint.block_id in _free_queue_ids(manager)


def _first_chunk_then_pass(manager: KVCacheManager, turn: Request):
    """Two chunks (32-64, 64-70): the second allocation moves the state column
    past the resumed checkpoint, which then leaves the table."""
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(turn)[:2]
    assert hit == 32
    assert manager.allocate_slots(turn, 64 - hit, hit, blocks) is not None
    turn.num_computed_tokens = 64
    manager.new_step_starts()
    assert manager.allocate_slots(turn, turn.num_prompt_tokens - 64) is not None


def test_resumed_checkpoint_reference_after_abort_mid_prefill(pinned_hit):
    manager = _manager()
    checkpoint = _previous_turn(manager)
    turn = _request("turn2", PROMPT + list(range(500, 530)))
    _first_chunk_then_pass(manager, turn)
    state = _state(manager)
    assert checkpoint in mamba_checkpoint_retention._pinned(state)["turn2"]
    assert checkpoint.ref_cnt == 1 and checkpoint.block_id not in _free_queue_ids(manager)
    manager.free(turn)
    assert checkpoint.ref_cnt == 0 and checkpoint.block_id in _free_queue_ids(manager)
    assert "turn2" not in mamba_checkpoint_retention._pinned(state)


def test_resumed_checkpoint_reference_after_preemption(pinned_hit):
    manager = _manager()
    checkpoint = _previous_turn(manager)
    turn = _request("turn2", PROMPT + list(range(500, 530)))
    _first_chunk_then_pass(manager, turn)
    manager.free(turn)  # preempted: the scheduler frees and recomputes
    turn.num_computed_tokens = 0
    assert checkpoint.ref_cnt == 0
    # The preempted first chunk published the state at its replay boundary.
    assert _run(manager, turn, num_decode=0) == 64
    _finish(manager, turn)
    assert checkpoint.ref_cnt == 0 and checkpoint.block_id in _free_queue_ids(manager)
    assert "turn2" not in mamba_checkpoint_retention._pinned(_state(manager))


def test_two_requests_resuming_from_one_checkpoint_free_it_once(pinned_hit):
    manager = _manager()
    checkpoint = _previous_turn(manager)
    a, b = _request("a", PROMPT + [1, 2, 3]), _request("b", PROMPT + [4, 5, 6])
    assert _run(manager, a, num_decode=0) == 32
    assert _run(manager, b, num_decode=0) == 32
    assert checkpoint.ref_cnt == 2
    _finish(manager, a)
    assert checkpoint.ref_cnt == 1 and checkpoint.block_id not in _free_queue_ids(manager)
    _finish(manager, b)
    assert checkpoint.ref_cnt == 0
    assert _free_queue_ids(manager).count(checkpoint.block_id) == 1


def test_pinned_checkpoints_leave_older_boundary_first(pinned_hit):
    """A turn that publishes its own checkpoint (64) frees it after the one it
    resumed from (32), whatever order they left the table in, and both after
    its KV."""
    manager = _manager()
    resumed = _previous_turn(manager)
    turn = _request("turn2", PROMPT + list(range(500, 530)))
    assert _run(manager, turn, num_decode=0) == 32
    own = _cached_state_block(manager, turn, 64)
    kv = _attention_ids(manager, "turn2")
    manager.remove_skipped_blocks("turn2", turn.num_computed_tokens, turn.num_prompt_tokens)
    pinned = mamba_checkpoint_retention._pinned(_state(manager))["turn2"]
    assert pinned == [resumed, own]
    pinned.reverse()  # one range removal releases its blocks highest first
    manager.free(turn)
    queue = _free_queue_ids(manager)
    assert queue.index(resumed.block_id) < queue.index(own.block_id)
    assert all(queue.index(resumed.block_id) > queue.index(k) for k in kv)
