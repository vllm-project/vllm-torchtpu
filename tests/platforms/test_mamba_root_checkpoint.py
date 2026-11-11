# SPDX-License-Identifier: Apache-2.0
"""Fork first turns resume at the parent's first-user-message boundary.

See `vllm_torchtpu.core.mamba_root_checkpoint`. The c32 run j-e3e88acd
shows the pattern: a flat chain's first turn re-sends the main
conversation's context as one user message, so its tokens match the main
chain's later prompts only up to the end of the main chain's first user
message. The main chain's first turn was not served in the window (aiperf
lanes start mid-trajectory) and its later turns publish checkpoints only at
their own prompt ends, all past that point, so the fork's hit is 0.

These tests drive a real scheduler-side `KVCacheManager` (Mamba + attention
groups, the served retention patch, MTP speculative blocks) with chunk
stops computed the way `Scheduler._mamba_block_aligned_split` computes
them, including the block-floored shared-prefix junction.
"""

import inspect

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import mamba_checkpoint_retention, mamba_root_checkpoint

B = 16
IM_START, IM_END, NL, USER, ASSIST = 100001, 100002, 100003, 100004, 100005
MARKER = f"{IM_START},{ASSIST},{IM_END}"
STATE = 0
_RETENTION_PATCHED = (
    (MambaManager, "cache_blocks"),
    (MambaManager, "_remove_blocks_in_range"),
    (MambaManager, "remove_skipped_blocks"),
    (MambaManager, "pop_blocks_for_free"),
    (KVCacheCoordinator, "free"),
    (KVCacheCoordinator, "pop_blocks_for_free"),
)


def _restore(monkeypatch):
    for owner, name in (
        *_RETENTION_PATCHED,
        (BlockPool, "get_cached_block"),
        (BlockPool, "_insert_block_hash"),
        (KVCacheManager, "__init__"),
        (KVCacheManager, "get_computed_blocks"),
        (Scheduler, "__init__"),
    ):
        monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    mamba_checkpoint_retention.patch_mamba_checkpoint_retention()


@pytest.fixture
def base(monkeypatch):
    _restore(monkeypatch)
    monkeypatch.setenv("VLLM_TPU_MAMBA_ROOT_CHECKPOINT", "0")
    mamba_root_checkpoint.patch_mamba_root_checkpoint()


@pytest.fixture
def fixed(monkeypatch):
    _restore(monkeypatch)
    monkeypatch.setenv("VLLM_TPU_MAMBA_ROOT_CHECKPOINT", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_ROOT_CHECKPOINT_MARKER", MARKER)
    mamba_root_checkpoint.patch_mamba_root_checkpoint()


def _manager() -> KVCacheManager:
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
        num_speculative_blocks=2,
    )
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
        max_model_len=2048,
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


def _split_end(request: Request, start: int, end: int) -> int:
    """The chunk end `Scheduler._mamba_block_aligned_split` picks (align
    mode, no eagle block drop, no internal checkpoint)."""
    prompt = request.num_prompt_tokens
    if end < prompt:
        end = end // B * B
    last_cache_position = request.num_tokens - request.num_tokens % B
    next_boundary = (start // B + 1) * B
    junction = request.shared_prefix_boundary
    block_floored = start + (junction - start) // B * B
    stops = (
        next_boundary if start % B else 0,
        last_cache_position,
        block_floored if start < junction < end else 0,
    )
    return min((s for s in stops if start < s < end), default=end)


def _run(manager: KVCacheManager, request: Request, num_decode: int = 2) -> int:
    manager.new_step_starts()
    blocks, hit, request.shared_prefix_boundary = manager.get_computed_blocks(
        request
    )
    prompt = request.num_tokens
    start, first = hit, True
    while start < prompt:
        end = _split_end(request, start, prompt)
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
    manager.free(request)
    return hit


def _msg(role: int, content: list[int], close: bool = True) -> list[int]:
    return [IM_START, role, NL, *content, *([IM_END, NL] if close else [])]


U0 = list(range(1, 57))  # first user message: 3 + 56 tokens, IM_END at 59
A0 = list(range(200, 220))
U1 = list(range(300, 320))
ROOT = 59


def _main_turn1() -> Request:
    tokens = (
        _msg(USER, U0) + _msg(ASSIST, A0) + _msg(USER, U1) + [IM_START, ASSIST, NL]
    )
    return _request("main-t1", tokens)


def _flat(request_id: str = "flat", tail: int = 50) -> Request:
    # The fork: the parent's first user message continued in the same
    # message, then its own assistant prompt.
    body = U0 + list(range(5000, 5000 + tail))
    return _request(request_id, _msg(USER, body) + [IM_START, ASSIST, NL])


def _cached_state(manager: KVCacheManager, request: Request, tokens: int):
    key = make_block_hash_with_group_id(request.block_hashes[tokens // B - 1], STATE)
    return manager.block_pool.cached_block_hash_to_block.get_one_block(key)


def test_root_boundary_marker():
    init_none_hash(sha256)
    marker = (IM_START, ASSIST, IM_END)
    assert mamba_root_checkpoint.root_boundary(_main_turn1().prompt_token_ids, marker) == ROOT
    # No assistant turn yet: nothing to checkpoint.
    assert mamba_root_checkpoint.root_boundary(_msg(USER, U0), marker) == 0
    assert mamba_root_checkpoint.root_boundary(_flat().prompt_token_ids, marker) > ROOT


def test_fork_first_turn_misses_on_base(base):
    """The j-e3e88acd pattern: the fork shares 59 tokens with the main turn
    (KV cached) but no state at or below them, so its hit is 0."""
    manager = _manager()
    main = _main_turn1()
    assert _run(manager, main) == 0
    flat = _flat()
    blocks, hit, junction = manager.get_computed_blocks(flat)
    # The attention KV of [0, 48) is cached; the Mamba state is not.
    assert hit == 0 and junction == ROOT // B * B
    assert _run(manager, _flat()) == 0


def test_fork_first_turn_hits_the_root_checkpoint(fixed):
    manager = _manager()
    main = _main_turn1()
    assert _run(manager, main) == 0
    assert _cached_state(manager, main, ROOT // B * B) is not None
    assert _run(manager, _flat()) == ROOT // B * B
    # The main turn's root, and the fork's own (end of its single message).
    assert manager._tpu_root_checkpoint.stats["junctions"] == 2


def test_root_checkpoint_is_kept_warm_by_later_turns(fixed):
    manager = _manager()
    main = _main_turn1()
    _run(manager, main)
    root_state = _cached_state(manager, main, ROOT // B * B)
    other = _request("other", list(range(7000, 7100)))
    _run(manager, other)
    queue = manager.block_pool.free_block_queue
    ids = [b.block_id for b in queue.get_all_free_blocks()]
    other_ids = {b.block_id for b in manager.block_pool.blocks if b.block_hash is not None}
    assert ids.index(root_state.block_id) < max(ids.index(i) for i in other_ids if i in ids)
    # The main chain's next turn passes the root: the checkpoint moves to the
    # most-recently-used end, behind the other session's blocks.
    turn2 = _request(
        "main-t2", list(main.all_token_ids) + [IM_END, NL] + _msg(USER, [9, 9, 9])
        + [IM_START, ASSIST, NL]
    )
    manager.new_step_starts()
    _, hit, junction = manager.get_computed_blocks(turn2)
    assert hit > ROOT and junction == 0
    ids = [b.block_id for b in queue.get_all_free_blocks()]
    assert ids[-1] == root_state.block_id
    assert manager._tpu_root_checkpoint.stats["refreshes"] == 1


def test_detected_junction_wins(fixed):
    """An upstream junction (attention hit past the Mamba hit) is kept."""
    manager = _manager()
    _run(manager, _flat("flat-a", tail=60))
    # A second fork sharing 99 tokens with the first: the upstream junction
    # (96, the attention hit past the Mamba hit 0) is kept.
    _, hit, junction = manager.get_computed_blocks(_flat("flat-b", tail=40))
    assert (hit, junction) == (0, 96)
    assert manager._tpu_root_checkpoint.stats["junctions"] == 1  # flat-a only


def test_prompts_without_assistant_turn_unchanged(fixed):
    manager = _manager()
    request = _request("single", _msg(USER, U0))
    _, hit, junction = manager.get_computed_blocks(request)
    assert (hit, junction) == (0, 0)


def test_flag_off_installs_nothing(base):
    assert not getattr(
        KVCacheManager.get_computed_blocks, mamba_root_checkpoint._PATCH_FLAG, False
    )
    assert not hasattr(_manager(), "_tpu_root_checkpoint")


def test_served_split_stops_at_the_root_without_the_fold():
    """c32 geometry with the tail fold off (main-KV DCP2): the upstream split
    ends a chunk at the block-floored root, so the state there is written by
    the plain (non-fold) path and cached as a reachable boundary."""
    from types import SimpleNamespace

    split = inspect.unwrap(Scheduler._mamba_block_aligned_split)
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        block_size=1024,
        use_eagle=True,
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=512,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=256,
    )
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=50_000,
        num_tokens=50_000,
        shared_prefix_boundary=0,
    )
    assert split(scheduler, request, 8192) == 8192
    request.shared_prefix_boundary = 4_100  # root 4100 -> stop at 4096
    assert split(scheduler, request, 8192) == 4096
    request.num_computed_tokens = 4096
    assert split(scheduler, request, 8192) == 8192


def test_marker_from_tokenizer(monkeypatch):
    from types import SimpleNamespace

    import vllm.tokenizers.registry as registry

    vocab = {"<|im_start|>": 151644, "<|im_end|>": 151645}
    fake = SimpleNamespace(
        convert_tokens_to_ids=lambda t: vocab.get(t, 0),
        encode=lambda text, add_special_tokens=False: [77091],
        unk_token_id=None,
    )
    monkeypatch.setattr(registry, "cached_tokenizer_from_config", lambda cfg: fake)
    assert mamba_root_checkpoint._marker_from_tokenizer(None) == (151644, 77091, 151645)
    fake.encode = lambda text, add_special_tokens=False: [1, 2]
    assert mamba_root_checkpoint._marker_from_tokenizer(None) is None
    assert mamba_root_checkpoint._parse_marker("1, 2,3") == (1, 2, 3)
    assert mamba_root_checkpoint._parse_marker("1,2") is None
