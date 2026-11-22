# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""A prefix hit at the last block boundary of its writer's prompt adopts the
writer's sliding-window pages and skips the replay; every other hit replays as
before."""

import random
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import block_hash_bytes, packed_prompt_ids
from vllm_torchtpu.core import prefix_swa_tail as tail

BLOCK = 256
PAGE = 64
WINDOW = 128
CHUNK = 2044
DRAFTS = 5
PATCHED = {
    KVCacheManager: ("__init__", "get_computed_blocks"),
    SlidingWindowManager: ("cache_blocks", "free"),
    Scheduler: ("_mark_prefix_replay",),
}


@pytest.fixture(autouse=True)
def none_hash(monkeypatch):
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", None, raising=False)
    kv_cache_utils.init_none_hash(sha256)


@pytest.fixture
def switch():
    """switch(True) installs the patch, switch(False) restores vLLM as served."""
    # SlidingWindowManager inherits cache_blocks and free: restore by removal.
    originals = {
        (c, n): c.__dict__.get(n) for c, names in PATCHED.items() for n in names
    }

    def restore():
        for (cls, name), value in originals.items():
            if value is None:
                cls.__dict__.get(name) and delattr(cls, name)
            else:
                setattr(cls, name, value)

    def set_patch(on):
        restore()
        if on:
            tail._patch_window_manager(SlidingWindowManager)
            tail._patch_kv_cache_manager(KVCacheManager)
            tail._patch_scheduler(Scheduler)

    yield set_patch
    restore()


def window(head_size):
    return SlidingWindowMLASpec(
        block_size=PAGE,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.float32,
        sliding_window=WINDOW,
        bounded_replay=True,
    )


# DeepSeek-V4.1 with dspark as served: compressed caches in full-attention
# groups, the target's sliding-window layers dealt into 2 groups, and the
# drafter's sliding-window group last.
SERVED = [
    FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=64, dtype=torch.float32
    ),
    window(64),
    window(64),
    window(32),
]
REPLAYING = (1, 2, 3)


def make_manager(num_blocks=900):
    groups = [
        KVCacheGroupSpec([f"layer{i}"], spec, is_eagle_group=(i == len(SERVED) - 1))
        for i, spec in enumerate(SERVED)
    ]
    return KVCacheManager(
        KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        max_model_len=1 << 20,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
    )


def tokens(seed, count):
    rng = random.Random(seed)
    return [rng.randrange(150_000) for _ in range(count)]


class Served(SimpleNamespace):
    """What one request's admission and run looked like to the cache."""


def serve(manager, request_id, prompt, outputs=(), free=True, array_prompt=False):
    """Admit a request as the scheduler does (replaying the hit's last window
    unless the hit adopted it), prefill it in chunks, decode the outputs as
    accepted drafts, and record every sliding-window page it held by position.
    ``array_prompt`` builds the request as the engine serves it with
    VLLM_TPU_ENGINE_ARRAY_PROMPT and VLLM_TPU_BLOCK_HASH_BYTES."""
    if array_prompt:
        request = packed_prompt_ids._array_prompt_request(Request)(
            request_id,
            packed_prompt_ids.PromptIds(packed_prompt_ids.TYPECODE, prompt),
            SamplingParams(max_tokens=16),
            None,
            block_hasher=block_hash_bytes.get_request_block_hasher(BLOCK, sha256),
        )
    else:
        request = Request(
            request_id,
            list(prompt),
            SamplingParams(max_tokens=16),
            None,
            block_hasher=kv_cache_utils.get_request_block_hasher(BLOCK, sha256),
        )
    blocks, hit, _ = manager.get_computed_blocks(request)
    adopted = hit > 0 and getattr(request, tail._HIT, 0) == hit
    lists = [list(group) for group in blocks.blocks]
    replay = 0 if adopted or hit == 0 else min(WINDOW, hit)
    managers = manager.coordinator.single_type_managers
    held = {g: {} for g in REPLAYING}

    def snapshot():
        for g in REPLAYING:
            for index, block in enumerate(managers[g].req_to_blocks[request_id]):
                if not block.is_null:
                    held[g].setdefault(index, block)

    step = min(CHUNK, request.num_tokens - hit)
    assert manager.allocate_slots(request, step, hit, blocks) is not None
    admitted = {g: list(managers[g].req_to_blocks[request_id]) for g in REPLAYING}
    request.num_computed_tokens = hit + step
    snapshot()
    while request.num_computed_tokens < request.num_tokens:
        step = min(CHUNK, request.num_tokens - request.num_computed_tokens)
        assert manager.allocate_slots(request, step) is not None
        request.num_computed_tokens += step
        snapshot()
    for start in range(0, len(outputs), 1 + DRAFTS):
        accepted = list(outputs[start : start + 1 + DRAFTS])
        request.append_output_token_ids(accepted)
        assert manager.allocate_slots(request, len(accepted)) is not None
        request.num_computed_tokens += len(accepted)
        snapshot()
    if free:
        manager.free(request)
    return Served(
        request=request,
        hit=hit,
        adopted=adopted,
        replay=replay,
        lists=lists,
        admitted=admitted,
        held=held,
    )


# The writer's prompt ends 221 tokens past a block. The next turn renders the
# writer's 300 output tokens as other tokens, so its hit ends at the writer's
# last prompt boundary, as in the served workload.
PROMPT = tokens(1, 5 * BLOCK + 221)
OUTPUT = tokens(2, 300)
ALIGNED = len(PROMPT) // BLOCK * BLOCK
NEXT = PROMPT + tokens(3, 700)


def test_extending_turn_adopts_the_writers_window(switch):
    switch(True)
    manager = make_manager()
    writer = serve(manager, "r0", PROMPT, OUTPUT)
    turn = serve(manager, "r1", NEXT)
    assert (turn.hit, turn.adopted, turn.replay) == (ALIGNED, True, 0)
    first = (ALIGNED - WINDOW) // PAGE
    for g in REPLAYING:
        pages = turn.lists[g][first:]
        # The window pages the writer held at those positions, nulls below.
        assert [p.block_id for p in pages] == [
            writer.held[g][first].block_id,
            writer.held[g][first + 1].block_id,
        ]
        assert all(p.is_null for p in turn.lists[g][:first])
        assert turn.admitted[g][first : first + 2] == pages
        # Pages from the hit on are the turn's own new pages.
        own = turn.admitted[g][first + 2 :]
        assert own and not {p.block_id for p in own} & {p.block_id for p in pages}
    # vLLM as served replays the window and adopts nothing.
    switch(False)
    manager = make_manager()
    serve(manager, "r0", PROMPT, OUTPUT)
    head = serve(manager, "r1", NEXT)
    assert (head.hit, head.adopted, head.replay) == (ALIGNED, False, WINDOW)
    assert all(not head.lists[g] for g in REPLAYING)


def test_engine_prompt_array_adopts_the_same_window(switch):
    """With the prompt kept as the received array and hashed as bytes, the
    turn hits, adopts and skips the replay as with vLLM's list prompt."""
    switch(True)
    manager = make_manager()
    writer = serve(manager, "r0", PROMPT, OUTPUT, array_prompt=True)
    turn = serve(manager, "r1", NEXT, array_prompt=True)
    assert type(turn.request.prompt_token_ids) is packed_prompt_ids.PromptIds
    # The tail keys derive from these hashes: the array's equal the list's.
    listed = Request(
        "r2",
        list(NEXT),
        SamplingParams(max_tokens=16),
        None,
        block_hasher=block_hash_bytes.get_request_block_hasher(BLOCK, sha256),
    )
    assert turn.request.block_hashes == listed.block_hashes
    assert (turn.hit, turn.adopted, turn.replay) == (ALIGNED, True, 0)
    first = (ALIGNED - WINDOW) // PAGE
    for g in REPLAYING:
        assert [p.block_id for p in turn.lists[g][first:]] == [
            writer.held[g][first].block_id,
            writer.held[g][first + 1].block_id,
        ]


def test_adopted_pages_stay_cached_and_counted(switch):
    """A running reader holds a reference; after it ends, the pages go back to
    the free queue still cached, and the next turn adopts them again."""
    switch(True)
    manager = make_manager()
    serve(manager, "r0", PROMPT, OUTPUT)
    reader = serve(manager, "r1", NEXT, free=False)
    pages = [reader.lists[g][-1] for g in REPLAYING]
    assert all(p.ref_cnt == 1 for p in pages)
    manager.free(reader.request)
    assert all(p.ref_cnt == 0 and p.block_hash is not None for p in pages)
    again = serve(manager, "r2", PROMPT + tokens(4, 500))
    assert again.adopted and [again.lists[g][-1] for g in REPLAYING] == pages


@pytest.mark.parametrize(
    "new, why",
    [
        # The new prompt extends the hit by less than a window: the CED
        # decoder's window would start below the hit.
        (PROMPT[:ALIGNED] + tokens(3, WINDOW - 1), "short suffix"),
        # The hit ends at an inner boundary of the writer's prompt.
        (PROMPT[: 4 * BLOCK] + tokens(3, 700), "no tail in group 1"),
        # The hit ends in the writer's decoded tokens, where no tail is kept.
        (PROMPT + OUTPUT + tokens(3, 700), "no tail in group 1"),
    ],
    ids=["short-suffix", "inner-boundary", "decoded-boundary"],
)
def test_other_hits_replay(switch, new, why):
    switch(True)
    manager = make_manager()
    serve(manager, "r0", PROMPT, OUTPUT)
    turn = serve(manager, "r1", new)
    assert turn.hit > 0
    assert (turn.adopted, turn.replay) == (False, WINDOW)
    assert all(not turn.lists[g] for g in REPLAYING)
    assert tail._state(manager).replayed == {why: 1}


def test_chunk_starting_inside_the_window(switch):
    """The prompt's last boundary lies 4 tokens past the first chunk: the
    second chunk still holds the window pages the first chunk wrote."""
    switch(True)
    manager = make_manager()
    prompt = tokens(5, CHUNK + 4 + 100)
    aligned = len(prompt) // BLOCK * BLOCK
    assert aligned - WINDOW < CHUNK < aligned
    writer = serve(manager, "r0", prompt, OUTPUT)
    turn = serve(manager, "r1", prompt + tokens(6, 400))
    assert (turn.hit, turn.adopted, turn.replay) == (aligned, True, 0)
    first = (aligned - WINDOW) // PAGE
    for g in REPLAYING:
        assert turn.lists[g][first:] == [
            writer.held[g][first],
            writer.held[g][first + 1],
        ]


def test_evicted_page_replays(switch):
    """A tail one of whose pages left the cache in any group is not adopted."""
    switch(True)
    manager = make_manager()
    writer = serve(manager, "r0", PROMPT, OUTPUT)
    page = writer.held[3][(ALIGNED - PAGE) // PAGE]
    pool = manager.coordinator.block_pool
    assert page.block_hash is not None
    pool._maybe_evict_cached_block(page)
    turn = serve(manager, "r1", NEXT)
    assert (turn.hit, turn.adopted, turn.replay) == (ALIGNED, False, WINDOW)


def test_reused_pages_lose_their_tail(switch):
    """Pages evicted and reused by other requests are no longer found."""
    switch(True)
    manager = make_manager(num_blocks=120)
    serve(manager, "r0", PROMPT, OUTPUT)
    for i in range(6):
        serve(manager, f"f{i}", tokens(10 + i, 9 * BLOCK))
    turn = serve(manager, "r1", NEXT)
    assert not turn.adopted


def test_writer_stores_its_prompts_last_boundary_only(switch):
    switch(True)
    manager = make_manager()
    serve(manager, "r0", PROMPT, OUTPUT)
    state = tail._state(manager)
    # 1280 is the prompt's last boundary; 1024 lies inside it and 1536 and
    # 1792 in the decoded tokens.
    assert state.stored == len(REPLAYING)
    assert all(
        not m._tpu_swa_tail_done for m in manager.coordinator.single_type_managers[1:]
    )


def test_scheduler_skips_the_replay_only_for_the_adopted_hit(switch):
    switch(True)
    scheduler = SimpleNamespace(prefix_replay_tokens=WINDOW)
    request = SimpleNamespace(replay_start=7)
    setattr(request, tail._HIT, ALIGNED)
    assert Scheduler._mark_prefix_replay(scheduler, request, ALIGNED) == 0
    assert request.replay_start == 0
    # Any other hit length (a reset hit, another lookup) replays.
    assert Scheduler._mark_prefix_replay(scheduler, request, ALIGNED - BLOCK) == WINDOW
    assert request.replay_start == ALIGNED - BLOCK - WINDOW
    setattr(request, tail._HIT, 0)
    assert Scheduler._mark_prefix_replay(scheduler, request, ALIGNED) == WINDOW


def test_off_without_replaying_groups(switch):
    switch(True)
    groups = [KVCacheGroupSpec(["layer0"], SERVED[0])]
    manager = KVCacheManager(
        KVCacheConfig(num_blocks=100, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=1 << 20,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
    )
    assert tail._state(manager) is None
