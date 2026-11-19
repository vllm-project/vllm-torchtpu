# SPDX-License-Identifier: Apache-2.0
"""Exercise SWA lifecycle through the upstream unified KV-cache pool."""

from collections import deque

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

WINDOW = 128


@pytest.fixture(autouse=True)
def register_specs():
    KVCacheSpecRegistry.check_kv_cache_spec_registry({})
    init_none_hash(sha256)


def spec(block_size=32, extra_retained_tokens=0):
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        bounded_replay=True,
        extra_retained_tokens=extra_retained_tokens,
    )


def manager(swa_spec=None, *, caching=False, groups=1, num_blocks=4096):
    swa_spec = swa_spec or spec()
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["main"],
                FullAttentionSpec(
                    block_size=swa_spec.block_size,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.bfloat16,
                ),
            ),
            *[KVCacheGroupSpec([f"swa.{i}"], swa_spec) for i in range(groups)],
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=swa_spec.block_size,
        hash_block_size=swa_spec.block_size,
        max_in_flight_tokens=4096,
        enable_caching=caching,
    )


def request(name="request", tokens=32768, prompt=None, block_size=32):
    return Request(
        name,
        prompt_token_ids=prompt if prompt is not None else [1] * tokens,
        sampling_params=SamplingParams(max_tokens=1 << 16),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def swa_managers(cache):
    return [
        m
        for m in cache.coordinator.single_type_managers
        if isinstance(m.kv_cache_spec, SlidingWindowMLASpec)
    ]


@pytest.mark.parametrize("caching", [False, True])
def test_swa_uses_shared_block_pool(caching):
    cache = manager(caching=caching, groups=2, num_blocks=32)
    assert all(
        m.block_pool is cache.block_pool for m in cache.coordinator.single_type_managers
    )
    free = cache.block_pool.get_num_free_blocks()
    req = request()
    assert cache.allocate_slots(req, 128) is not None
    assert cache.block_pool.get_num_free_blocks() == free - 3 * 4
    assert cache.allocate_slots(request("too_large"), 512) is None
    cache.free(req)
    assert cache.block_pool.get_num_free_blocks() == free


def test_swa_reclamation_waits_for_inflight_prefill():
    cache = manager()
    req = request()
    assert cache.allocate_slots(req, 256) is not None
    (swa,) = swa_managers(cache)
    old = list(swa.req_to_blocks[req.request_id][:4])
    req.num_computed_tokens = 256
    req.num_in_flight_tokens = 256
    assert cache.allocate_slots(req, 6) is not None
    assert all(block.ref_cnt == 1 for block in old)
    req.num_in_flight_tokens = 0
    assert cache.allocate_slots(req, 6) is not None
    assert all(block.ref_cnt == 0 for block in old)
    assert all(block.is_null for block in swa.req_to_blocks[req.request_id][:4])
    assert not any(
        block.is_null
        for block in cache.coordinator.single_type_managers[0].req_to_blocks[
            req.request_id
        ]
    )


def test_rejection_preserves_cross_page_history():
    cache = manager()
    req = request()
    assert cache.allocate_slots(req, 160) is not None
    (swa,) = swa_managers(cache)
    history = swa.req_to_blocks[req.request_id][0]
    # CPU positions advance to 160 while the preceding six queries are pending.
    req.num_computed_tokens = 160
    req.num_in_flight_tokens = 6
    assert cache.allocate_slots(req, 6) is not None
    assert cache.get_block_ids(req.request_id)[1][0] == history.block_id
    # Rejection rewinds to 158; token 31 remains in the 128-token window.
    req.num_computed_tokens = 158
    req.num_in_flight_tokens = 0
    assert cache.allocate_slots(req, 6) is not None
    assert history.ref_cnt == 1
    assert cache.get_block_ids(req.request_id)[1][0] == history.block_id
    req.num_computed_tokens = 166
    assert cache.allocate_slots(req, 6) is not None
    assert history.ref_cnt == 0


def test_deferred_free_returns_blocks_to_shared_pool():
    cache = manager(groups=2)
    req = request()
    before = cache.block_pool.get_num_free_blocks()
    assert cache.allocate_slots(req, 128) is not None
    blocks = cache.pop_blocks_for_free(req)
    assert cache.block_pool.get_num_free_blocks() == before - 12
    cache.block_pool.free_blocks(reversed(blocks))
    assert cache.block_pool.get_num_free_blocks() == before


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("block_size", [16, 32])
def test_dspark_c48_schedule_fits_pool_and_keeps_kernel_windows(caching, block_size):
    """48 requests per rank: chunked prefill, K=5 DSpark verify, async depth 2.

    Each step frees on the processed basis before it allocates. A step reads
    SWA from its first query - 127 (draft blocks start one token later and
    reach back 128) through its last lookahead slot; those pages must stay
    live until the step completes, which is after the next step is scheduled.
    """
    seqs, budget, spec_tokens, depth = 48, 2048, 5, 2
    swa_spec = spec(block_size=block_size)
    cache = manager(swa_spec, caching=caching, num_blocks=40000)
    (swa,) = swa_managers(cache)
    rng = np.random.default_rng(block_size + caching)
    waiting = deque(
        request(
            f"r{i}",
            prompt=list(map(int, rng.integers(0, 1 << 20, int(n)))),
            block_size=block_size,
        )
        for i, n in enumerate(rng.integers(200, 1500, 160))
    )
    outputs = dict.fromkeys((r.request_id for r in waiting), 0)
    limits = {r.request_id: int(rng.integers(8, 120)) for r in waiting}
    running: list[Request] = []
    in_flight: deque[list] = deque()
    peak = 0

    def live(req, lo, hi):
        blocks = swa.req_to_blocks[req.request_id]
        return all(
            not blocks[i].is_null
            for i in range(max(0, lo) // block_size, hi // block_size + 1)
        )

    def complete(step):
        for req, first, count, decode in step:
            # The step read its window while the next step was scheduled.
            assert live(req, first - (WINDOW - 1), first + count + spec_tokens - 1)
            req.num_in_flight_tokens -= count
            if decode:
                accepted = int(rng.integers(1, spec_tokens + 2))
                req.num_computed_tokens -= count - accepted
                req.append_output_token_ids([1] * accepted)
                outputs[req.request_id] += accepted
            elif req.num_computed_tokens >= req.num_prompt_tokens:
                req.append_output_token_ids([1])
                outputs[req.request_id] += 1

    while waiting or running or in_flight:
        while waiting and len(running) < seqs:
            running.append(waiting.popleft())
        tokens, step = budget, []
        for req in running:
            if outputs[req.request_id] >= limits[req.request_id]:
                # Finished: wait for its in-flight steps, then free it.
                continue
            remaining = req.num_prompt_tokens - req.num_computed_tokens
            decode = remaining <= 0
            count = 1 + spec_tokens if decode else min(remaining, tokens)
            if count <= 0 or count > tokens:
                continue
            # Actual first position: in-flight rejections rewind the worker.
            first = req.num_computed_tokens
            assert (
                cache.allocate_slots(req, count, num_lookahead_tokens=spec_tokens)
                is not None
            )
            assert live(req, first - (WINDOW - 1), first + count + spec_tokens - 1)
            req.num_computed_tokens += count
            req.num_in_flight_tokens += count
            tokens -= count
            step.append([req, first, count, decode])
        pool = swa.block_pool
        peak = max(peak, pool.num_gpu_blocks - pool.get_num_free_blocks())
        if step:
            in_flight.append(step)
        if in_flight and (len(in_flight) >= depth or not step):
            done = in_flight.popleft()
            complete(done)
            # Later in-flight steps start where the rejections left them.
            for later in in_flight:
                for entry in later:
                    entry[1] = min(entry[1], entry[0].num_computed_tokens - entry[2])
        for req in list(running):
            if outputs[req.request_id] >= limits[req.request_id] and not any(
                entry[0] is req for later in in_flight for entry in later
            ):
                running.remove(req)
                cache.free(req)
    assert swa.block_pool.get_num_free_blocks() == swa.block_pool.num_gpu_blocks - 1
    assert peak < cache.block_pool.num_gpu_blocks


def test_prefix_hit_replays_window_into_shared_pages():
    cache = manager(spec(), caching=True)
    (swa,) = swa_managers(cache)
    prompt = list(range(1024))
    first = request("first", prompt=prompt)
    assert cache.allocate_slots(first, 1024) is not None
    first.num_computed_tokens = 1024
    cache.free(first)
    assert swa.block_pool.get_num_free_blocks() == swa.block_pool.num_gpu_blocks - 1

    second = request("second", prompt=prompt)
    hits, hit, _ = cache.get_computed_blocks(second)
    assert hit > WINDOW and hit % 32 == 0
    assert hits.blocks[1] == []
    # The scheduler adopts the main-KV hit and replays its last window; the
    # replayed tokens are already counted in the hit.
    assert (
        cache.allocate_slots(
            second,
            1024 - hit,
            num_new_computed_tokens=hit,
            new_computed_blocks=hits,
        )
        is not None
    )
    blocks = swa.req_to_blocks[second.request_id]
    replay = (hit - WINDOW) // 32
    assert all(block.is_null for block in blocks[: (hit - WINDOW + 1) // 32])
    assert not any(block.is_null for block in blocks[replay:])
    assert len(blocks) == 1024 // 32


def test_prefill_lookahead_keeps_extra_history():
    cache = manager(spec(extra_retained_tokens=8))
    req = request()
    assert cache.allocate_slots(req, 160) is not None
    (swa,) = swa_managers(cache)
    history = swa.req_to_blocks[req.request_id][0]
    req.num_computed_tokens = 160
    assert cache.allocate_slots(req, 8) is not None
    assert history.ref_cnt == 1
    req.num_computed_tokens = 168
    assert cache.allocate_slots(req, 1) is not None
    assert history.ref_cnt == 0
