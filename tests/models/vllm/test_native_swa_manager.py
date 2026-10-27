# SPDX-License-Identifier: Apache-2.0
"""Exercise the DeepSeek-V4.1 SWA pool through the upstream KV-cache control plane."""

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request

from vllm_torchtpu.models.vllm.deepseek_v41.v2.swa_cache import (
    DeepseekV41SWAManager,
    DeepseekV41SWASpec,
    num_physical_blocks,
    register_swa_specs,
    swa_pool_bytes,
)

pytestmark = pytest.mark.cpu_test

WINDOW = 128


@pytest.fixture(autouse=True)
def register_specs():
    KVCacheSpecRegistry.check_kv_cache_spec_registry({})
    register_swa_specs()
    init_none_hash(sha256)


def spec(sequences=2, in_flight=512, lookahead=0, block_size=32, bounded=True):
    return DeepseekV41SWASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        bounded_replay=bounded,
        max_num_seqs=sequences,
        max_in_flight_tokens=in_flight,
        num_lookahead_tokens=lookahead,
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
        max_in_flight_tokens=swa_spec.max_in_flight_tokens,
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
        if isinstance(m, DeepseekV41SWAManager)
    ]


def test_private_pool_is_charged_alone_and_exhaustion_is_an_error():
    cache = manager(spec(sequences=1, in_flight=32))
    (swa,) = swa_managers(cache)
    assert swa.block_pool is not cache.block_pool
    capacity = spec(sequences=1, in_flight=32).pool_capacity
    assert swa.block_pool.num_gpu_blocks == capacity
    free = swa.block_pool.get_num_free_blocks()
    first = request("first")
    assert cache.allocate_slots(first, free * 32) is not None
    assert cache.block_pool.get_num_free_blocks() == 4095 - free
    assert swa.block_pool.get_num_free_blocks() == 0
    # Whole-sequence admission never consults the private pool.
    assert (
        swa.get_num_blocks_to_allocate(
            "second",
            num_tokens=32768,
            new_computed_blocks=[],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=32768,
            apply_admission_cap=True,
        )
        == 0
    )
    # A shortfall means the capacity bound is wrong: never turn it into
    # an admission failure that the scheduler would answer with preemption.
    with pytest.raises(RuntimeError, match="SWA pool exhausted"):
        cache.allocate_slots(request("third"), 32)
    cache.free(first)
    assert cache.allocate_slots(request("third"), 32) is not None


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


def test_deferred_free_returns_each_block_to_its_own_pool():
    cache = manager(groups=2)
    req = request()
    pools = [mgr.block_pool for mgr in cache.coordinator.single_type_managers]
    before = [pool.get_num_free_blocks() for pool in pools]
    assert len(set(map(id, pools))) == 3
    assert cache.allocate_slots(req, 128) is not None
    blocks = cache.pop_blocks_for_free(req)
    assert [pool.get_num_free_blocks() for pool in pools] == [n - 4 for n in before]
    cache.block_pool.free_blocks(reversed(blocks))
    assert [pool.get_num_free_blocks() for pool in pools] == before
    assert all(pool.null_block.is_null for pool in pools)


def test_private_ids_do_not_enter_shared_zeroing_or_copy_commands():
    cache = manager()
    req = request()
    assert cache.allocate_slots(req, 128) is not None
    (swa,) = swa_managers(cache)
    assert not swa.records_new_block_ids
    assert swa.take_new_block_ids() == []
    assert swa.take_pending_cow_copies() == []


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("caching", [False, True])
def test_one_manager_with_and_without_prefix_caching(caching, bounded):
    swa_spec = spec(bounded=bounded)
    assert not swa_spec.prefix_cacheable
    cache = manager(swa_spec, caching=caching)
    (swa,) = swa_managers(cache)
    assert type(swa) is DeepseekV41SWAManager
    if caching:
        # Only the main KV takes part in prefix-hit lookup.
        assert [g.group_ids for g in cache.coordinator.attention_groups] == [[0]]
    req = request(tokens=512)
    assert cache.allocate_slots(req, 512) is not None
    assert [b.block_id for b in swa.req_to_blocks[req.request_id]] == list(range(1, 17))


def test_prefix_hit_replays_the_window_into_fresh_private_pages():
    cache = manager(spec(sequences=4, in_flight=2048), caching=True)
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
    swa_spec = spec(
        sequences=seqs,
        in_flight=depth * budget,
        lookahead=spec_tokens,
        block_size=block_size,
    )
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
    assert peak <= swa_spec.pool_capacity - 1
    # The bound is not grossly loose: at least half the pages were used.
    assert peak > swa_spec.pool_capacity // 2


def test_capacity_counts_window_lookahead_and_in_flight_tokens():
    base = spec(sequences=48, in_flight=4096, lookahead=0)
    dspark = replace(base, num_lookahead_tokens=5)
    assert base.pool_capacity == 1 + 48 * (4 + 2) + 128
    assert dspark.pool_capacity == 1 + 48 * (5 + 2) + 128
    assert num_physical_blocks(dspark, 100000) == dspark.pool_capacity
    assert (
        num_physical_blocks(
            FullAttentionSpec(
                block_size=32, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
            ),
            100000,
        )
        == 100000
    )
    specs = {
        "a": dspark,
        "b": dspark,
        "main": FullAttentionSpec(
            block_size=32, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
        ),
    }
    assert swa_pool_bytes(specs) == 2 * dspark.pool_capacity * 32 * 1024


def test_spec_merge_keeps_pool_inputs():
    original = spec(lookahead=5)
    assert DeepseekV41SWASpec.merge([original, original]) == original
    with pytest.raises(AssertionError):
        DeepseekV41SWASpec.merge([original, replace(original, num_lookahead_tokens=0)])


def test_upstream_grouping_preserves_private_pool_contract():
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=32768, original_max_model_len=32768),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
        max_in_flight_tokens=512,
        num_prefill_lookahead_tokens=0,
    )
    specs = {
        "main": MLAAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            tokens_per_state=2,
        ),
        "swa.0": spec(lookahead=5),
        "swa.1": spec(lookahead=5),
    }
    reservation = swa_pool_bytes(specs)
    result = get_kv_cache_configs(config, [specs], [(256 << 20) - reservation])[0]
    swa_groups = [
        g
        for g in result.kv_cache_groups
        if isinstance(g.kv_cache_spec, DeepseekV41SWASpec)
    ]
    assert sum(len(g.layer_names) for g in swa_groups) == 2
    assert all(
        g.kv_cache_spec.pool_capacity == spec(lookahead=5).pool_capacity
        for g in swa_groups
    )
