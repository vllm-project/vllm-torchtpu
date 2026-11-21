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

import os
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core import prefix_aware_scheduler
from vllm_torchtpu.core.prefix_aware_scheduler import (
    PrefixAwareScheduler,
    _Follower,
    _Rejoin,
)


def _make_request(
    request_id: str,
    num_tokens: int = 1000,
    arrival_time: float = 0.0,
    num_preemptions: int = 0,
    num_computed_tokens: int = 0,
    skip_reading_prefix_cache: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        arrival_time=arrival_time,
        num_preemptions=num_preemptions,
        num_computed_tokens=num_computed_tokens,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
        block_hashes=[object()],
    )


def _make_scheduler(
    requests: list[SimpleNamespace],
    warmth_map: dict[str, int],
    window: int = 64,
    enable_caching: bool = True,
    raise_on_lookup: bool = False,
    tier_warmth: bool = False,
    connector: object | None = None,
) -> PrefixAwareScheduler:
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    q = FCFSRequestQueue()
    for r in requests:
        q.add_request(r)
    s.waiting = q
    s.skipped_waiting = FCFSRequestQueue()
    s.policy = SchedulingPolicy.FCFS
    s.WINDOW = window
    s._paw_n = 0
    s._paw_cold_tokens = 0
    s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False
    s._init_tier_warmth(tier_warmth)
    s.connector = connector
    s._tier = None

    coordinator = MagicMock()

    def find_hit(block_hashes, max_len):
        if raise_on_lookup:
            raise RuntimeError("simulated coordinator failure")
        for r in requests:
            if r.block_hashes is block_hashes:
                return ({}, warmth_map.get(r.request_id, 0), 0)
        return ({}, 0, 0)

    coordinator.find_longest_cache_hit.side_effect = find_hit

    kvm = MagicMock()
    kvm.enable_caching = enable_caching
    kvm.prefix_cache_lookup_enabled.side_effect = lambda request: (
        enable_caching and not request.skip_reading_prefix_cache
    )
    kvm.coordinator = coordinator
    s.kv_cache_manager = kvm
    return s


# --- Offload-tier warmth (TPU_PREFIX_AWARE_TIER_WARMTH) ---------------------

_CHUNK_TOKENS = 4096
_HASHES_PER_CHUNK = 16
_HASH_TOKENS = _CHUNK_TOKENS // _HASHES_PER_CHUNK


def _hashed_request(request_id: str, num_tokens: int, arrival_time: float):
    r = _make_request(request_id, num_tokens=num_tokens, arrival_time=arrival_time)
    r.block_hashes = [
        f"{request_id}:{i}".encode() for i in range(num_tokens // _HASH_TOKENS)
    ]
    return r


def _chunk_key(request: SimpleNamespace, chunk: int) -> bytes:
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(
        request.block_hashes[(chunk + 1) * _HASHES_PER_CHUNK - 1], 0
    )


class _FakeTierManager:
    """Answers tier probes from a key set; records each probe's keys.

    Mirrors RaidenOffloadingManager.probe_warm_chunks: the number of leading
    keys that are held (`keys`) or still being stored (`pending`).
    """

    def __init__(self, keys=(), pending=(), fail=False):
        self.keys = set(keys)
        self.pending = set(pending)
        self.fail = fail
        self.calls = 0
        self.probed: list[list[bytes]] = []

    def probe_warm_chunks(self, keys):
        self.calls += 1
        keys = list(keys)
        self.probed.append(keys)
        if self.fail:
            raise RuntimeError("simulated store failure")
        warm = 0
        for key in keys:
            if key not in self.keys and key not in self.pending:
                break
            warm += 1
        return warm


def _offload_connector(
    manager,
    num_groups: int = 1,
    multi: bool = True,
    is_eagle_group: bool = False,
):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    group = SimpleNamespace(
        group_idx=0,
        tokens_per_chunk=_CHUNK_TOKENS,
        hashes_per_chunk=_HASHES_PER_CHUNK,
        is_eagle_group=is_eagle_group,
    )
    sched = TPURaidenOffloadingScheduler.__new__(TPURaidenOffloadingScheduler)
    sched._raiden_manager = manager
    sched.config = SimpleNamespace(kv_group_configs=[group] * num_groups)
    offload = SimpleNamespace(connector_scheduler=sched)
    if not multi:
        return offload
    # A P/D producer child first, as in TPUMultiConnector configs.
    return SimpleNamespace(_connectors=[SimpleNamespace(), offload])


def test_default_knobs():
    assert PrefixAwareScheduler.WINDOW == 64


def test_promotes_warmest_within_window():
    now = time.time()
    r_cold = _make_request("cold", num_tokens=1000, arrival_time=now - 10)
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r_cold, r_warm], {"cold": 0, "warm": 800})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["warm", "cold"]


def test_ranks_by_uncached_tokens_not_hit_ratio():
    now = time.time()
    # 100k tokens, 90% cached -> 10k uncached tokens
    r_long = _make_request("long", num_tokens=100_000, arrival_time=now - 10)
    # 20k tokens, 80% cached -> 4k uncached tokens (fewer uncached tokens wins)
    r_short = _make_request("short", num_tokens=20_000, arrival_time=now - 5)
    s = _make_scheduler([r_long, r_short], {"long": 90_000, "short": 16_000})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["short", "long"]


def test_fcfs_order_preserved_when_all_cold():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["r1", "r2"]


def test_preempted_tier_outranks_normal():
    now = time.time()
    r_preempted = _make_request(
        "preempted", num_tokens=1000, arrival_time=now - 5, num_preemptions=1
    )
    r_warm = _make_request("warm", num_tokens=1000, arrival_time=now - 10)
    s = _make_scheduler([r_preempted, r_warm], {"preempted": 0, "warm": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order == ["preempted", "warm"]


def test_window_bounds_evaluation():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 15)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 10)
    r_outside = _make_request("outside", num_tokens=1000, arrival_time=now - 5)
    # window = 2, so r_outside is beyond the inspection window
    s = _make_scheduler(
        [r1, r2, r_outside], {"r1": 100, "r2": 200, "outside": 999}, window=2
    )

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    order = [r.request_id for r in q]
    assert order[0] == "r2"  # r2 was warmest in window [r1, r2]


def test_single_or_empty_queue_is_noop():
    s = _make_scheduler([_make_request("single")], {"single": 500})
    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["single"]


def test_uses_num_computed_tokens_when_present():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, num_computed_tokens=900
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 0})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r2", "r1"]


def test_skip_reading_prefix_cache_or_caching_disabled():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request(
        "r2", num_tokens=1000, arrival_time=now - 5, skip_reading_prefix_cache=True
    )
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 900})

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]

    s_disabled = _make_scheduler(
        [r1, _make_request("r3", num_tokens=1000, arrival_time=now - 5)],
        {"r1": 0, "r3": 900},
        enable_caching=False,
    )
    q_disabled = s_disabled._select_waiting_queue_for_scheduling()
    assert q_disabled is not None
    assert [r.request_id for r in q_disabled] == ["r1", "r3"]


def test_coordinator_failure_falls_back_to_fcfs():
    now = time.time()
    r1 = _make_request("r1", num_tokens=1000, arrival_time=now - 10)
    r2 = _make_request("r2", num_tokens=1000, arrival_time=now - 5)
    s = _make_scheduler([r1, r2], {"r1": 0, "r2": 500}, raise_on_lookup=True)

    q = s._select_waiting_queue_for_scheduling()
    assert q is not None
    assert [r.request_id for r in q] == ["r1", "r2"]
    assert s._paw_scoring_broken is True


def _tier_resident_and_warm():
    now = time.time()
    # HBM evicted this 25-chunk session; the tier holds its first 24 chunks,
    # so it needs ~4k tokens of compute.
    resident = _hashed_request("resident", num_tokens=25 * 4096, arrival_time=now - 10)
    # 20k tokens, 16k of them in HBM: 4k uncached, a bit more than `resident`
    # once the tier counts.
    warm = _hashed_request("warm", num_tokens=20_480 + 512, arrival_time=now - 5)
    manager = _FakeTierManager(keys=[_chunk_key(resident, c) for c in range(24)])
    return resident, warm, manager, {"resident": 0, "warm": 16_384}


def test_tier_warmth_off_ranks_on_hbm_only():
    resident, warm, manager, hbm = _tier_resident_and_warm()
    s = _make_scheduler([resident, warm], hbm, connector=_offload_connector(manager))

    q = s._select_waiting_queue_for_scheduling()
    assert [r.request_id for r in q] == ["warm", "resident"]
    assert manager.calls == 0


def test_tier_warmth_promotes_tier_resident_request():
    resident, warm, manager, hbm = _tier_resident_and_warm()
    s = _make_scheduler(
        [warm, resident], hbm, tier_warmth=True, connector=_offload_connector(manager)
    )

    q = s._select_waiting_queue_for_scheduling()
    assert [r.request_id for r in q] == ["resident", "warm"]
    assert s._prefix_warmth(resident) == 24 * 4096


def test_tier_probe_stops_at_first_miss_and_counts_pending_stores():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(
        keys=[_chunk_key(r, 0), _chunk_key(r, 1), _chunk_key(r, 4)],
        pending=[_chunk_key(r, 2)],
    )
    s = _make_scheduler(
        [r], {"r": 0}, tier_warmth=True, connector=_offload_connector(manager)
    )

    # Chunks 0-2 (one still being stored), then chunk 3 misses.
    assert s._prefix_warmth(r) == 3 * 4096


def test_tier_probe_starts_at_the_hbm_hit():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in (2, 3, 4)])
    s = _make_scheduler(
        [r], {"r": 2 * 4096}, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 5 * 4096


def test_tier_probe_needs_a_chunk_aligned_hbm_hit():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(10)])
    s = _make_scheduler(
        [r], {"r": 4096 + 256}, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 4096 + 256
    assert manager.calls == 0


def test_tier_probe_skips_nearly_warm_requests():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(10)])
    s = _make_scheduler(
        [r], {"r": 8 * 4096}, tier_warmth=True, connector=_offload_connector(manager)
    )

    # One whole chunk left past the hit: below TIER_PROBE_MIN_CHUNKS.
    assert s._prefix_warmth(r) == 8 * 4096
    assert manager.calls == 0


def test_tier_probe_is_cached_until_the_hbm_hit_changes():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(4)])
    hbm = {"r": 0}
    s = _make_scheduler(
        [r], hbm, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 4 * 4096
    calls = manager.calls
    assert s._prefix_warmth(r) == 4 * 4096
    assert manager.calls == calls
    hbm["r"] = 4096
    assert s._prefix_warmth(r) == 4 * 4096
    assert manager.calls > calls


def test_tier_probe_failure_falls_back_to_hbm_only():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(fail=True)
    s = _make_scheduler(
        [r], {"r": 4096}, tier_warmth=True, connector=_offload_connector(manager)
    )

    assert s._prefix_warmth(r) == 4096
    assert s._tier_probe_broken is True
    assert s._prefix_warmth(r) == 4096
    assert manager.calls == 1


def test_tier_warmth_needs_a_single_group_offloading_connector():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    keys = [_chunk_key(r, c) for c in range(10)]

    no_connector = _make_scheduler([r], {"r": 0}, tier_warmth=True)
    assert no_connector._prefix_warmth(r) == 0

    plain = _make_scheduler(
        [r], {"r": 0}, tier_warmth=True, connector=SimpleNamespace()
    )
    assert plain._prefix_warmth(r) == 0

    two_groups = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(_FakeTierManager(keys=keys), num_groups=2),
    )
    assert two_groups._prefix_warmth(r) == 0

    single = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(_FakeTierManager(keys=keys), multi=False),
    )
    # A hit covers at most num_tokens - 1 tokens: 9 whole chunks.
    assert single._prefix_warmth(r) == 9 * 4096


def test_tier_probe_accounts_for_flagged_eagle_group_tail_pop():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    # 3 chunks in tier -> 2 chunks usable when is_eagle_group=True.
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(3)])
    s = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(manager, is_eagle_group=True),
    )
    assert s._prefix_warmth(r) == 2 * 4096

    # 1 chunk in tier -> dropped to 0 when is_eagle_group=True.
    one_chunk = _FakeTierManager(keys=[_chunk_key(r, 0)])
    s_one = _make_scheduler(
        [r],
        {"r": 0},
        tier_warmth=True,
        connector=_offload_connector(one_chunk, is_eagle_group=True),
    )
    assert s_one._prefix_warmth(r) == 0


def test_tier_probe_caps_chunks_in_one_store_probe():
    now = time.time()
    r = _hashed_request("r", num_tokens=100 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(100)])
    s = _make_scheduler(
        [r], {"r": 0}, tier_warmth=True, connector=_offload_connector(manager)
    )
    assert s._prefix_warmth(r) == s.TIER_PROBE_MAX_CHUNKS * 4096
    # One probe per request covers the whole capped run, not one per chunk.
    assert manager.calls == 1
    assert manager.probed == [
        [_chunk_key(r, c) for c in range(s.TIER_PROBE_MAX_CHUNKS)]
    ]


def test_tier_probe_asks_for_the_chunks_past_the_hbm_hit():
    now = time.time()
    r = _hashed_request("r", num_tokens=10 * 4096, arrival_time=now)
    manager = _FakeTierManager(keys=[_chunk_key(r, c) for c in range(10)])
    s = _make_scheduler(
        [r], {"r": 3 * 4096}, tier_warmth=True, connector=_offload_connector(manager)
    )
    # A hit covers at most num_tokens - 1 tokens: chunks 3-8 are probed.
    assert s._prefix_warmth(r) == 9 * 4096
    assert manager.probed == [[_chunk_key(r, c) for c in range(3, 9)]]


def test_paw_tier_tokens_excludes_already_computed_requests():
    now = time.time()
    r1 = _hashed_request("r1", num_tokens=10 * 4096, arrival_time=now - 10)
    r2 = _hashed_request("r2", num_tokens=10 * 4096, arrival_time=now - 5)
    manager = _FakeTierManager(keys=[_chunk_key(r1, c) for c in range(4)])
    s = _make_scheduler(
        [r1, r2],
        {"r1": 0, "r2": 0},
        tier_warmth=True,
        connector=_offload_connector(manager),
    )
    # First step probes r1 (4 chunks = 16384 tier tokens).
    s._select_waiting_queue_for_scheduling()
    assert s._paw_tier_tokens == 4 * 4096

    # Once r1 has num_computed_tokens > 0, it does not re-credit cached tier tokens.
    r1.num_computed_tokens = 9 * 4096
    s._select_waiting_queue_for_scheduling()
    assert s._paw_tier_tokens == 4 * 4096


# PrefixAwareScheduler with VLLM_TPU_FOLLOW_SHARED_PREFIX on a real vLLM
# scheduler and KV cache manager, driven step by step with a pipelined (async)
# engine loop and a fake model runner.
#
# Every scheduled chunk is checked against what the worker would hold: each
# block a request reads before its chunk must hold KV of the same prompt prefix
# (same block hash), written by a step that ran before this one.

BLOCK = 16
CHUNK = 64
BUDGET = 192


def make_scheduler(
    follow: bool,
    num_blocks: int = 4000,
    max_model_len: int = 2048,
    budget: int = BUDGET,
    max_num_seqs: int = 8,
    policy: str = "fcfs",
    prefix_caching: bool = True,
    compile_sizes: list[int] | None = None,
) -> PrefixAwareScheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        policy=policy,
        max_model_len=max_model_len,
        long_prefill_token_threshold=CHUNK,
        enable_chunked_prefill=True,
        async_scheduling=True,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=prefix_caching,
    )
    cache_config.num_gpu_blocks = num_blocks
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        device_config=DeviceConfig(device="cpu"),
    )
    if compile_sizes is not None:
        vllm_config.compilation_config.compile_sizes = compile_sizes
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    with patch.dict(os.environ, {"VLLM_TPU_FOLLOW_SHARED_PREFIX": str(int(follow))}):
        scheduler = PrefixAwareScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=BLOCK,
            log_stats=True,
        )
    assert scheduler.follow_shared_prefix is follow
    return scheduler


def make_request(request_id: str, prompt: list[int], max_tokens: int = 1) -> Request:
    init_none_hash(sha256)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


@dataclass
class Run:
    computed: Counter = field(default_factory=Counter)
    first_token_step: dict = field(default_factory=dict)
    steps: int = 0
    reads_checked: int = 0


class Worker:
    """Block tables as the worker applies them, and the content of each block:
    the block hash of the prompt block written into it and the step that
    completed it."""

    def __init__(self):
        self.tables: dict[str, list[int]] = {}
        self.written: dict[int, tuple[int, object]] = {}

    def apply(self, out, step: int, requests: dict[str, Request], run: Run):
        for request_id in out.preempted_req_ids or ():
            self.tables.pop(request_id, None)
        starts = {}
        for new in out.scheduled_new_reqs:
            self.tables[new.req_id] = list(new.block_ids[0])
            starts[new.req_id] = new.num_computed_tokens
        cached = out.scheduled_cached_reqs
        for i, request_id in enumerate(cached.req_ids):
            block_ids = cached.new_block_ids[i]
            if request_id in cached.resumed_req_ids:
                self.tables[request_id] = list(block_ids[0])
            elif block_ids is not None:
                self.tables[request_id].extend(block_ids[0])
            starts[request_id] = cached.num_computed_tokens[i]
        for request_id, num_tokens in out.num_scheduled_tokens.items():
            request = requests[request_id]
            start = starts[request_id]
            table = self.tables[request_id]
            for i in range(start // BLOCK):
                step_written, block_hash = self.written[table[i]]
                assert block_hash == request.block_hashes[i], (
                    f"{request_id} reads block {i} holding another prefix"
                )
                assert step_written < step, (
                    f"{request_id} at step {step} reads block {i} that step "
                    f"{step_written} writes"
                )
                run.reads_checked += 1
            end = start + num_tokens
            for i in range(start // BLOCK, end // BLOCK):
                if i < len(request.block_hashes):
                    self.written[table[i]] = (step, request.block_hashes[i])
            run.computed[request_id] += num_tokens


def model_output(out, requests: dict[str, Request], starts: dict) -> ModelRunnerOutput:
    req_ids = list(out.num_scheduled_tokens)
    sampled = []
    for request_id in req_ids:
        end = starts[request_id] + out.num_scheduled_tokens[request_id]
        sampled.append([7] if end >= requests[request_id].num_prompt_tokens else [])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def run_engine(
    follow: bool,
    arrivals: dict[int, list[tuple[str, list[int]]]],
    max_steps: int = 2000,
    scheduler=None,
) -> Run:
    scheduler = scheduler or make_scheduler(follow)
    worker = Worker()
    requests: dict[str, Request] = {}
    run = Run()
    in_flight = deque()
    step = 0
    last_arrival = max(arrivals)
    while step < max_steps and (
        step <= last_arrival or scheduler.has_requests() or in_flight
    ):
        for request_id, prompt in arrivals.get(step, []):
            requests[request_id] = make_request(request_id, prompt)
            scheduler.add_request(requests[request_id])
        out = scheduler.schedule()
        starts = {n.req_id: n.num_computed_tokens for n in out.scheduled_new_reqs}
        starts.update(
            zip(
                out.scheduled_cached_reqs.req_ids,
                out.scheduled_cached_reqs.num_computed_tokens,
            )
        )
        worker.apply(out, step, requests, run)
        in_flight.append((out, starts))
        if len(in_flight) == 2 or not out.num_scheduled_tokens:
            done, done_starts = in_flight.popleft()
            engine_outputs = scheduler.update_from_output(
                done, model_output(done, requests, done_starts)
            )
            for outputs in engine_outputs.values():
                for o in outputs.outputs:
                    if o.new_token_ids:
                        run.first_token_step.setdefault(o.request_id, step)
        step += 1
    run.steps = step
    assert set(run.first_token_step) == set(requests), "a request never finished"
    return run


def prompt(seed: int, num_tokens: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 30000) for _ in range(num_tokens)]


def sibling_arrivals():
    shared = prompt(1, 40 * BLOCK)
    return {
        0: [("leader", shared + prompt(2, 3 * BLOCK))],
        1: [("other", prompt(3, 30 * BLOCK))],
        2: [("follower", shared + prompt(4, 5 * BLOCK))],
    }


def test_follower_skips_the_chunks_its_sibling_computes():
    off = run_engine(False, sibling_arrivals())
    on = run_engine(True, sibling_arrivals())
    assert off.computed["follower"] >= 30 * BLOCK
    assert on.computed["follower"] <= 6 * BLOCK + CHUNK
    assert on.computed["leader"] == off.computed["leader"]
    assert on.first_token_step["follower"] <= off.first_token_step["follower"] + 2
    assert on.first_token_step["other"] <= off.first_token_step["other"]
    assert on.reads_checked > 0


def test_off_changes_nothing():
    a = run_engine(False, sibling_arrivals())
    b = run_engine(False, sibling_arrivals())
    assert a.computed == b.computed and a.first_token_step == b.first_token_step


def tree_arrivals(seed: int):
    rng = random.Random(seed)
    arrivals: dict[int, list] = {}
    roots = [prompt(seed * 100 + i, rng.randrange(20, 80) * BLOCK) for i in range(4)]
    for n in range(24):
        root = rng.choice(roots)
        cut = rng.randrange(1, len(root) // BLOCK) * BLOCK
        tail = prompt(
            seed * 1000 + n, rng.randrange(1, 12) * BLOCK + rng.randrange(BLOCK)
        )
        arrivals.setdefault(rng.randrange(0, 60), []).append(
            (f"r{n}", root[:cut] + tail)
        )
    return arrivals


@pytest.mark.parametrize("seed", range(12))
def test_shared_trees_keep_every_read_correct(seed):
    off = run_engine(False, tree_arrivals(seed))
    on = run_engine(True, tree_arrivals(seed))
    assert sum(on.computed.values()) <= sum(off.computed.values())
    assert on.reads_checked > 0


def test_read_check_fails_without_the_same_step_patch(monkeypatch):
    """The read check catches a request that reads a block its sibling writes
    in the same step, which the same-step prefix-hit patch prevents."""
    monkeypatch.setattr(
        BlockPool, "get_cached_block", BlockPool.get_cached_block.__wrapped__
    )
    with pytest.raises(AssertionError, match="that step"):
        run_engine(False, sibling_arrivals())


def test_follower_needs_to_see_blocks_hashed_this_step(monkeypatch):
    """Counting only blocks written before this step, the follower never sees
    its chunk cached and computes it beside its sibling, as with follow off."""
    off = run_engine(False, sibling_arrivals())
    monkeypatch.setattr(prefix_aware_scheduler, "BLOCKS_HASHED_THIS_STEP", "x")
    on = run_engine(True, sibling_arrivals())
    assert on.computed["follower"] == off.computed["follower"]


def test_rejoined_follower_keeps_its_running_place():
    scheduler = make_scheduler(True)
    restore = scheduler._restore_rejoined_places
    checked = []

    def checked_restore():
        restore()
        for request_id, rejoin in scheduler._rejoins.items():
            request = scheduler.requests.get(request_id)
            if request is None or request not in scheduler.running:
                continue
            place = scheduler.running.index(request)
            before = [r.request_id for r in scheduler.running[:place]]
            behind = {r.request_id for r in scheduler.running[place + 1 :]}
            assert not behind & rejoin.ahead
            assert not before or before[-1] in rejoin.ahead
            checked.append(request_id)

    scheduler._restore_rejoined_places = checked_restore
    arrivals = sibling_arrivals()
    arrivals[3] = [("late", prompt(5, 30 * BLOCK))]
    run_engine(True, arrivals, scheduler=scheduler)
    assert set(checked) == {"follower"}


def test_follower_rejoins_under_a_connector_that_needs_kv_delivery():
    """With a KV connector that hands off KV, a rejoin preempts the follower
    dropping its in-flight output, as vLLM's own preemptions then do, and the
    follower still takes its sibling's chunks and reads only written KV."""
    scheduler = make_scheduler(True)
    scheduler.requires_kv_delivery = True
    preempt = scheduler._preempt_request
    drops = {}

    def recorded_preempt(request, timestamp, drop_stale_output=False):
        drops[request.request_id] = drop_stale_output
        preempt(request, timestamp, drop_stale_output=drop_stale_output)

    scheduler._preempt_request = recorded_preempt
    off = run_engine(False, sibling_arrivals())
    on = run_engine(True, sibling_arrivals(), scheduler=scheduler)
    assert drops == {"follower": True}
    assert on.computed["follower"] <= 6 * BLOCK + CHUNK
    assert on.computed["leader"] == off.computed["leader"]
    assert on.reads_checked > 0


def test_paused_scheduler_preempts_no_follower():
    for state, kept in ((PauseState.PAUSED_NEW, True), (PauseState.UNPAUSED, False)):
        scheduler = make_scheduler(True)
        request = make_request("r", prompt(6, 8 * BLOCK))
        scheduler.add_request(request)
        request.status = RequestStatus.RUNNING
        scheduler._followers["r"] = _Follower(cached_at_step_start=0)
        scheduler._pause_state = state
        scheduler._update_followers()
        assert ("r" in scheduler._followers) is kept
        assert request.status == RequestStatus.RUNNING
        assert not scheduler._rejoins


def test_follower_the_last_step_did_not_reach_waits_while_its_prefix_grows():
    """A follower that the last step did not reach, for example because the
    requests ahead of it used the token budget, is kept while its cached
    prefix grows between step starts, and preempted once it stops."""
    scheduler = make_scheduler(True)
    request = make_request("r", prompt(9, 8 * BLOCK))
    scheduler.requests["r"] = request
    scheduler.running.append(request)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 2 * BLOCK
    cached = iter([3 * BLOCK, 4 * BLOCK, 5 * BLOCK, 5 * BLOCK])
    scheduler._cached_prompt_tokens = lambda r: next(cached)
    scheduler._followers["r"] = _Follower()

    for expected in (3 * BLOCK, 4 * BLOCK, 5 * BLOCK):
        scheduler._update_followers()
        assert scheduler._followers["r"].cached_at_step_start == expected
        assert request.status == RequestStatus.RUNNING
        assert not scheduler._rejoins

    scheduler._update_followers()
    assert request.status == RequestStatus.PREEMPTED
    assert "r" in scheduler.reset_preempted_req_ids
    assert "r" not in scheduler._followers
    assert scheduler._rejoins["r"].computed_tokens == 2 * BLOCK
    assert request not in scheduler.running


def test_finished_and_waiting_requests_leave_or_keep_their_entries():
    scheduler = make_scheduler(True)
    waiting = make_request("waiting", prompt(7, 8 * BLOCK))
    rejoined = make_request("rejoined", prompt(8, 8 * BLOCK))
    for request in (waiting, rejoined):
        scheduler.add_request(request)
    waiting.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    rejoined.status = RequestStatus.RUNNING
    for request_id in ("aborted", "waiting", "rejoined"):
        scheduler._rejoins[request_id] = _Rejoin(computed_tokens=0, ahead=frozenset())
    scheduler._rejoins["rejoined"].hit_tokens = 4 * BLOCK
    scheduler._followers["aborted"] = _Follower()

    scheduler._update_followers()

    assert set(scheduler._rejoins) == {"waiting"}
    assert not scheduler._followers
    assert scheduler._num_rejoins == 1
    assert scheduler._num_followed_tokens == 4 * BLOCK


# PrefixAwareScheduler with TPU_PREFIX_AWARE_SHORT_PROMPTS, driven one
# finished step at a time. A step of two chunks, so two running long prefills
# fill it; a prompt is short when it takes at most one full step from each.

CHUNKS_PER_STEP = 2
SHORT = CHUNKS_PER_STEP * CHUNKS_PER_STEP * CHUNK


def short_first_scheduler(
    monkeypatch,
    on: bool = True,
    follow: bool = False,
    budget: int = CHUNKS_PER_STEP * CHUNK,
    max_num_seqs: int = 8,
    hold: bool = False,
    short_step: bool = False,
    compile_sizes: tuple[int, ...] = (16, CHUNK, 2 * CHUNK, BUDGET),
    num_blocks: int = 4000,
    take_oldest: bool = False,
) -> PrefixAwareScheduler:
    monkeypatch.setenv(
        "TPU_PREFIX_AWARE_SHORT_PROMPTS",
        ("first-short-step" if short_step else "first") if on else "off",
    )
    monkeypatch.setenv("TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX", "1" if hold else "0")
    monkeypatch.setenv(
        "TPU_PREFIX_AWARE_SHORTS_TAKE_OLDEST_CHUNK", "1" if take_oldest else "0"
    )
    return make_scheduler(
        follow,
        budget=budget,
        max_num_seqs=max_num_seqs,
        compile_sizes=list(compile_sizes),
        num_blocks=num_blocks,
    )


def add(scheduler, request_id: str, num_tokens: int, first_token: int = 0):
    prompt = list(range(first_token, first_token + num_tokens))
    scheduler.add_request(make_request(request_id, prompt))


def step(scheduler) -> dict[str, int]:
    """Schedule one step and finish it at once; the tokens each request got."""
    out = scheduler.schedule()
    req_ids = list(out.num_scheduled_tokens)
    sampled = [[] if scheduler.requests[r].is_prefill_chunk else [7] for r in req_ids]
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
        ),
    )
    return dict(out.num_scheduled_tokens)


def two_long_prefills(scheduler):
    add(scheduler, "long1", 480, first_token=20_000)
    add(scheduler, "long2", 2000, first_token=30_000)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


def test_a_short_prompt_shares_the_step_with_two_long_prefills(monkeypatch):
    off = short_first_scheduler(monkeypatch, on=False)
    on = short_first_scheduler(monkeypatch)
    for scheduler in (off, on):
        two_long_prefills(scheduler)
        add(scheduler, "short", 55, first_token=50_000)
    assert step(off) == {"long1": CHUNK, "long2": CHUNK}
    assert step(on) == {"long1": CHUNK, "long2": CHUNK - 55, "short": 55}


@pytest.mark.parametrize(
    ("budget", "limit"), [(2 * CHUNK, SHORT), (3 * CHUNK, 3 * 3 * CHUNK)]
)
def test_a_prompt_is_short_when_it_fits_in_a_long_prefills_step_of_steps(
    monkeypatch, budget, limit
):
    scheduler = short_first_scheduler(monkeypatch, budget=budget)
    assert scheduler.short_prompt_tokens == limit
    assert short_first_scheduler(monkeypatch, on=False).short_prompt_tokens is None


@pytest.mark.parametrize("budget", [CHUNK // 2, CHUNK, CHUNK + CHUNK // 2])
def test_short_prompts_first_needs_a_step_of_two_chunks(monkeypatch, budget):
    with pytest.raises(NotImplementedError, match="--long-prefill-token-threshold"):
        short_first_scheduler(monkeypatch, budget=budget)


@pytest.mark.parametrize(
    "env, value",
    [
        ("TPU_PREFIX_AWARE_SHORT_PROMPTS", "first"),
        ("TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX", "1"),
    ],
)
def test_short_prompts_first_and_the_hold_need_the_fcfs_policy(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(NotImplementedError, match="FCFS"):
        make_scheduler(False, policy="priority")


def test_the_hold_needs_prefix_caching(monkeypatch):
    monkeypatch.setenv("TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX", "1")
    with pytest.raises(NotImplementedError, match="--enable-prefix-caching"):
        make_scheduler(False, prefix_caching=False)


def test_an_unknown_short_prompts_mode_is_refused(monkeypatch):
    monkeypatch.setenv("TPU_PREFIX_AWARE_SHORT_PROMPTS", "sometimes")
    with pytest.raises(ValueError, match="sometimes"):
        make_scheduler(False)


@pytest.mark.parametrize("take_oldest", [False, True])
@pytest.mark.parametrize("short_step", [False, True])
def test_long_prefills_alone_keep_the_whole_budget(
    monkeypatch, short_step, take_oldest
):
    scheduler = short_first_scheduler(
        monkeypatch, short_step=short_step, take_oldest=take_oldest
    )
    two_long_prefills(scheduler)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


@pytest.mark.parametrize("take_oldest", [False, True])
@pytest.mark.parametrize("short_step", [False, True])
def test_a_waiting_long_prompt_gets_no_reservation(
    monkeypatch, short_step, take_oldest
):
    scheduler = short_first_scheduler(
        monkeypatch, short_step=short_step, take_oldest=take_oldest
    )
    two_long_prefills(scheduler)
    add(scheduler, "other", SHORT + 25, first_token=50_000)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


def test_a_short_prompt_keeps_its_chunks_after_its_first_step(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch)
    two_long_prefills(scheduler)
    add(scheduler, "short", 2 * CHUNK + 60, first_token=50_000)
    assert step(scheduler) == {"long1": CHUNK, "short": CHUNK}
    assert step(scheduler) == {"long1": CHUNK, "short": CHUNK}
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK - 60, "short": 60}


def test_cached_tokens_do_not_count_toward_the_short_limit(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch)
    add(scheduler, "warm", SHORT + 8 * BLOCK)
    while "warm" in scheduler.requests:
        step(scheduler)
    two_long_prefills(scheduler)
    add(scheduler, "turn", SHORT + 8 * BLOCK + 12)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK - 12, "turn": 12}


def test_blocks_the_last_step_cached_count_toward_the_short_limit(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch)
    two_long_prefills(scheduler)
    add(scheduler, "warm", SHORT)
    while "warm" in scheduler.requests:
        step(scheduler)
    add(scheduler, "turn", SHORT + 200)
    assert step(scheduler) == {"long1": CHUNK, "turn": CHUNK}


def test_the_oldest_long_prefill_keeps_its_full_chunk(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch)
    two_long_prefills(scheduler)
    for i in range(3):
        add(scheduler, f"s{i}", 47, first_token=50_000 + 100 * i)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK - 47, "s0": 47}


@pytest.mark.parametrize("short_step", [False, True])
def test_a_short_prompt_behind_a_short_prefill_takes_the_oldest_long_chunk(
    monkeypatch, short_step
):
    off = short_first_scheduler(monkeypatch, short_step=short_step)
    on = short_first_scheduler(monkeypatch, short_step=short_step, take_oldest=True)
    for scheduler in (off, on):
        two_long_prefills(scheduler)
        add(scheduler, "s1", 2 * CHUNK + 60, first_token=50_000)
        step(scheduler)
        add(scheduler, "s2", 31, first_token=60_000)
    assert step(off) == {"long1": CHUNK, "s1": CHUNK}
    assert step(on) == {"long1": CHUNK - 31, "s1": CHUNK, "s2": 31}


def test_waiting_short_prompts_take_the_oldest_long_chunk(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch, take_oldest=True)
    two_long_prefills(scheduler)
    for i in range(3):
        add(scheduler, f"s{i}", 47, first_token=50_000 + 100 * i)
    assert step(scheduler) == {"long1": 2 * CHUNK - 2 * 47, "s0": 47, "s1": 47}
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK - 47, "s2": 47}
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


def test_short_prompts_take_the_oldest_long_chunk_only_with_short_prompts_first(
    monkeypatch,
):
    with pytest.raises(NotImplementedError, match="short prompts first"):
        short_first_scheduler(monkeypatch, on=False, take_oldest=True)


def test_a_long_prefill_a_prompt_waits_for_keeps_its_chunk_from_short_prompts(
    monkeypatch,
):
    scheduler = short_first_scheduler(monkeypatch, hold=True, take_oldest=True)
    two_long_prefills(scheduler)
    add(scheduler, "child", 2000 + 16, first_token=30_000)
    add(scheduler, "short", 31, first_token=50_000)
    assert step(scheduler) == {"long1": CHUNK - 31, "long2": CHUNK, "short": 31}


def test_fewest_uncached_tokens_go_first(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch)
    two_long_prefills(scheduler)
    add(scheduler, "big", 94, first_token=50_000)
    add(scheduler, "small", 31, first_token=60_000)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK - 31, "small": 31}


def test_a_preempted_short_prompt_gets_the_reservation_first(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch)
    two_long_prefills(scheduler)
    add(scheduler, "preempted", 94, first_token=50_000)
    add(scheduler, "small", 31, first_token=60_000)
    scheduler.requests["preempted"].num_preemptions = 1
    assert step(scheduler) == {"long1": CHUNK, "preempted": CHUNK}


def test_a_long_prefill_resumed_a_step_later_stays_long(monkeypatch):
    """A preempted prefill keeps the uncached tokens it had at admission when
    it is resumed in a later step (vLLM resumes the requests it preempts
    itself no earlier than the next step), so the tokens it has left later
    on do not make it short."""
    scheduler = short_first_scheduler(monkeypatch)
    add(scheduler, "long1", 480, first_token=20_000)
    add(scheduler, "resumed", SHORT + 2 * CHUNK + 16, first_token=30_000)
    assert step(scheduler) == {"long1": CHUNK, "resumed": CHUNK}
    resumed = scheduler.requests["resumed"]
    scheduler.running.remove(resumed)
    scheduler._preempt_request(resumed, time.monotonic())
    scheduler._pause_state = PauseState.PAUSED_NEW
    assert step(scheduler) == {"long1": CHUNK}
    scheduler._pause_state = PauseState.UNPAUSED
    for _ in range(2):
        assert step(scheduler) == {"long1": CHUNK, "resumed": CHUNK}
    assert resumed.num_tokens - resumed.num_computed_tokens < SHORT
    add(scheduler, "short", 31, first_token=50_000)
    assert step(scheduler) == {"long1": CHUNK, "resumed": CHUNK - 31, "short": 31}


def test_a_failing_prefix_cache_lookup_raises_only_in_the_short_prompt_count(
    monkeypatch,
):
    """The short-prompt count raises on a failed lookup instead of counting
    the prompt as cold; admission ranking keeps its fallback to FCFS."""
    scheduler = short_first_scheduler(monkeypatch)
    add(scheduler, "r", 100)

    def fail(block_hashes, max_cache_hit_length):
        raise RuntimeError("lookup failed")

    coordinator = scheduler.kv_cache_manager.coordinator
    monkeypatch.setattr(coordinator, "find_longest_cache_hit", fail)
    with pytest.raises(RuntimeError, match="lookup failed"):
        scheduler._uncached_waiting_tokens(scheduler.requests["r"])
    assert scheduler._prefix_warmth(scheduler.requests["r"]) == 0
    assert scheduler._paw_scoring_broken is True


def test_a_paused_scheduler_reserves_nothing(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch)
    two_long_prefills(scheduler)
    add(scheduler, "short", 55, first_token=50_000)
    scheduler._pause_state = PauseState.PAUSED_NEW
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


def test_no_free_sequence_slot_means_no_reservation(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch, max_num_seqs=2)
    two_long_prefills(scheduler)
    add(scheduler, "short", 16, first_token=50_000)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


@pytest.mark.parametrize("take_oldest", [False, True])
@pytest.mark.parametrize("short_step", [False, True])
@pytest.mark.parametrize("follow", [False, True])
def test_every_prompt_token_is_computed_once(
    monkeypatch, follow, short_step, take_oldest
):
    rng = random.Random(0)
    sizes = [2, 22, 59, CHUNK + 1, 156, SHORT, SHORT + 1, 937]
    prompts = {f"r{i}": rng.choice(sizes) for i in range(24)}
    scheduler = short_first_scheduler(
        monkeypatch,
        follow=follow,
        max_num_seqs=6,
        short_step=short_step,
        take_oldest=take_oldest,
    )
    computed = dict.fromkeys(prompts, 0)
    pending = list(prompts.items())
    for n in range(1000):
        if pending and n % 2 == 0:
            request_id, num_tokens = pending.pop(0)
            add(
                scheduler,
                request_id,
                num_tokens,
                first_token=10_000 * int(request_id[1:]),
            )
        scheduled = step(scheduler)
        assert sum(scheduled.values()) <= 2 * CHUNK
        assert all(tokens <= CHUNK for tokens in scheduled.values())
        for request_id, tokens in scheduled.items():
            computed[request_id] += tokens
        if not pending and not scheduler.requests:
            break
    assert computed == prompts
    scheduler.schedule()
    assert not scheduler._uncached_admitted
    assert not scheduler._uncached_by_request


def run_with_short_prompts_first(
    monkeypatch, follow: bool, arrivals, short_step: bool = False
) -> tuple[Run, int]:
    scheduler = short_first_scheduler(
        monkeypatch, follow=follow, budget=BUDGET, short_step=short_step
    )
    return run_engine(follow, arrivals, scheduler=scheduler), scheduler._num_rejoins


@pytest.mark.parametrize("short_step", [False, True])
@pytest.mark.parametrize("seed", range(12))
def test_short_prompts_first_keeps_followers_and_every_read_correct(
    monkeypatch, seed, short_step
):
    """Short prompts first caps a long prefill's chunk before the follower
    check, so a capped prefill whose capped chunk is cached still follows."""
    off, _ = run_with_short_prompts_first(
        monkeypatch, False, tree_arrivals(seed), short_step
    )
    on, rejoins = run_with_short_prompts_first(
        monkeypatch, True, tree_arrivals(seed), short_step
    )
    assert sum(on.computed.values()) <= sum(off.computed.values())
    assert on.reads_checked > 0
    if seed == 0:
        assert rejoins > 0


def prefilled(scheduler, request_id: str) -> int:
    """Steps on until the request finishes; the prompt tokens it computed."""
    tokens = 0
    for _ in range(100):
        tokens += step(scheduler).get(request_id, 0)
        if request_id not in scheduler.requests:
            return tokens
    raise AssertionError(f"{request_id} did not finish")


def parent_and_child(scheduler):
    add(scheduler, "parent", 40 * BLOCK)
    assert step(scheduler) == {"parent": CHUNK}
    add(scheduler, "child", 41 * BLOCK)


@pytest.mark.parametrize("short_first", [False, True])
def test_a_prompt_sharing_a_running_prefix_waits_for_it(monkeypatch, short_first):
    off = short_first_scheduler(monkeypatch, on=short_first, hold=False)
    on = short_first_scheduler(monkeypatch, on=short_first, hold=True)
    for scheduler in (off, on):
        parent_and_child(scheduler)
    assert step(off) == {"parent": CHUNK, "child": CHUNK}
    assert step(on) == {"parent": CHUNK}
    assert prefilled(on, "child") == BLOCK


def test_hold_off_holds_no_prompt(monkeypatch):
    scheduler = short_first_scheduler(
        monkeypatch, follow=True, hold=False, budget=BUDGET
    )

    def hold():
        raise AssertionError("a prompt was held")

    scheduler._hold_prompts_behind_running_prefixes = hold
    run = run_engine(True, tree_arrivals(0), scheduler=scheduler)
    assert run.reads_checked > 0


def test_a_waiting_prompt_skips_the_chunk_a_follower_computes(monkeypatch):
    """Following the shared prefix alone, the prompt computes one chunk beside
    the running prefill before it follows; waiting in the queue, it computes
    only its own tokens."""
    follow = short_first_scheduler(monkeypatch, on=False, follow=True, hold=False)
    both = short_first_scheduler(monkeypatch, follow=True, hold=True)
    for scheduler in (follow, both):
        parent_and_child(scheduler)
    assert prefilled(follow, "child") == CHUNK + BLOCK
    assert prefilled(both, "child") == BLOCK


@pytest.mark.parametrize("short_first", [False, True])
def test_a_held_prompt_is_looked_up_once(monkeypatch, short_first):
    scheduler = short_first_scheduler(monkeypatch, on=short_first, hold=True)
    parent_and_child(scheduler)
    lookups = []
    look_up = scheduler._look_up_uncached
    scheduler._look_up_uncached = lambda request: lookups.append(
        request.request_id
    ) or look_up(request)
    for _ in range(3):
        assert "child" not in step(scheduler)
    assert lookups == ["child"]
    prefilled(scheduler, "child")
    assert scheduler._uncached_by_request == {}


def test_a_released_prompt_counts_the_blocks_cached_while_it_was_held(monkeypatch):
    """The child is first looked up while the parent has computed 4 of the 40
    blocks they share, too many uncached tokens for a short prompt. Once the
    parent has computed them, the child has one block left and starts as a
    short prompt beside the two long prefills, before the parent ends."""
    scheduler = short_first_scheduler(monkeypatch, hold=True)
    parent_tokens = list(range(60 * BLOCK))
    scheduler.add_request(make_request("parent", parent_tokens))
    assert step(scheduler) == {"parent": CHUNK}
    add(scheduler, "long", 2000, first_token=30_000)
    child_tokens = parent_tokens[: 40 * BLOCK] + list(range(90_000, 90_000 + BLOCK))
    scheduler.add_request(make_request("child", child_tokens))
    assert step(scheduler) == {"parent": CHUNK, "long": CHUNK}
    assert scheduler._uncached_by_request["child"] > SHORT
    parent = scheduler.requests["parent"]
    while "child" not in (scheduled := step(scheduler)):
        assert parent.num_computed_tokens <= 40 * BLOCK, "the child missed its step"
    assert scheduled["child"] == BLOCK
    assert parent.num_computed_tokens == 40 * BLOCK + CHUNK


def test_a_long_prefill_a_prompt_waits_for_keeps_its_full_chunk(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch, hold=True)
    two_long_prefills(scheduler)
    add(scheduler, "child", 2000 + 16, first_token=30_000)
    add(scheduler, "short", 31, first_token=50_000)
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


@pytest.mark.parametrize(
    "child_tokens, eagle, computed",
    [
        # Shares 10 blocks, and the EAGLE/MTP drop leaves its hit a block
        # short of them, so the block before its own is computed again.
        (list(range(10 * BLOCK)) + list(range(90_000, 90_000 + 20 * BLOCK)), True, 21),
        # Is the first 10 blocks: a hit covers at most 9 of them.
        (list(range(10 * BLOCK)), False, 1),
    ],
)
def test_a_prompt_waits_only_for_the_shared_blocks_it_can_read(
    monkeypatch, child_tokens, eagle, computed
):
    """The prompt waits while the parent computes the blocks it shares and
    can read, and starts in the step after the parent has computed them,
    long before the parent ends."""
    scheduler = short_first_scheduler(monkeypatch, on=False, hold=True)
    if eagle:
        scheduler.kv_cache_manager.coordinator.eagle_group_ids = {0}
    add(scheduler, "parent", 40 * BLOCK)
    assert step(scheduler) == {"parent": CHUNK}
    scheduler.add_request(make_request("child", child_tokens))
    parent = scheduler.requests["parent"]
    while "child" not in (scheduled := step(scheduler)):
        assert "parent" in scheduler.requests, "the child waited for the whole parent"
    assert parent.num_computed_tokens - CHUNK == 12 * BLOCK
    first = scheduled["child"]
    assert first + prefilled(scheduler, "child") == computed * BLOCK


@pytest.mark.parametrize(
    "status, skips_cache, held",
    [
        (RequestStatus.WAITING, False, True),
        (RequestStatus.PREEMPTED, False, False),
        (RequestStatus.WAITING, True, False),
    ],
)
def test_only_a_new_prompt_that_reads_the_cache_waits(
    monkeypatch, status, skips_cache, held
):
    scheduler = short_first_scheduler(monkeypatch, hold=True)
    parent_and_child(scheduler)
    child = scheduler.requests["child"]
    child.status = status
    child.skip_reading_prefix_cache = skips_cache
    carrier = scheduler._carrier_of(child, list(scheduler.running))
    assert carrier == ("parent" if held else None)


def test_a_sibling_held_in_the_queue_never_follows(monkeypatch):
    """The hold keeps the sibling waiting until the leader has cached the
    shared prefix, so it computes only its own blocks and never runs as a
    follower."""
    follow_alone = make_scheduler(True)
    alone = run_engine(True, sibling_arrivals(), scheduler=follow_alone)
    held = short_first_scheduler(monkeypatch, follow=True, hold=True, budget=BUDGET)
    both = run_engine(True, sibling_arrivals(), scheduler=held)
    assert follow_alone._num_followed_steps > 0
    assert held._num_followed_steps == 0
    assert alone.computed["follower"] > 5 * BLOCK
    assert both.computed["follower"] == 5 * BLOCK
    assert both.computed["leader"] == alone.computed["leader"]
    assert both.reads_checked > 0


def test_a_short_step_starts_a_short_prompt_in_one_chunk_of_tokens(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch, short_step=True)
    two_long_prefills(scheduler)
    add(scheduler, "short", 55, first_token=50_000)
    assert step(scheduler) == {"long1": CHUNK - 55, "short": 55}
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK}


def test_a_short_step_needs_room_for_a_waiting_short_prompt(monkeypatch):
    """While a short prefill's chunk fills the short step, a waiting short
    prompt cannot start in it, so the step keeps its full size."""
    scheduler = short_first_scheduler(monkeypatch, short_step=True)
    two_long_prefills(scheduler)
    add(scheduler, "s1", 2 * CHUNK + 60, first_token=50_000)
    assert step(scheduler) == {"s1": CHUNK}
    add(scheduler, "s2", 31, first_token=60_000)
    assert step(scheduler) == {"long1": CHUNK, "s1": CHUNK}
    assert step(scheduler) == {"long1": CHUNK, "long2": CHUNK - 60, "s1": 60}
    assert step(scheduler) == {"long1": CHUNK - 31, "s2": 31}


@pytest.mark.parametrize("short_step", [False, True])
def test_a_short_prompt_the_kv_cache_cannot_hold_does_not_stall_the_prefill(
    monkeypatch, short_step
):
    """31 blocks hold the long prompt's 30 and leave 3 once it has computed 28.
    The short prompt then needs 4, so it cannot start. A short step capping
    the long prefill at 0 for it would schedule nothing on every step: the
    prefill never reaches allocate_slots and frees no block."""
    scheduler = short_first_scheduler(monkeypatch, short_step=short_step, num_blocks=32)
    add(scheduler, "long", 30 * BLOCK, first_token=20_000)
    while scheduler.requests["long"].num_computed_tokens < 28 * BLOCK:
        assert step(scheduler) == {"long": CHUNK}
    add(scheduler, "short", CHUNK, first_token=50_000)
    assert prefilled(scheduler, "short") == CHUNK
    assert "long" not in scheduler.requests


def test_a_short_step_needs_a_running_long_prefill(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch, short_step=True)
    add(scheduler, "a", 31, first_token=10_000)
    add(scheduler, "b", 47, first_token=20_000)
    add(scheduler, "c", 94, first_token=30_000)
    assert sum(step(scheduler).values()) == 2 * CHUNK


def test_a_decoding_request_keeps_the_full_step(monkeypatch):
    scheduler = short_first_scheduler(monkeypatch, short_step=True)
    scheduler.add_request(make_request("decoder", list(range(16)), max_tokens=50))
    step(scheduler)
    add(scheduler, "long1", 480, first_token=20_000)
    add(scheduler, "long2", 2000, first_token=30_000)
    step(scheduler)
    add(scheduler, "short", 55, first_token=50_000)
    scheduled = step(scheduler)
    assert "short" in scheduled
    assert sum(scheduled.values()) == 2 * CHUNK


@pytest.mark.parametrize(
    "compile_sizes, short_step",
    [((16, 32, 96, 128), 96), ((16, CHUNK, 128), CHUNK)],
)
def test_the_short_step_is_the_smallest_compiled_size_for_a_chunk(
    monkeypatch, compile_sizes, short_step
):
    scheduler = short_first_scheduler(
        monkeypatch, short_step=True, compile_sizes=compile_sizes
    )
    assert scheduler.short_step_tokens == short_step
    assert short_first_scheduler(monkeypatch).short_step_tokens is None


@pytest.mark.parametrize("compile_sizes", [(16, 32), (16, 32, 2 * CHUNK)])
def test_a_short_step_needs_a_compiled_size_below_the_full_step(
    monkeypatch, compile_sizes
):
    with pytest.raises(NotImplementedError, match="compile_sizes"):
        short_first_scheduler(monkeypatch, short_step=True, compile_sizes=compile_sizes)
