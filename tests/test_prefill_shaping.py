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
"""The prefill shaping options of PrefixAwareScheduler (``tpu_scheduler``
``prefill_chunk_cap``, ``srpf_running`` and ``admission_tiers``).

The policies are tested as functions of core/prefill_shaping.py, their hooks
on PrefixAwareScheduler stand-ins, and the chunk cap on a real scheduler
(stock ``Scheduler.schedule()`` over a real KVCacheManager, with and without
a Mamba "align" group) driven step by step.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core import prefix_aware_scheduler, scheduler_options
from vllm_torchtpu.core.prefill_shaping import (
    TIER_COLD,
    TIER_FREE_BLOCK_FIT,
    TIER_PREEMPTED,
    TIER_WARM,
    ChunkCapLimits,
    PrefillChunkCap,
    admission_tier,
    free_blocks_for_fit,
    remaining_prompt_tokens,
)
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

SHAPING = ("prefill_chunk_cap", "srpf_running", "admission_tiers")

# ---------------- options ----------------


def test_the_shaping_options_default_off():
    opts = scheduler_options.parse({})
    assert not (opts.prefill_chunk_cap or opts.srpf_running or opts.admission_tiers)
    assert opts.prefix_aware_levers is False


@pytest.mark.parametrize("name", SHAPING)
def test_a_shaping_option_parses_with_the_levers(name):
    opts = scheduler_options.parse({name: True, "prefix_aware_levers": True})
    assert getattr(opts, name) is True


@pytest.mark.parametrize("levers", [None, False])
@pytest.mark.parametrize("name", SHAPING)
def test_a_shaping_option_is_refused_without_the_levers(name, levers):
    raw = {name: True}
    if levers is not None:
        raw["prefix_aware_levers"] = levers
    with pytest.raises(ValueError, match=f"{name} .*prefix_aware_levers to true"):
        scheduler_options.parse(raw)


@pytest.mark.parametrize("name", SHAPING)
def test_a_shaping_option_is_a_strict_bool(name):
    with pytest.raises(ValueError, match="must be a bool"):
        scheduler_options.parse({name: 1, "prefix_aware_levers": True})


# ---------------- chunk cap limits ----------------


@pytest.mark.parametrize(
    ("budget", "max_num_seqs", "expected"),
    [
        # (cold_chunk_cap, cold_threshold, cold_min_decoders, step_total_cap)
        (8192, 16, (2048, 8192, 12, 4096)),
        (8192, 64, (2048, 8192, 12, 4096)),
        (8192, 12, (2048, 8192, 9, 4096)),
        (8192, 14, (2048, 8192, 10, 4096)),
        (16384, 8, (4096, 16384, 6, 8192)),
        (2048, 4, (512, 2048, 3, 1024)),
        (1024, 1, (512, 1024, 1, 1024)),
    ],
)
def test_the_limits_derive_from_the_budget_and_max_num_seqs(
    budget, max_num_seqs, expected
):
    assert ChunkCapLimits.derive(budget, max_num_seqs) == ChunkCapLimits(*expected)


# ---------------- PrefillChunkCap ----------------

BLOCK = 256
# 2048 / 8192 / 12 decoders / 4096.
LIMITS = ChunkCapLimits.derive(8192, 16)


def _decoder(i: int, tokens: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=f"d{i}",
        num_prompt_tokens=100,
        num_tokens=101,
        num_computed_tokens=100,
        num_tokens_with_spec=100 + tokens,
        num_output_placeholders=0,
    )


def _prefill(rid: str, prompt: int, computed: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        num_computed_tokens=computed,
    )


def _cap(num_decoders: int, prefilling=(), tokens: int = 1) -> PrefillChunkCap:
    cap = PrefillChunkCap(LIMITS, BLOCK)
    cap.begin_step(
        [_decoder(i, tokens) for i in range(num_decoders)] + list(prefilling)
    )
    return cap


@pytest.mark.parametrize(
    ("num_decoders", "expected"),
    # From 12 decoders a cold prefill gets 2048 minus the decode tokens.
    [(12, 2048 - 12), (20, 2048 - 20), (11, 8000), (0, 8000)],
)
def test_a_cold_prefill_is_capped_from_cold_min_decoders(num_decoders, expected):
    cold = _prefill("cold", 20_000)
    assert _cap(num_decoders).cap(cold, 0, 8000, has_waiting=True) == expected


def test_a_warm_prefill_takes_its_chunk_alone():
    # 8192 tokens left is not more than the cold threshold.
    warm = _prefill("warm", 8192 + 640, computed=640)
    assert _cap(12).cap(warm, 640, 8000, has_waiting=True) == 8000


@pytest.mark.parametrize(("has_waiting", "expected"), [(True, 2048 - 2), (False, 8000)])
def test_a_running_cold_prefill_is_capped_while_requests_wait(has_waiting, expected):
    cold = _prefill("cold", 30_000, computed=2048)
    cap = _cap(2, prefilling=[cold])
    assert cap.cap(cold, 2048, 8000, has_waiting=has_waiting) == expected
    # A cold prefill admitted from the waiting queue is not running yet.
    new = _prefill("new", 30_000)
    assert cap.cap(new, 0, 8000, has_waiting=True) == 8000


def test_the_cold_cap_keeps_at_least_a_block():
    cold = _prefill("cold", 20_000)
    # 12 decoders x 200 tokens leave nothing of the 2048-token cold cap.
    assert _cap(12, tokens=200).cap(cold, 0, 8000, has_waiting=False) == BLOCK


def test_a_prefill_behind_another_fills_the_step_total():
    cap = _cap(12)
    assert cap.cap(_prefill("a", 20_000), 0, 8000, has_waiting=False) == 2036
    # 4096 - (12 decode + 2036) tokens are left for a warm prefill.
    assert cap.cap(_prefill("b", 5000), 0, 5000, has_waiting=False) == 2048
    # The step holds 4096; a third prefill is held ...
    assert cap.cap(_prefill("c", 5000), 0, 5000, has_waiting=False) == 0
    # ... and so is one that would finish within the room, since none is left.
    assert cap.cap(_prefill("d", 100, 99), 99, 1, has_waiting=False) == 0


@pytest.mark.parametrize(
    ("prompt", "num_new_tokens", "expected"),
    [(5000, 4900, 0), (105, 5, 5), (108, 8, 8), (109, 9, 0)],
)
def test_a_prefill_behind_another_is_held_under_a_block_unless_it_finishes(
    prompt, num_new_tokens, expected
):
    cap = _cap(12)
    cap.cap(_prefill("a", 20_000), 0, 8000, has_waiting=False)  # 2036
    cap.cap(_prefill("b", 5000), 0, 2040, has_waiting=False)  # 2040
    # 4096 - 4088 = 8 tokens are left.
    late = _prefill("late", prompt, 100)
    assert cap.cap(late, 100, num_new_tokens, has_waiting=False) == expected


@pytest.mark.parametrize(
    ("first", "expected"),
    [
        # 2048 - (12 + 1000) tokens of the cold cap are left: at least a block.
        (1000, 2048 - 1012),
        # Under a block left: the cold cap minus the decode tokens, within the
        # 4096 - 1912 tokens the step total leaves.
        (1900, 2048 - 12),
    ],
)
def test_a_cold_prefill_behind_another_takes_what_the_cold_cap_leaves(first, expected):
    cap = _cap(12)
    assert cap.cap(_prefill("a", first), 0, first, has_waiting=False) == first
    assert cap.cap(_prefill("cold", 20_000), 0, 8000, has_waiting=False) == expected


def test_no_cap_after_a_prefill_with_few_decoders():
    cap = _cap(11)
    assert cap.cap(_prefill("a", 5000), 0, 5000, has_waiting=True) == 5000
    assert cap.cap(_prefill("b", 5000), 0, 3000, has_waiting=True) == 3000


@pytest.mark.parametrize(
    ("request_", "start", "num_new_tokens"),
    [
        (_decoder(99), 100, 1),
        (_prefill("done", 4096, 4096), 4096, 1),
        (_prefill("empty", 20_000), 0, 0),
    ],
)
def test_requests_past_their_prefill_pass_through(request_, start, num_new_tokens):
    cap = _cap(12)
    assert cap.cap(request_, start, num_new_tokens, True) == num_new_tokens
    # They do not count as a prefill of the step.
    assert cap.cap(_prefill("cold", 20_000), 0, 8000, True) == 2036


def test_a_resumed_request_prefills_past_its_prompt():
    """A request resumed from preemption replays its output tokens as
    prefill; the cold threshold counts them."""
    resumed = SimpleNamespace(
        request_id="r", num_prompt_tokens=8000, num_tokens=9000, num_computed_tokens=0
    )
    assert _cap(12).cap(resumed, 0, 8192, has_waiting=False) == 2036


def test_record_keeps_the_chunk_the_scheduler_took():
    cap = _cap(12)
    a = _prefill("a", 20_000)
    assert cap.cap(a, 0, 8000, has_waiting=False) == 2036
    # The block alignment cut it to 1792; the decoders are not recorded.
    cap.record(a, 1792)
    cap.record(_decoder(0), 1)
    # 4096 - (12 + 1792) tokens are left.
    assert cap.cap(_prefill("b", 5000), 0, 5000, has_waiting=False) == 2292


def test_begin_step_forgets_the_last_step():
    cap = _cap(12)
    cap.cap(_prefill("a", 20_000), 0, 8000, has_waiting=False)
    cap.begin_step([_decoder(i) for i in range(12)])
    assert cap.cap(_prefill("b", 5000), 0, 5000, has_waiting=False) == 5000


def test_a_decoder_counts_at_least_one_token():
    """A decoder whose next token is already scheduled (async scheduling)
    still takes a token of the step."""
    assert _cap(12, tokens=0).cap(_prefill("cold", 20_000), 0, 8000, False) == 2036


def test_a_request_replaying_its_output_is_a_prefill_not_a_decoder():
    """A request resumed from preemption replays its output tokens as a
    prefill: it adds no decode tokens, and it is capped as a cold prefill."""
    replaying = SimpleNamespace(
        request_id="r",
        num_prompt_tokens=300,
        num_tokens=12_300,
        num_computed_tokens=2048,
        num_tokens_with_spec=12_300,
        num_output_placeholders=0,
    )
    cap = _cap(12, [replaying])
    assert cap.cap(replaying, 2048, 8000, has_waiting=False) == 2048 - 12
    # A cold prompt behind it still takes the cold cap.
    assert cap.cap(_prefill("cold", 20_000), 0, 8000, has_waiting=False) == 2036


# ---------------- SRPF ----------------


def test_remaining_prompt_tokens():
    assert remaining_prompt_tokens(_prefill("p", 1000, 300)) == 700
    assert remaining_prompt_tokens(_decoder(0)) == 0


def _step_scheduler(monkeypatch, running, consumer, **switches):
    """A PrefixAwareScheduler stand-in whose vLLM schedule() records the
    running order it saw."""
    seen = {}

    def fake_schedule(self, *args, **kwargs):
        seen["order"] = [r.request_id for r in self.running]
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._is_kv_consumer = consumer
    s.running = list(running)
    s.waiting = FCFSRequestQueue()
    s.kv_holding_waiting = FCFSRequestQueue()
    s.max_num_scheduled_tokens = 8192
    s._inflight_prefills = set()
    s._decode_throttle_switch = False
    s._kv_prefetch_n = 0
    s._kv_prefetch_cap = None
    s._promote_warmest_switch = False
    for name, value in switches.items():
        setattr(s, name, value)
    return s, seen


MIXED = [
    _prefill("p_long", 30_000, 2048),
    _decoder(1),
    _prefill("p_short", 9000, 4096),
    _decoder(2),
    _prefill("p_mid", 20_000, 0),
]


@pytest.mark.parametrize("consumer", [False, True], ids=["aggregated", "kv_consumer"])
def test_srpf_runs_prefills_fewest_remaining_first(monkeypatch, consumer):
    s, seen = _step_scheduler(monkeypatch, MIXED, consumer, _srpf_running=True)
    running_list = s.running
    assert s.schedule() == "out"
    assert seen["order"] == ["d1", "d2", "p_short", "p_mid", "p_long"]
    assert s.running is running_list


@pytest.mark.parametrize(
    "running",
    [MIXED, [_prefill("p", 9000), _decoder(1)]],
    ids=["off", "one_prefill"],
)
def test_srpf_leaves_running_alone_off_or_with_one_prefill(monkeypatch, running):
    """Without the levers the order is vLLM's unless SRPF has prefills to
    sort (decoders first is one of the levers)."""
    srpf = running is not MIXED
    s, seen = _step_scheduler(monkeypatch, running, False, _srpf_running=srpf)
    s.schedule()
    assert seen["order"] == [r.request_id for r in running]


def test_schedule_starts_the_chunk_cap_step_on_the_running_set(monkeypatch):
    cap = PrefillChunkCap(LIMITS, BLOCK)
    running = [_decoder(i) for i in range(12)] + [_prefill("p", 30_000, 2048)]
    s, _ = _step_scheduler(monkeypatch, running, False, _chunk_cap=cap)
    s.schedule()
    # 12 decoders: the running cold prefill is capped.
    assert cap.cap(running[-1], 2048, 8000, has_waiting=False) == 2036


@pytest.mark.parametrize(
    ("waiting", "holding", "expected"),
    [(False, False, 8000), (True, False, 2048 - 4), (False, True, 2048 - 4)],
    ids=["none", "waiting", "kv_holding"],
)
def test_a_running_cold_prefill_is_capped_while_either_queue_waits(
    monkeypatch, waiting, holding, expected
):
    """Four decoders are too few for the caps, but a request in either
    waiting queue, a KV load in flight included, caps a running cold
    prefill."""
    prefill = _prefill("p", 30_000, 2048)
    running = [_decoder(i) for i in range(4)] + [prefill]
    cap = PrefillChunkCap(LIMITS, BLOCK)
    s, _ = _step_scheduler(monkeypatch, running, False, _chunk_cap=cap)
    s.need_mamba_block_aligned_split = True
    if waiting:
        s.waiting.add_request(_waiting("w", 1000))
    if holding:
        s.kv_holding_waiting.add_request(_waiting("h", 1000))
    s._begin_prefill_shaping_step()
    assert s._cap_prefill_chunk(prefill, 2048, 8000) == expected


@pytest.mark.parametrize("mamba", [True, False], ids=["mamba_align", "attention"])
def test_a_held_prefill_ends_the_waiting_loop_only_under_the_mamba_split(
    monkeypatch, mamba
):
    """Under the Mamba split a 0 ends vLLM's waiting loop and is its stop
    reason; without it vLLM skips the request and keeps scanning, so the
    hold names no stop."""
    cap = PrefillChunkCap(LIMITS, BLOCK)
    running = [_decoder(i) for i in range(12)]
    s, _ = _step_scheduler(monkeypatch, running, False, _chunk_cap=cap)
    s.need_mamba_block_aligned_split = mamba
    s._levers_switch = True
    s._step_wait_event = None
    s._kv_prefetch_in_waiting = False
    s._begin_prefill_shaping_step()
    # A warm prefill leaves 84 tokens of the step total.
    assert cap.cap(_prefill("a", 4000), 0, 4000, has_waiting=True) == 4000
    held = _prefill("b", 4000)
    held.status = RequestStatus.WAITING
    if mamba:
        assert s._cap_prefill_chunk(held, 0, 4000) == 0
        assert s._step_wait_event == "chunk_cap0"
    else:
        assert s._reserve_prefill_lookahead(held, 0, 4000) == 0
        assert s._step_wait_event is None


def test_the_cap_keeps_the_chunk_the_lookahead_reserve_leaves(monkeypatch):
    """The MTP prefill lookahead reserve can trim the chunk after the Mamba
    split; the next prefill counts the trimmed chunk."""
    cap = PrefillChunkCap(LIMITS, BLOCK)
    running = [_decoder(i) for i in range(12)]
    s, _ = _step_scheduler(monkeypatch, running, False, _chunk_cap=cap)
    s.need_mamba_block_aligned_split = True
    s._kv_prefetch_in_waiting = False
    s.num_prefill_lookahead = 4
    s._begin_prefill_shaping_step()
    a = _prefill("a", 1794)
    assert cap.cap(a, 0, 1794, has_waiting=False) == 1794
    # The split ends it on a block; 2 tokens are left, under the 4-token
    # lookahead, so the reserve leaves 1790.
    cap.record(a, 1792)
    assert s._reserve_prefill_lookahead(a, 0, 1792) == 1790
    assert cap.cap(_prefill("b", 5000), 0, 5000, has_waiting=False) == 4096 - 12 - 1790


class _Retention:
    """Records the hybrid-state retention's split calls and stops every
    chunk at ``stop``."""

    def __init__(self, stop: int) -> None:
        self.stop = stop
        self.calls: list[tuple] = []

    def prepare_prefill_split(self, request, start, num_new_tokens):
        self.calls.append(("prepare", start, num_new_tokens))

    def adjust_prefill_split(self, request, start, aligned):
        self.calls.append(("adjust", start, aligned))
        return min(aligned, self.stop - start)


def test_the_chunk_cap_runs_before_the_retention_split(monkeypatch):
    """With hybrid_state_retention on too, vLLM's block alignment and the
    retention stops split the capped chunk, and the cap keeps the chunk
    they leave."""
    cap = PrefillChunkCap(LIMITS, BLOCK)
    cap.begin_step([_decoder(i) for i in range(12)])
    retention = _Retention(stop=1024)
    seen = []

    def vllm_split(self, request, num_new_tokens, local=0, external=0):
        seen.append(num_new_tokens)
        return num_new_tokens // BLOCK * BLOCK

    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", vllm_split)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._chunk_cap = cap
    s._retention = retention
    s.mamba_has_prefill_checkpoint_blocks = False
    s.need_mamba_block_aligned_split = True
    s.waiting = FCFSRequestQueue()
    s.kv_holding_waiting = FCFSRequestQueue()
    assert s._mamba_block_aligned_split(_prefill("cold", 20_000), 8000) == 1024
    # 2048 minus 12 decode tokens, ended on a block, then at the stop.
    assert seen == [2036]
    assert retention.calls == [("prepare", 0, 2036), ("adjust", 0, 1792)]
    # A prefill behind it fills the 4096 step total around the 1024 kept.
    assert cap.cap(_prefill("w", 4000), 0, 4000, has_waiting=False) == 4096 - 12 - 1024


# ---------------- admission tiers ----------------


def test_the_default_ranking_uses_the_tiered_ranking_values():
    assert TIER_PREEMPTED == prefix_aware_scheduler._TIER_PREEMPTED
    assert TIER_WARM == prefix_aware_scheduler._TIER_NORMAL


def _waiting(rid, num_tokens, preemptions=0, arrival=0.0):
    return SimpleNamespace(
        request_id=rid,
        num_tokens=num_tokens,
        num_preemptions=preemptions,
        arrival_time=arrival,
        num_computed_tokens=0,
        skip_reading_prefix_cache=False,
        block_hashes=[object()],
    )


@pytest.mark.parametrize(
    ("num_tokens", "warmth", "preemptions", "fit", "tier"),
    [
        (900, 800, 0, 7, TIER_FREE_BLOCK_FIT),  # 100 uncached tokens: 7 blocks
        (900, 800, 1, 7, TIER_FREE_BLOCK_FIT),
        (900, 800, 0, 6, TIER_WARM),
        (900, 800, 0, None, TIER_WARM),
        (900, 0, 0, 1000, TIER_COLD),  # a cold request never fits
        (900, 0, 1, None, TIER_PREEMPTED),
        (900, 800, 1, None, TIER_PREEMPTED),
    ],
)
def test_admission_tier(num_tokens, warmth, preemptions, fit, tier):
    request = _waiting("r", num_tokens, preemptions)
    assert admission_tier(request, warmth, fit, 16) == tier


@pytest.mark.parametrize(
    ("free", "head_cold", "expected"),
    [(10, 161, 10), (11, 161, None), (None, 161, None), (0, 0, None)],
)
def test_free_blocks_for_fit(free, head_cold, expected):
    assert free_blocks_for_fit(free, head_cold, 16) == expected


def _ranking_scheduler(requests, warmth, free_blocks, tiers):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s.waiting = FCFSRequestQueue()
    for r in requests:
        s.waiting.add_request(r)
    s.policy = SchedulingPolicy.FCFS
    s.block_size = 16
    s._paw_n = s._paw_cold_tokens = s._paw_prompt_tokens = 0
    s._paw_scoring_broken = False
    s._admission_tiers = tiers

    def find_hit(block_hashes, max_len):
        (r,) = (r for r in requests if r.block_hashes is block_hashes)
        return ((), warmth.get(r.request_id, 0), 0)

    pool = SimpleNamespace(get_num_free_blocks=MagicMock(return_value=free_blocks))
    s.kv_cache_manager = SimpleNamespace(
        enable_caching=True,
        coordinator=SimpleNamespace(find_longest_cache_hit=find_hit),
        block_pool=pool,
    )
    return s


@pytest.mark.parametrize(("tiers", "first"), [(False, "cold"), (True, "warm")])
def test_warm_before_cold(tiers, first):
    cold = _waiting("cold", 1000)
    warm = _waiting("warm", 5000)  # 4900 uncached tokens, but warm
    s = _ranking_scheduler([cold, warm], {"warm": 100}, 10_000, tiers)
    s._promote_warmest(s.waiting)
    assert s.waiting.peek_request().request_id == first


@pytest.mark.parametrize(
    ("tiers", "free_blocks", "first"),
    [
        (False, 10, "preempted"),
        (True, 10, "fits"),
        # The head fits 1000 free blocks: no fit tier.
        (True, 1000, "preempted"),
    ],
)
def test_a_warm_request_that_fits_goes_first_when_the_head_does_not(
    tiers, free_blocks, first
):
    head = _waiting("head", 2000)  # 125 blocks
    preempted = _waiting("preempted", 1000, preemptions=1)
    fits = _waiting("fits", 900)  # 100 uncached tokens: 7 blocks
    s = _ranking_scheduler([head, preempted, fits], {"fits": 800}, free_blocks, tiers)
    s._promote_warmest(s.waiting)
    assert s.waiting.peek_request().request_id == first
    assert s.kv_cache_manager.block_pool.get_num_free_blocks.called is tiers


# ---------------- PrefixAwareScheduler.__init__ ----------------


def _init(monkeypatch, options, role=None):
    kv_transfer_config = (
        None
        if role is None
        else KVTransferConfig(kv_connector="TPURaidenConnector", kv_role=role)
    )

    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            additional_config={"tpu_scheduler": options},
        )
        self.connector = None
        self.kv_cache_manager = MagicMock()
        self.max_num_scheduled_tokens = 8192
        self.max_num_running_reqs = 14
        self.block_size = 640

    monkeypatch.setattr(AsyncScheduler, "__init__", fake_init)
    return PrefixAwareScheduler()


def test_init_leaves_the_shaping_off_by_default(monkeypatch):
    s = _init(monkeypatch, {})
    assert s._chunk_cap is None
    assert s._srpf_running is False
    assert s._admission_tiers is False


def test_init_reads_the_shaping_options(monkeypatch):
    options = dict.fromkeys(SHAPING + ("prefix_aware_levers",), True)
    s = _init(monkeypatch, options)
    assert s._srpf_running is True
    assert s._admission_tiers is True
    assert s._chunk_cap is not None
    assert s._chunk_cap.limits == ChunkCapLimits.derive(8192, 14)
    assert s._chunk_cap.block_size == 640


@pytest.mark.parametrize("name", SHAPING)
def test_engine_start_refuses_a_shaping_option_without_the_levers(monkeypatch, name):
    with pytest.raises(ValueError, match="prefix_aware_levers"):
        _init(monkeypatch, {name: True})


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_both"])
def test_the_step_cap_ships_with_the_resend_trim(monkeypatch, role):
    """prefill_chunk_cap needs prefix_aware_levers, so an engine that is not
    a KV consumer and caps prefill chunks also trims the token ids it
    re-sends for running requests to the ones the workers read."""
    options = {"prefix_aware_levers": True, "prefill_chunk_cap": True}
    s = _init(monkeypatch, options, role)
    assert s._chunk_cap is not None

    def cached(self, *args, **kwargs):
        return CachedRequestData(
            req_ids=["a"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={"a": [1, 2, 3, 4, 5]},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[2],
        )

    monkeypatch.setattr(AsyncScheduler, "_make_cached_request_data", cached)
    assert s._make_cached_request_data().all_token_ids == {"a": [4, 5]}


# ---------------- the chunk cap on a real scheduler ----------------

E2E_BLOCK = 256
E2E_BUDGET = 8192
# Three quarters of 12 running requests: 9 decoders turn the caps on.
E2E_SEQS = 12
E2E_DECODERS = 9


def _model_dir(tmp_path):
    d = tmp_path / "tiny-opt"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 32768,
                "vocab_size": 50272,
                "torch_dtype": "float16",
                "word_embed_proj_dim": 64,
                "do_layer_norm_before": True,
            }
        )
    )
    return str(d)


def _real_scheduler(tmp_path, hybrid: bool, options: dict) -> PrefixAwareScheduler:
    """A PrefixAwareScheduler over vLLM's own scheduler and KV cache manager,
    on an aggregated engine (no KV connector). ``hybrid`` adds a Mamba
    "align" group, which turns the block-aligned split on."""
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=E2E_SEQS,
            max_num_batched_tokens=E2E_BUDGET,
            max_model_len=32768,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        model_config=ModelConfig(
            model=_model_dir(tmp_path),
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(
            block_size=E2E_BLOCK,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        device_config=DeviceConfig("cpu"),
        additional_config={"tpu_scheduler": options},
    )
    num_blocks = 4000
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=E2E_BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        )
    ]
    if hybrid:
        mamba = MambaSpec(
            block_size=E2E_BLOCK,
            shapes=(1, 1),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
        groups.append(KVCacheGroupSpec(["mamba"], mamba))
    scheduler = PrefixAwareScheduler(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=E2E_BLOCK,
        log_stats=True,
    )
    assert scheduler.need_mamba_block_aligned_split is hybrid
    return scheduler


def _request(rid: str, num_tokens: int) -> Request:
    init_none_hash(sha256)
    seed = sum(ord(c) for c in rid) * 7919
    return Request(
        request_id=rid,
        prompt_token_ids=[(seed + i) % 50000 + 1 for i in range(num_tokens)],
        sampling_params=SamplingParams(max_tokens=10_000, ignore_eos=True),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(E2E_BLOCK, sha256),
    )


def _step(s: PrefixAwareScheduler) -> dict[str, int]:
    """One schedule() and its model output; a request still prefilling
    samples nothing."""
    out = s.schedule()
    ids = list(out.num_scheduled_tokens)
    s.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[
                [] if s.requests[rid].is_prefill_chunk else [0] for rid in ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
        ),
    )
    return dict(out.num_scheduled_tokens)


def _with_decoders(tmp_path, hybrid, cap_on):
    options = {"prefix_aware_levers": True, "prefill_chunk_cap": True} if cap_on else {}
    s = _real_scheduler(tmp_path, hybrid, options)
    decoders = [f"d{i}" for i in range(E2E_DECODERS)]
    for rid in decoders:
        s.add_request(_request(rid, 300))
    for _ in range(4):
        _step(s)
    assert all(s.requests[rid].num_computed_tokens > 300 for rid in decoders)
    return s, decoders


@pytest.mark.parametrize("hybrid", [True, False], ids=["mamba_align", "attention"])
@pytest.mark.parametrize("cap_on", [False, True], ids=["off", "on"])
def test_a_cold_prompt_prefills_in_capped_chunks_beside_the_decoders(
    tmp_path, hybrid, cap_on
):
    s, decoders = _with_decoders(tmp_path, hybrid, cap_on)
    s.add_request(_request("cold", 20_000))
    cold = s.requests["cold"]
    # (prompt tokens left, chunk) per step.
    chunks = []
    while cold.num_computed_tokens < cold.num_prompt_tokens:
        left = cold.num_prompt_tokens - cold.num_computed_tokens
        scheduled = _step(s)
        # The decoders take a token every step.
        assert all(scheduled[rid] == 1 for rid in decoders)
        chunks.append((left, scheduled["cold"]))
    assert sum(chunk for _, chunk in chunks) == 20_000

    def aligned(tokens):
        return tokens // E2E_BLOCK * E2E_BLOCK if hybrid else tokens

    if not cap_on:
        # The cold prompt takes the rest of the budget.
        assert chunks[0][1] == aligned(E2E_BUDGET - E2E_DECODERS)
        return
    # While more than the budget (the cold threshold) is left: 2048 minus
    # the decode tokens, ended on a block by the Mamba split. Then the rest.
    capped = [chunk for left, chunk in chunks if left > E2E_BUDGET]
    rest = [chunk for left, chunk in chunks if left <= E2E_BUDGET]
    assert capped == [aligned(2048 - E2E_DECODERS)] * len(capped)
    assert len(capped) == (7 if hybrid else 6)
    assert rest[0] > 2048


@pytest.mark.parametrize(
    ("hybrid", "expected"),
    [
        # The third prefill takes the 4096 - (9 + 1792 + 1792) tokens left,
        # cut to a block.
        (True, [1792, 1792, 256]),
        # 9 + 2039 + 2039 leave 9 tokens: the third prefill is held.
        (False, [2039, 2039, 0]),
    ],
    ids=["mamba_align", "attention"],
)
def test_prefills_of_a_step_stay_within_the_step_total(tmp_path, hybrid, expected):
    s, decoders = _with_decoders(tmp_path, hybrid, cap_on=True)
    colds = ["c0", "c1", "c2"]
    for rid in colds:
        s.add_request(_request(rid, 20_000))
    scheduled = _step(s)
    assert [scheduled.get(rid, 0) for rid in colds] == expected
    assert sum(scheduled.values()) <= 4096
    assert all(scheduled[rid] == 1 for rid in decoders)


@pytest.mark.parametrize(
    ("hybrid", "chunk"), [(True, 1792), (False, 2039)], ids=["mamba_align", "attention"]
)
def test_a_resumed_request_replays_its_output_beside_a_cold_prompt(
    tmp_path, hybrid, chunk
):
    """A request preempted after more output than the budget replays it as
    a prefill: the cap counts no decode tokens for it, so it and a cold
    prompt both take the cold cap's chunk."""
    s, decoders = _with_decoders(tmp_path, hybrid, cap_on=True)
    s.add_request(_request("resumed", 300))
    _step(s)
    resumed = s.requests["resumed"]
    for i in range(12_000):
        resumed.append_output_token_ids(7 + i % 1000)
    s.running.remove(resumed)
    s._preempt_request(resumed, 0.0)
    s.add_request(_request("cold", 20_000))
    for _ in range(2):
        scheduled = _step(s)
        assert scheduled["resumed"] == scheduled["cold"] == chunk
        assert all(scheduled[rid] == 1 for rid in decoders)
