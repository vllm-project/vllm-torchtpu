import json
from types import SimpleNamespace

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
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.core.prefix_aware_short_first_scheduler import (
    PrefixAwareShortFirstScheduler,
)
from vllm_torchtpu.core.short_prefill_first_scheduler import (
    ShortPrefillFirstScheduler,
)

BUDGET = 4096
BLOCK = 16
# The served prefill leg: each request at most half the step budget a step,
# and prompts of at most half the step budget count as short.
LONG = 2048
SHORT = 2048


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-llama")
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32000,
                "max_position_embeddings": 131072,
                "torch_dtype": "bfloat16",
            }
        )
    )
    return str(path)


def _scheduler(
    cls,
    model_dir,
    monkeypatch,
    num_blocks=40_000,
    max_long=0,
    long_tokens=16_384,
    step_fill=False,
):
    monkeypatch.setenv("TPU_SHORT_PREFILL_FIRST_TOKENS", str(SHORT))
    monkeypatch.setenv("TPU_PREFILL_MAX_LONG_RUNNING", str(max_long))
    monkeypatch.setenv("TPU_PREFILL_LONG_TOKENS", str(long_tokens))
    monkeypatch.setenv("TPU_PREFILL_STEP_FILL", "1" if step_fill else "0")
    model_config = ModelConfig(
        model=model_dir, skip_tokenizer_init=True, max_model_len=131072
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=BUDGET,
            max_num_seqs=16,
            max_model_len=131072,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=LONG,
            async_scheduling=True,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK, cache_dtype="auto", enable_prefix_caching=True
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["l0"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    config.cache_config.num_gpu_blocks = num_blocks
    return cls(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=BLOCK,
    )


init_none_hash(sha256)
_HASHER = get_request_block_hasher(BLOCK, sha256)


def _request(req_id, prompt, first_token=0, max_tokens=4):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + prompt)),
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        block_hasher=_HASHER,
    )


def _step(scheduler):
    output = scheduler.schedule()
    scheduled = dict(output.num_scheduled_tokens)
    req_ids = list(scheduled)
    sampled = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        sampled.append([] if request.is_prefill_chunk else [7])
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
        ),
    )
    return scheduled


def _loaded_request(scheduler, req_id, prompt, computed, first_token):
    """A prompt whose tier load finished, waiting in skipped_waiting with
    ``computed`` of its ``prompt`` tokens computed."""
    loaded = _request(req_id, prompt, first_token=first_token)
    loaded.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    loaded.num_computed_tokens = computed
    scheduler.requests[req_id] = loaded
    scheduler.skipped_waiting.add_request(loaded)
    scheduler.finished_recving_kv_req_ids.add(req_id)

    # Promotion without the connector's block caching, which these
    # schedulers do not have.
    def promote(request, scheduler=scheduler):
        if request.request_id not in scheduler.finished_recving_kv_req_ids:
            return False
        scheduler.finished_recving_kv_req_ids.remove(request.request_id)
        request.status = RequestStatus.WAITING
        return True

    scheduler._try_promote_blocked_waiting_request = promote
    return loaded


def _two_long_prefills(scheduler, prompt=20_000, max_tokens=4):
    scheduler.add_request(_request("long1", prompt, max_tokens=max_tokens))
    scheduler.add_request(
        _request("long2", prompt, first_token=100_000, max_tokens=max_tokens)
    )
    assert _step(scheduler) == {"long1": LONG, "long2": LONG}


def test_a_short_prompt_gets_tokens_beside_two_running_long_prefills(
    model_dir, monkeypatch
):
    base = _scheduler(PrefixAwareScheduler, model_dir, monkeypatch)
    combined = _scheduler(PrefixAwareShortFirstScheduler, model_dir, monkeypatch)
    for scheduler in (base, combined):
        _two_long_prefills(scheduler)
        scheduler.add_request(_request("short", 500, first_token=200_000))
    assert _step(base) == {"long1": LONG, "long2": LONG}
    assert _step(combined) == {
        "short": 500,
        "long1": LONG,
        "long2": BUDGET - LONG - 500,
    }


def test_short_prompts_are_reserved_for_fewest_uncached_first(model_dir, monkeypatch):
    arrival = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    combined = _scheduler(PrefixAwareShortFirstScheduler, model_dir, monkeypatch)
    for scheduler in (arrival, combined):
        _two_long_prefills(scheduler)
        for i, prompt in enumerate((1800, 1700, 300, 400)):
            scheduler.add_request(
                _request(f"s{i + 1}", prompt, first_token=200_000 + 10_000 * i)
            )
    # Arrival order reserves s1, s2 and s3 (3,800 tokens; s4 does not fit).
    assert _step(arrival) == {"s1": 1800, "s2": 1700, "s3": 300, "long1": 296}
    # Prefix-aware order reserves s3, s4 and s2 (2,400 tokens; s1 does not fit).
    assert _step(combined) == {"s3": 300, "s4": 400, "s2": 1700, "long1": 1696}


def test_prefix_aware_order_holds_without_a_running_prefill(model_dir, monkeypatch):
    fcfs = _scheduler(AsyncScheduler, model_dir, monkeypatch)
    prefix_aware = _scheduler(PrefixAwareScheduler, model_dir, monkeypatch)
    combined = _scheduler(PrefixAwareShortFirstScheduler, model_dir, monkeypatch)
    for scheduler in (fcfs, prefix_aware, combined):
        scheduler.add_request(_request("warm", 6000, max_tokens=1))
        while "warm" in scheduler.requests:
            _step(scheduler)
        scheduler.add_request(_request("cold1", 9000, first_token=50_000))
        scheduler.add_request(_request("cold2", 9000, first_token=70_000))
        scheduler.add_request(_request("turn", 6400))
    assert combined._promote_short_waiting() == []
    assert [r.request_id for r in combined.waiting] == ["cold1", "cold2", "turn"]
    assert _step(fcfs) == {"cold1": LONG, "cold2": LONG}
    expected = {"turn": 400, "cold1": LONG, "cold2": BUDGET - LONG - 400}
    for scheduler in (prefix_aware, combined):
        scheduled = _step(scheduler)
        assert scheduled == expected
        assert list(scheduled) == ["turn", "cold1", "cold2"]


def test_a_prompt_back_from_a_tier_load_is_reserved_for(model_dir, monkeypatch):
    base = _scheduler(PrefixAwareScheduler, model_dir, monkeypatch)
    combined = _scheduler(PrefixAwareShortFirstScheduler, model_dir, monkeypatch)
    for scheduler in (base, combined):
        _two_long_prefills(scheduler)
        # A prompt whose tier load finished: 19,700 of its 20,000 tokens
        # are computed and it waits in skipped_waiting.
        loaded = _request("loaded", 20_000, first_token=200_000)
        loaded.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        loaded.num_computed_tokens = 19_700
        scheduler.requests["loaded"] = loaded
        scheduler.skipped_waiting.add_request(loaded)

        # Promotion without the connector's block caching, which these
        # schedulers do not have.
        def promote(request, scheduler=scheduler):
            if request.request_id not in scheduler.finished_recving_kv_req_ids:
                return False
            scheduler.finished_recving_kv_req_ids.remove(request.request_id)
            request.status = RequestStatus.WAITING
            return True

        scheduler._try_promote_blocked_waiting_request = promote

    assert combined._promote_short_waiting() == []
    for scheduler in (base, combined):
        scheduler.finished_recving_kv_req_ids.add("loaded")
    assert [r.request_id for r in combined._promote_short_waiting()] == ["loaded"]
    assert combined._uncached_by_request["loaded"] == 300
    assert _step(base) == {"long1": LONG, "long2": LONG}
    assert _step(combined) == {
        "loaded": 300,
        "long1": LONG,
        "long2": BUDGET - LONG - 300,
    }


def test_a_reservation_the_kv_cache_cannot_hold_does_not_stall_the_prefills(
    model_dir, monkeypatch
):
    # 380 blocks hold 6,080 tokens. The two long prefills' first chunks take
    # 256 of them; the two short prompts reserve the whole budget but need
    # 128 blocks each, more than the 124 left. The step that reserves for them
    # schedules nothing, and the next one reserves nothing, so the prefills
    # run on and free their blocks.
    scheduler = _scheduler(
        PrefixAwareShortFirstScheduler, model_dir, monkeypatch, num_blocks=380
    )
    _two_long_prefills(scheduler, prompt=3_000, max_tokens=1)
    scheduler.add_request(_request("a", SHORT, first_token=50_000, max_tokens=1))
    scheduler.add_request(_request("b", SHORT, first_token=60_000, max_tokens=1))
    empty_in_a_row = 0
    for _ in range(100):
        if not scheduler.requests:
            break
        scheduled = _step(scheduler)
        empty_in_a_row = 0 if scheduled else empty_in_a_row + 1
        assert empty_in_a_row < 2
    assert not scheduler.requests


def test_an_async_load_admission_keeps_the_next_reservation(model_dir, monkeypatch):
    scheduler = _scheduler(PrefixAwareShortFirstScheduler, model_dir, monkeypatch)
    _two_long_prefills(scheduler)
    scheduler.add_request(_request("tiered", 9000, first_token=200_000))
    # The tier holds all but 500 of its tokens, so it is short.
    monkeypatch.setattr(scheduler, "_prefix_warmth", lambda request: 8500)
    async_load = True

    # Stands in for the step that sends "tiered" to its tier load (or, with
    # async_load False, fails to give it KV blocks): no tokens are scheduled.
    def schedule_nothing(self, throttle_prefills=False):
        if async_load:
            request = self.waiting.pop_request()
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            self.skipped_waiting.add_request(request)
        return SimpleNamespace(num_scheduled_tokens={})

    monkeypatch.setattr(
        PrefixAwareScheduler, "schedule", schedule_nothing, raising=False
    )
    scheduler.schedule()
    assert not scheduler._skip_promotion

    scheduler.skipped_waiting.remove_requests(list(scheduler.skipped_waiting))
    scheduler.add_request(_request("tiered2", 9000, first_token=300_000))
    async_load = False
    scheduler.schedule()
    assert scheduler._skip_promotion


def test_every_prompt_token_is_computed_once(model_dir, monkeypatch):
    prompts = {"a": 30_000, "b": 700, "c": 12_000, "d": 1900, "e": 64, "f": 2049}
    prompts |= {"g": 25_000, "h": 300, "i": 5000}
    scheduler = _scheduler(PrefixAwareShortFirstScheduler, model_dir, monkeypatch)
    prefilled = dict.fromkeys(prompts, 0)
    pending = list(prompts.items())
    steps = 0
    while scheduler.requests or pending:
        if pending and steps % 2 == 0:
            req_id, prompt = pending.pop(0)
            scheduler.add_request(
                _request(req_id, prompt, first_token=100_000 * (ord(req_id) - 96))
            )
        scheduled = _step(scheduler)
        assert sum(scheduled.values()) <= BUDGET
        assert all(tokens <= LONG for tokens in scheduled.values())
        for req_id, tokens in scheduled.items():
            prefilled[req_id] += min(tokens, prompts[req_id] - prefilled[req_id])
        steps += 1
        assert steps < 200
    assert prefilled == prompts


# --- Limit on long prefills running at once (TPU_PREFILL_MAX_LONG_RUNNING) ---

# Long: more than 4,096 uncached tokens left. Requests of 2,049-4,096 are
# neither short nor long.
L = 4096


def _limited(model_dir, monkeypatch, max_long=1, **kwargs):
    return _scheduler(
        PrefixAwareShortFirstScheduler,
        model_dir,
        monkeypatch,
        max_long=max_long,
        long_tokens=L,
        **kwargs,
    )


def _ids(queue):
    return [r.request_id for r in queue]


def test_no_second_long_prefill_is_admitted_while_one_runs(model_dir, monkeypatch):
    off = _limited(model_dir, monkeypatch, max_long=0)
    on = _limited(model_dir, monkeypatch)
    for scheduler in (off, on):
        scheduler.add_request(_request("long1", 20_000))
        assert _step(scheduler) == {"long1": LONG}
        scheduler.add_request(_request("long2", 20_000, first_token=100_000))
        scheduler.add_request(_request("long3", 20_000, first_token=200_000))
        scheduler.add_request(_request("mid", 3000, first_token=300_000))
        # The request behind the held long ones is admitted in the same step.
        assert _step(scheduler) == {"long1": LONG, "mid": LONG}
    assert _step(off) == {"long1": LONG, "mid": 952, "long2": BUDGET - LONG - 952}
    assert _step(on) == {"long1": LONG, "mid": 952}
    assert _ids(on.waiting) == ["long2", "long3"]
    assert not on.skipped_waiting


def test_a_prompt_back_from_a_tier_load_is_held_when_long(model_dir, monkeypatch):
    scheduler = _limited(model_dir, monkeypatch)
    scheduler.add_request(_request("long1", 20_000))
    assert _step(scheduler) == {"long1": LONG}
    _loaded_request(scheduler, "far", 20_000, 10_000, first_token=100_000)
    _loaded_request(scheduler, "near", 20_000, 17_000, first_token=200_000)
    # 10,000 tokens left is long and waits; 3,000 left is not and runs.
    assert _step(scheduler) == {"long1": LONG, "near": LONG}
    assert _ids(scheduler.skipped_waiting) == ["far"]


def test_the_limit_never_caps_or_preempts_running_requests(model_dir, monkeypatch):
    scheduler = _limited(model_dir, monkeypatch, max_long=0)
    _two_long_prefills(scheduler)
    scheduler.max_long_prefills = 1
    scheduler.add_request(_request("long3", 20_000, first_token=200_000))
    assert _step(scheduler) == {"long1": LONG, "long2": LONG}
    assert _ids(scheduler.running) == ["long1", "long2"]
    assert _ids(scheduler.waiting) == ["long3"]


def test_held_requests_keep_their_queue_positions(model_dir, monkeypatch):
    scheduler = _limited(model_dir, monkeypatch)
    scheduler.add_request(_request("long1", 20_000))
    _step(scheduler)
    for i, (req_id, prompt) in enumerate(
        (("l2", 20_000), ("m1", 3000), ("l3", 30_000), ("m2", 3000), ("l4", 9000))
    ):
        scheduler.add_request(_request(req_id, prompt, first_token=100_000 * (i + 1)))
    assert _step(scheduler) == {"long1": LONG, "m1": LONG}
    assert _ids(scheduler.waiting) == ["l2", "l3", "m2", "l4"]
    assert not scheduler.skipped_waiting


def test_with_no_long_running_the_first_long_in_prefix_aware_order_starts(
    model_dir, monkeypatch
):
    off = _limited(model_dir, monkeypatch, max_long=0)
    on = _limited(model_dir, monkeypatch)
    for scheduler in (off, on):
        scheduler.add_request(_request("big", 30_000))
        scheduler.add_request(_request("small", 10_000, first_token=100_000))
    assert _step(off) == {"small": LONG, "big": LONG}
    assert _step(on) == {"small": LONG}
    assert _ids(on.waiting) == ["big"]


def test_only_long_requests_waiting_never_gives_an_empty_step(model_dir, monkeypatch):
    prompts = {"long1": 20_000, "long2": 12_000, "long3": 9000, "mid": 3000}
    scheduler = _limited(model_dir, monkeypatch)
    for i, (req_id, prompt) in enumerate(prompts.items()):
        scheduler.add_request(
            _request(req_id, prompt, first_token=100_000 * i, max_tokens=1)
        )
    prefilled = dict.fromkeys(prompts, 0)
    for _ in range(100):
        if not scheduler.requests:
            break
        scheduled = _step(scheduler)
        assert scheduled
        running_long = [
            r for r in scheduled if prompts[r] - prefilled[r] > L and prompts[r] > L
        ]
        assert len(running_long) <= 1
        for req_id, tokens in scheduled.items():
            prefilled[req_id] += min(tokens, prompts[req_id] - prefilled[req_id])
    assert not scheduler.requests
    assert prefilled == prompts


def test_a_running_long_prefill_stops_counting_at_the_threshold(model_dir, monkeypatch):
    scheduler = _limited(model_dir, monkeypatch)
    scheduler.add_request(_request("long1", 10_000))
    assert _step(scheduler) == {"long1": LONG}
    scheduler.add_request(_request("long2", 20_000, first_token=100_000))
    # long1 has 7,952 and then 5,904 tokens left: long2 waits.
    assert _step(scheduler) == {"long1": LONG}
    assert _step(scheduler) == {"long1": LONG}
    # 3,856 left is at or under the threshold, so long2 starts beside it.
    assert _step(scheduler) == {"long1": LONG, "long2": LONG}


class _NoLimit(PrefixAwareShortFirstScheduler):
    def _hold_long_requests(self):
        return []


def test_the_limit_off_schedules_as_the_class_without_it(model_dir, monkeypatch):
    prompts = {
        "a": 30_000,
        "b": 700,
        "c": 12_000,
        "d": 1900,
        "e": 64,
        "f": 2049,
        "g": 25_000,
        "h": 300,
        "i": 5000,
        "j": 3500,
        "k": 40_000,
    }
    schedulers = (
        _scheduler(_NoLimit, model_dir, monkeypatch),
        _limited(model_dir, monkeypatch, max_long=0),
    )
    traces = []
    for scheduler in schedulers:
        trace = []
        pending = list(prompts.items())
        while scheduler.requests or pending:
            if pending and len(trace) % 2 == 0:
                req_id, prompt = pending.pop(0)
                scheduler.add_request(
                    _request(req_id, prompt, first_token=100_000 * (ord(req_id) - 96))
                )
            trace.append(_step(scheduler))
            assert len(trace) < 300
        traces.append(trace)
    assert traces[0] == traces[1]
    assert scheduler._held_steps == 0


# --- Step fill (TPU_PREFILL_STEP_FILL) ----------------------------------------


def _filling(model_dir, monkeypatch, step_fill=True, max_long=1, **kwargs):
    return _scheduler(
        PrefixAwareShortFirstScheduler,
        model_dir,
        monkeypatch,
        max_long=max_long,
        long_tokens=L,
        step_fill=step_fill,
        **kwargs,
    )


def test_a_lone_long_prefill_with_empty_queues_takes_the_whole_step(
    model_dir, monkeypatch
):
    off = _filling(model_dir, monkeypatch, step_fill=False)
    on = _filling(model_dir, monkeypatch)
    for scheduler in (off, on):
        scheduler.add_request(_request("long", 20_000))
        assert _step(scheduler) == {"long": LONG}
    assert _step(off) == {"long": LONG}
    assert _step(on) == {"long": BUDGET}


def test_a_lone_long_prefill_beside_an_admittable_short_keeps_its_cap(
    model_dir, monkeypatch
):
    scheduler = _filling(model_dir, monkeypatch)
    scheduler.add_request(_request("long", 20_000))
    _step(scheduler)
    scheduler.add_request(_request("short", 500, first_token=100_000))
    assert _step(scheduler) == {"long": LONG, "short": 500}


def test_a_lone_long_prefill_with_only_held_longs_waiting_takes_the_whole_step(
    model_dir, monkeypatch
):
    scheduler = _filling(model_dir, monkeypatch)
    scheduler.add_request(_request("long1", 20_000))
    _step(scheduler)
    scheduler.add_request(_request("long2", 20_000, first_token=100_000))
    assert _step(scheduler) == {"long1": BUDGET}
    assert _ids(scheduler.waiting) == ["long2"]


def test_running_prefills_fill_the_step_in_running_order(model_dir, monkeypatch):
    off = _filling(model_dir, monkeypatch, step_fill=False, max_long=0)
    on = _filling(model_dir, monkeypatch, max_long=0)
    for scheduler in (off, on):
        scheduler.add_request(_request("long1", 5000))
        scheduler.add_request(_request("long2", 20_000, first_token=100_000))
        assert _step(scheduler) == {"long1": LONG, "long2": LONG}
    assert _step(off) == {"long1": LONG, "long2": LONG}
    # long1 has 2,952 tokens left; long2 takes what the step has after it.
    assert _step(on) == {"long1": 2952, "long2": BUDGET - 2952}


class _NoFill(PrefixAwareShortFirstScheduler):
    def _admittable_waiting(self):
        return True


def test_step_fill_off_schedules_as_the_class_without_it(model_dir, monkeypatch):
    prompts = {
        "a": 30_000,
        "b": 700,
        "c": 12_000,
        "d": 1900,
        "e": 64,
        "f": 2049,
        "g": 25_000,
        "h": 300,
        "i": 5000,
        "j": 3500,
        "k": 40_000,
    }
    traces = []
    for scheduler in (
        _scheduler(_NoFill, model_dir, monkeypatch, max_long=1, long_tokens=L),
        _filling(model_dir, monkeypatch, step_fill=False),
    ):
        trace = []
        pending = list(prompts.items())
        while scheduler.requests or pending:
            if pending and len(trace) % 2 == 0:
                req_id, prompt = pending.pop(0)
                scheduler.add_request(
                    _request(req_id, prompt, first_token=100_000 * (ord(req_id) - 96))
                )
            trace.append(_step(scheduler))
            assert len(trace) < 300
        traces.append(trace)
    assert traces[0] == traces[1]


def test_every_prompt_token_is_computed_once_with_step_fill(model_dir, monkeypatch):
    prompts = {"a": 30_000, "b": 700, "c": 12_000, "d": 1900, "e": 64, "f": 2049}
    prompts |= {"g": 25_000, "h": 300, "i": 5000}
    scheduler = _filling(model_dir, monkeypatch)
    prefilled = dict.fromkeys(prompts, 0)
    pending = list(prompts.items())
    steps = 0
    while scheduler.requests or pending:
        if pending and steps % 3 == 0:
            req_id, prompt = pending.pop(0)
            scheduler.add_request(
                _request(req_id, prompt, first_token=100_000 * (ord(req_id) - 96))
            )
        scheduled = _step(scheduler)
        assert scheduled
        assert sum(scheduled.values()) <= BUDGET
        for req_id, tokens in scheduled.items():
            prefilled[req_id] += min(tokens, prompts[req_id] - prefilled[req_id])
        steps += 1
        assert steps < 200
    assert prefilled == prompts
