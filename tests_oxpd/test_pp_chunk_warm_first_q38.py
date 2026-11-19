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
"""q38 HIGHC-WARMFIRST: TPU_PP_WARM_FIRST (port of benliangw d523c2ce0) on the
REAL vLLM scheduler machinery, plus the port's own contracts.

* Real prefix cache: a warm request (its prefix cached by an earlier, finished
  request) is admitted before a cold request that arrived earlier, and the
  uncached estimate the order uses equals the hit the scheduler then admits.
* Aging on real Requests: a cold request is passed only by requests arriving
  within U / rate seconds of it, and a stream of warm arrivals cannot starve it
  past that bound (aging 0 does starve it, which is why the default is 10,000).
* Off: the order code never runs (no lookups, no sorts, the waiting queue is
  untouched) and decisions equal first come, first served.
* The keys stay out of the compile-cache key (host-only), and the tier key falls
  back cleanly on this tree, where no connector reports host-tier hits.
"""

import logging
import time as _time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import envs
from vllm_torchtpu.core import pp_chunk_scheduler
from vllm_torchtpu.core.pp_chunk_scheduler import TpuPipelineChunkScheduler

BLOCK = 16
_KEYS = ("TPU_PP_WARM_FIRST", "TPU_PP_WARM_FIRST_AGING", "TPU_PP_WARM_FIRST_TIER")


# --------------------------------------------------------------------------- real scheduler


@pytest.fixture(scope="module")
def model_config():
    errors = []
    tiny = Path(__file__).resolve().parent / "q38_dstash" / "tiny_opt"
    for model in (str(tiny), "facebook/opt-125m"):
        try:
            return ModelConfig(
                model=model,
                tokenizer=model,
                dtype="float16",
                seed=42,
                max_model_len=2048,
                skip_tokenizer_init=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    # A real-scheduler test that cannot build its model config must fail, not skip: a skip would hide it.
    pytest.fail("no ModelConfig: " + " | ".join(errors))


@pytest.fixture
def clock(monkeypatch):
    """Virtual wall clock for _order_waiting (schedule() reads time.time())."""
    now = [0.0]
    monkeypatch.setattr(
        pp_chunk_scheduler,
        "time",
        SimpleNamespace(time=lambda: now[0], monotonic=_time.monotonic),
    )
    return now


def _set_keys(monkeypatch, on, aging=0.0, tier=False):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    if on is not None:
        monkeypatch.setenv("TPU_PP_WARM_FIRST", "1" if on else "0")
        monkeypatch.setenv("TPU_PP_WARM_FIRST_AGING", str(aging))
        monkeypatch.setenv("TPU_PP_WARM_FIRST_TIER", "1" if tier else "0")


def _build(model_config, max_num_seqs=1, budget=64, num_blocks=256):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=budget,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=64, dtype=torch.float16
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = TpuPipelineChunkScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK,
    )
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(BLOCK, sha256)


def _req(rid, tokens, hasher, arrival):
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    return Request(rid, list(tokens), params, None, arrival_time=arrival, block_hasher=hasher)


def _fresh(seed, n):
    return [seed * 100_000 + i for i in range(n)]


def _step(scheduler):
    """One schedule + output; a request whose prompt is done samples one token
    (a PD-prefill request, max_tokens=1, finishes there)."""
    out = scheduler.schedule()
    ids = list(out.num_scheduled_tokens)
    sampled = [
        [7]
        if scheduler.requests[r].num_computed_tokens
        >= scheduler.requests[r].num_prompt_tokens
        else []
        for r in ids
    ]
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=sampled,
        ),
    )
    return out


def _run_until_idle(scheduler, limit=500):
    trace = []
    for _ in range(limit):
        if not scheduler.has_requests():
            return trace
        out = _step(scheduler)
        trace.append(
            (
                [(nr.req_id, nr.num_computed_tokens) for nr in out.scheduled_new_reqs],
                sorted(out.num_scheduled_tokens.items()),
            )
        )
    raise AssertionError("scheduler did not drain")


def _seed_cache(scheduler, hasher, rid, tokens, clock):
    scheduler.add_request(_req(rid, tokens, hasher, clock[0]))
    _run_until_idle(scheduler)
    assert rid not in scheduler.requests


@pytest.mark.parametrize("on", [True, False])
def test_real_cache_warm_request_admitted_before_earlier_cold_one(
    monkeypatch, model_config, clock, on
):
    _set_keys(monkeypatch, on, aging=0.0)
    scheduler, hasher = _build(model_config)
    prefix = _fresh(1, 64)
    _seed_cache(scheduler, hasher, "seed", prefix, clock)
    clock[0] = 1.0
    scheduler.add_request(_req("cold", _fresh(2, 64), hasher, 1.0))
    clock[0] = 2.0
    # 80 tokens, 64 of them cached: 16 uncached against cold's 64.
    scheduler.add_request(_req("warm", prefix + _fresh(3, 16), hasher, 2.0))
    clock[0] = 3.0
    out = _step(scheduler)
    first = [(nr.req_id, nr.num_computed_tokens) for nr in out.scheduled_new_reqs]
    if on:
        assert first == [("warm", 64)]
        # The order's estimate is the hit the scheduler then admitted.
        assert scheduler._uncached["warm"][0] == 80 - 64
        assert scheduler._uncached["cold"][0] == 64
        assert scheduler._paw_scoring_broken is False
        assert scheduler._wf_reorders == 1
    else:
        assert first == [("cold", 0)]
        assert scheduler._wf_sorts == 0 and not scheduler._uncached


@pytest.mark.parametrize("warm_arrival, passes", [(4.0, True), (5.0, False)])
def test_real_requests_aging_bound(monkeypatch, model_config, clock, warm_arrival, passes):
    # cold: 64 uncached; warm: 16 uncached; rate 10 tokens/s -> warm passes
    # only if it arrives less than (64 - 16) / 10 = 4.8 s after cold.
    _set_keys(monkeypatch, True, aging=10.0)
    scheduler, hasher = _build(model_config)
    prefix = _fresh(1, 64)
    _seed_cache(scheduler, hasher, "seed", prefix, clock)
    scheduler.add_request(_req("cold", _fresh(2, 64), hasher, 0.0))
    scheduler.add_request(_req("warm", prefix + _fresh(3, 16), hasher, warm_arrival))
    clock[0] = warm_arrival + 0.01
    out = _step(scheduler)
    got = [nr.req_id for nr in out.scheduled_new_reqs]
    assert got == (["warm"] if passes else ["cold"])


def _starvation_run(monkeypatch, model_config, clock, aging, horizon=40):
    """A cold 64-token prompt waits while warm 80-token requests (64 cached)
    arrive one per step (1 s per step). Returns the step the cold prompt is
    admitted at, or None if never within the horizon."""
    _set_keys(monkeypatch, True, aging=aging)
    scheduler, hasher = _build(model_config, max_num_seqs=1, budget=16)
    prefix = _fresh(1, 64)
    _seed_cache(scheduler, hasher, "seed", prefix, clock)
    clock[0] = 0.0
    scheduler.add_request(_req("cold", _fresh(2, 64), hasher, 0.0))
    for step in range(horizon):
        clock[0] = float(step)
        scheduler.add_request(_req(f"w{step}", prefix + _fresh(10 + step, 16), hasher, float(step)))
        out = _step(scheduler)
        if any(nr.req_id == "cold" for nr in out.scheduled_new_reqs):
            return step
    return None


def test_real_scheduler_aging_bounds_starvation(monkeypatch, model_config, clock):
    # Without aging, warm arrivals (16 uncached) keep passing the cold prompt
    # (64 uncached) for the whole horizon.
    assert _starvation_run(monkeypatch, model_config, clock, aging=0.0) is None
    # At 4 tokens/s the cold prompt is passed only by requests arriving less
    # than (64 - 16) / 4 = 12 s after it; the ones that arrived earlier still
    # drain first, one 16-token step each, so admission lands by ~2x the bound.
    step = _starvation_run(monkeypatch, model_config, clock, aging=4.0)
    assert step is not None and 10 <= step <= 26


def _mixed_workload(hasher):
    """Prefix-sharing conversations of mixed lengths, arriving out of size order."""
    base = {c: _fresh(50 + c, 200) for c in range(4)}
    reqs = []
    t = 0.0
    for turn in range(3):
        for c in range(4):
            length = 40 + 30 * turn + 17 * c
            tail = _fresh(500 + 10 * c + turn, 5 + 3 * c)
            reqs.append((f"c{c}t{turn}", base[c][:length] + tail, t))
            t += 0.05
    return reqs


def _trace(monkeypatch, model_config, clock, on, spy=None, aging=0.0):
    _set_keys(monkeypatch, on, aging=aging)
    scheduler, hasher = _build(model_config, max_num_seqs=2, budget=48)
    if spy is not None:
        original = scheduler._order_waiting

        def counted(now):
            spy.append(now)
            return original(now)

        scheduler._order_waiting = counted
    trace = []
    pending = list(_mixed_workload(hasher))
    while pending or scheduler.has_requests():
        while pending and pending[0][2] <= clock[0]:
            rid, toks, t = pending.pop(0)
            scheduler.add_request(_req(rid, toks, hasher, t))
        if not scheduler.has_requests():
            clock[0] = pending[0][2]
            continue
        before = [r.request_id for r in scheduler.waiting]
        out = _step(scheduler)
        trace.append(
            (
                before,
                [(nr.req_id, nr.num_computed_tokens) for nr in out.scheduled_new_reqs],
                sorted(out.num_scheduled_tokens.items()),
            )
        )
        clock[0] += 0.1
    return scheduler, trace


def test_off_runs_no_order_code_and_keeps_first_come_first_served(
    monkeypatch, model_config, clock
):
    calls = []
    scheduler, trace = _trace(monkeypatch, model_config, clock, on=False, spy=calls)
    assert calls == []
    assert scheduler._warm_first is False
    assert scheduler._wf_sorts == 0 and scheduler._wf_lookups == 0
    assert not scheduler._uncached
    # Admission order is arrival order when off.
    admitted = [rid for _, new, _ in trace for rid, _ in new]
    order = [rid for rid, _, _ in _mixed_workload(None)]
    assert admitted == order
    # And the same workload with the key on does reorder (the workload can
    # tell the two apart).
    clock[0] = 0.0
    calls_on = []
    _, trace_on = _trace(monkeypatch, model_config, clock, on=True, spy=calls_on)
    admitted_on = [rid for _, new, _ in trace_on for rid, _ in new]
    assert calls_on and admitted_on != order
    assert sorted(admitted_on) == sorted(order)


def test_unset_keys_mean_off_and_default_aging(monkeypatch, model_config):
    _set_keys(monkeypatch, None)
    assert envs.TPU_PP_WARM_FIRST is False
    assert envs.TPU_PP_WARM_FIRST_AGING == 10000.0
    assert envs.TPU_PP_WARM_FIRST_TIER is False
    scheduler, _ = _build(model_config)
    assert scheduler._warm_first is False and scheduler._tier is None
    monkeypatch.setenv("TPU_PP_WARM_FIRST", "1")
    monkeypatch.setenv("TPU_PP_WARM_FIRST_AGING", "2500.5")
    assert envs.TPU_PP_WARM_FIRST is True
    assert envs.TPU_PP_WARM_FIRST_AGING == 2500.5
    scheduler, _ = _build(model_config)
    assert scheduler._warm_first is True and scheduler._warm_first_aging == 2500.5


# --------------------------------------------------------------------------- compile key


def test_warm_first_keys_stay_out_of_the_compile_cache_key(monkeypatch):
    from unittest.mock import patch

    from vllm_torchtpu.compilation.tpu_compiler import (
        _TPU_COMPILE_ENV_IGNORED,
        _tpu_compile_env_factors,
        compute_tpu_compilation_hash,
    )

    assert set(_KEYS) <= set(envs.environment_variables)
    assert set(_KEYS) <= _TPU_COMPILE_ENV_IGNORED
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9,
            num_gpu_blocks=1024,
            num_gpu_blocks_override=None,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=8),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )
    hashes = []
    with patch(
        "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
        return_value=[],
    ):
        for values in (None, ("0", "10000", "0"), ("1", "0", "1"), ("1", "123.5", "0")):
            for key in _KEYS:
                monkeypatch.delenv(key, raising=False)
            if values is not None:
                for key, value in zip(_KEYS, values):
                    monkeypatch.setenv(key, value)
            factors = _tpu_compile_env_factors()
            assert not set(_KEYS) & set(factors)
            hashes.append(compute_tpu_compilation_hash(config))
    assert len(set(hashes)) == 1


# --------------------------------------------------------------------------- tier fallback


def _stub_scheduler(connector, tier, warmth):
    scheduler = TpuPipelineChunkScheduler.__new__(TpuPipelineChunkScheduler)
    scheduler.connector = connector
    scheduler._init_warm_first(True, 0.0, tier)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.running = []

    class _Cache:
        def find_longest_cache_hit(self, block_hashes, max_len):
            return ((), min(warmth.get(block_hashes, 0), max_len), 0)

    scheduler.kv_cache_manager = SimpleNamespace(enable_caching=True, coordinator=_Cache())
    return scheduler


def _stub_req(rid, n, arrival):
    return SimpleNamespace(
        request_id=rid,
        num_tokens=n,
        num_prompt_tokens=n,
        num_computed_tokens=0,
        num_preemptions=0,
        arrival_time=arrival,
        skip_reading_prefix_cache=False,
        block_hashes=rid,
    )


@pytest.mark.parametrize(
    "connector",
    [None, SimpleNamespace(), SimpleNamespace(sub_connectors=[SimpleNamespace()])],
)
def test_tier_without_a_host_tier_connector_falls_back_to_the_local_cache(
    connector, caplog
):
    caplog.set_level(logging.WARNING)
    scheduler = _stub_scheduler(connector, True, {"warm": 9_000})
    assert scheduler._tier is None
    assert any("TPU_PP_WARM_FIRST_TIER is set" in r.getMessage() for r in caplog.records)
    for r in (_stub_req("cold", 10_000, 0.0), _stub_req("warm", 10_000, 1.0)):
        scheduler.waiting.add_request(r)
    scheduler._order_waiting(now=2.0)
    assert [r.request_id for r in scheduler.waiting] == ["warm", "cold"]
    assert scheduler._wf_tier_hits == 0


def test_a_failing_tier_lookup_latches_off_and_keeps_ordering(caplog):
    class _Tier:
        calls = 0

        def host_tier_hit_tokens(self, request, local):
            _Tier.calls += 1
            raise RuntimeError("store gone")

    scheduler = _stub_scheduler(SimpleNamespace(sub_connectors=[_Tier()]), True, {"warm": 9_000})
    assert scheduler._tier is not None
    for r in (_stub_req("cold", 10_000, 0.0), _stub_req("warm", 10_000, 1.0)):
        scheduler.waiting.add_request(r)
    scheduler._order_waiting(now=2.0)
    assert scheduler._tier is None and _Tier.calls == 1
    assert [r.request_id for r in scheduler.waiting] == ["warm", "cold"]
