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
"""MultiStepAsyncScheduler (TPU_MULTISTEP_DECODE) on a real scheduler: which
steps burst, how long, and the token accounting through update_from_output."""

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
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.multistep_decode import (
    MultiStepAsyncScheduler,
    MultiStepPrefillFirstScheduler,
)
from vllm_torchtpu.platforms import tpu_platform
from vllm_torchtpu.runner import multistep_decode as runner_multistep

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _build(
    model_config,
    monkeypatch,
    k=4,
    log_stats=True,
    max_model_len=2048,
    max_num_seqs=8,
    cls=MultiStepAsyncScheduler,
):
    monkeypatch.setenv("TPU_MULTISTEP_DECODE", str(k))
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=256,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = 256
    register_all_kvcache_specs(vllm_config)
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        log_stats=log_stats,
    )


def _request(request_id, num_tokens=16, max_tokens=16, temperature=0.0, **kw):
    params = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, ignore_eos=True, **kw
    )
    params.update_from_generation_config({}, 2)
    prompt = [hash(request_id) % 1000 + i for i in range(num_tokens)]
    return Request(request_id, prompt, params, None)


def _output(scheduler_output, tokens_per_req):
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[7] * tokens_per_req(r) for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _run(scheduler, steps):
    """Async loop: schedule step n, then deliver step n - 1's tokens (one per
    scheduled token: a burst's tokens, or one for a decode/last prefill)."""
    trace = []
    in_flight = None
    for _ in range(steps):
        out = scheduler.schedule()
        trace.append(dict(out.num_scheduled_tokens))
        if in_flight is not None:
            prev = in_flight
            scheduler.update_from_output(
                prev,
                _output(
                    prev,
                    lambda r, p=prev: (
                        p.num_scheduled_tokens[r]
                        if r in p.scheduled_spec_decode_tokens
                        else 1
                    ),
                ),
            )
        in_flight = out
    return trace


def test_decode_bursts_after_prefill_and_cut_at_max_tokens(model_config, monkeypatch):
    scheduler = _build(model_config, monkeypatch, k=4)
    scheduler.add_request(_request("a", max_tokens=10))
    scheduler.add_request(_request("b", max_tokens=10))
    trace = _run(scheduler, 6)
    # prefill; bursts of 4 (9 tokens left after the prefill's first); then the
    # most any request has left (5 -> 4 is still a full burst, then 1 left).
    assert trace[0] == {"a": 16, "b": 16}
    assert trace[1] == {"a": 4, "b": 4}
    assert trace[2] == {"a": 4, "b": 4}
    assert trace[3] == {"a": 1, "b": 1}
    # All 10 tokens are in flight or delivered: nothing more is scheduled.
    assert trace[4] == {} and trace[5] == {}
    assert scheduler.multistep_counts["burst 4"] == 2
    for request in (scheduler.requests.get("a"), scheduler.requests.get("b")):
        assert request is None or request.num_output_tokens <= 10


def test_burst_output_is_uniform_and_marks_placeholders(model_config, monkeypatch):
    scheduler = _build(model_config, monkeypatch, k=3)
    scheduler.add_request(_request("a"))
    scheduler.schedule()
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"a": 3}
    assert out.scheduled_spec_decode_tokens == {"a": [-1, -1]}
    assert runner_multistep.burst_len(out) == 3
    view = runner_multistep.decode_view(out)
    assert view.num_scheduled_tokens == {"a": 1}
    assert view.total_num_scheduled_tokens == 1
    assert view.scheduled_spec_decode_tokens == {}
    assert scheduler.requests["a"].num_output_placeholders == 4  # 1 + 3


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"temperature": 1.0}, "single: sampling"),
        ({"logprobs": 1}, "single: logprobs"),
        ({"min_tokens": 4}, "single: min_tokens"),
    ],
)
def test_requests_the_top_token_path_cannot_serve_keep_single_steps(
    model_config, monkeypatch, kwargs, reason
):
    scheduler = _build(model_config, monkeypatch)
    scheduler.add_request(_request("a"))
    scheduler.add_request(_request("x", **kwargs))
    trace = _run(scheduler, 3)
    assert trace[1] == {"a": 1, "x": 1} and trace[2] == {"a": 1, "x": 1}
    assert scheduler.multistep_counts[reason] == 2


def test_waiting_request_and_prefill_chunk_stop_bursts(model_config, monkeypatch):
    scheduler = _build(model_config, monkeypatch)
    scheduler.add_request(_request("a"))
    scheduler.schedule()
    scheduler.add_request(_request("long", num_tokens=400))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["a"] == 1
    assert out.num_scheduled_tokens["long"] == 255  # chunked prefill
    # The first step admitted "a" from the queue too.
    assert scheduler.multistep_counts["single: waiting requests"] == 2
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["a"] == 1
    assert scheduler.multistep_counts["single: prefill chunk"] == 1


def test_burst_is_cut_to_the_model_length_room(model_config, monkeypatch):
    scheduler = _build(model_config, monkeypatch, k=8, max_model_len=20)
    scheduler.add_request(_request("a", num_tokens=16, max_tokens=4))
    scheduler.schedule()
    out = scheduler.schedule()
    # 16 computed of 20: three more positions fit (the last one is sampled).
    assert out.num_scheduled_tokens == {"a": 3}


def test_spec_decoding_stats_are_not_reported_for_bursts(model_config, monkeypatch):
    scheduler = _build(model_config, monkeypatch, k=4, log_stats=True)
    scheduler.add_request(_request("a"))
    trace = _run(scheduler, 4)
    assert trace[1] == {"a": 4}
    stats = scheduler.make_spec_decoding_stats(None, 3, 3, None, "a")
    assert stats is None


def test_platform_installs_the_scheduler_idempotently(monkeypatch):
    monkeypatch.setenv("TPU_MULTISTEP_DECODE", "4")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=True),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None,
    )
    tpu_platform._configure_multistep_decode_scheduler(config)
    tpu_platform._configure_multistep_decode_scheduler(config)
    assert config.scheduler_config.scheduler_cls.endswith("MultiStepAsyncScheduler")


@pytest.mark.parametrize(
    "change, need",
    [
        (("scheduler_config", "async_scheduling", False), "--async-scheduling"),
        (("parallel_config", "data_parallel_size", 2), "no data parallelism"),
        (("scheduler_config", "scheduler_cls", "x.Other"), "the default scheduler"),
    ],
)
def test_platform_refuses_layouts_bursts_cannot_serve(monkeypatch, change, need):
    monkeypatch.setenv("TPU_MULTISTEP_DECODE", "4")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=True),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None,
    )
    section, field, value = change
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=need):
        tpu_platform._configure_multistep_decode_scheduler(config)


def test_finished_in_flight_requests_sit_out_in_place(model_config, monkeypatch):
    scheduler = _build(model_config, monkeypatch, k=4)
    scheduler.add_request(_request("a", max_tokens=5))
    scheduler.add_request(_request("b", max_tokens=16))
    scheduler.schedule()
    assert scheduler.schedule().num_scheduled_tokens == {"a": 4, "b": 4}
    # "a" has all 5 tokens in flight: it sits the burst out, keeping its place.
    assert scheduler.schedule().num_scheduled_tokens == {"b": 4}
    assert [r.request_id for r in scheduler.running] == ["a", "b"]


def test_held_requests_count_against_admission(model_config, monkeypatch):
    scheduler = _build(model_config, monkeypatch, k=4, max_num_seqs=2)
    scheduler.add_request(_request("a", max_tokens=5))
    scheduler.add_request(_request("b"))
    scheduler.schedule()
    assert scheduler.schedule().num_scheduled_tokens == {"a": 4, "b": 4}
    scheduler.add_request(_request("w"))
    out = scheduler.schedule()
    # "a" (all tokens in flight) sits out without a token past max_tokens,
    # and still holds its slot: "w" is not admitted beyond max_num_seqs.
    assert out.num_scheduled_tokens == {"b": 1}
    assert [r.request_id for r in scheduler.running] == ["a", "b"]
    assert scheduler.max_num_active_reqs == 2


def test_burst_duration_cap_follows_the_measured_iteration_time(
    model_config, monkeypatch
):
    from vllm_torchtpu.core import multistep_decode as scheduler_module

    monkeypatch.setenv("TPU_MULTISTEP_DECODE_MAX_BURST_MS", "25")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: clock.now)
    scheduler = _build(model_config, monkeypatch, k=8)
    scheduler.add_request(_request("a", max_tokens=200))
    in_flight = None
    sizes = []
    for _ in range(8):
        out = scheduler.schedule()
        sizes.append(out.num_scheduled_tokens.get("a"))
        if in_flight is not None:
            # Every decode iteration takes 10 ms on the device.
            clock.now += 0.010 * in_flight.num_scheduled_tokens["a"]
            scheduler.update_from_output(
                in_flight,
                _output(
                    in_flight,
                    lambda r, p=in_flight: (
                        p.num_scheduled_tokens[r]
                        if r in p.scheduled_spec_decode_tokens
                        else 1
                    ),
                ),
            )
        in_flight = out
    # Prefill, then full bursts until two decode-only outputs give a measured
    # 10 ms per iteration; 25 ms then allows bursts of 2.
    assert sizes[0] == 16 and sizes[1] == 8
    assert sizes[-1] == 2


def test_decode_step_flags_are_out_of_the_compile_key():
    from vllm_torchtpu.compilation import tpu_compiler

    for name in (
        "TPU_MULTISTEP_DECODE",
        "TPU_MULTISTEP_DECODE_MAX_BURST_MS",
        "TPU_K_STEP_DECODE_BACKEND",
    ):
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_platform_mixes_bursts_into_prefill_first_idempotently(monkeypatch):
    monkeypatch.setenv("TPU_MULTISTEP_DECODE", "4")
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", "8192")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=True),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None,
    )
    # The engine core re-runs both hooks on the config they already set.
    for _ in range(2):
        tpu_platform._configure_prefill_first_scheduler(config)
        tpu_platform._configure_multistep_decode_scheduler(config)
    assert config.scheduler_config.scheduler_cls.endswith(
        "MultiStepPrefillFirstScheduler"
    )


def test_bursts_resume_around_a_solo_prefill(model_config, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", "64")
    scheduler = _build(
        model_config, monkeypatch, k=4, cls=MultiStepPrefillFirstScheduler
    )
    scheduler.add_request(_request("a", max_tokens=40))
    trace = _run(scheduler, 2)
    assert trace[1] == {"a": 4}
    scheduler.add_request(_request("long", num_tokens=200, max_tokens=40))
    trace = _run(scheduler, 2)
    # The waiting prompt takes a solo step ("a" sits it out); with nothing
    # left waiting or prefilling, both burst again.
    assert trace[0] == {"long": 200}
    assert trace[1] == {"a": 4, "long": 4}
    # "a"'s admission and the solo step.
    assert scheduler.multistep_counts["single: waiting requests"] == 2


def test_burst_fits_the_regular_step_budget(model_config, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", "24")
    scheduler = _build(
        model_config, monkeypatch, k=8, cls=MultiStepPrefillFirstScheduler
    )
    for name in "abcd":
        scheduler.add_request(_request(name, max_tokens=40))
    trace = _run(scheduler, 6)
    # The steps after the prefills, where every request has the same count.
    decodes = [t for t in trace if len(t) == 4 and len(set(t.values())) == 1]
    # Regular steps schedule at most 24 tokens: 4 requests burst 6 each.
    assert decodes and all(t == dict.fromkeys("abcd", 6) for t in decodes)


def test_held_back_decodes_keep_no_placeholder_drafts(model_config, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", "64")
    scheduler = _build(
        model_config, monkeypatch, k=4, cls=MultiStepPrefillFirstScheduler
    )
    scheduler.add_request(_request("a", max_tokens=40))
    scheduler.add_request(_request("b", max_tokens=40))
    _run(scheduler, 1)
    # A policy holding "a" back in a burst step, as decode pacing would.
    monkeypatch.setattr(scheduler, "_tpot_paced", lambda decodes: decodes[:1])
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"b": 4}
    assert scheduler.requests["a"].spec_token_ids == []
    monkeypatch.setattr(scheduler, "_tpot_paced", lambda decodes: [])
    scheduler.add_request(_request("w"))
    out = scheduler.schedule()
    assert runner_multistep.burst_len(out) == 1


def test_platform_refuses_policies_that_hold_decodes_back(monkeypatch):
    monkeypatch.setenv("TPU_MULTISTEP_DECODE", "4")
    monkeypatch.setenv("VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", "8192")
    monkeypatch.setenv("VLLM_TPU_MIN_DECODE_INTERVAL_MS", "5")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=True),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None,
    )
    tpu_platform._configure_prefill_first_scheduler(config)
    with pytest.raises(ValueError, match="set: VLLM_TPU_MIN_DECODE_INTERVAL_MS"):
        tpu_platform._configure_multistep_decode_scheduler(config)
