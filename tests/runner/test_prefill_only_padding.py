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
"""Prefill-only MTP: the worker strips draft-placeholder rows vLLM padded onto a
final prompt token (a serving crash: computed 126208, scheduled 2, prompt
126209), and vLLM's scheduler counts them as rejected drafts."""

import dataclasses
import json
import types

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner, _strip_prefill_only_padding


def _so(num_sched, spec, new=(), cached=None):
    return dataclasses.replace(
        SchedulerOutput.make_empty(),
        num_scheduled_tokens=dict(num_sched),
        total_num_scheduled_tokens=sum(num_sched.values()),
        scheduled_spec_decode_tokens=dict(spec),
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
    )


def _cached(req_ids, computed):
    return dataclasses.replace(
        CachedRequestData.make_empty(),
        req_ids=list(req_ids),
        num_computed_tokens=list(computed),
    )


def _requests(prompts=()):
    """The runner's request states: request id -> prompt length."""
    return {r: types.SimpleNamespace(num_prompt_tokens=p) for r, p in prompts}


def _batch(req_ids=()):
    """An input batch holding these requests."""
    return types.SimpleNamespace(req_id_to_index={r: i for i, r in enumerate(req_ids)})


def _new(rid, computed, prompt_len):
    return NewRequestData(
        req_id=rid,
        prompt_token_ids=[0] * prompt_len,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=computed,
        lora_request=None,
    )


def test_strips_padding_on_new_request():
    so = _so(
        {"crash": 2, "other": 4096},
        {"crash": [-1]},
        new=[_new("crash", 126208, 126209), _new("other", 0, 9000)],
    )
    out = _strip_prefill_only_padding(so, _requests())
    assert out.num_scheduled_tokens == {"crash": 1, "other": 4096}
    assert out.total_num_scheduled_tokens == 4097
    assert "crash" not in out.scheduled_spec_decode_tokens
    assert out.scheduled_new_reqs is so.scheduled_new_reqs


def test_the_scheduler_s_step_is_left_untouched():
    """Under a uniprocess executor the scheduler holds this object, and it
    rolls the placeholder back as a rejected draft only if it still sees it."""
    so = _so({"crash": 2}, {"crash": [-1]}, new=[_new("crash", 126208, 126209)])
    out = _strip_prefill_only_padding(so, _requests())
    assert out is not so
    assert so.num_scheduled_tokens == {"crash": 2}
    assert so.total_num_scheduled_tokens == 2
    assert so.scheduled_spec_decode_tokens == {"crash": [-1]}


def test_strips_padding_on_resumed_cached_request():
    so = _so({"r": 2}, {"r": [-1]}, cached=_cached(["r"], [126208]))
    out = _strip_prefill_only_padding(so, _requests([("r", 126209)]))
    assert out.num_scheduled_tokens["r"] == 1


def test_leaves_real_drafts_and_in_prompt_steps_alone():
    # Real draft ids (not placeholders): untouched (the runner's guard still
    # refuses them).
    so = _so({"a": 2}, {"a": [17]}, new=[_new("a", 126208, 126209)])
    assert _strip_prefill_only_padding(so, _requests()) is so
    # Placeholders but the step stays inside the prompt: untouched.
    so = _so({"b": 2}, {"b": [-1]}, new=[_new("b", 100, 5000)])
    assert _strip_prefill_only_padding(so, _requests()) is so
    assert so.num_scheduled_tokens["b"] == 2


@pytest.mark.parametrize("computed", [126209, 126300])
def test_leaves_steps_past_the_prompt_alone(computed):
    """Under async scheduling a decode or verify step also carries -1
    placeholders. It starts at or past the end of the prompt, so the step
    without them still crosses it: untouched, and the runner's guard refuses
    it."""
    cached = _so({"r": 2}, {"r": [-1]}, cached=_cached(["r"], [computed]))
    assert _strip_prefill_only_padding(cached, _requests([("r", 126209)])) is cached
    new = _so({"n": 2}, {"n": [-1]}, new=[_new("n", computed, 126209)])
    assert _strip_prefill_only_padding(new, _requests()) is new
    assert new.num_scheduled_tokens == {"n": 2}


class _Stop(Exception):
    pass


def _stub_runner(prefill_only, **fields):
    return types.SimpleNamespace(
        execute_model_state=None,
        _adaptive_verification=None,
        _mtp_prefill_only=prefill_only,
        **fields,
    )


def _execute_until_update_states(monkeypatch, prefill_only):
    """Run TPUModelRunner.execute_model on a stub runner up to _update_states,
    the first step after the strip, and report whether the strip ran first and
    on the runner's requests."""
    calls = []
    requests = {}
    stripped = _so({"r": 1}, {})

    def _strip(so, reqs):
        calls.append(("strip", so, reqs))
        return stripped

    monkeypatch.setattr(tpu_runner, "_strip_prefill_only_padding", _strip)
    runner = _stub_runner(prefill_only, requests=requests)

    def _update_states(so):
        calls.append(("update", so))
        raise _Stop

    runner._update_states = _update_states
    so = _so({"r": 2}, {"r": [-1]})
    with pytest.raises(_Stop):
        TPUModelRunner.execute_model(runner, so)
    return calls, so, requests, stripped


def test_execute_model_strips_before_reading_the_step(monkeypatch):
    calls, so, requests, stripped = _execute_until_update_states(
        monkeypatch, prefill_only=True
    )
    # The rest of the step runs on the stripped copy.
    assert calls == [("strip", so, requests), ("update", stripped)]


def test_execute_model_does_not_strip_off_the_prefill_only_path(monkeypatch):
    calls, so, _, _ = _execute_until_update_states(monkeypatch, prefill_only=False)
    assert calls == [("update", so)]


def test_a_request_skipped_in_the_last_step_is_stripped():
    """vLLM's _update_states drops a request it did not schedule from the input
    batch and adds it back when it is scheduled again; the strip runs before
    that, so it reads the prompt length from the runner's requests."""
    seen = []

    def _update_states(so):
        seen.append(so)
        raise _Stop

    runner = _stub_runner(
        True,
        input_batch=_batch(),  # "r" was not scheduled in the previous step
        requests=_requests([("r", 126209)]),
        _update_states=_update_states,
    )
    so = _so({"r": 2}, {"r": [-1]}, cached=_cached(["r"], [126208]))
    with pytest.raises(_Stop):
        TPUModelRunner.execute_model(runner, so)
    (out,) = seen
    assert out.num_scheduled_tokens == {"r": 1}
    assert "r" not in out.scheduled_spec_decode_tokens


def _scheduler(tmp_path, scheduler_cls, chunk):
    """vLLM's scheduler over a 2-layer OPT config and a 16-block cache."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 2048,
                "vocab_size": 1000,
            }
        )
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=chunk,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=scheduler_cls is AsyncScheduler,
        ),
        model_config=ModelConfig(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float16",
            seed=42,
            max_model_len=2048,
        ),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = 16
    register_all_kvcache_specs(vllm_config)
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=16,
    )


def _sampled(so, tokens):
    req_ids = list(so.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[tokens] * len(req_ids),
    )


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
def test_the_scheduler_counts_the_dropped_placeholder_as_rejected(
    tmp_path, scheduler_cls
):
    """vLLM's scheduler, reading its own step, rolls the placeholder the worker
    dropped back as a rejected draft: the request ends the step at its prompt
    length with no output placeholder left, holding the sampled token."""
    prompt = 33
    scheduler = _scheduler(tmp_path, scheduler_cls, chunk=prompt - 1)
    request = Request(
        "r",
        list(range(1, prompt + 1)),
        SamplingParams(max_tokens=4, detokenize=False),
        None,
    )
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"r": prompt - 1}
    scheduler.update_from_output(first, _sampled(first, []))
    # The serving crash's state: one prompt token left and a -1 placeholder.
    request.spec_token_ids = [-1]
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"r": 2}
    assert step.scheduled_spec_decode_tokens == {"r": [-1]}
    ran = _strip_prefill_only_padding(step, _requests([("r", prompt)]))
    assert ran.num_scheduled_tokens == {"r": 1}
    scheduler.update_from_output(step, _sampled(ran, [7]))
    assert request.num_computed_tokens == prompt
    assert request.num_output_placeholders == 0
    assert list(request.output_token_ids) == [7]
