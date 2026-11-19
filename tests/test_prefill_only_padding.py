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
126209)."""

import dataclasses
import types

import numpy as np
import pytest
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

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


def _batch(prompts=()):
    """An input batch holding these requests: request id -> prompt length."""
    ids = {r: i for i, (r, _) in enumerate(prompts)}
    return types.SimpleNamespace(
        req_id_to_index=ids, num_prompt_tokens=np.array([p for _, p in prompts] or [0])
    )


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


class _Stop(Exception):
    pass


def _stub_runner(prefill_only, **fields):
    return types.SimpleNamespace(
        _pp_wave=None,
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
