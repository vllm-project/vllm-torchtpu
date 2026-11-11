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

import types

import numpy as np
import pytest

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner, _strip_prefill_only_padding


def _so(num_sched, spec, new=(), cached=None):
    return types.SimpleNamespace(
        num_scheduled_tokens=dict(num_sched),
        total_num_scheduled_tokens=sum(num_sched.values()),
        scheduled_spec_decode_tokens=dict(spec),
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached
        or types.SimpleNamespace(req_ids=[], num_computed_tokens=[]),
    )


def _batch(prompts=()):
    ids = {r: i for i, (r, _) in enumerate(prompts)}
    return types.SimpleNamespace(
        req_id_to_index=ids, num_prompt_tokens=np.array([p for _, p in prompts] or [0])
    )


def _new(rid, computed, prompt_len):
    return types.SimpleNamespace(
        req_id=rid, num_computed_tokens=computed, prompt_token_ids=[0] * prompt_len
    )


def test_strips_padding_on_new_request():
    so = _so(
        {"crash": 2, "other": 4096},
        {"crash": [-1]},
        new=[_new("crash", 126208, 126209), _new("other", 0, 9000)],
    )
    assert _strip_prefill_only_padding(so, _batch()) == 1
    assert so.num_scheduled_tokens == {"crash": 1, "other": 4096}
    assert so.total_num_scheduled_tokens == 4097
    assert "crash" not in so.scheduled_spec_decode_tokens


def test_strips_padding_on_resumed_cached_request():
    cached = types.SimpleNamespace(req_ids=["r"], num_computed_tokens=[126208])
    so = _so({"r": 2}, {"r": [-1]}, cached=cached)
    assert _strip_prefill_only_padding(so, _batch([("r", 126209)])) == 1
    assert so.num_scheduled_tokens["r"] == 1


def test_leaves_real_drafts_and_in_prompt_steps_alone():
    # Real draft ids (not placeholders): untouched (the runner's guard still
    # refuses them).
    so = _so({"a": 2}, {"a": [17]}, new=[_new("a", 126208, 126209)])
    assert _strip_prefill_only_padding(so, _batch()) == 0
    # Placeholders but the step stays inside the prompt: untouched.
    so = _so({"b": 2}, {"b": [-1]}, new=[_new("b", 100, 5000)])
    assert _strip_prefill_only_padding(so, _batch()) == 0
    assert so.num_scheduled_tokens["b"] == 2


class _Stop(Exception):
    pass


def _execute_until_update_states(monkeypatch, prefill_only):
    """Run TPUModelRunner.execute_model on a stub runner up to _update_states,
    the first step after the strip, and report whether the strip ran first and
    on the runner's input batch."""
    calls = []
    batch = types.SimpleNamespace()
    monkeypatch.setattr(
        tpu_runner,
        "_strip_prefill_only_padding",
        lambda so, input_batch: calls.append(("strip", so, input_batch)),
    )
    runner = types.SimpleNamespace(
        _pp_wave=None,
        execute_model_state=None,
        _adaptive_verification=None,
        _mtp_prefill_only=prefill_only,
        input_batch=batch,
    )

    def _update_states(so):
        calls.append(("update", so))
        raise _Stop

    runner._update_states = _update_states
    so = _so({"r": 2}, {"r": [-1]})
    with pytest.raises(_Stop):
        TPUModelRunner.execute_model(runner, so)
    return calls, so, batch


def test_execute_model_strips_before_reading_the_step(monkeypatch):
    calls, so, batch = _execute_until_update_states(monkeypatch, prefill_only=True)
    assert calls == [("strip", so, batch), ("update", so)]


def test_execute_model_does_not_strip_off_the_prefill_only_path(monkeypatch):
    calls, so, _ = _execute_until_update_states(monkeypatch, prefill_only=False)
    assert calls == [("update", so)]
