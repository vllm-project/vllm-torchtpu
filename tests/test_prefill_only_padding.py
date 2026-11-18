# SPDX-License-Identifier: Apache-2.0
"""Prefill-only MTP: the worker strips draft-placeholder rows vLLM padded onto a final prompt token
(band-A crash: computed 126208, scheduled 2, prompt 126209), prefill-kernel agent."""
import types

import numpy as np

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _so(num_sched, spec, new=(), cached=None):
    return types.SimpleNamespace(
        num_scheduled_tokens=dict(num_sched), total_num_scheduled_tokens=sum(num_sched.values()),
        scheduled_spec_decode_tokens=dict(spec), scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached or types.SimpleNamespace(req_ids=[], num_computed_tokens=[]))


def _runner(prompts=()):
    ids = {r: i for i, (r, _) in enumerate(prompts)}
    ib = types.SimpleNamespace(req_id_to_index=ids, num_prompt_tokens=np.array([p for _, p in prompts] or [0]))
    return types.SimpleNamespace(input_batch=ib)


def _new(rid, computed, prompt_len):
    return types.SimpleNamespace(req_id=rid, num_computed_tokens=computed, prompt_token_ids=[0] * prompt_len)


def test_strips_padding_on_new_request():
    so = _so({"crash": 2, "other": 4096}, {"crash": [-1]}, new=[_new("crash", 126208, 126209), _new("other", 0, 9000)])
    assert TPUModelRunner._strip_prefill_only_padding(_runner(), so) == 1
    assert so.num_scheduled_tokens == {"crash": 1, "other": 4096}
    assert so.total_num_scheduled_tokens == 4097
    assert "crash" not in so.scheduled_spec_decode_tokens


def test_strips_padding_on_resumed_cached_request():
    cached = types.SimpleNamespace(req_ids=["r"], num_computed_tokens=[126208])
    so = _so({"r": 2}, {"r": [-1]}, cached=cached)
    assert TPUModelRunner._strip_prefill_only_padding(_runner([("r", 126209)]), so) == 1
    assert so.num_scheduled_tokens["r"] == 1


def test_leaves_real_drafts_and_in_prompt_steps_alone():
    # Real draft ids (not placeholders): untouched (the runner's guard still refuses them).
    so = _so({"a": 2}, {"a": [17]}, new=[_new("a", 126208, 126209)])
    assert TPUModelRunner._strip_prefill_only_padding(_runner(), so) == 0
    # Placeholders but the step stays inside the prompt: untouched.
    so = _so({"b": 2}, {"b": [-1]}, new=[_new("b", 100, 5000)])
    assert TPUModelRunner._strip_prefill_only_padding(_runner(), so) == 0
    assert so.num_scheduled_tokens["b"] == 2


def test_execute_model_strips_before_the_states_and_the_refusal():
    """q38 BLSTACK-RFIX (Benliang Wang's runner): execute_model calls the strip under _mtp_prefill_only before
    _update_states absorbs the draft placeholders and before the prefill-only refusal raises on them."""
    import inspect

    src = inspect.getsource(TPUModelRunner.execute_model)
    gate = src.index("if self._mtp_prefill_only:\n")
    strip = src.index("self._strip_prefill_only_padding(scheduler_output)")
    states = src.index("self._update_states(scheduler_output)")
    refusal = src.index("refusing decode/verify")
    assert gate < strip < states < refusal


def test_strip_count_moves_only_when_a_request_is_trimmed():
    """q38 BLSTACK-RFIX: the per-process count adds trimmed requests and rows, and nothing otherwise."""
    from vllm_torchtpu.runner import tpu_runner as tr

    before = dict(tr._PREFILL_ONLY_STRIP)
    so = _so({"x": 3, "y": 2}, {"x": [-1, -1], "y": [17]}, new=[_new("x", 50, 51), _new("y", 50, 51)])
    assert TPUModelRunner._strip_prefill_only_padding(_runner(), so) == 1
    assert tr._PREFILL_ONLY_STRIP == {"requests": before["requests"] + 1, "rows": before["rows"] + 2}
    so = _so({"b": 2}, {"b": [-1]}, new=[_new("b", 100, 5000)])
    assert TPUModelRunner._strip_prefill_only_padding(_runner(), so) == 0
    assert tr._PREFILL_ONLY_STRIP == {"requests": before["requests"] + 1, "rows": before["rows"] + 2}


def test_strip_count_log_cadence():
    from vllm_torchtpu.runner.tpu_runner import _prefill_only_strip_log_due as due

    assert [n for n in range(0, 2100) if due(n)] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
