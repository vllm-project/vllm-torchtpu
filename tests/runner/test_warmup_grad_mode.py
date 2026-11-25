# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Each precompile pass warms its programs in the grad mode their serving
caller runs in. Dynamo guards on grad mode, so an entry warmed in the other
mode never serves: the first served call of every bucket compiles again.

execute_model runs under no_grad; sample_tokens (sampling, verify, the draft
propose, the boundary-state copies) keeps grad mode on.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

HSIZE = 8
REQS = [1, 2, 4]
TOKENS = [8, 16]


def _counting_compile(fn):
    frames = []

    def backend(gm, _example_inputs):
        frames.append(gm)
        return gm.forward

    return torch.compile(fn, backend=backend, fullgraph=True, dynamic=False), frames


def _lm_head(hidden):
    return hidden.float().sum(-1)


def _selected_lm_head(hidden, indices):
    return hidden.index_select(0, indices).float().sum(-1)


def _logits_runner(**kw):
    return SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: HSIZE),
        num_reqs_paddings=REQS,
        num_tokens_paddings=TOKENS,
        max_num_reqs=REQS[-1],
        _hidden_states_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        _candidate_bounds=None,
        _precompile_timed=lambda name: contextlib.nullcontext(),
        **kw,
    )


@pytest.fixture(autouse=True)
def _no_sync(monkeypatch):
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *a, **k: None)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class _StateProbe:
    """`execute_model_state` reads record the grad mode they happen in."""

    def __init__(self, value):
        self.value = value
        self.grad = []
        self._pp_wave = None
        self._pp_is_last = False

    @property
    def execute_model_state(self):
        self.grad.append(torch.is_grad_enabled())
        return self.value

    def _sample_tokens_pp_intermediate_stage(self):
        return None


def test_execute_model_runs_under_no_grad():
    probe = _StateProbe(value=object())  # set state -> "State error" raise
    with pytest.raises(RuntimeError, match="sample_tokens"):
        TPUModelRunner.execute_model(probe, None)
    assert probe.grad == [False]


def test_sample_tokens_keeps_grad_mode_on():
    probe = _StateProbe(value=None)
    TPUModelRunner.sample_tokens(probe, None)
    assert probe.grad and all(probe.grad)


def test_compute_logits_from_hidden_states_warm_hits_serving_call():
    compiled, frames = _counting_compile(_lm_head)
    runner = _logits_runner(compute_logits_from_hidden_states=compiled)
    TPUModelRunner._precompile_compute_logits_from_hidden_states(runner)
    warmed = len(frames)
    assert warmed == len(REQS)
    with torch.no_grad():  # execute_model's mode
        for n in REQS:
            compiled(torch.zeros((n, HSIZE), dtype=torch.bfloat16))
    assert len(frames) == warmed


@pytest.mark.parametrize("from_hidden", [False, True])
def test_compute_top_tokens_warm_hits_serving_call(monkeypatch, from_hidden):
    selected, sel_frames = _counting_compile(_selected_lm_head)
    plain, plain_frames = _counting_compile(_lm_head)
    runner = _logits_runner(
        _greedy_head_owner=object(),
        compute_selected_top_tokens=selected,
        compute_top_tokens_from_hidden_states=plain,
    )
    monkeypatch.setattr(
        runner_mod,
        "_get_sequence_layout_planner_for_runner",
        lambda _r: SimpleNamespace(uses_selected_logits_hidden_states=from_hidden),
    )
    TPUModelRunner._precompile_compute_top_tokens(runner)
    warmed_sel, warmed_plain = len(sel_frames), len(plain_frames)
    assert warmed_sel > 0 and warmed_plain == (len(REQS) if from_hidden else 0)
    with torch.no_grad():  # execute_model's mode
        for t in TOKENS:
            for n in REQS:
                if n > t:
                    continue
                selected(
                    torch.zeros((t, HSIZE), dtype=torch.bfloat16),
                    torch.zeros(n, dtype=torch.int32),
                )
        if from_hidden:
            for n in REQS:
                plain(torch.zeros((n, HSIZE), dtype=torch.bfloat16))
    assert len(sel_frames) == warmed_sel
    assert len(plain_frames) == warmed_plain


def test_spec_boundary_pairs_warm_hits_serving_call(monkeypatch):
    """The boundary-pair program is served from sample_tokens with grad mode
    on, so its warmup must not run under no_grad."""
    compiled, frames = _counting_compile(runner_mod._spec_boundary_pairs_impl)
    monkeypatch.setattr(runner_mod, "_spec_boundary_pairs_compiled", compiled)
    width, plan = 2, [(0, []), (1, [])]
    runner = SimpleNamespace(
        _copies_spec_boundary_states=lambda: True,
        _mamba_ckpt_window=width - 1,
        _mamba_copy_plan=plan,
        max_num_reqs=4,
        _bucket_len=TPUModelRunner._bucket_len,
        device=torch.device("cpu"),
        _mamba_column_stride=lambda: 16,
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    TPUModelRunner._precompile_spec_boundary_pairs(runner)
    warmed = len(frames)
    assert warmed > 0
    num_cols = 3 + len(plan) * width
    # The padded candidate counts `_plan_spec_boundary_copies` hands over.
    padded = {
        min(TPUModelRunner._bucket_len(n), max(n, runner.max_num_reqs))
        for n in range(1, runner.max_num_reqs + 1)
    }
    with torch.enable_grad():  # sample_tokens' mode
        for n in sorted(padded):
            compiled(
                torch.zeros((n, num_cols), dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
                16,
                width,
            )
    assert len(frames) == warmed


def test_warm_guard_flags_default_on(monkeypatch):
    """Renders that never set them (za11 A16) recompiled every warmed
    substitute bucket (thread-count guard) on its first served step."""
    monkeypatch.delenv("VLLM_TPU_WARM_SERVING_THREADS", raising=False)
    monkeypatch.delenv("VLLM_TPU_SUBTRACT_REJECTED_WARM_NO_GRAD", raising=False)
    assert envs.VLLM_TPU_WARM_SERVING_THREADS is True
    assert envs.VLLM_TPU_SUBTRACT_REJECTED_WARM_NO_GRAD is True
