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
"""Warm-ups of the compiled async spec-step programs: each runs at every shape
serving gives it, under execute_model's no_grad (dynamo guards on grad mode),
so the first serving call at a bucket finds its graph."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
import vllm_torchtpu.runner.tpu_runner_async_output as async_output
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

NUM_TOKENS_PADDINGS = [16, 32, 64]
NUM_REQS_PADDINGS = [8, 16]
K = 3


def _runner():
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        _is_async_drafter=True,
        max_num_reqs=16,
        num_reqs_paddings=NUM_REQS_PADDINGS,
        num_tokens_paddings=NUM_TOKENS_PADDINGS,
        device=torch.device("cpu"),
        _precompile_timed=lambda _label: contextlib.nullcontext(),
    )
    runner._precompile_draft_extract = lambda: (
        TPUModelRunner._precompile_draft_extract(runner)
    )
    return runner


@pytest.fixture
def recorded(monkeypatch):
    calls: dict[str, list] = {"substitute": [], "fused": [], "extract": []}

    def substitute(input_ids, indices, next_tokens):
        calls["substitute"].append(torch.is_grad_enabled())
        return input_ids

    def fused(input_ids, indices, next_tokens, logits_indices, target_indices):
        calls["fused"].append(torch.is_grad_enabled())
        return input_ids, input_ids[: target_indices.shape[0]]

    def extract(input_ids, logits_indices, target_indices):
        calls["extract"].append(
            (
                input_ids.shape[0],
                logits_indices.shape[0],
                target_indices.shape[0],
                input_ids.dtype,
                torch.is_grad_enabled(),
            )
        )
        return input_ids[: target_indices.shape[0]]

    monkeypatch.setattr(runner_mod, "_substitute_placeholder_token", substitute)
    monkeypatch.setattr(runner_mod, "_substitute_and_extract_draft_token_ids", fused)
    monkeypatch.setattr(async_output, "_compiled_extract_draft_token_ids", extract)
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *_a, **_k: None)
    return calls


def test_substitution_precompile_runs_under_no_grad(recorded):
    TPUModelRunner._precompile_substitute_placeholder_token(_runner())
    assert recorded["substitute"] and recorded["fused"]
    assert not any(recorded["substitute"] + recorded["fused"])


def test_the_separate_draft_extract_is_warmed_at_every_reachable_shape(recorded):
    """A step whose logits rows do not cover its tokens extracts its drafts
    apart: logits and target indices of the padded logits length L, and an
    input cut to L, or the step's N > L tokens when a draft lies past L."""
    TPUModelRunner._precompile_substitute_placeholder_token(_runner())
    expected = {
        (num_tokens, length, length)
        for length in NUM_TOKENS_PADDINGS
        for num_tokens in NUM_TOKENS_PADDINGS
        if num_tokens >= length
    }
    shapes = [call[:3] for call in recorded["extract"]]
    assert sorted(shapes) == sorted(expected)
    assert {call[3] for call in recorded["extract"]} == {torch.int32}
    assert not any(call[4] for call in recorded["extract"])


@pytest.mark.parametrize(
    "num_draft_tokens, num_scheduled_tokens",
    [
        # Three requests verify one draft each before a 40-token prefill
        # chunk: every draft lies below the logits length, so the input is
        # cut to it.
        ([1, 1, 1, 0], [2, 2, 2, 40]),
        # The prefill chunk comes first: the drafts lie past the logits
        # length, so the extract reads the step's padded tokens.
        ([0, 1, 1, 1], [40, 2, 2, 2]),
    ],
)
def test_the_warmed_draft_extract_serves_a_spec_step_without_compiling(
    monkeypatch, num_draft_tokens, num_scheduled_tokens
):
    """The real extract, compiled whole with static shapes on a backend that
    counts its compiles (dynamo's guards decide whether a call reuses a warmed
    graph, whatever the backend), warmed by the substitution precompile and
    then called as execute_model calls it, on metadata from the real
    SpeculativeDecodingManager."""
    compiles = []

    def counting_backend(gm, example_inputs):
        compiles.append([tuple(t.shape) for t in example_inputs])
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        async_output.extract_draft_token_ids,
        backend=counting_backend,
        fullgraph=True,
        dynamic=False,
    )
    monkeypatch.setattr(async_output, "_compiled_extract_draft_token_ids", compiled)
    monkeypatch.setattr(
        runner_mod, "_substitute_placeholder_token", lambda ids, *_: ids
    )
    monkeypatch.setattr(
        runner_mod,
        "_substitute_and_extract_draft_token_ids",
        lambda ids, *_: (ids, ids),
    )
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *_a, **_k: None)

    TPUModelRunner._precompile_substitute_placeholder_token(_runner())
    warmed = len(compiles)
    assert warmed == 6  # (N, L, L) for every L and every N >= L

    manager = SpeculativeDecodingManager(
        SimpleNamespace(
            device=torch.device("cpu"),
            num_tokens_paddings=NUM_TOKENS_PADDINGS,
            arange_np=np.arange(256, dtype=np.int32),
            input_ids_cpu=torch.zeros(256, dtype=torch.int32),
            _unfenced_h2d_sources=[],
            _registered_h2d_sources=set(),
        )
    )
    metadata = manager.get_spec_decode_metadata(
        np.array(num_draft_tokens, dtype=np.int32),
        np.cumsum(np.array(num_scheduled_tokens, dtype=np.int32)),
        padded_num_reqs=8,
        defer_draft_token_ids=True,
    )
    input_ids = torch.zeros(64, dtype=torch.int32)
    with torch.no_grad():
        compiled(
            runner_mod._draft_extract_input_ids(input_ids, metadata),
            metadata.final_logits_indices,
            metadata.target_logits_indices,
        )
    assert len(compiles) == warmed, compiles[warmed:]


def test_non_spec_chunk_padding_is_warmed_at_every_request_bucket(monkeypatch):
    calls = []

    def pad(selected, k):
        calls.append(
            (tuple(selected.shape), selected.dtype, k, torch.is_grad_enabled())
        )
        return selected

    monkeypatch.setattr(async_output, "_compiled_pad_non_spec_next_tokens", pad)
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *_a, **_k: None)
    TPUModelRunner._precompile_non_spec_chunk_padding(_runner())
    assert calls == [((n, 1), torch.int64, K, False) for n in NUM_REQS_PADDINGS]


class _WarmupStep:
    """A private runner method of the stand-in below: calling it records its
    name. It is falsy, so a private flag the stand-in does not set reads as
    off."""

    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    def __call__(self, *_args, **_kwargs):
        self.calls.append(self.name)

    def __bool__(self):
        return False


class _WarmupFlowRunner(SimpleNamespace):
    """A runner for capture_model whose private methods only record their
    names, so the test follows capture_model's own flow."""

    def __getattr__(self, name):
        if name.startswith("_") and not name.startswith("__"):
            return _WarmupStep(self.calls, name)
        raise AttributeError(name)


def test_the_async_drafter_warmup_pads_draft_free_chunks(monkeypatch):
    """capture_model runs the non-spec chunk padding warm-up for an async
    drafter that verifies: serving pads a draft-free chunk in a spec step."""
    calls = []
    runner = _WarmupFlowRunner(
        calls=calls,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        _pp_handoff=None,
        enforce_eager=False,
        lora_config=None,
        maybe_setup_dummy_loras=lambda _config: contextlib.nullcontext(),
        _precompile_timed=lambda _label: contextlib.nullcontext(),
        vllm_config=None,
        device=torch.device("cpu"),
        model=None,
        _mtp_prefill_only=False,
        _is_async_drafter=True,
        _async_drafter=SimpleNamespace(
            precompile=lambda: calls.append("drafter.precompile")
        ),
        _adaptive_verification=None,
    )
    monkeypatch.setattr(
        runner_mod, "maybe_create_mm_encoder_manager", lambda *_a, **_k: None
    )
    TPUModelRunner.capture_model(runner)
    assert "_precompile_rejection_sampler" in calls
    assert "_precompile_non_spec_chunk_padding" in calls
