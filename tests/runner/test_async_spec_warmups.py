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
serving gives it, in its serving call's grad mode (dynamo guards on grad mode):
execute_model's programs under its no_grad, sample_tokens' in the caller's
mode. So the first serving call at a bucket finds its graph."""

import contextlib
import inspect
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
from vllm_torchtpu.spec_decode.utils import _static_shape_args
from vllm_torchtpu.worker.tpu_worker import TPUWorker

NUM_TOKENS_PADDINGS = [16, 32, 64]
NUM_REQS_PADDINGS = [8, 16]
K = 3
MAX_NUM_REQS = 16


def _runner(most_model_len_num_reqs=None):
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        _is_async_drafter=True,
        max_num_reqs=MAX_NUM_REQS,
        num_reqs_paddings=NUM_REQS_PADDINGS,
        num_tokens_paddings=NUM_TOKENS_PADDINGS,
        num_reqs_max_model_len=MAX_NUM_REQS,
        most_model_len=None if most_model_len_num_reqs is None else 1024,
        num_reqs_most_model_len=most_model_len_num_reqs,
        uses_mrope=False,
        device=torch.device("cpu"),
        _precompile_timed=lambda _label: contextlib.nullcontext(),
    )
    runner._precompile_draft_extract = lambda: (
        TPUModelRunner._precompile_draft_extract(runner)
    )
    return runner


@pytest.fixture
def fresh_dynamo():
    """Dynamo's caches start and end empty, so compile counts are this test's
    and its graphs do not outlive it."""
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


def _count_compiles(monkeypatch, module, name) -> list:
    """Replace `module.name`, a static-shape compiled program, with its
    function compiled whole on a backend that records each compile. Dynamo's
    guards decide whether a call reuses a graph, whatever the backend."""
    compiles = []

    def counting_backend(gm, example_inputs):
        compiles.append(
            [tuple(t.shape) for t in example_inputs if isinstance(t, torch.Tensor)]
        )
        return gm.forward

    fn = inspect.unwrap(getattr(module, name))
    compiled = torch.compile(
        fn, backend=counting_backend, fullgraph=True, dynamic=False
    )
    monkeypatch.setattr(module, name, _static_shape_args(compiled))
    return compiles


class _StepStopped(Exception):
    pass


class _EntryRunner:
    """A runner stand-in that runs `call` at the entry point's first read of
    it, inside the grad mode the entry point gives its body, then stops."""

    def __init__(self, call):
        self._call = call

    def __getattr__(self, name):
        self._call()
        raise _StepStopped


def _inside_sample_tokens(call) -> None:
    """Run `call` where sample_tokens' body runs, in the grad mode serving
    gives it: inside TPUWorker.sample_tokens and TPUModelRunner.sample_tokens,
    called as the engine calls them."""
    runner = _EntryRunner(call)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sample_tokens=lambda grammar_output: TPUModelRunner.sample_tokens(
                runner, grammar_output
            )
        )
    )
    with pytest.raises(_StepStopped):
        TPUWorker.sample_tokens(worker, None)


def _inside_execute_model(call) -> None:
    """Run `call` where execute_model prepares a step's inputs, in the grad
    mode serving gives it: inside TPUModelRunner.execute_model."""
    with pytest.raises(_StepStopped):
        TPUModelRunner.execute_model(_EntryRunner(call), None)


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


@pytest.mark.parametrize(
    "max_num_reqs, max_logits_len",
    # The logits length L is at most the bucket of max_num_reqs * (1 + K) rows.
    [(MAX_NUM_REQS, 64), (4, 16)],
)
def test_the_separate_draft_extract_is_warmed_at_every_reachable_shape(
    recorded, max_num_reqs, max_logits_len
):
    """A step whose logits rows do not cover its tokens extracts its drafts
    apart: logits and target indices of the padded logits length L, and an
    input cut to L, or the step's N > L tokens when a draft lies past L."""
    runner = _runner()
    runner.max_num_reqs = max_num_reqs
    TPUModelRunner._precompile_substitute_placeholder_token(runner)
    expected = {
        (num_tokens, length, length)
        for length in NUM_TOKENS_PADDINGS
        for num_tokens in NUM_TOKENS_PADDINGS
        if max_logits_len >= length and num_tokens >= length
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
@pytest.mark.usefixtures("fresh_dynamo")
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
    assert calls == [((n, 1), torch.int64, K, True) for n in NUM_REQS_PADDINGS]


@pytest.mark.usefixtures("fresh_dynamo")
@pytest.mark.parametrize("num_reqs", NUM_REQS_PADDINGS)
def test_the_warmed_chunk_padding_serves_sample_tokens_without_compiling(
    monkeypatch, num_reqs
):
    """The real padding program, warmed as capture_model warms it and then
    called from inside sample_tokens, as a spec step pads a draft-free chunk:
    the serving call reuses a warmed graph. A warm-up in another grad mode
    than sample_tokens' leaves the call to compile while serving."""
    compiles = _count_compiles(
        monkeypatch, async_output, "_compiled_pad_non_spec_next_tokens"
    )
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *_a, **_k: None)
    TPUModelRunner._precompile_non_spec_chunk_padding(_runner())
    warmed = len(compiles)
    assert warmed == len(NUM_REQS_PADDINGS)

    selected = torch.zeros((num_reqs, 1), dtype=torch.int64)
    _inside_sample_tokens(
        lambda: async_output._compiled_pad_non_spec_next_tokens(selected, K)
    )
    assert len(compiles) == warmed, compiles[warmed:]


def _subtract_rejected(num_tokens, num_scheduled, target_num_reqs):
    """Call the subtraction with the inputs `_prepare_inputs` gives it for a
    step of `num_scheduled` tokens per request, every request in the previous
    step's output, padded to `num_tokens`."""
    seq_lens_idx = np.full(target_num_reqs, -1, dtype=np.int32)
    positions_idx = np.full(num_tokens, -1, dtype=np.int32)
    start = 0
    for i, n in enumerate(num_scheduled):
        seq_lens_idx[i] = i
        positions_idx[start : start + n] = i
        start += n
    rejected = torch.zeros(len(num_scheduled), dtype=torch.int32)
    async_output._compiled_subtract_num_rejected_tokens(
        torch.zeros(256, dtype=torch.int32)[:target_num_reqs],
        torch.zeros(256, dtype=torch.int32)[:num_tokens],
        torch.nn.functional.pad(rejected, (0, MAX_NUM_REQS - rejected.shape[0])),
        torch.from_numpy(seq_lens_idx),
        torch.from_numpy(positions_idx),
    )


@pytest.mark.usefixtures("fresh_dynamo")
@pytest.mark.parametrize(
    "num_tokens, num_scheduled, most_model_len_num_reqs",
    [
        # Three requests verify K drafts each beside two draft-free ones.
        (16, [1 + K] * 3 + [1] * 2, None),
        (32, [1 + K] * 6 + [1] * 2, None),
        (64, [1 + K] * 12 + [1] * 4, None),
        # A step in the most-model-len request layout.
        (16, [1 + K] * 2 + [1] * 3, 8),
    ],
)
def test_the_warmed_rejected_token_subtraction_serves_a_step_without_compiling(
    monkeypatch, num_tokens, num_scheduled, most_model_len_num_reqs
):
    """The real subtraction, warmed as capture_model warms it and then called
    from inside execute_model at a step that mixes verify windows with
    draft-free requests: the serving call reuses a warmed graph."""
    compiles = _count_compiles(
        monkeypatch, async_output, "_compiled_subtract_num_rejected_tokens"
    )
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *_a, **_k: None)
    runner = _runner(most_model_len_num_reqs)
    TPUModelRunner._precompile_rejected_token_subtraction(runner)
    warmed = len(compiles)
    layouts = 1 if most_model_len_num_reqs is None else 2
    assert warmed == layouts * len(NUM_TOKENS_PADDINGS)

    target_num_reqs = most_model_len_num_reqs or MAX_NUM_REQS
    _inside_execute_model(
        lambda: _subtract_rejected(num_tokens, num_scheduled, target_num_reqs)
    )
    assert len(compiles) == warmed, compiles[warmed:]


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
    """capture_model runs the non-spec chunk padding and rejected-token
    subtraction warm-ups for an async drafter that verifies: serving pads a
    draft-free chunk in a spec step and subtracts the previous step's
    rejected drafts."""
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
    assert "_precompile_rejected_token_subtraction" in calls
