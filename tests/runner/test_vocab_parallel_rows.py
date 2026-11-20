# SPDX-License-Identifier: Apache-2.0
"""The runner's logits row counts for a vocab-parallel LM head.

Every DP rank must call the head with the same row count, so the runner pads
each chunk's logits rows to a DP-wide bound inside the compiled logits graph
and cuts them back there; warmup must compile every (num_tokens, rows, padded)
shape that serving calls it with.
"""

import contextlib
from functools import cache
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _get_padded_num_reqs_with_upper_limit,
    _get_padded_token_len,
)

HIDDEN, VOCAB = 8, 16
# (max_num_reqs, token paddings, speculative tokens): the measured decode
# setup (64 requests, K=5, compile sizes [384, 2048]), the 48-request setup,
# a full token ladder, and no speculative decoding.
SETUPS = [
    (64, [384, 2048], 5),
    (48, [288, 2048], 5),
    (64, [16, 32, 64, 128, 256, 512, 1024, 2048], 5),
    (8, [16, 32, 64, 128], None),
]
RUNNER_METHODS = (
    "_uses_vocab_parallel_head",
    "_vocab_parallel_rows_for",
    "_vocab_parallel_rows",
    "_vocab_parallel_logits_shapes",
    "_padded_logits_rows",
    "_compute_padded_logits",
    "_selected_logits",
    "_logits_from_hidden",
    "_precompile_compute_selected_logits",
    "_precompile_compute_logits_from_hidden_states",
)


def _runner(monkeypatch, max_num_reqs, paddings, k):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **kw: None)
    weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    calls = SimpleNamespace(selected=[], hidden=[], head=[])

    def compute_logits(hidden):
        calls.head.append(hidden.shape[0])
        return hidden @ weight.T

    runner = SimpleNamespace(
        model=SimpleNamespace(
            lm_head=SimpleNamespace(vocab_parallel_op=object()),
            compute_logits=compute_logits,
        ),
        is_pooling_model=False,
        speculative_config=(
            None if k is None else SimpleNamespace(num_speculative_tokens=k)
        ),
        num_tokens_paddings=list(paddings),
        num_reqs_paddings=[],
        max_num_reqs=max_num_reqs,
        _dp_step_max_reqs=0,
        _dp_lockstep_enabled=lambda: True,
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.float32,
        model_config=SimpleNamespace(get_hidden_size=lambda: HIDDEN),
        _precompile_timed=lambda name: contextlib.nullcontext(),
        weight=weight,
        calls=calls,
    )
    for name in RUNNER_METHODS:
        setattr(runner, name, MethodType(getattr(TPUModelRunner, name), runner))
    selected = MethodType(TPUModelRunner.compute_selected_logits.__wrapped__, runner)
    from_hidden = MethodType(
        TPUModelRunner.compute_logits_from_hidden_states.__wrapped__, runner
    )

    def compute_selected_logits(hidden, indices, padded_rows=None):
        calls.selected.append((hidden.shape[0], indices.shape[0], padded_rows))
        return selected(hidden, indices, padded_rows)

    def compute_logits_from_hidden_states(hidden, padded_rows=None):
        calls.hidden.append((hidden.shape[0], padded_rows))
        return from_hidden(hidden, padded_rows)

    runner.compute_selected_logits = compute_selected_logits
    runner.compute_logits_from_hidden_states = compute_logits_from_hidden_states
    return runner


def _chunk_logits_rows(runner):
    """The logits row counts of a chunk of n requests, built as serving builds
    them: the padded request count, and with speculative decoding the padded
    length of every draft count, n to n * (1 + K) positions."""
    paddings, spec = runner.num_tokens_paddings, runner.speculative_config

    @cache
    def rows(n):
        counts = {_get_padded_num_reqs_with_upper_limit(n, runner.max_num_reqs)}
        if spec is not None:
            most = min(n * (1 + spec.num_speculative_tokens), paddings[-1])
            counts.update(
                _get_padded_token_len(paddings, length) for length in range(n, most + 1)
            )
        return counts

    return rows


def _serve_every_step(runner):
    """Call the head as serving does: under every DP-wide largest chunk, a
    chunk of every smaller size on each token bucket, and an idle rank."""
    chunk_rows = _chunk_logits_rows(runner)
    for max_reqs in range(runner.max_num_reqs + 1):
        runner._dp_step_max_reqs = max_reqs
        rows = {runner._vocab_parallel_rows()}
        for n in range(1, max_reqs + 1):
            rows |= chunk_rows(n)
        for r in rows:
            indices = torch.zeros(r, dtype=torch.int32)
            for num_tokens in runner.num_tokens_paddings:
                runner._selected_logits(torch.zeros(num_tokens, HIDDEN), indices)
            runner._logits_from_hidden(torch.zeros(r, HIDDEN))


@pytest.mark.parametrize("setup", SETUPS)
def test_warmup_compiles_every_shape_serving_calls(monkeypatch, setup):
    runner = _runner(monkeypatch, *setup)
    runner._precompile_compute_selected_logits()
    runner._precompile_compute_logits_from_hidden_states()
    warmed_selected = set(runner.calls.selected)
    warmed_hidden = set(runner.calls.hidden)
    runner.calls.selected.clear()
    runner.calls.hidden.clear()

    _serve_every_step(runner)

    assert set(runner.calls.selected) == warmed_selected
    assert set(runner.calls.hidden) == warmed_hidden
    assert all(padded is not None for *_, padded in warmed_selected)


@pytest.mark.parametrize("setup", SETUPS)
def test_bound_covers_every_chunk_of_the_step(monkeypatch, setup):
    """A chunk of n requests on any rank, under a DP-wide largest chunk of
    max_reqs, has at most the bound's rows."""
    runner = _runner(monkeypatch, *setup)
    chunk_rows = _chunk_logits_rows(runner)
    for max_reqs in range(1, runner.max_num_reqs + 1):
        runner._dp_step_max_reqs = max_reqs
        bound = runner._vocab_parallel_rows()
        for n in range(1, max_reqs + 1):
            for rows in chunk_rows(n):
                assert runner._padded_logits_rows(rows) == bound


def test_rows_over_the_bound_raise(monkeypatch):
    runner = _runner(monkeypatch, *SETUPS[0])
    runner._dp_step_max_reqs = 1
    bound = runner._vocab_parallel_rows()
    with pytest.raises(RuntimeError, match="would not pair"):
        runner._padded_logits_rows(bound + 1)


@pytest.mark.parametrize("setup", SETUPS)
def test_padded_logits_equal_the_unpadded_call(monkeypatch, setup):
    """For every warmed shape the head runs on the padded rows, and the
    logits functions return the unpadded call's result."""
    runner = _runner(monkeypatch, *setup)
    weight = runner.weight
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(runner.num_tokens_paddings[0], HIDDEN, generator=generator)
    for rows, padded in runner._vocab_parallel_logits_shapes():
        indices = torch.randint(0, hidden.shape[0], (rows,), generator=generator)
        indices = indices.to(torch.int32)
        got = runner.compute_selected_logits(hidden, indices, padded)
        assert runner.calls.head[-1] == padded
        torch.testing.assert_close(got, hidden[indices.long()] @ weight.T)

        selected = hidden[indices.long()]
        got = runner.compute_logits_from_hidden_states(selected, padded)
        assert runner.calls.head[-1] == padded
        torch.testing.assert_close(got, selected @ weight.T)


def test_runner_passes_the_bound_to_the_logits_functions(monkeypatch):
    runner = _runner(monkeypatch, *SETUPS[2])
    runner._dp_step_max_reqs = 3
    bound = runner._vocab_parallel_rows()
    hidden = torch.randn(64, HIDDEN)
    indices = torch.tensor([5, 0, 17, 63, 2], dtype=torch.int32)

    got = runner._selected_logits(hidden, indices)
    assert runner.calls.selected[-1] == (64, 5, bound)
    torch.testing.assert_close(got, hidden[indices.long()] @ runner.weight.T)

    rows = hidden[:5]
    got = runner._logits_from_hidden(rows)
    assert runner.calls.hidden[-1] == (5, bound)
    torch.testing.assert_close(got, rows @ runner.weight.T)


def test_whole_head_keeps_the_rows(monkeypatch):
    runner = _runner(monkeypatch, *SETUPS[0])
    runner.model.lm_head = SimpleNamespace()
    assert runner._vocab_parallel_rows() is None
    assert runner._vocab_parallel_logits_shapes() == []
    assert runner._padded_logits_rows(7) is None

    hidden = torch.randn(16, HIDDEN)
    runner._selected_logits(hidden, torch.arange(7, dtype=torch.int32))
    assert runner.calls.selected[-1] == (16, 7, None)
    assert runner.calls.head[-1] == 7
