# SPDX-License-Identifier: Apache-2.0
"""The runner's logits row counts for a vocab-parallel LM head.

Every DP rank must call the head with the same row count, so the runner pads
each chunk's logits rows to a DP-wide bound; warmup must compile every shape
that bound produces.
"""

import contextlib
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
    "_vocab_parallel_row_counts",
    "_padded_logits_rows",
    "_selected_logits",
    "_logits_from_hidden",
    "_precompile_compute_selected_logits",
    "_precompile_compute_logits_from_hidden_states",
)


def _runner(monkeypatch, max_num_reqs, paddings, k):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **kw: None)
    weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    calls = SimpleNamespace(selected=[], hidden=[])

    def compute_selected_logits(hidden, indices):
        calls.selected.append((hidden.shape[0], indices.shape[0]))
        return hidden[indices.long()] @ weight.T

    def compute_logits_from_hidden_states(hidden):
        calls.hidden.append(hidden.shape[0])
        return hidden @ weight.T

    runner = SimpleNamespace(
        model=SimpleNamespace(lm_head=SimpleNamespace(vocab_parallel_op=object())),
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
        compute_selected_logits=compute_selected_logits,
        compute_logits_from_hidden_states=compute_logits_from_hidden_states,
        calls=calls,
    )
    for name in RUNNER_METHODS:
        setattr(runner, name, MethodType(getattr(TPUModelRunner, name), runner))
    return runner


@pytest.mark.parametrize("setup", SETUPS)
def test_warmup_compiles_every_shape_the_padded_head_takes(monkeypatch, setup):
    runner = _runner(monkeypatch, *setup)
    runner._precompile_compute_selected_logits()
    runner._precompile_compute_logits_from_hidden_states()
    warmed = set(runner.calls.selected)
    for max_reqs in range(runner.max_num_reqs + 1):
        runner._dp_step_max_reqs = max_reqs
        rows = runner._vocab_parallel_rows()
        assert rows in runner.calls.hidden
        for num_tokens in runner.num_tokens_paddings:
            assert (num_tokens, rows) in warmed


@pytest.mark.parametrize("setup", SETUPS)
def test_bound_covers_every_chunk_of_the_step(monkeypatch, setup):
    """A chunk of n requests on any rank, under a DP-wide largest chunk of
    max_reqs, has at most the bound's rows: the padded request count, and
    padded_logits_length with speculative decoding."""
    runner = _runner(monkeypatch, *setup)
    paddings, spec = runner.num_tokens_paddings, runner.speculative_config
    for max_reqs in range(1, runner.max_num_reqs + 1):
        runner._dp_step_max_reqs = max_reqs
        bound = runner._vocab_parallel_rows()
        for n in range(1, max_reqs + 1):
            counts = [_get_padded_num_reqs_with_upper_limit(n, runner.max_num_reqs)]
            if spec is not None:
                indices = min(n * (1 + spec.num_speculative_tokens), paddings[-1])
                counts.append(_get_padded_token_len(paddings, indices))
            for rows in counts:
                assert runner._padded_logits_rows(rows) == bound


def test_rows_over_the_bound_raise(monkeypatch):
    runner = _runner(monkeypatch, *SETUPS[0])
    runner._dp_step_max_reqs = 1
    bound = runner._vocab_parallel_rows()
    with pytest.raises(RuntimeError, match="would not pair"):
        runner._padded_logits_rows(bound + 1)


def test_padded_logits_equal_the_unpadded_call(monkeypatch):
    runner = _runner(monkeypatch, *SETUPS[2])
    runner._dp_step_max_reqs = 3
    bound = runner._vocab_parallel_rows()
    hidden = torch.randn(64, HIDDEN)
    indices = torch.tensor([5, 0, 17, 63, 2], dtype=torch.int32)

    got = runner._selected_logits(hidden, indices)
    assert runner.calls.selected[-1] == (64, bound)
    torch.testing.assert_close(got, runner.compute_selected_logits(hidden, indices))

    rows = hidden[:5]
    got = runner._logits_from_hidden(rows)
    assert runner.calls.hidden[-1] == bound
    torch.testing.assert_close(got, runner.compute_logits_from_hidden_states(rows))


def test_whole_head_keeps_the_rows(monkeypatch):
    runner = _runner(monkeypatch, *SETUPS[0])
    runner.model.lm_head = SimpleNamespace()
    assert runner._vocab_parallel_rows() is None
    assert runner._vocab_parallel_row_counts() == []
    assert runner._padded_logits_rows(7) == 7
