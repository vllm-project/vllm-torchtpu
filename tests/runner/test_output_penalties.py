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
"""``presence_penalty`` and ``frequency_penalty`` on TPU: the device op against
vLLM's ``apply_penalties``, the tracker on a real ``InputBatch``, and the
runner through ``sample_tokens``."""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from runner_input_test_utils import _input_batch, _request
from runner_output_test_utils import plain_step_runner, run_sample_tokens
from vllm import SamplingParams
from vllm.model_executor.layers.utils import apply_penalties
from vllm.v1.outputs import LogprobsTensors

from vllm_torchtpu.layers.adapter.sample import output_penalties as penalty_op
from vllm_torchtpu.layers.adapter.sample.repetition_penalty import (
    scale_seen_logits,
)
from vllm_torchtpu.runner import min_tokens, output_penalties, tpu_runner
from vllm_torchtpu.runner.repetition_penalty import RepetitionPenaltyTracker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

VOCAB = 32
NEG_INF = float("-inf")


@pytest.fixture(autouse=True)
def _eager():
    with torch.compiler.set_stance("force_eager"):
        yield


def _pad(rows, value):
    width = max(1, max(len(r) for r in rows))
    return torch.tensor([r + [value] * (width - len(r)) for r in rows])


def _vllm(logits, prompts, outputs, rep, pres, freq):
    """vLLM's apply_penalties on float32 logits, as its sampler calls it (the
    padding id is the vocabulary size)."""
    return apply_penalties(
        logits.to(torch.float32).clone(),
        _pad(prompts, VOCAB),
        _pad(outputs, VOCAB),
        torch.tensor(pres, dtype=torch.float32),
        torch.tensor(freq, dtype=torch.float32),
        torch.tensor(rep, dtype=torch.float32),
    )


def _counts_table(outputs, num_slots=8):
    """The device table with request ``i``'s output counted in row ``i``."""
    counts = torch.zeros((num_slots + 1, VOCAB), dtype=torch.int32)
    for slot, output in enumerate(outputs):
        ids = torch.full((16,), -1, dtype=torch.int32)
        ids[: len(output)] = torch.tensor(output, dtype=torch.int32)
        penalty_op.add_to_slot(counts, torch.tensor([slot]), ids)
    return counts


def _ours(logits, outputs, pres, freq):
    counts = _counts_table(outputs)
    slots = torch.arange(len(outputs), dtype=torch.int32)
    params = torch.tensor(list(zip(pres, freq)), dtype=torch.float32)
    return penalty_op.apply_output_penalties(logits, counts, slots, params)


OUTPUTS = [[2, 7, 7, 7], [], [5, 5, 11], [8, 1, 1]]
CASES = {
    "presence": ([0.5] * 4, [0.0] * 4),
    "frequency": ([0.0] * 4, [0.7] * 4),
    "per_row": ([0.0, 0.4, -0.3, 1.2], [0.2, 0.0, 0.5, -0.1]),
    "negative": ([-0.5] * 4, [-0.25] * 4),
    "zero": ([0.0] * 4, [0.0] * 4),
}


def _logits(dtype=torch.float32, masked=False):
    logits = torch.randn(4, VOCAB, generator=torch.Generator().manual_seed(0)) * 3
    if masked:  # a grammar or min_tokens mask came first
        logits[0, 7] = logits[2, 5] = logits[3, 0] = NEG_INF
    return logits.to(dtype)


class TestOpAgainstVllm:
    @pytest.mark.parametrize("case", list(CASES))
    def test_float32(self, case):
        pres, freq = CASES[case]
        logits = _logits()
        want = _vllm(logits, [[]] * 4, OUTPUTS, [1.0] * 4, pres, freq)
        torch.testing.assert_close(_ours(logits, OUTPUTS, pres, freq), want)

    @pytest.mark.parametrize("case", list(CASES))
    def test_bfloat16_goes_back_to_the_logits_dtype(self, case):
        """As repetition_penalty: float32 math, then the logits dtype."""
        pres, freq = CASES[case]
        logits = _logits(torch.bfloat16)
        got = _ours(logits, OUTPUTS, pres, freq)
        assert got.dtype == torch.bfloat16
        want = _vllm(logits, [[]] * 4, OUTPUTS, [1.0] * 4, pres, freq)
        torch.testing.assert_close(got, want.to(torch.bfloat16), atol=0, rtol=0)

    def test_float32_logits_flag(self, monkeypatch):
        monkeypatch.setattr(penalty_op, "FLOAT32_LOGITS", True)
        pres, freq = CASES["per_row"]
        logits = _logits(torch.bfloat16)
        got = _ours(logits, OUTPUTS, pres, freq)
        assert got.dtype == torch.float32
        torch.testing.assert_close(
            got, _vllm(logits, [[]] * 4, OUTPUTS, [1.0] * 4, pres, freq)
        )

    def test_masked_logits_stay_masked(self):
        pres, freq = CASES["per_row"]
        logits = _logits(masked=True)
        got = _ours(logits, OUTPUTS, pres, freq)
        want = _vllm(logits, [[]] * 4, OUTPUTS, [1.0] * 4, pres, freq)
        torch.testing.assert_close(got, want)
        assert torch.isneginf(got[0, 7]) and torch.isneginf(got[2, 5])

    def test_after_repetition_penalty_matches_vllm(self):
        """repetition_penalty (over prompt + output) first, then these, as in
        vLLM's apply_penalties."""
        prompts = [[1, 2, 2, 9], [3, 4], [5], [6, 7, 8]]
        rep = [1.3, 1.0, 0.8, 2.0]
        pres, freq = CASES["per_row"]
        logits = _logits()
        seen = torch.zeros((4, VOCAB), dtype=torch.bool)
        for r, ids in enumerate(p + o for p, o in zip(prompts, OUTPUTS)):
            seen[r, ids] = True
        repeated = scale_seen_logits(logits, seen, torch.tensor(rep)[:, None])
        got = _ours(repeated, OUTPUTS, pres, freq)
        torch.testing.assert_close(
            got, _vllm(logits, prompts, OUTPUTS, rep, pres, freq)
        )

    def test_padded_vocabulary_columns_are_unchanged(self):
        logits = torch.randn(4, VOCAB + 8, generator=torch.Generator().manual_seed(1))
        pres, freq = CASES["per_row"]
        got = _ours(logits, OUTPUTS, pres, freq)
        assert torch.equal(got[:, VOCAB:], logits[:, VOCAB:])
        want = _vllm(logits[:, :VOCAB], [[]] * 4, OUTPUTS, [1.0] * 4, pres, freq)
        torch.testing.assert_close(got[:, :VOCAB], want)

    def test_the_zero_row_leaves_a_row_unchanged(self):
        counts = _counts_table(OUTPUTS)
        logits = _logits()
        slots = torch.full((4,), 8, dtype=torch.int32)  # the all-zero row
        params = torch.ones((4, 2))
        got = penalty_op.apply_output_penalties(logits, counts, slots, params)
        assert torch.equal(got, logits)

    def test_add_sampled_tokens(self):
        counts = torch.zeros((9, VOCAB), dtype=torch.int32)
        slots = torch.tensor([0, 8, 3, 0], dtype=torch.int32)
        tokens = torch.tensor([[5], [6], [7], [5]])
        weights = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
        penalty_op.add_sampled_tokens(counts, slots, tokens, weights)
        assert counts[0, 5] == 2 and counts[3, 7] == 1
        assert counts[8].sum() == 0  # weight 0 changes nothing

    def test_out_of_range_tokens_are_not_counted(self):
        counts = torch.zeros((2, VOCAB), dtype=torch.int32)
        tokens = torch.tensor([[VOCAB + 3], [-1]])
        weights = torch.ones(2, dtype=torch.int32)
        penalty_op.add_sampled_tokens(
            counts, torch.tensor([1, 1], dtype=torch.int32), tokens, weights
        )
        assert counts.sum() == 0


def _batch(*requests, max_model_len=64):
    return _input_batch(*requests, vocab_size=VOCAB, max_model_len=max_model_len)


def _tracker(**kwargs):
    kwargs.setdefault("async_scheduling", False)
    kwargs.setdefault("speculative_decoding", False)
    return output_penalties.OutputPenaltyTracker(
        VOCAB, 4, torch.device("cpu"), **kwargs
    )


def _row_counts(tracker, chunk, row):
    return tracker._counts[chunk._slots[row]].tolist()


def _bincount(ids):
    return np.bincount(np.asarray(ids, dtype=np.int64), minlength=VOCAB).tolist()


class TestTracker:
    def test_seeds_rows_from_the_output_after_a_reorder(self):
        a = _request("a", SamplingParams(frequency_penalty=0.5), [4, 4, 9])
        b = _request("b", SamplingParams(), [6])  # no penalty
        c = _request("c", SamplingParams(presence_penalty=0.25), [7])
        batch = _batch(a, b, c)
        batch.swap_states(0, 2)  # c, b, a
        tracker = _tracker()
        assert tracker.begin_step(batch)
        chunk = tracker.plain_chunk(batch, 0, 3, 4)
        assert chunk._weights.tolist() == [1, 0, 1, 0]
        np.testing.assert_allclose(
            chunk._params.numpy(), [[0.25, 0.0], [0, 0], [0.0, 0.5], [0, 0]]
        )
        assert chunk._slots[1] == chunk._slots[3] == 0  # the null row
        assert _row_counts(tracker, chunk, 0) == _bincount([7])
        assert _row_counts(tracker, chunk, 2) == _bincount([4, 4, 9])

    def test_chunk_offset(self):
        batch = _batch(
            _request("a", SamplingParams(frequency_penalty=0.5), [2]),
            _request("b", SamplingParams(presence_penalty=0.5), [4]),
        )
        tracker = _tracker()
        tracker.begin_step(batch)
        chunk = tracker.plain_chunk(batch, 1, 2, 2)
        assert chunk._weights.tolist() == [1, 0]
        assert _row_counts(tracker, chunk, 0) == _bincount([4])

    def test_none_without_a_penalized_row_in_the_chunk(self):
        batch = _batch(
            _request("a", SamplingParams(), [2]), _request("b", SamplingParams(), [4])
        )
        tracker = _tracker()
        assert not tracker.begin_step(batch)
        assert tracker.plain_chunk(batch, 0, 2, 2) is None

    def test_partial_prefill_rows_are_neither_penalized_nor_counted(self):
        batch = _batch(
            _request("a", SamplingParams(frequency_penalty=0.5), []),
            _request("b", SamplingParams(frequency_penalty=0.5), [3]),
        )
        tracker = _tracker()
        tracker.begin_step(batch)
        assert tracker.plain_chunk(batch, 0, 1, 1, discarded_req_ids={"a"}) is None
        chunk = tracker.plain_chunk(batch, 0, 2, 2, discarded_req_ids={"a"})
        assert chunk._weights.tolist() == [0, 1]
        assert chunk._slots[0] == 0  # the null row
        assert tracker._slots.slot_for("a") is None

    def test_on_sampled_counts_without_a_reseed(self):
        a = _request("a", SamplingParams(frequency_penalty=0.5), [3])
        batch = _batch(a)
        tracker = _tracker()
        tracker.begin_step(batch)
        chunk = tracker.plain_chunk(batch, 0, 1, 2)
        chunk.on_sampled(torch.tensor([[3], [9]]))
        assert _row_counts(tracker, chunk, 0) == _bincount([3, 3])
        a.output_token_ids.append(3)  # the scheduler's update, same list
        with patch.object(tracker, "_seed", wraps=tracker._seed) as seed:
            tracker.begin_step(batch)
            chunk = tracker.plain_chunk(batch, 0, 1, 2)
        seed.assert_not_called()
        assert _row_counts(tracker, chunk, 0) == _bincount([3, 3])

    def test_reseeds_a_stale_row(self):
        """A resumed request, or another output list, is counted again."""
        a = _request("a", SamplingParams(frequency_penalty=0.5), [3, 4])
        batch = _batch(a)
        tracker = _tracker()
        tracker.begin_step(batch)
        tracker.plain_chunk(batch, 0, 1, 1)
        a.output_token_ids.extend([5, 5])  # tokens the tracker never saw
        chunk = tracker.plain_chunk(batch, 0, 1, 1)
        assert _row_counts(tracker, chunk, 0) == _bincount([3, 4, 5, 5])
        # Another list object of the same length (a streaming continuation).
        batch.req_output_token_ids[0] = [6, 6, 6, 6]
        chunk = tracker.plain_chunk(batch, 0, 1, 1)
        assert _row_counts(tracker, chunk, 0) == _bincount([6, 6, 6, 6])

    def test_long_output_is_seeded_in_passes(self, monkeypatch):
        monkeypatch.setattr(output_penalties, "SEED_WIDTH", 4)
        output = [1, 2, 3, 3, 3, 4, 5, 6, 6, 7]
        batch = _batch(_request("a", SamplingParams(presence_penalty=1.0), output))
        tracker = _tracker()
        tracker.begin_step(batch)
        chunk = tracker.plain_chunk(batch, 0, 1, 1)
        assert _row_counts(tracker, chunk, 0) == _bincount(output)

    def test_rows_are_released_and_reused_zeroed(self):
        batch = _batch(_request("a", SamplingParams(frequency_penalty=0.5), [5, 5]))
        tracker = _tracker()
        tracker.begin_step(batch)
        tracker.plain_chunk(batch, 0, 1, 1)
        assert not tracker.begin_step(_batch(_request("x", SamplingParams(), [2])))
        assert not tracker._slots.allocated_req_ids()
        batch = _batch(_request("b", SamplingParams(presence_penalty=0.5), [6]))
        tracker.begin_step(batch)
        chunk = tracker.plain_chunk(batch, 0, 1, 1)
        assert _row_counts(tracker, chunk, 0) == _bincount([6])

    def test_a_finished_row_is_released_when_another_joins(self):
        """One request leaves and another joins in the same step, with every
        row leased: the old row must be released before the new one leases."""
        tracker = output_penalties.OutputPenaltyTracker(
            VOCAB,
            1,
            torch.device("cpu"),
            async_scheduling=False,
            speculative_decoding=False,
        )
        batch = _batch(_request("a", SamplingParams(frequency_penalty=0.5), [2]))
        tracker.begin_step(batch)
        tracker.plain_chunk(batch, 0, 1, 1)
        batch = _batch(_request("b", SamplingParams(presence_penalty=0.5), [3]))
        assert tracker.begin_step(batch)
        chunk = tracker.plain_chunk(batch, 0, 1, 1)
        assert set(tracker._slots.allocated_req_ids()) == {"b"}
        assert _row_counts(tracker, chunk, 0) == _bincount([3])

    def test_active_for_a_penalized_batch(self):
        tracker = _tracker()
        assert tracker.active(
            _batch(_request("a", SamplingParams(frequency_penalty=0.5), [2]))
        )
        assert not tracker.active(_batch(_request("a", SamplingParams(), [2])))

    def test_a_runner_that_never_samples_holds_nothing(self):
        """A pipeline stage before the last, or a pooling model."""
        tracker = _tracker(samples=False)
        assert tracker._counts is None and not tracker.supported
        batch = _batch(_request("a", SamplingParams(frequency_penalty=0.5), [2]))
        with patch.object(output_penalties.logger, "warning_once") as warn:
            assert not tracker.begin_step(batch)
        warn.assert_not_called()

    @pytest.mark.parametrize(
        "mode", [{"async_scheduling": True}, {"speculative_decoding": True}]
    )
    def test_unsupported_mode_warns_and_skips(self, mode):
        batch = _batch(_request("a", SamplingParams(frequency_penalty=0.5), [2]))
        tracker = _tracker(**mode)
        assert tracker._counts is None  # no HBM in a mode that cannot use it
        with patch.object(output_penalties.logger, "warning_once") as warn:
            assert not tracker.begin_step(batch)
        warn.assert_called_once()
        assert not tracker.active(batch)

    def test_override_top_tokens_is_refused(self):
        batch = _batch(_request("a", SamplingParams(frequency_penalty=0.5), [2]))
        tracker = _tracker()
        tracker.begin_step(batch)
        chunk = tracker.plain_chunk(batch, 0, 1, 1)
        with pytest.raises(RuntimeError, match="full logits"):
            chunk.override_top_tokens(torch.zeros((1, 1), dtype=torch.int64))


def _penalized_runner(logits, output, **penalties):
    """`plain_step_runner` with request r0 penalized and `output` so far."""
    runner = plain_step_runner(logits)
    runner._output_penalties = output_penalties.OutputPenaltyTracker(
        logits.shape[1],
        8,
        torch.device("cpu"),
        async_scheduling=False,
        speculative_decoding=False,
    )
    request = runner.requests["r0"]
    request.output_token_ids.extend(output)
    batch = runner.input_batch
    batch.presence_penalties_cpu = np.array(
        [penalties.get("presence", 0.0)], dtype=np.float32
    )
    batch.frequency_penalties_cpu = np.array(
        [penalties.get("frequency", 0.0)], dtype=np.float32
    )
    batch.presence_penalties_reqs = {"r0"} if "presence" in penalties else set()
    batch.frequency_penalties_reqs = {"r0"} if "frequency" in penalties else set()
    return runner


def _step_logits():
    logits = torch.zeros(1, VOCAB)
    logits[0, 3], logits[0, 5] = 4.0, 3.5
    return logits


@pytest.mark.parametrize(
    "penalties, expected",
    [({}, 3), ({"frequency": 0.3}, 5), ({"presence": 0.6}, 5)],
)
def test_sample_tokens_applies_them(penalties, expected):
    """Token 3 is twice in the output: 4.0 - 2 * 0.3 < 3.5, and 4.0 - 0.6 < 3.5."""
    runner = _penalized_runner(_step_logits(), [3, 3], **penalties)
    assert run_sample_tokens(runner).sampled_token_ids == [[expected]]


def test_sampled_token_is_counted_for_the_next_step():
    """Step 1 picks 3 (4.0 - 1 * 0.3 > 3.5); counted on device, so step 2
    sees two 3s (4.0 - 0.6 < 3.5) and picks 5."""
    runner = _penalized_runner(_step_logits(), [3], frequency=0.3)
    tracker = runner._output_penalties
    assert run_sample_tokens(runner).sampled_token_ids == [[3]]
    # sample_tokens appended the token to the request's output, as in serving.
    assert runner.requests["r0"].output_token_ids == [3, 3]
    runner.execute_model_state = _fresh_state(_step_logits())
    with patch.object(tracker, "_seed", wraps=tracker._seed) as seed:
        assert run_sample_tokens(runner).sampled_token_ids == [[5]]
    seed.assert_not_called()


def _fresh_state(logits):
    """A new execute_model_state for one chunk with r0, as the next step has."""
    return plain_step_runner(logits).execute_model_state


def test_logprobs_come_from_the_unpenalized_logits():
    runner = _penalized_runner(_step_logits(), [3, 3], frequency=0.3)
    runner.input_batch.max_num_logprobs = 0
    seen = []

    def gather_logprobs(x, tokens):
        seen.append(x.clone())
        return LogprobsTensors(
            tokens.view(-1, 1), torch.zeros(1, 1), torch.zeros(1, dtype=torch.int64)
        )

    runner.gather_logprobs = gather_logprobs
    assert run_sample_tokens(runner).sampled_token_ids == [[5]]
    assert seen[0][0, 3] == 4.0


def test_top_tokens_step_with_these_penalties_is_refused():
    runner = _penalized_runner(_step_logits(), [3], frequency=0.3)
    runner.execute_model_state.top_tokens = True
    with pytest.raises(AssertionError, match="presence_penalty"):
        TPUModelRunner.sample_tokens(runner, None)


def test_top_tokens_path_is_off_with_these_penalties():
    """With real trackers and InputBatch: a greedy batch takes the top-tokens
    path until a request in it sets one of the penalties."""
    penalized = _request("a", SamplingParams(temperature=0, frequency_penalty=0.5), [2])
    plain = _request("b", SamplingParams(temperature=0), [3])
    runner = object.__new__(TPUModelRunner)
    runner._greedy_head_owner = object()
    runner._min_tokens = min_tokens.MinTokensTracker()
    runner._repetition_penalty = RepetitionPenaltyTracker(
        VOCAB, 4, async_scheduling=False, speculative_decoding=False
    )
    runner._output_penalties = _tracker()
    for requests, expected in (([plain], True), ([penalized, plain], False)):
        runner.input_batch = _batch(*requests)
        runner.requests = {r.req_id: r for r in requests}
        assert runner._greedy_top_tokens_this_step() is expected


def _precompile_runner(tracker, paddings):
    return SimpleNamespace(
        _output_penalties=tracker,
        num_reqs_paddings=paddings,
        _precompile_timed=lambda _name: contextlib.nullcontext(),
        _dummy_logits=lambda n: torch.zeros((n, VOCAB), dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    "mode",
    [{"async_scheduling": True}, {"speculative_decoding": True}, {"samples": False}],
)
def test_precompile_is_a_no_op_where_they_cannot_apply(monkeypatch, mode):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *_args: None)
    with patch.object(penalty_op, "apply_output_penalties") as op:
        TPUModelRunner._precompile_output_penalties(
            _precompile_runner(_tracker(**mode), [4])
        )
    op.assert_not_called()


def test_precompile_covers_the_graphs_serving_runs():
    """Every graph a penalized chunk runs gets the shapes, dtypes and input
    kinds (no views) that _precompile_output_penalties gave it."""
    calls = set()
    real = {
        name: getattr(penalty_op, name)
        for name in (
            "apply_output_penalties",
            "add_sampled_tokens",
            "seed_slot",
            "add_to_slot_compiled",
        )
    }

    def spy(name):
        def wrapped(*args):
            calls.add((name, *((a.shape, a.dtype, a._base is None) for a in args)))
            return real[name](*args)

        return wrapped

    with patch.multiple(penalty_op, **{n: spy(n) for n in real}):
        tracker = output_penalties.OutputPenaltyTracker(
            VOCAB,
            8,
            torch.device("cpu"),
            async_scheduling=False,
            speculative_decoding=False,
        )
        with patch.object(tpu_runner, "synchronize_tensors", lambda *_args: None):
            TPUModelRunner._precompile_output_penalties(
                _precompile_runner(tracker, [4, 8])
            )
        warmed, calls = calls, set()
        long_output = list(range(VOCAB)) * 40  # several seeding passes
        batch = _batch(
            _request("a", SamplingParams(frequency_penalty=0.5), long_output),
            _request("b", SamplingParams(presence_penalty=0.5), [3]),
            max_model_len=2048,
        )
        tracker.begin_step(batch)
        for num_rows in (4, 8):
            chunk = tracker.plain_chunk(batch, 0, 2, num_rows)
            out = chunk.apply(torch.zeros((num_rows, VOCAB), dtype=torch.bfloat16))
            chunk.on_sampled(torch.zeros((num_rows, 1), dtype=torch.int64))
            assert out.dtype == torch.bfloat16
    assert {c[0] for c in calls} == set(real)
    assert calls <= warmed


class _RecordingProcessor:
    """A stand-in for a later chunk processor (the thinking budget's)."""

    def __init__(self):
        self.seen = []

    def apply(self, logits):
        self.seen.append(logits.clone())
        return logits

    def override_top_tokens(self, token_ids):
        return token_ids

    def on_sampled(self, token_ids):
        pass


def test_they_run_before_the_thinking_budget():
    """vLLM's Sampler.apply_logits_processors applies the penalties before
    the thinking-token budget, so its processor gets the penalized logits."""
    runner = _penalized_runner(_step_logits(), [3, 3], frequency=0.3)
    later = _RecordingProcessor()
    tb_step = MagicMock()
    tb_step.plain_chunk.return_value = later
    runner._thinking_budget = MagicMock()
    runner._thinking_budget.step_state.return_value = tb_step
    assert run_sample_tokens(runner).sampled_token_ids == [[5]]
    assert len(later.seen) == 1
    assert later.seen[0][0, 3] == pytest.approx(4.0 - 2 * 0.3)
