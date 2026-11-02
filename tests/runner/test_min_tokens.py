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
"""min_tokens on the TPU sampler: the requests the runner tracks, the
stop-token mask built for them, and where the runner applies it."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import min_tokens
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncPreResults,
    AsyncTPUCopyState,
)

NEG_INF = float("-inf")


class TestTracker:
    def test_tracks_requests_with_min_tokens_and_stop_ids(self):
        tracker = min_tokens.MinTokensTracker()
        tracker.add("a", SamplingParams(min_tokens=3, stop_token_ids=[9, 2]))
        tracker.add("b", SamplingParams(stop_token_ids=[9]))  # no min_tokens
        tracker.add("c", None)  # pooling request
        assert list(tracker.requests) == ["a"]
        assert tracker.requests["a"].stop_ids == (2, 9)
        assert not tracker.requests["a"].structured

    def test_tracks_the_eos_added_from_the_generation_config(self):
        """The usual request names no stop ids: the EOS comes from the
        generation config and is only in all_stop_token_ids."""
        tracker = min_tokens.MinTokensTracker()
        params = SamplingParams(min_tokens=5)
        params.update_from_generation_config({}, eos_token_id=2)
        tracker.add("a", params)
        assert tracker.requests["a"].stop_ids == (2,)

    def test_structured_output_flag(self):
        tracker = min_tokens.MinTokensTracker()
        params = SamplingParams(
            min_tokens=3,
            stop_token_ids=[9],
            structured_outputs=StructuredOutputsParams(regex="a+"),
        )
        tracker.add("a", params)
        assert tracker.requests["a"].structured

    def test_batch_state_uses_the_current_index_and_drops_reached_requests(self):
        tracker = min_tokens.MinTokensTracker()
        for req_id in "abc":
            tracker.add(req_id, SamplingParams(min_tokens=3, stop_token_ids=[9]))
        outputs = {"a": 1, "b": 3, "c": 0}
        state = tracker.batch_state({"a": 2, "b": 0}, outputs.get)
        assert state == {2: (3, 1, (9,), False)}  # c: not in this batch
        assert set(tracker.requests) == {"a", "c"}  # b reached min_tokens

    def test_batch_state_keeps_requests_outside_the_batch(self):
        """A preempted request's output can still end with unsettled async
        placeholders (it left the batch before the rollback); resuming
        restores the real output, so it must still be tracked then."""
        tracker = min_tokens.MinTokensTracker()
        tracker.add("p", SamplingParams(min_tokens=10, stop_token_ids=[9]))
        assert tracker.batch_state({}, {"p": 12}.get) == {}
        assert tracker.batch_state({"p": 0}, {"p": 9}.get) == {0: (10, 9, (9,), False)}

    def test_re_adding_a_request_id_replaces_its_entry(self):
        """A streaming-input continuation comes back as a new request with the
        same id and new params."""
        tracker = min_tokens.MinTokensTracker()
        tracker.add(
            "a", SamplingParams(min_tokens=50, max_tokens=64, stop_token_ids=[9])
        )
        tracker.add("a", SamplingParams(stop_token_ids=[9]))
        assert not tracker

    def test_batch_state_drops_requests_that_are_gone(self):
        tracker = min_tokens.MinTokensTracker()
        tracker.add("a", SamplingParams(min_tokens=3, stop_token_ids=[9]))
        assert tracker.batch_state({"a": 0}, lambda _req_id: None) == {}
        assert not tracker

    def test_any_in(self):
        tracker = min_tokens.MinTokensTracker()
        tracker.add("a", SamplingParams(min_tokens=3, stop_token_ids=[9]))
        assert tracker.any_in({"a": 0}) and not tracker.any_in({"b": 0})


class TestStopTable:
    def test_masks_rows_below_min_tokens_only(self):
        state = {
            0: (5, 2, (3, 7), False),  # 2 of 5: masked
            2: (3, 3, (9,), False),  # reached: not masked
        }
        stop_ids, restore = min_tokens.build_stop_table(
            state, min_tokens.plain_rows(state, 0, 3), num_rows=4
        )
        assert stop_ids.shape == (4, min_tokens.STOP_IDS_WIDTH)
        # Sorted ids, then the first id repeated over the width.
        assert stop_ids[0].tolist() == [3, 7] + [3] * 6
        assert (stop_ids[1:] == -1).all()
        assert not restore.any()

    def test_rows_ahead_of_the_output(self):
        # 3 output tokens, min 5: positions 0 and 1 ahead are still below.
        state = {1: (5, 3, (4,), False)}
        target, bonus = min_tokens.verify_rows(state, 1, [3])
        stop_ids, _ = min_tokens.build_stop_table(state, target, num_rows=4)
        assert [row[0] for row in stop_ids.tolist()] == [4, 4, -1, -1]
        assert min_tokens.build_stop_table(state, bonus, num_rows=2) is None

    def test_rows_only_for_tracked_requests_of_the_chunk(self):
        state = {2: (9, 0, (1,), False), 4: (9, 0, (1,), False), 7: (9, 0, (1,), False)}
        assert min_tokens.plain_rows(state, 2, 5) == {2: (0, 0, 1), 4: (2, 0, 1)}
        target, bonus = min_tokens.verify_rows(state, 2, [2, 0, 1])
        assert target == {2: (0, 0, 2), 4: (2, 0, 1)}
        assert bonus == {2: (0, 2, 1), 4: (2, 1, 1)}

    def test_verify_logits_rows_are_draft_positions_then_the_bonus(self):
        state = {2: (9, 0, (1,), False), 3: (9, 0, (1,), False), 4: (9, 0, (1,), False)}
        assert min_tokens.verify_logits_rows(state, 2, [2, 0, 1]) == {
            2: (0, 0, 3),
            3: (3, 0, 1),
            4: (4, 0, 2),
        }

    def test_requests_sharing_stop_ids_each_get_their_own_rows(self):
        """One (stop ids, structured) group with several runs: the usual case,
        every request stopping on the generation-config EOS."""
        state = {0: (9, 0, (2,), False), 1: (9, 0, (2,), False)}
        rows = min_tokens.verify_logits_rows(state, 0, [3, 1])
        stop_ids, _ = min_tokens.build_stop_table(state, rows, 8)
        assert [row[0] for row in stop_ids] == [2] * 6 + [-1] * 2
        state = {0: (9, 0, (2,), False), 2: (9, 0, (2,), False)}
        stop_ids, _ = min_tokens.build_stop_table(
            state, min_tokens.plain_rows(state, 0, 4), 4
        )
        assert [row[0] for row in stop_ids] == [2, -1, 2, -1]

    def test_nothing_to_mask(self):
        assert (
            min_tokens.build_stop_table({}, min_tokens.plain_rows({}, 0, 2), 2) is None
        )

    def test_more_stop_ids_take_more_columns_of_the_precompiled_width(self):
        ids = tuple(range(min_tokens.STOP_IDS_WIDTH + 1))
        stop_ids, _ = min_tokens.build_stop_table(
            {0: (2, 0, ids, False)}, {0: (0, 0, 1)}, 1
        )
        assert stop_ids.shape == (1, 2 * min_tokens.STOP_IDS_WIDTH)
        assert set(stop_ids[0].tolist()) == set(ids)

    def test_restore_only_for_structured_output(self):
        state = {0: (5, 0, (1,), True), 1: (5, 0, (1,), False)}
        _, restore = min_tokens.build_stop_table(
            state, min_tokens.plain_rows(state, 0, 2), 2
        )
        assert restore.tolist() == [True, False]


@pytest.fixture
def eager():
    with torch.compiler.set_stance("force_eager"):
        yield


@pytest.mark.usefixtures("eager")
class TestMask:
    def _mask(self, logits, state, rows=None):
        rows = (
            min_tokens.plain_rows(state, 0, logits.shape[0]) if rows is None else rows
        )
        table = min_tokens.build_stop_table(state, rows, logits.shape[0])
        return min_tokens.apply_stop_table(logits, *table)

    def test_masks_stop_tokens_of_masked_rows(self):
        logits = torch.arange(24, dtype=torch.float32).view(3, 8)
        state = {0: (5, 0, (0, 5), False), 2: (5, 0, (7,), False)}
        out = self._mask(logits, state)
        expected = logits.clone()
        expected[0, [0, 5]] = NEG_INF
        expected[2, 7] = NEG_INF
        assert torch.equal(out, expected)

    def test_token_zero_with_an_unmasked_row_present(self):
        # Unmasked rows write their own column-0 logit back; token 0 of a
        # masked row must still be masked.
        logits = torch.ones(4, 8)
        out = self._mask(logits, {2: (3, 0, (0,), False)})
        assert torch.isneginf(out[2, 0])
        assert torch.isneginf(out).sum() == 1

    def test_more_stop_ids_than_the_width_in_passes_of_that_width(self):
        width = min_tokens.STOP_IDS_WIDTH
        ids = tuple(range(1, width + 3))
        logits = torch.zeros(2, 32)
        widths = []
        mask = min_tokens.mask_stop_tokens

        def spy(x, stop_ids):
            widths.append(stop_ids.shape[1])
            return mask(x, stop_ids)

        with patch.object(min_tokens, "mask_stop_tokens", side_effect=spy):
            out = self._mask(logits, {1: (5, 0, ids, False)})
        assert widths == [width, width]
        assert torch.isneginf(out[1, list(ids)]).all()
        assert torch.isneginf(out).sum() == len(ids)

    def test_no_restore_pass_without_structured_rows(self):
        logits = torch.zeros(1, 8)
        with patch.object(min_tokens, "restore_all_masked_rows") as restore:
            self._mask(logits, {0: (5, 0, (2,), False)})
        restore.assert_not_called()

    def test_structured_row_left_all_neg_inf_gets_its_stops_back(self):
        logits = torch.full((2, 6), NEG_INF)
        logits[:, 2] = 1.0  # the grammar allows only token 2, a stop token
        state = {0: (5, 0, (2,), True), 1: (5, 0, (2,), False)}
        out = self._mask(logits, state)
        assert out[0, 2] == 1.0
        assert torch.isneginf(out[1]).all()

    def test_structured_row_with_other_tokens_left_stays_masked(self):
        logits = torch.zeros(1, 6)
        out = self._mask(logits, {0: (5, 0, (2,), True)})
        assert torch.isneginf(out[0, 2]) and torch.isneginf(out).sum() == 1


def _request(req_id, params, output):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(4)),
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=4,
        output_token_ids=output,
    )


def _runner(input_batch, requests=(), tracked=()):
    """A runner with `requests` (CachedRequestState) of which `tracked` asked
    for min_tokens."""
    runner = object.__new__(TPUModelRunner)
    runner.input_batch = input_batch
    runner.requests = {r.req_id: r for r in requests}
    runner._pre_async_results = None
    runner._min_tokens = min_tokens.MinTokensTracker()
    for r in requests:
        if r.req_id in tracked:
            runner._min_tokens.add(r.req_id, r.sampling_params)
    return runner


def _input_batch():
    return InputBatch(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        vocab_size=16,
        block_sizes=[16],
        max_num_blocks_per_req=[4],
        logitsprocs=LogitsProcessors(),
        logitsprocs_need_output_token_ids=True,
    )


@pytest.mark.usefixtures("eager")
def test_mask_follows_the_request_when_the_batch_is_reordered():
    """The TPU runner reorders the batch (swap_states) after vLLM refreshes
    its logits-processor state; the mask must use the post-reorder rows."""
    batch = _input_batch()
    a = _request("a", SamplingParams(min_tokens=4, stop_token_ids=[9]), [5, 6])
    b = _request("b", SamplingParams(stop_token_ids=[9]), [5])
    batch.add_request(a)
    batch.add_request(b)
    batch.refresh_metadata()
    batch.swap_states(0, 1)  # as _reorder_batch_for_rpa does
    runner = _runner(batch, [a, b], tracked={"a"})

    min_toks = runner._min_tokens_batch_state()
    out = runner._mask_min_tokens(
        torch.zeros(4, 16), min_tokens.plain_rows(min_toks, 0, 2), min_toks
    )
    assert torch.isneginf(out[1, 9]) and torch.isneginf(out).sum() == 1

    a.output_token_ids.extend([7, 8])  # the request reaches min_tokens
    assert runner._min_tokens_batch_state() == {}
    assert not runner._min_tokens


def _async_runner(num_draft, tracked=True, min_toks=4):
    """Request "a" whose output ends with the previous step's 1 + num_draft
    optimistic placeholders (one real token before them); the step accepted
    one token. K = 3 drafts per step."""
    batch = SimpleNamespace(req_id_to_index={"a": 0})
    output = [5] + [0] * (1 + num_draft)
    params = SamplingParams(min_tokens=min_toks, max_tokens=128, stop_token_ids=[9])
    a = _request("a", params, output)
    runner = _runner(batch, [a], tracked={"a"} if tracked else ())
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    runner._pre_async_results = AsyncPreResults(
        req_ids=["a"],
        next_tokens_tpu=torch.zeros(1, 1 + num_draft, dtype=torch.int32),
        req_id_to_index_copy={"a": 0},
        request_seq_lens=[(0, a, 0, "a")],
        discard_sampled_tokens_req_indices=[],
        num_draft_per_req={0: num_draft},
    )

    def rollback():  # stands in for _modify_prev_results: one token accepted
        del a.output_token_ids[-(1 + num_draft) :]
        a.output_token_ids.append(6)

    runner._modify_prev_results = MagicMock(side_effect=rollback)
    return runner


@pytest.mark.parametrize(
    "min_toks", [3, 4, 8]
)  # the edges of 2..5 outputs + 0..3 ahead
def test_async_draft_placeholders_are_settled_before_counting(min_toks):
    runner = _async_runner(
        num_draft=3, min_toks=min_toks
    )  # 1 + 4 placeholders: counts as 5
    assert runner._min_tokens_batch_state() == {0: (min_toks, 2, (9,), False)}
    runner._modify_prev_results.assert_called_once()
    assert runner._pre_async_results is None


@pytest.mark.parametrize(
    "num_draft, tracked, min_toks",
    [
        (0, True, 4),  # no drafts: the one placeholder is the committed token
        (3, False, 4),  # not tracked
        (3, True, 100),  # 2..5 outputs + 0..3 ahead < 100: every row masked anyway
        (3, True, 2),  # at least 2 outputs: nothing masked either way
        (3, "out", 4),  # tracked but not in this batch: nothing to mask
    ],
)
def test_no_early_settle_unless_it_can_change_the_mask(num_draft, tracked, min_toks):
    runner = _async_runner(num_draft, bool(tracked), min_toks)
    if tracked == "out":
        runner.input_batch.req_id_to_index.clear()
    if tracked is False:  # another request is tracked, so the settle check runs
        b = _request(
            "b", SamplingParams(min_tokens=9, max_tokens=16, stop_token_ids=[9]), [5]
        )
        runner.requests["b"] = b
        runner.input_batch.req_id_to_index["b"] = 1
        runner._min_tokens.add("b", b.sampling_params)
    runner._min_tokens_batch_state()
    runner._modify_prev_results.assert_not_called()
    assert runner._pre_async_results is not None


def test_settle_uses_the_real_rollback_with_a_moved_request():
    """The previous step had "a" at index 0 with 3 drafts, of which only the
    bonus was accepted; this step "a" sits at index 1."""
    from vllm_torchtpu.runner.tpu_runner import INVALID_TOKEN_ID

    params = SamplingParams(min_tokens=4, max_tokens=64, stop_token_ids=[9])
    a = _request("a", params, [5, 0, 0, 0, 0])  # 1 real + 1 + 3 placeholders
    z = _request("z", SamplingParams(max_tokens=64), [5])
    batch = SimpleNamespace(
        req_id_to_index={"z": 0, "a": 1},
        num_tokens_no_spec=np.array([5, 9]),
        token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
    )
    runner = _runner(batch, [a, z], tracked={"a"})
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    sampled = torch.tensor([[6, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID]])
    runner._pre_async_results = AsyncPreResults(
        req_ids=["a"],
        next_tokens_tpu=sampled,
        request_seq_lens=[(0, a, 4, "a")],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"a": 0},
        copy_state=SimpleNamespace(wait=lambda: None, sampled_token_ids_cpu=sampled),
        num_draft_per_req={0: 3},
    )
    assert runner._min_tokens_batch_state() == {1: (4, 2, (9,), False)}
    assert a.output_token_ids == [5, 6] and runner._pre_async_results is None
    assert batch.num_tokens_no_spec[1] == 6  # rolled back at a's current index


def test_update_states_tracks_new_requests_and_forgets_finished_ones():
    runner = object.__new__(TPUModelRunner)
    runner._min_tokens = min_tokens.MinTokensTracker()
    runner._min_tokens.add("old", SamplingParams(min_tokens=3, stop_token_ids=[9]))
    runner._mamba_state_pos, runner._mamba_slot_by_req_id = {}, {}
    runner._free_mamba_slots, runner._mamba_offset_seeded = [], set()
    runner._pre_async_results = None
    runner.requests = {}
    runner._install_spec_token_room_guard = lambda: None
    new_req = SimpleNamespace(
        req_id="new", sampling_params=SamplingParams(min_tokens=2, stop_token_ids=[9])
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids={"old"},
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        kv_cache_block_copies=None,
        scheduled_new_reqs=[new_req],
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=None):
        TPUModelRunner._update_states(runner, scheduler_output)
    assert list(runner._min_tokens.requests) == ["new"]

    # Aborted and resubmitted under the same id in one step: forget, then track.
    scheduler_output.finished_req_ids = {"new"}
    with patch.object(GPUModelRunner, "_update_states", return_value=None):
        TPUModelRunner._update_states(runner, scheduler_output)
    assert list(runner._min_tokens.requests) == ["new"]


def test_update_states_keeps_preempted_and_resumed_requests():
    """Under the V1 model runner (the TPU default) a preempted request stays
    in `requests` and comes back as a cached (resumed) request, not a new
    one, so nothing would re-add it."""
    runner = object.__new__(TPUModelRunner)
    runner._min_tokens = min_tokens.MinTokensTracker()
    runner._min_tokens.add("p", SamplingParams(min_tokens=10, stop_token_ids=[9]))
    runner._mamba_state_pos, runner._mamba_slot_by_req_id = {}, {}
    runner._free_mamba_slots, runner._mamba_offset_seeded = [], set()
    runner._pre_async_results = None
    runner.requests = {}
    runner._install_spec_token_room_guard = lambda: None
    for preempted, resumed in (({"p"}, set()), (None, {"p"})):
        scheduler_output = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=preempted,
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=resumed),
            kv_cache_block_copies=None,
            scheduled_new_reqs=[],
        )
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, scheduler_output)
        assert list(runner._min_tokens.requests) == ["p"]


def _precompile_runner(speculative_config, mtp_prefill_only):
    runner = object.__new__(TPUModelRunner)
    runner.num_reqs_paddings = [8, 16]
    runner.num_tokens_paddings = [16, 32, 64, 128]
    runner.speculative_config = speculative_config
    runner._mtp_prefill_only = mtp_prefill_only
    runner.device = torch.device("cpu")
    runner._dummy_logits = lambda n: torch.zeros(n, 4)
    runner._precompile_timed = MagicMock()
    # Prefill-only MTP never allocates the target-row bitmask.
    runner.structured_decoding_manager = (
        SimpleNamespace()
        if mtp_prefill_only
        else SimpleNamespace(target_grammar_bitmask_cpu=torch.zeros(64, 1))
    )
    return runner


@pytest.mark.usefixtures("eager")
@pytest.mark.parametrize(
    "speculative_config, mtp_prefill_only, expected",
    [
        (None, False, [8, 16]),
        (object(), False, [8, 16, 32, 64]),
        # PP > 1 (or PCP) producer: drafts over prefill chunks only, no verify.
        (object(), True, [8, 16]),
    ],
)
def test_precompiled_row_buckets(speculative_config, mtp_prefill_only, expected):
    runner = _precompile_runner(speculative_config, mtp_prefill_only)
    seen = []
    mask = min_tokens.mask_stop_tokens

    def spy(logits, stop_ids):
        seen.append(logits.shape[0])
        return mask(logits, stop_ids)

    with patch.object(min_tokens, "mask_stop_tokens", side_effect=spy):
        runner._precompile_min_tokens_mask()
    assert seen == expected


@pytest.mark.usefixtures("eager")
def test_precompile_covers_the_mask_calls_serving_makes():
    """At every row bucket (request and verify buckets alike), the mask and
    restore calls of a masked step (more stop ids than the width, and a
    structured row) use the stop-id width and dtypes that were precompiled.
    Both sides take their logits from `_dummy_logits` here, so the logits
    dtype itself is not compared."""
    runner = _precompile_runner(object(), False)
    calls = set()
    mask, restore = min_tokens.mask_stop_tokens, min_tokens.restore_all_masked_rows

    def spy_mask(logits, stop_ids):
        calls.add(("mask", logits.shape, logits.dtype, stop_ids.shape, stop_ids.dtype))
        return mask(logits, stop_ids)

    def spy_restore(logits, masked, flags):
        calls.add(("restore", logits.shape, logits.dtype, flags.shape, flags.dtype))
        return restore(logits, masked, flags)

    with (
        patch.object(min_tokens, "mask_stop_tokens", side_effect=spy_mask),
        patch.object(min_tokens, "restore_all_masked_rows", side_effect=spy_restore),
    ):
        runner._precompile_min_tokens_mask()
        precompiled, calls = calls, set()
        # Vocabulary 4 in the dummy logits: ids repeat, the table width is
        # what matters.
        stop_ids = tuple(i % 4 for i in range(min_tokens.STOP_IDS_WIDTH + 1))
        for num_rows in (8, 16, 32, 64):
            state = {0: (5, 0, stop_ids, True)}
            table = min_tokens.build_stop_table(
                state, min_tokens.plain_rows(state, 0, 1), num_rows
            )
            assert table is not None
            min_tokens.apply_stop_table(runner._dummy_logits(num_rows), *table)
    assert len(calls) == 8  # mask and restore at 4 buckets
    assert calls <= precompiled


def test_top_tokens_step_needs_the_full_logits_below_min_tokens():
    runner = object.__new__(TPUModelRunner)
    runner._greedy_head_owner = object()
    params = SamplingParams(temperature=0)
    runner.requests = {"r0": SimpleNamespace(sampling_params=params)}
    runner._min_tokens = min_tokens.MinTokensTracker()
    batch = SimpleNamespace(
        all_greedy=True,
        max_num_logprobs=None,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
    )
    runner.input_batch = batch
    assert runner._greedy_top_tokens_this_step()

    runner._min_tokens.add("r0", SamplingParams(min_tokens=3, stop_token_ids=[2]))
    assert not runner._greedy_top_tokens_this_step()

    # A tracked request outside the batch (preempted) does not hold it back.
    runner._min_tokens.discard(["r0"])
    runner._min_tokens.add("p", SamplingParams(min_tokens=3, stop_token_ids=[2]))
    assert runner._greedy_top_tokens_this_step()


def _plain_step_runner(num_output, logits, req_id="r0"):
    """The runner state sample_tokens reads for one greedy request in a plain
    (non-speculative, synchronous) step, picking the argmax of what it gets.
    With `num_output`, request `req_id` has that many outputs, min_tokens 4
    and stop token 3."""
    runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"r0": 1},
                finished_req_ids=set(),
                has_structured_output_requests=False,
            ),
            draft_chunks=[],
            logits_list=[logits],
            num_reqs_list=[1],
            spec_decode_metadata_list=[None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
            kv_save_nan_check=None,
        ),
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=1,
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            all_greedy=True,
            num_tokens_no_spec=[1],
            token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
        ),
        requests={
            "r0": SimpleNamespace(
                num_computed_tokens=0, num_tokens=1, output_token_ids=[]
            )
        },
        _pre_async_results=None,
        _pp_handoff=None,
        _dp_step_other_decode=False,
        _min_tokens=min_tokens.MinTokensTracker(),
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _is_async_drafter=False,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        is_pooling_model=False,
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda x: torch.argmax(x, dim=-1),
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _modify_prev_results=lambda: None,
        _stage_timing=MagicMock(),
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    if num_output is not None:
        runner._min_tokens.add(req_id, SamplingParams(min_tokens=4, stop_token_ids=[3]))
        runner.requests[req_id] = SimpleNamespace(
            num_computed_tokens=0, num_tokens=1, output_token_ids=[0] * num_output
        )
    for name in (
        "_mask_min_tokens",
        "_min_tokens_batch_state",
        "_settle_draft_placeholders_for_min_tokens",
        "_num_output_tokens",
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


@pytest.mark.parametrize(
    "num_output, expected",
    [(None, 3), (4, 3), (1, 5)],  # controls: EOS (3) is the argmax
)
def test_plain_step_masks_below_min_tokens(num_output, expected):
    logits = torch.zeros(1, 8)
    logits[0, 3], logits[0, 5] = 4.0, 2.0
    runner = _plain_step_runner(num_output, logits)

    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=chunks,
            chunk_real_lens=lengths,
            copy_ready_event=MagicMock(),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
        torch.compiler.set_stance("force_eager"),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    assert output.sampled_token_ids == [[expected]]


@pytest.mark.parametrize("num_output, expected", [(None, 3), (1, 5)])
def test_plain_step_masks_random_sampling_below_min_tokens(num_output, expected):
    """Temperature 1 through the runner's own sampler: stop token 3 holds
    all but e^-10 of the mass."""
    logits = torch.full((1, 8), NEG_INF)
    logits[0, 3], logits[0, 5] = 10.0, 0.0
    runner = _plain_step_runner(num_output, logits)
    runner.input_batch.all_greedy = False
    runner._build_padded_sampling_params = lambda *_args: (
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.ones((1, 1)),
    )
    runner._sampler_topk_candidates = lambda *_args: 0
    runner._sampler_applies_topk_topp = (
        TPUModelRunner._sampler_applies_topk_topp.__get__(runner)
    )
    runner._get_sampling_generator = lambda: torch.Generator().manual_seed(0)
    runner._apply_temperature = TPUModelRunner._apply_temperature.__get__(runner)
    runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(runner)
    assert _run_sample_tokens(runner).sampled_token_ids == [[expected]]


def test_sample_tokens_settles_async_placeholders_before_it_counts():
    runner = _plain_step_runner(1, torch.zeros(1, 8))
    calls = []
    runner._settle_draft_placeholders_for_min_tokens = lambda: calls.append("settle")
    batch_state = runner._min_tokens.batch_state

    def counted(*args):
        calls.append("count")
        return batch_state(*args)

    runner._min_tokens.batch_state = counted
    _run_sample_tokens(runner)
    assert calls == ["settle", "count"]


def test_top_tokens_step_with_a_request_below_min_tokens_is_refused():
    """top_tokens hands sample_tokens each row's greedy token, which cannot
    be masked; execute_model never picks it with a tracked request in the
    batch, and sample_tokens refuses it loudly if that ever changes."""
    logits = torch.zeros(1, 8)
    runner = _plain_step_runner(1, logits)
    runner.execute_model_state.top_tokens = True
    with pytest.raises(AssertionError, match="below min_tokens"):
        TPUModelRunner.sample_tokens(runner, None)


def test_plain_step_logprobs_come_from_the_unmasked_logits():
    """As vLLM's raw logprobs, which precede its logits processors."""
    from vllm.v1.outputs import LogprobsTensors

    logits = torch.zeros(1, 8)
    logits[0, 3], logits[0, 5] = 4.0, 2.0
    runner = _plain_step_runner(1, logits)
    runner.input_batch.max_num_logprobs = 0
    seen = []

    def gather_logprobs(x, tokens):
        seen.append(x.clone())
        return LogprobsTensors(
            tokens.view(-1, 1), torch.zeros(1, 1), torch.zeros(1, dtype=torch.int64)
        )

    runner.gather_logprobs = gather_logprobs

    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=chunks,
            chunk_real_lens=lengths,
            copy_ready_event=MagicMock(),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
        torch.compiler.set_stance("force_eager"),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    assert output.sampled_token_ids == [[5]]  # sampled from the masked logits
    assert len(seen) == 1 and seen[0][0, 3] == 4.0  # logprobs from the raw ones


@pytest.mark.parametrize(
    "num_output, expected",
    [(None, 3), (1, 5)],  # control: EOS (3) is the argmax
)
def test_draft_less_chunk_of_a_spec_step_masks_below_min_tokens(num_output, expected):
    """Two chunks: r0 verifies drafts (stubbed), r1 has none yet and samples
    from its own logits, which must be masked."""
    logits = torch.zeros(1, 8)
    logits[0, 3], logits[0, 5] = 4.0, 2.0
    runner = _plain_step_runner(num_output, logits, req_id="r1")
    st = runner.execute_model_state
    st.scheduler_output.num_scheduled_tokens = {"r0": 2, "r1": 1}
    st.logits_list = [torch.zeros(2, 8), logits]
    st.num_reqs_list = [1, 1]
    st.spec_decode_metadata_list = [object(), None]
    batch = runner.input_batch
    batch.num_reqs, batch.req_ids = 2, ["r0", "r1"]
    batch.req_id_to_index = {"r0": 0, "r1": 1}
    batch.num_tokens_no_spec = [1, 1]
    batch.token_ids_cpu = np.zeros((2, 8), dtype=np.int32)
    runner.requests.setdefault(
        "r1", SimpleNamespace(num_computed_tokens=0, num_tokens=1, output_token_ids=[])
    )
    runner.speculative_config = SimpleNamespace(
        method="ngram", num_speculative_tokens=1
    )
    runner._sample_spec_verify_chunk = lambda *_args: torch.tensor([[7, -1]])
    runner.drafter = SimpleNamespace()
    runner.spec_decode_manager = SimpleNamespace(
        propose_draft_token_ids=lambda **_kwargs: None
    )
    runner._sync_replicated_drafts_across_tp = lambda drafts: drafts

    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=chunks,
            chunk_real_lens=lengths,
            copy_ready_event=MagicMock(),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
        torch.compiler.set_stance("force_eager"),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    assert output.sampled_token_ids == [[7], [expected]]


def _spec_step_runner(num_output, logits_list, md_list, req_ids):
    """sample_tokens state for a spec step with real verify chunks: request
    `req_ids[-1]` has `num_output` outputs, min_tokens 4, stop token 3."""
    runner = _plain_step_runner(num_output, logits_list[-1], req_id=req_ids[-1])
    st = runner.execute_model_state
    st.scheduler_output.num_scheduled_tokens = {r: 1 for r in req_ids}
    st.logits_list = logits_list
    st.num_reqs_list = [1] * len(req_ids)
    st.spec_decode_metadata_list = md_list
    batch = runner.input_batch
    batch.num_reqs, batch.req_ids = len(req_ids), list(req_ids)
    batch.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
    batch.num_tokens_no_spec = [1] * len(req_ids)
    batch.token_ids_cpu = np.zeros((len(req_ids), 8), dtype=np.int32)
    for r in req_ids:
        runner.requests.setdefault(
            r, SimpleNamespace(num_computed_tokens=0, num_tokens=1, output_token_ids=[])
        )
    spec_config = SimpleNamespace(
        method="ngram",
        num_speculative_tokens=3,
        rejection_sample_method="strict",
        synthetic_acceptance_rates=None,
    )
    runner.speculative_config = spec_config
    runner.rejection_sampler = RejectionSampler(spec_config, torch.device("cpu"))
    for name in (
        "_sample_spec_verify_chunk",
        "spec_bonus_and_target_logits",
        "spec_gather_bonus_and_target_logits",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    runner.drafter = SimpleNamespace()
    runner.spec_decode_manager = SimpleNamespace(
        propose_draft_token_ids=lambda **_kwargs: None
    )
    runner._sync_replicated_drafts_across_tp = lambda drafts: drafts
    return runner


def _run_sample_tokens(runner):
    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=chunks,
            chunk_real_lens=lengths,
            copy_ready_event=MagicMock(),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
        torch.compiler.set_stance("force_eager"),
    ):
        return TPUModelRunner.sample_tokens(runner, None)


def _one_request_verify_md():
    """One request, drafts [3, 3, 3]; every row prefers token 3 (logit 4),
    then token 6 (logit 2)."""
    logits = torch.zeros(4, 8)
    logits[:, 3], logits[:, 6] = 4.0, 2.0
    md = SimpleNamespace(
        draft_token_ids=torch.tensor([3, 3, 3], dtype=torch.int32),
        draft_lengths=torch.tensor([3], dtype=torch.int32),
        draft_lengths_cpu=np.array([3]),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        final_logits_indices=torch.arange(4, dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0], dtype=torch.int64),
        group_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    return logits, md


@pytest.mark.parametrize(
    "num_output, expected",
    [(None, [3, 3, 3, 3]), (1, [6])],  # control: every draft (the stop) accepted
)
def test_spec_step_masks_verify_chunks_through_sample_tokens(num_output, expected):
    """sample_tokens hands this step's min_tokens rows to the real verify."""
    logits, md = _one_request_verify_md()
    runner = _spec_step_runner(num_output, [logits], [md], ["r0"])
    assert _run_sample_tokens(runner).sampled_token_ids == [expected]


def test_spec_step_draft_less_chunk_logprobs_come_from_the_unmasked_logits():
    verify_logits, md = _one_request_verify_md()
    plain = torch.zeros(1, 8)
    plain[0, 3], plain[0, 5] = 4.0, 2.0
    runner = _spec_step_runner(1, [verify_logits, plain], [md, None], ["r0", "r1"])
    runner.input_batch.max_num_logprobs = 0
    runner.spec_logprob_row_tokens = lambda tokens, *_args: torch.zeros(
        (4, 1), dtype=torch.int64
    )
    seen = []

    def gather_logprobs(x, tokens):
        from vllm.v1.outputs import LogprobsTensors

        seen.append(x.clone())
        n = x.shape[0]
        return LogprobsTensors(
            tokens.view(-1, 1), torch.zeros(n, 1), torch.zeros(n, dtype=torch.int64)
        )

    runner.gather_logprobs = gather_logprobs
    output = _run_sample_tokens(runner)
    assert output.sampled_token_ids[1] == [5]  # r1 sampled from the masked logits
    assert seen[-1][0, 3] == 4.0  # its logprobs from the raw ones


def _verify(min_toks, grammar=False, start=0, grammar_only_stop=None, greedy=True):
    """Two requests, drafts of 3 and 1, every draft the target argmax.
    Vocabulary 8; each row prefers token `t` (logit 4), then token 6 (logit 2).
    Not `greedy`: random sampling at temperature 0.01, where a logit gap of 2
    is a factor e^200. Returns the selected tokens and whether the fused
    greedy prelude ran."""
    draft_lengths = [3, 1]
    prefer = [[1, 1, 1, 3], [2, 5]]  # per request: draft rows, then the bonus
    rows = [t for req in prefer for t in req]
    logits = torch.zeros(len(rows), 8)
    for r, t in enumerate(rows):
        logits[r, t] = 4.0
        logits[r, 6] = 2.0
    md = SimpleNamespace(
        draft_token_ids=torch.tensor([1, 1, 1, 2], dtype=torch.int32),
        draft_lengths=torch.tensor(draft_lengths, dtype=torch.int32),
        draft_lengths_cpu=np.array(draft_lengths),
        target_logits_indices=torch.tensor([0, 1, 2, 4], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3, 5], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
        group_indices=torch.tensor([0, 1, 2, 0], dtype=torch.int32),
    )
    spec_config = SimpleNamespace(
        num_speculative_tokens=3,
        rejection_sample_method="strict",
        synthetic_acceptance_rates=None,
    )
    runner = _runner(SimpleNamespace(no_top_k=True, no_top_p=True))
    runner.speculative_config = spec_config
    runner.rejection_sampler = RejectionSampler(spec_config, torch.device("cpu"))
    runner.sample_from_logits_func = runner.sample_from_logits
    runner._build_padded_sampling_params = lambda *_args: (
        torch.full((2, 1), 0.01),
        torch.zeros((2, 1), dtype=torch.int32),
        torch.ones((2, 1)),
    )
    runner._sampler_topk_candidates = lambda *_args: 0
    fused = MagicMock(wraps=runner.spec_bonus_and_target_logits)
    runner.spec_bonus_and_target_logits = fused
    grammar_output = None
    if grammar:  # a grammar that allows every token ...
        grammar_output = object()

        def mask_spec_logits(target, bonus, *_args):
            if grammar_only_stop is not None:  # ... or only the stop at target row 0
                target = target.clone()
                target[0] = NEG_INF
                target[0, grammar_only_stop] = 4.0
            return target, bonus

        runner.structured_decoding_manager = SimpleNamespace(
            mask_spec_logits=mask_spec_logits
        )
    with torch.compiler.set_stance("force_eager"):
        out = TPUModelRunner._sample_spec_verify_chunk(
            runner,
            logits,
            md,
            grammar_output,
            SimpleNamespace(scheduled_spec_decode_tokens={}),
            start,
            start + 2,
            greedy,
            None if greedy else torch.Generator().manual_seed(0),
            False,
            min_toks,
        ).tolist()
    return out, fused.called


def test_verify_without_min_tokens_accepts_every_draft():
    assert _verify({}) == ([[1, 1, 1, 3], [2, 5, -1, -1]], True)


@pytest.mark.parametrize("grammar", [False, True])
@pytest.mark.parametrize(
    "num_output, expected_req0",
    [
        # Below min_tokens = 4 at draft positions 0-1 (2 outputs): token 1 is
        # a stop there, so the first draft is rejected for the runner-up, 6.
        (2, [6, -1, -1, -1]),
        # 3 outputs: only draft position 0 is below; still rejected there.
        (3, [6, -1, -1, -1]),
    ],
)
def test_verify_masks_draft_positions_below_min_tokens(
    num_output, expected_req0, grammar
):
    got, fused = _verify({0: (4, num_output, (1,), False)}, grammar)
    assert got[0] == expected_req0
    assert got[1] == [2, 5, -1, -1]  # the other request is untouched
    assert fused is not grammar  # a greedy chunk keeps the fused prelude


@pytest.mark.parametrize("grammar", [False, True])
def test_verify_masks_the_bonus_row_at_its_position(grammar):
    # 1 output, min 5: draft positions 0-2 and the bonus (position 3) are
    # below. Stop token 3 is only the bonus row's favourite.
    assert _verify({0: (5, 1, (3,), False)}, grammar)[0][0] == [1, 1, 1, 6]
    # 2 outputs: the bonus is position 3 -> output 5, reached; not masked.
    assert _verify({0: (5, 2, (3,), False)}, grammar)[0][0] == [1, 1, 1, 3]


@pytest.mark.parametrize("grammar", [False, True])
def test_verify_masks_the_second_request_at_its_own_rows(grammar):
    """Request 1's draft row comes after request 0's three drafts and bonus
    (and after its three target rows once gathered)."""
    got, _ = _verify({1: (4, 2, (2,), False)}, grammar)
    assert got == [[1, 1, 1, 3], [6, -1, -1, -1]]


def test_verify_with_random_sampling_masks_drafts_and_bonus():
    assert _verify({}, greedy=False)[0] == [[1, 1, 1, 3], [2, 5, -1, -1]]
    # The stop draft is rejected and the recovered token is not a stop ...
    assert _verify({0: (4, 2, (1,), False)}, greedy=False)[0][0] == [6, -1, -1, -1]
    # ... and the bonus is sampled from its masked row.
    assert _verify({0: (5, 1, (3,), False)}, greedy=False)[0][0] == [1, 1, 1, 6]


def test_verify_chunk_at_an_offset_masks_its_own_requests():
    """Verify chunks start after the decode rows of the batch."""
    got, _ = _verify({2: (4, 2, (1,), False)}, start=2)
    assert got == [[6, -1, -1, -1], [2, 5, -1, -1]]


def test_verify_with_grammar_restores_a_structured_row_left_all_neg_inf():
    """The grammar allows only stop token 1 at the first draft position: the
    min_tokens mask leaves the row all -inf, so a structured request gets its
    stop back (as in vLLM) and the draft is accepted; an unstructured one
    keeps it masked."""
    structured, _ = _verify({0: (4, 2, (1,), True)}, grammar=True, grammar_only_stop=1)
    plain, _ = _verify({0: (4, 2, (1,), False)}, grammar=True, grammar_only_stop=1)
    assert structured[0][0] == 1
    assert plain[0][0] != 1


@pytest.mark.tpu_test
def test_compiled_mask_on_tpu_matches_reference():
    import torch_tpu  # noqa: F401

    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(16, 1024, generator=gen)
    logits[3] = NEG_INF
    logits[3, 7] = 1.0  # structured row allowing only stop token 7
    state = {
        0: (5, 0, (0, 5, 1023), False),
        3: (5, 0, (7,), True),
        9: (5, 0, tuple(range(min_tokens.STOP_IDS_WIDTH + 2)), False),  # 2 passes
    }
    table = min_tokens.build_stop_table(
        state, min_tokens.plain_rows(state, 0, 12), logits.shape[0]
    )
    with torch.compiler.set_stance("force_eager"):
        ref = min_tokens.apply_stop_table(logits, *table)
    got = min_tokens.apply_stop_table(logits.to(torch.device("tpu")), *table).cpu()
    assert torch.equal(got, ref)
    assert torch.isneginf(ref[0, [0, 5, 1023]]).all() and ref[3, 7] == 1.0
    assert torch.isneginf(ref[9, : min_tokens.STOP_IDS_WIDTH + 2]).all()
