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
"""``repetition_penalty`` on the TPU sampler: request tracking, scaling math,
zero-stall async pending-token handling, spec verify, and TPU compilation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import min_tokens
from vllm_torchtpu.runner import repetition_penalty as rep_pen
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncPreResults,
    AsyncTPUCopyState,
)

NEG_INF = float("-inf")


@pytest.fixture
def eager():
    with torch.compiler.set_stance("force_eager"):
        yield


class TestTracker:
    def test_tracks_only_non_unit_repetition_penalty(self):
        tracker = rep_pen.RepetitionPenaltyTracker()
        tracker.add("a", SamplingParams(repetition_penalty=1.05), [1, 3])
        tracker.add("b", SamplingParams(repetition_penalty=1.0), [2])
        tracker.add("c", None, [4])
        assert list(tracker.requests) == ["a"]
        assert tracker.requests["a"].penalty == pytest.approx(1.05)

    def test_incremental_seen_mask_and_rollback_and_out_of_range_ids(self):
        tracker = rep_pen.RepetitionPenaltyTracker()
        # Includes -1 (MM placeholder) and 151645 (text token above codec head).
        tracker.add(
            "a",
            SamplingParams(repetition_penalty=1.2),
            [-1, 2, 5, 151645],
        )
        entry = tracker.requests["a"]

        # Prefill: only in-range prompt ids [2, 5] marked.
        m0 = entry.seen_mask([], vocab_size=8)
        assert np.where(m0)[0].tolist() == [2, 5]

        # Incremental single-token decode update (+ out-of-range ignored).
        m1 = entry.seen_mask([6, 999], vocab_size=8)
        assert np.where(m1)[0].tolist() == [2, 5, 6]

        # Large delta (>8 tokens) takes the vectorized numpy branch.
        m2 = entry.seen_mask([6, 999] + [1] * 10, vocab_size=8)
        assert np.where(m2)[0].tolist() == [1, 2, 5, 6]

        # Shrunk output (e.g. rollback) rebuilds cleanly from prompt + output.
        m3 = entry.seen_mask([7], vocab_size=8)
        assert np.where(m3)[0].tolist() == [2, 5, 7]

    def test_preempted_kept_and_gone_evicted_and_empty_sentinel_immutable(self):
        tracker = rep_pen.RepetitionPenaltyTracker()
        tracker.add("keep", SamplingParams(repetition_penalty=1.1), [1])
        tracker.add("gone", SamplingParams(repetition_penalty=1.1), [2])

        def lookup(req_id: str):
            return ([3], -1) if req_id == "keep" else None

        # "keep" is temporarily absent from req_id_to_index (preempted).
        assert tracker.batch_state({}, lookup) == {}
        assert list(tracker.requests) == ["keep"]
        state = tracker.batch_state({"keep": 1}, lookup)
        assert list(state) == [1]

        assert not rep_pen.EMPTY_TRACKER
        assert not rep_pen.EMPTY_TRACKER.any_in({"keep": 0})
        with pytest.raises(RuntimeError, match="EMPTY_TRACKER"):
            rep_pen.EMPTY_TRACKER.add("x", SamplingParams(repetition_penalty=1.1))


@pytest.mark.usefixtures("eager")
class TestScalingMath:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_upstream_vllm_reference(self, dtype):
        from vllm._custom_ops import apply_repetition_penalties

        gen = torch.Generator().manual_seed(42)
        num_rows, vocab_size = 6, 64
        logits = torch.randn((num_rows, vocab_size), generator=gen).to(dtype)
        logits[0, 0] = 0.0
        logits[1, 5] = NEG_INF

        prompt_mask = torch.zeros((num_rows, vocab_size), dtype=torch.bool)
        output_mask = torch.zeros((num_rows, vocab_size), dtype=torch.bool)
        prompt_mask[0, [0, 3, 12]] = True
        output_mask[0, [7, 12]] = True
        prompt_mask[1, [5, 9]] = True
        output_mask[3, [1, 2, 63]] = True
        # Row 2 has penalty == 1.0 (untracked), row 4 has penalty < 1.0.
        penalties_1d = torch.tensor(
            [1.05, 1.5, 1.0, 2.0, 0.8, 1.0], dtype=torch.float32
        )
        output_mask[4, [10, 20]] = True

        ref = logits.clone()
        apply_repetition_penalties(ref, prompt_mask, output_mask, penalties_1d)

        seen_np = (prompt_mask | output_mask).numpy()
        pen_np = penalties_1d.view(-1, 1).numpy()
        got = rep_pen.apply_penalties_table(logits, seen_np, pen_np)

        assert got.dtype == dtype
        assert torch.isneginf(got[1, 5])
        torch.testing.assert_close(
            got[~torch.isneginf(ref)],
            ref[~torch.isneginf(ref)],
            rtol=1e-2 if dtype == torch.bfloat16 else 1e-6,
            atol=1e-2 if dtype == torch.bfloat16 else 1e-6,
        )


def _plain_runner(logits: torch.Tensor, async_scheduling: bool = False):
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
            num_tokens_no_spec=[2],
            token_ids_cpu=np.zeros((1, 16), dtype=np.int32),
        ),
        requests={
            "r0": SimpleNamespace(
                req_id="r0",
                prompt_token_ids=[4],
                num_computed_tokens=1,
                num_tokens=2,
                output_token_ids=[],
            )
        },
        _pre_async_results=None,
        _pp_wave=None,
        _dp_step_other_decode=False,
        _min_tokens=min_tokens.MinTokensTracker(),
        _repetition_penalty=rep_pen.RepetitionPenaltyTracker(),
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _is_async_drafter=False,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        is_pooling_model=False,
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda x: torch.argmax(x, dim=-1, keepdim=True),
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _modify_prev_results=MagicMock(),
        _sync_replicated_drafts_across_tp=lambda drafts: drafts,
        _update_placeholder=lambda *_args: {"r0": 0},
        _stage_timing=MagicMock(),
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    for name in (
        "_apply_repetition_penalty_plain",
        "_lookup_repetition_penalty_output",
        "_make_model_runner_output",
        "_mask_min_tokens",
        "_min_tokens_batch_state",
        "_num_output_tokens",
        "_repetition_penalty_batch_state",
        "_settle_async_drafts_for_repetition_penalty",
        "_settle_draft_placeholders_for_min_tokens",
        "get_finished_kv_transfers",
        "maybe_wait_for_kv_save",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
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


@pytest.mark.usefixtures("eager")
class TestRunnerIntegration:
    def test_plain_step_penalizes_seen_tokens_and_preserves_raw_logprobs(self):
        # Token 4 (in prompt) has logit 3.0; token 2 (unseen) has logit 2.5.
        # With penalty 1.5, token 4 drops to 2.0, so greedy picks token 2,
        # while raw logprobs still read 3.0 on token 4.
        logits = torch.zeros(1, 8)
        logits[0, 4], logits[0, 2] = 3.0, 2.5
        runner = _plain_runner(logits)
        runner._repetition_penalty.add(
            "r0", SamplingParams(repetition_penalty=1.5), [4]
        )
        runner.input_batch.max_num_logprobs = 0
        seen_logprob_logits: list[torch.Tensor] = []

        def gather_logprobs(x, tokens):
            seen_logprob_logits.append(x.clone())
            return LogprobsTensors(
                tokens.view(-1, 1), torch.zeros(1, 1), torch.zeros(1, dtype=torch.int64)
            )

        runner.gather_logprobs = gather_logprobs
        out = _run_sample_tokens(runner)
        assert out.sampled_token_ids == [[2]]
        assert seen_logprob_logits[0][0, 4].item() == pytest.approx(3.0)

    def test_plain_async_step_penalizes_pending_device_token_without_zero_leak(self):
        """Under async_scheduling=True (no drafts), output_token_ids ends with
        placeholder [0] while the true previous token (5) lives in
        pre.next_tokens_tpu. Token 0 must NOT be penalized, token 5 MUST be
        penalized, and _modify_prev_results must NOT be called early."""
        logits = torch.zeros(1, 8)
        # Token 5 (pending on TPU): 3.0 -> 3.0 / 2.0 = 1.5
        # Token 0 (placeholder in output_token_ids, NOT truly seen): 2.6 -> 2.6
        # Token 3 (unseen runner-up): 2.4 -> 2.4
        logits[0, 5], logits[0, 0], logits[0, 3] = 3.0, 2.6, 2.4
        runner = _plain_runner(logits, async_scheduling=True)
        r0 = runner.requests["r0"]
        r0.output_token_ids = [0]  # single optimistic async placeholder!
        runner._repetition_penalty.add(
            "r0", SamplingParams(repetition_penalty=2.0), [4]
        )
        runner._pre_async_results = AsyncPreResults(
            req_ids=["r0"],
            next_tokens_tpu=torch.tensor([5, 0, 0, 0], dtype=torch.int64),
            request_seq_lens=[(0, r0, 2, "r0")],
            discard_sampled_tokens_req_indices=[],
            req_id_to_index_copy={"r0": 0},
            num_draft_per_req=None,
        )

        # Verify batch_state itself never calls _modify_prev_results early.
        state = runner._repetition_penalty_batch_state()
        runner._modify_prev_results.assert_not_called()
        assert list(state[0].committed_output) == []
        assert state[0].pending_slot == 0

        async_out = _run_sample_tokens(runner)
        # sample_tokens calls _modify_prev_results once at its normal post-sample point.
        assert runner._modify_prev_results.call_count == 1
        assert async_out.get_output().sampled_token_ids == [[0]]

    def test_async_spec_drafts_settle_before_building_state(self):
        logits = torch.zeros(1, 8)
        runner = _plain_runner(logits, async_scheduling=True)
        r0 = runner.requests["r0"]
        r0.output_token_ids = [6, 0, 0, 0]  # 1 real + 1 bonus + 2 draft placeholders
        runner._repetition_penalty.add(
            "r0", SamplingParams(repetition_penalty=1.5), [4]
        )

        def settle():
            r0.output_token_ids[:] = [6, 7]

        runner._modify_prev_results = MagicMock(side_effect=settle)
        runner._pre_async_results = AsyncPreResults(
            req_ids=["r0"],
            next_tokens_tpu=torch.zeros(4, dtype=torch.int64),
            request_seq_lens=[(0, r0, 3, "r0")],
            discard_sampled_tokens_req_indices=[],
            req_id_to_index_copy={"r0": 0},
            num_draft_per_req={0: 2},
        )
        state = runner._repetition_penalty_batch_state()
        runner._modify_prev_results.assert_called_once()
        assert runner._pre_async_results is None
        assert list(state[0].committed_output) == [6, 7]
        assert state[0].pending_slot == -1

    def test_batch_reorder_and_top_tokens_gate(self):
        batch = InputBatch(
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
        a = CachedRequestState(
            req_id="a",
            prompt_token_ids=[3],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0, repetition_penalty=2.0),
            generator=None,
            block_ids=([0],),
            num_computed_tokens=1,
            output_token_ids=[5],
        )
        b = CachedRequestState(
            req_id="b",
            prompt_token_ids=[3],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0),
            generator=None,
            block_ids=([1],),
            num_computed_tokens=1,
            output_token_ids=[5],
        )
        batch.add_request(a)
        batch.add_request(b)
        batch.refresh_metadata()
        batch.swap_states(0, 1)  # "a" moves to row 1

        runner = object.__new__(TPUModelRunner)
        runner.input_batch = batch
        runner.requests = {"a": a, "b": b}
        runner._pre_async_results = None
        runner._greedy_head_owner = object()
        runner._min_tokens = min_tokens.MinTokensTracker()
        runner._repetition_penalty = rep_pen.RepetitionPenaltyTracker()
        runner._repetition_penalty.add("a", a.sampling_params, a.prompt_token_ids)

        assert not runner._greedy_top_tokens_this_step()
        state = runner._repetition_penalty_batch_state()
        logits = torch.full((4, 16), 2.0)
        out = runner._apply_repetition_penalty_plain(logits, 0, 2, state)
        # Row 0 ("b") untouched; row 1 ("a") scaled at tokens 3 and 5.
        assert torch.equal(out[0], logits[0])
        assert out[1, 3].item() == pytest.approx(1.0)
        assert out[1, 5].item() == pytest.approx(1.0)
        assert out[1, 0].item() == pytest.approx(2.0)

    def test_update_states_tracks_and_discards_requests(self):
        runner = object.__new__(TPUModelRunner)
        runner._min_tokens = min_tokens.MinTokensTracker()
        runner._mamba_state_pos, runner._mamba_slot_by_req_id = {}, {}
        runner._free_mamba_slots, runner._mamba_offset_seeded = [], set()
        runner._pre_async_results = None
        runner.requests = {
            "new": SimpleNamespace(prompt_token_ids=[2, 4]),
        }
        runner._install_spec_token_room_guard = lambda: None
        # Starts from class-level EMPTY_TRACKER and upgrades automatically.
        assert runner._repetition_penalty is rep_pen.EMPTY_TRACKER
        new_req = SimpleNamespace(
            req_id="new",
            sampling_params=SamplingParams(repetition_penalty=1.1),
        )
        so = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=None,
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            kv_cache_block_copies=None,
            scheduled_new_reqs=[new_req],
        )
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, so)
        assert list(runner._repetition_penalty.requests) == ["new"]
        so.scheduled_new_reqs = []
        so.finished_req_ids = {"new"}
        with patch.object(GPUModelRunner, "_update_states", return_value=None):
            TPUModelRunner._update_states(runner, so)
        assert not runner._repetition_penalty

    @pytest.mark.parametrize("use_grammar", [False, True])
    def test_spec_verify_chunk_penalizes_draft_prefixes_and_bonus(self, use_grammar):
        """Draft [2, 3]: draft pos 0 sees prompt [1]; draft pos 1 sees [1, 2];
        bonus pos 2 sees [1, 2, 3]."""
        logits = torch.zeros(3, 8)
        # Pos 0: token 2 (unseen, 3.0) accepted.
        logits[0, 2], logits[0, 6] = 3.0, 2.0
        # Pos 1: token 2 (now seen from draft[0]!, 3.0 / 2.0 = 1.5 < token 3's 2.5)
        # -> token 3 accepted!
        logits[1, 2], logits[1, 3] = 3.0, 2.5
        # Bonus pos 2: token 3 (now seen from draft[1]!, 3.0 / 2.0 = 1.5 < 6's 2.0)
        # -> bonus picks 6 instead of repeating 3!
        logits[2, 3], logits[2, 6] = 3.0, 2.0

        md = SimpleNamespace(
            draft_token_ids=torch.tensor([2, 3], dtype=torch.int32),
            draft_lengths=torch.tensor([2], dtype=torch.int32),
            draft_lengths_cpu=np.array([2]),
            target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([2], dtype=torch.int32),
            segment_ids=torch.tensor([0, 0], dtype=torch.int64),
            group_indices=torch.tensor([0, 1], dtype=torch.int32),
        )
        spec_cfg = SimpleNamespace(
            num_speculative_tokens=2,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
        runner = object.__new__(TPUModelRunner)
        runner.input_batch = SimpleNamespace(
            req_ids=["r0"], no_top_k=True, no_top_p=True
        )
        runner.speculative_config = spec_cfg
        runner.rejection_sampler = RejectionSampler(spec_cfg, torch.device("cpu"))
        runner.structured_decoding_manager = SimpleNamespace(
            mask_spec_logits=lambda t, b, *_a: (t, b)
        )
        runner.sample_from_logits_func = lambda x, *_a, **_kw: torch.argmax(
            x, dim=-1, keepdim=True
        )
        tracker = rep_pen.RepetitionPenaltyTracker()
        tracker.add("r0", SamplingParams(repetition_penalty=2.0), [1])
        state = tracker.batch_state({"r0": 0}, lambda _rid: ([], -1))

        out = TPUModelRunner._sample_spec_verify_chunk(
            runner,
            logits,
            md,
            object() if use_grammar else None,
            SimpleNamespace(scheduled_spec_decode_tokens={"r0": [2, 3]}),
            0,
            1,
            True,
            None,
            False,
            {},
            state,
        )
        assert out.tolist() == [[2, 3, 6]]

    def test_precompile_covers_sync_spec_and_async_buckets(self):
        runner = object.__new__(TPUModelRunner)
        runner.num_reqs_paddings = [8, 16]
        runner.num_tokens_paddings = [16, 32, 64]
        runner.speculative_config = object()
        runner._mtp_prefill_only = False
        runner.scheduler_config = SimpleNamespace(async_scheduling=True)
        runner.device = torch.device("cpu")
        runner._dummy_logits = lambda n: torch.zeros(n, 12, dtype=torch.bfloat16)
        runner._precompile_timed = MagicMock()
        runner.structured_decoding_manager = SimpleNamespace(
            target_grammar_bitmask_cpu=torch.zeros(32, 1)
        )
        apply_shapes: list[tuple[int, int]] = []
        async_shapes: list[tuple[int, int]] = []
        orig_apply = rep_pen.apply_repetition_penalty
        orig_mark = rep_pen.mark_pending_async_tokens

        def spy_apply(logits, seen_mask, penalties):
            apply_shapes.append(tuple(logits.shape))
            return orig_apply(logits, seen_mask, penalties)

        def spy_mark(seen_mask, next_tokens, pre_indices):
            async_shapes.append((seen_mask.shape[0], next_tokens.shape[0]))
            return orig_mark(seen_mask, next_tokens, pre_indices)

        with (
            patch.object(rep_pen, "apply_repetition_penalty", side_effect=spy_apply),
            patch.object(rep_pen, "mark_pending_async_tokens", side_effect=spy_mark),
        ):
            runner._precompile_repetition_penalty()

        assert apply_shapes == [(8, 12), (16, 12), (32, 12)]
        assert async_shapes == [(8, 8), (8, 16), (16, 8), (16, 16)]


@pytest.mark.tpu_test
def test_compiled_repetition_penalty_and_async_pending_on_tpu():
    import torch_tpu  # noqa: F401

    device = torch.device("tpu")
    gen = torch.Generator().manual_seed(7)
    logits_cpu = torch.randn((8, 512), generator=gen, dtype=torch.bfloat16)
    logits_cpu[0, 10] = NEG_INF

    seen_np = np.zeros((8, 512), dtype=np.bool_)
    seen_np[0, [3, 10, 42]] = True
    seen_np[2, [100]] = True
    penalties_np = np.ones((8, 1), dtype=np.float32)
    penalties_np[0, 0] = 1.05
    penalties_np[2, 0] = 1.5
    pre_indices_np = np.full((8,), -1, dtype=np.int32)
    pre_indices_np[0] = 3  # pending async token at slot 3 -> token id 77
    pre_indices_np[2] = 1  # out-of-range token id 9999 -> ignored
    next_tokens_cpu = torch.tensor([0, 9999, 0, 77, 0, 0, 0, 0], dtype=torch.int64)

    with torch.compiler.set_stance("force_eager"):
        ref = rep_pen.apply_penalties_table(
            logits_cpu,
            seen_np,
            penalties_np,
            pre_indices_np,
            next_tokens_cpu,
        )

    got = rep_pen.apply_penalties_table(
        logits_cpu.to(device),
        seen_np,
        penalties_np,
        pre_indices_np,
        next_tokens_cpu.to(device),
    ).cpu()

    assert got.dtype == torch.bfloat16
    assert torch.equal(got, ref)
