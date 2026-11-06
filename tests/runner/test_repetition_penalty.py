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
sync decode integration, and TPU compilation."""

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

from vllm_torchtpu.runner import min_tokens
from vllm_torchtpu.runner import repetition_penalty as rep_pen
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

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
            return [3] if req_id == "keep" else None

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


def _plain_runner(logits: torch.Tensor, async_scheduling: bool = False, spec=None):
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
        speculative_config=spec,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _is_async_drafter=False,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        _adaptive_verification=None,
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
        "_mask_min_tokens",
        "_min_tokens_batch_state",
        "_num_output_tokens",
        "_repetition_penalty_batch_state",
        "_repetition_penalty_supported",
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

    @pytest.mark.parametrize(
        ("async_scheduling", "spec"), [(True, None), (False, object())]
    )
    def test_unsupported_modes_skip_penalty_with_warning(self, async_scheduling, spec):
        logits = torch.zeros(1, 8)
        runner = _plain_runner(logits, async_scheduling=async_scheduling, spec=spec)
        runner._repetition_penalty.add(
            "r0", SamplingParams(repetition_penalty=1.5), [4]
        )
        with patch("vllm_torchtpu.runner.tpu_runner.logger") as log:
            assert runner._repetition_penalty_batch_state() == {}
        log.warning_once.assert_called_once()
        assert "async_scheduling=False" in log.warning_once.call_args.args[0]

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
        runner.scheduler_config = SimpleNamespace(async_scheduling=False)
        runner.speculative_config = None
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

    @pytest.mark.parametrize(
        ("async_scheduling", "spec", "expected"),
        [
            (False, None, [(8, 12), (16, 12)]),
            (True, None, []),
            (False, object(), []),
        ],
    )
    def test_precompile_covers_supported_buckets_only(
        self, async_scheduling, spec, expected
    ):
        runner = object.__new__(TPUModelRunner)
        runner.num_reqs_paddings = [8, 16]
        runner.speculative_config = spec
        runner.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
        runner.device = torch.device("cpu")
        runner._dummy_logits = lambda n: torch.zeros(n, 12, dtype=torch.bfloat16)
        runner._precompile_timed = MagicMock()
        apply_shapes: list[tuple[int, int]] = []
        orig_apply = rep_pen.apply_repetition_penalty

        def spy_apply(logits, seen_mask, penalties):
            apply_shapes.append(tuple(logits.shape))
            return orig_apply(logits, seen_mask, penalties)

        with patch.object(rep_pen, "apply_repetition_penalty", side_effect=spy_apply):
            runner._precompile_repetition_penalty()

        assert apply_shapes == expected


@pytest.mark.tpu_test
def test_compiled_repetition_penalty_on_tpu():
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

    with torch.compiler.set_stance("force_eager"):
        ref = rep_pen.apply_penalties_table(logits_cpu, seen_np, penalties_np)

    got = rep_pen.apply_penalties_table(
        logits_cpu.to(device), seen_np, penalties_np
    ).cpu()

    assert got.dtype == torch.bfloat16
    assert torch.equal(got, ref)
