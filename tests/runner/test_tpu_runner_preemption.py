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
"""A request that rejoins the batch must not consume stale asynchronous output."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.min_tokens import MinTokensTracker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncPreResults
from vllm_torchtpu.runner.vllm_v1.worker.gpu_model_runner import GPUModelRunner


@pytest.mark.parametrize("transition", ["preempt", "finish", "continue"])
@pytest.mark.parametrize("num_drafts", [0, 3])
def test_async_results_do_not_overwrite_recomputed_request(transition, num_drafts):
    runner = object.__new__(TPUModelRunner)
    runner._min_tokens = MinTokensTracker()
    stride = 1 + num_drafts
    invalid = transition != "continue"
    states = {
        rid: SimpleNamespace(
            prev_num_draft_len=num_drafts, output_token_ids=[5] + [0] * stride
        )
        for rid in ["resumed", "continuing"]
    }
    runner.requests = states
    runner.input_batch = SimpleNamespace(
        req_ids=["resumed", "continuing"],
        req_id_to_index={"resumed": 0, "continuing": 1},
        num_tokens_no_spec=np.array([16 + stride, 16 + stride]),
        token_ids_cpu=np.zeros((2, 256), dtype=np.int32),
    )
    runner._last_sequence_layout_plan = None
    runner.pcp_manager = None
    runner.speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_drafts) if num_drafts else None
    )
    copied = torch.tensor(
        [[71] + [-1] * num_drafts, [81] + [-1] * num_drafts], dtype=torch.int32
    )
    rejected = torch.tensor([num_drafts, num_drafts], dtype=torch.int32)
    previous = SimpleNamespace(
        req_ids=["resumed", "continuing"],
        req_id_to_index_copy={"resumed": 0, "continuing": 1},
        request_seq_lens=[(i, states[rid], 16, rid) for i, rid in enumerate(states)],
        discard_sampled_tokens_req_indices=[],
        spec_decode_num_rejected_tokens=rejected if num_drafts else None,
        num_draft_per_req={0: num_drafts, 1: num_drafts},
        wait_for_copy=lambda: copied,
    )
    runner._pre_async_results = previous
    runner._install_spec_token_room_guard = lambda: None
    runner._mamba_state_pos = {"resumed": 9, "continuing": 12}
    runner._mamba_offset_seeded = {"resumed", "continuing"}
    runner._mamba_slot_by_req_id = {"resumed": 1, "continuing": 1 + stride}
    runner._free_mamba_slots = []
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = {"resumed"} if transition == "finish" else set()
    # vLLM >= 0.34 reports a preempted request in preempted_req_ids and
    # re-sends it as a new request when it resumes.
    output.preempted_req_ids = {"resumed"} if transition == "preempt" else None

    def restore_from_scheduler(_output):
        if invalid:
            # Upstream replaces the resumed request's optimistic output with
            # authoritative prompt/output tokens before its local recompute.
            states["resumed"].output_token_ids[:] = [11, 12, 13]
            runner.input_batch.token_ids_cpu[0, :64] = np.arange(64)
            runner.input_batch.num_tokens_no_spec[0] = 64

    with patch.object(
        GPUModelRunner, "_update_states", side_effect=restore_from_scheduler
    ):
        runner._update_states(output)

    n_recompute = 64 if invalid else stride
    dst, src = runner._prepare_async_token_substitution_indices(
        0, 2, np.array([n_recompute, stride], dtype=np.int32)
    )
    if invalid:
        # Old [bonus,drafts] must never overwrite the recomputed prompt/output.
        assert dst.tolist() == list(range(n_recompute, n_recompute + stride))
        assert src.tolist() == list(range(stride, 2 * stride))
    else:
        assert dst.tolist() == list(range(2 * stride))
        assert src.tolist() == list(range(2 * stride))
    assert runner._mamba_state_pos == (
        {"continuing": 12} if invalid else {"resumed": 9, "continuing": 12}
    )
    assert runner._mamba_offset_seeded == (
        {"continuing"} if invalid else {"resumed", "continuing"}
    )
    assert runner._mamba_slot_by_req_id == (
        {"continuing": 1 + stride}
        if invalid
        else {"resumed": 1, "continuing": 1 + stride}
    )
    assert runner._free_mamba_slots == ([1] if invalid else [])
    # Preserve source row indices and rejection correction for the other request.
    assert previous.req_id_to_index_copy["continuing"] == 1
    assert previous.spec_decode_num_rejected_tokens is (
        rejected if num_drafts else None
    )
    runner._modify_prev_results()
    assert states["continuing"].output_token_ids == [5, 81]
    if invalid:
        assert states["resumed"].output_token_ids == [11, 12, 13]
        assert runner.input_batch.token_ids_cpu[0, :64].tolist() == list(range(64))
        assert runner.input_batch.num_tokens_no_spec[0] == 64


class _RejoiningBatch:
    """The InputBatch surface vLLM's _update_states reaches when a running
    request leaves the batch and is added back."""

    def __init__(self):
        self.req_ids: list[str] = []
        self.req_id_to_index: dict[str, int] = {}
        self.token_ids_cpu = np.zeros((1, 16), dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(1, dtype=np.int64)
        self.num_computed_tokens_cpu = np.zeros(1, dtype=np.int64)
        self.prev_req_id_to_index = None

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    def add_request(self, req_state) -> None:
        index = len(self.req_ids)
        self.req_ids.append(req_state.req_id)
        self.req_id_to_index[req_state.req_id] = index
        tokens = req_state.prompt_token_ids + req_state.output_token_ids
        self.token_ids_cpu[index, : len(tokens)] = tokens
        self.num_tokens_no_spec[index] = len(tokens)
        self.num_computed_tokens_cpu[index] = req_state.num_computed_tokens

    def remove_request(self, req_id: str) -> int:
        self.req_ids.remove(req_id)
        return self.req_id_to_index.pop(req_id)

    def update_req_spec_token_ids(self, _req_state, _scheduled_spec_tokens) -> None:
        pass

    def condense(self) -> None:
        pass

    def refresh_metadata(self) -> None:
        pass


def test_request_skipped_by_zero_token_step_rejoins_with_settled_drafts():
    """A request verifies K=2 drafts and one is rejected; the next step
    schedules no tokens; then the request rejoins the batch.

    By then vLLM has rolled the rejected draft out of num_computed_tokens and
    num_output_tokens. The zero-token step must settle the request's
    1 + K placeholders and drop the results, so _update_states adds it back
    with its real tokens and nothing maps for _prepare_inputs to subtract the
    rejected draft from its positions a second time.
    """
    req_state = SimpleNamespace(
        req_id="req-a",
        prompt_token_ids=[1, 2, 3],
        # 5 from an earlier step, then the 1 + K placeholders of the step that
        # verified drafts 71 and 9.
        output_token_ids=[5, 0, 0, 0],
        num_computed_tokens=3,
        prev_num_draft_len=2,
        block_ids=([0],),
    )

    class FakeCopyState:
        # Draft 71 accepted, then bonus 72; draft 9 rejected.
        sampled_token_ids_cpu = torch.tensor([[71, 72, -1]], dtype=torch.int32)

        def wait(self):
            pass

    runner = object.__new__(TPUModelRunner)
    runner.requests = {"req-a": req_state}
    runner.input_batch = _RejoiningBatch()
    runner.input_batch.add_request(req_state)
    runner._pre_async_results = AsyncPreResults(
        req_ids=["req-a"],
        next_tokens_tpu=torch.tensor([72, 0, 0], dtype=torch.int32),
        request_seq_lens=[(0, req_state, 3, "req-a")],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"req-a": 0},
        copy_state=FakeCopyState(),
        spec_decode_num_rejected_tokens=torch.tensor([1], dtype=torch.int32),
        num_draft_per_req={0: 2},
    )
    runner.execute_model_state = None
    runner._adaptive_verification = None
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.use_async_scheduling = True
    runner.use_async_spec_decode = False
    runner.late_interaction_runner = SimpleNamespace(
        on_requests_finished=lambda _req_ids: None
    )
    runner._may_reorder_batch = lambda _scheduler_output: None
    runner._install_spec_token_room_guard = lambda: None
    runner._min_tokens = MinTokensTracker()
    runner._thinking_budget = None
    runner._mamba_state_pos = {}
    runner._mamba_offset_seeded = set()
    runner._mamba_slot_by_req_id = {}
    runner._free_mamba_slots = []

    # vLLM after the drafts' output: prompt + [5, 71] computed, three output
    # tokens; the bonus 72 is the next input.
    rejoining = SchedulerOutput.make_empty()
    rejoining.scheduled_cached_reqs = CachedRequestData(
        req_ids=["req-a"],
        new_token_ids=[],
        new_block_ids=[None],
        num_computed_tokens=[5],
        num_output_tokens=[3],
    )
    rejoining.num_scheduled_tokens = {"req-a": 1}
    rejoining.total_num_scheduled_tokens = 1

    with (
        patch(
            "vllm_torchtpu.runner.vllm_v1.worker.gpu_model_runner.get_pp_group",
            return_value=SimpleNamespace(is_last_rank=True),
        ),
        patch.object(tpu_runner, "has_kv_transfer_group", return_value=False),
    ):
        output = runner.execute_model(SchedulerOutput.make_empty())
        assert output is tpu_runner.EMPTY_MODEL_RUNNER_OUTPUT
        assert runner.input_batch.num_reqs == 0
        assert runner._pre_async_results is None
        assert req_state.output_token_ids == [5, 71, 72]

        runner._update_states(rejoining)

    assert req_state.output_token_ids == [5, 71, 72]
    assert runner.input_batch.req_ids == ["req-a"]
    assert runner.input_batch.token_ids_cpu[0, :6].tolist() == [1, 2, 3, 5, 71, 72]
    assert runner.input_batch.num_tokens_no_spec[0] == 6
    # Positions start at num_computed_tokens_cpu; with no previous results
    # _prepare_inputs neither substitutes nor subtracts rejected drafts.
    assert runner.input_batch.num_computed_tokens_cpu[0] == 5
    dst, src = runner._prepare_async_token_substitution_indices(
        0, 1, np.array([1], dtype=np.int32)
    )
    assert dst.size == 0 and src.size == 0
