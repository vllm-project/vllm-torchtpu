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
"""A resumed request must not consume asynchronous output from its old KV state."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


@pytest.mark.parametrize("transition", ["preempt", "resume", "finish", "continue"])
@pytest.mark.parametrize("num_drafts", [0, 3])
def test_async_results_do_not_overwrite_recomputed_request(transition, num_drafts):
    runner = object.__new__(TPUModelRunner)
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
    output.preempted_req_ids = {"resumed"} if transition == "preempt" else None
    output.scheduled_cached_reqs.resumed_req_ids = (
        {"resumed"} if transition == "resume" else set()
    )

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
