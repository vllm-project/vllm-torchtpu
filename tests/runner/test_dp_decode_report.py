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
"""The decode report on the DP metadata all-reduce of _dp_coordinated_step.

SrptPrefillScheduler applies its attention work cap only while some DP rank
decodes, and learns about the other ranks from this report. The ranks are
simulated here: all_reduce is the element-wise MAX of what each rank sends.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

DP_SIZE = 4


def _runner() -> TPUModelRunner:
    runner = object.__new__(TPUModelRunner)
    runner._dp_lockstep_enabled = lambda: True
    runner.num_tokens_paddings = [16, 32, 64]
    return runner


def _step(monkeypatch, runner, rank, decodes, all_reduce):
    group = SimpleNamespace(rank_in_group=rank, world_size=DP_SIZE, cpu_group=None)
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: group)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    return runner._dp_coordinated_step(16, 1, local_any_decode=decodes)


def _other_decode_per_rank(monkeypatch, decoding: set[int]) -> list[bool]:
    runners = [_runner() for _ in range(DP_SIZE)]
    sent: list[torch.Tensor] = []
    for rank, runner in enumerate(runners):
        _step(
            monkeypatch,
            runner,
            rank,
            rank in decoding,
            lambda t, op=None, group=None: sent.append(t.clone()),
        )
    reduced = torch.stack(sent).amax(dim=0)
    for rank, runner in enumerate(runners):
        _step(
            monkeypatch,
            runner,
            rank,
            rank in decoding,
            lambda t, op=None, group=None: t.copy_(reduced),
        )
    return [runner._dp_step_other_decode for runner in runners]


@pytest.mark.parametrize(
    ("decoding", "expected"),
    [
        (set(), [False, False, False, False]),
        # The only decoding rank is not told that another one decodes.
        ({2}, [True, True, False, True]),
        ({0, 3}, [True, True, True, True]),
    ],
)
def test_each_rank_learns_whether_another_rank_decodes(monkeypatch, decoding, expected):
    assert _other_decode_per_rank(monkeypatch, decoding) == expected


def test_no_decode_report_without_dp_lockstep():
    runner = _runner()
    runner._dp_lockstep_enabled = lambda: False
    runner._dp_step_other_decode = True
    assert runner._dp_coordinated_step(16, 1, local_any_decode=True) == (None, 1)
    assert runner._dp_step_other_decode is False


@pytest.mark.parametrize(
    ("cap", "computed", "expected"),
    [
        # Off with the cap, which is the only reader.
        ("0", [8, 20], False),
        ("25000000", [8, 19], False),
        ("25000000", [8, 20], True),
    ],
)
def test_any_decode_means_a_request_past_its_prompt(
    monkeypatch, cap, computed, expected
):
    monkeypatch.setenv("VLLM_TPU_SRPT_ATTN_WORK_CAP", cap)
    runner = object.__new__(TPUModelRunner)
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1},
        num_computed_tokens_cpu=np.array(computed),
        num_prompt_tokens=np.array([16, 20]),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"a": 8, "b": 1})
    assert runner._any_decode(scheduler_output) is expected
