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
"""Every target/draft LM-head call passes the sharded head's fixed rows."""

import importlib
import types
from types import SimpleNamespace

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

_h = importlib.import_module("test_pcp_runner_integration")
ROWS = 32  # stands in for logits_rows(...) = 128 in the served config


def _bind_rows(runner, rows=ROWS):
    runner._sharded_head_rows = rows
    runner._logit_row_arange = {}
    runner._logits_call_rows = types.MethodType(
        TPUModelRunner._logits_call_rows, runner
    )
    runner._logit_rows_index = types.MethodType(
        TPUModelRunner._logit_rows_index, runner
    )
    seen = []

    def head(hidden, idx):
        seen.append(idx.shape[0])
        return torch.zeros((idx.shape[0], 10))

    runner.compute_selected_logits = head
    runner.select_logit_rows = lambda logits, rows: torch.index_select(logits, 0, rows)
    return seen


def test_busy_non_spec_logits_indices_pad_to_head_rows(monkeypatch):
    monkeypatch.setattr(envs, "TPU_SHARDED_LM_HEAD", True)
    sched = [3, 2]
    r = _h._make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=sched,
        scheduled_tokens=sched,
        token_paddings=[16, 64],
        pcp_size=1,
    )
    r.num_reqs_paddings = [r.max_num_reqs]
    _bind_rows(r)
    _, idx, padded_num_reqs, *_ = TPUModelRunner._prepare_inputs(
        r, _h._scheduler_output(sched), 0, 0
    )
    assert idx.shape[0] == ROWS and padded_num_reqs == 8
    assert idx[:2].tolist() == [2, 4] and not idx[8:].any()


def test_idle_dummy_head_call_projects_head_rows(monkeypatch):
    monkeypatch.setattr(envs, "TPU_SHARDED_LM_HEAD", True)
    r = _h._make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[32],
        scheduled_tokens=[32],
        token_paddings=[256],
        pcp_size=1,
    )
    r._dp_num_tokens_across_dp = lambda n: None
    seen = _bind_rows(r)
    _h._run_dummy_run(monkeypatch, r, num_tokens=16)  # no head call: not lockstep
    assert seen == []
    TPUModelRunner._dummy_run(r, 16, r.num_reqs_max_model_len, 4, dp_lockstep=True)
    assert seen == [ROWS]  # was num_reqs_max_model_len (8)


def test_precompile_head_rows_and_trim(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors", lambda *a, **k: None
    )
    r = SimpleNamespace(
        device=torch.device("cpu"),
        num_tokens_paddings=[16, 64],
        num_reqs_paddings=[8],
        max_num_reqs=8,
        _hidden_states_dtype=torch.float32,
        model_config=SimpleNamespace(get_hidden_size=lambda: 4),
        _precompile_timed=lambda *_: __import__("contextlib").nullcontext(),
    )
    seen = _bind_rows(r)
    trimmed = []
    r.select_logit_rows = (
        lambda logits, rows: trimmed.append(rows.shape[0]) or logits[: rows.shape[0]]
    )
    TPUModelRunner._precompile_compute_selected_logits(r)
    assert seen == [ROWS, ROWS] and trimmed == [8, 8]


def test_draft_head_rows_equal_on_busy_and_idle_ranks():
    d = SimpleNamespace(
        runner=SimpleNamespace(num_tokens_paddings=[16, 32, 64, 128], max_num_reqs=32),
        _dp_lockstep_sharded=lambda: True,
    )
    busy = Eagle3Proposer._loop_bucket(d, 1)  # one live request
    idle = Eagle3Proposer._loop_bucket(d, 8)  # run_dp_dummy_draft
    assert busy == idle == 32
