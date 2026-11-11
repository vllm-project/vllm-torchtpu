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
"""VLLM_TPU_MK_DRAFT: compile ranges past the draft kernel's rows trace the stock draft.

za9e c4 j-c6b8b17f: the fused draft loop's warmup compiled every range up to 8192 rows,
built the MK draft kernel at 8192 rows (1872 MiB VMEM scratch) and the server died.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode import mtp_fused_loop as fl


def _inner(max_rows):
    return SimpleNamespace(mk_draft=SimpleNamespace(max_rows=max_rows))


def test_rows_gate():
    inner = _inner({4: 32, 1: 16})
    assert fl.mk_draft_rows_ok(inner, 16, 1) and fl.mk_draft_rows_ok(inner, 32, 4)
    assert not fl.mk_draft_rows_ok(inner, 8192, 1)
    assert not fl.mk_draft_rows_ok(inner, 8192, 4)
    assert not fl.mk_draft_rows_ok(inner, 17, 1)
    assert not fl.mk_draft_rows_ok(SimpleNamespace(mk_draft=None), 4, 1)
    assert not fl.mk_draft_rows_ok(_inner({}), 4, 1)  # not set by the driver: never MK


class _StubMtp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _inner({4: 32, 1: 16})
        self.mk_calls = 0

    def mk_draft_step(self, input_ids, positions, hidden, rows_per_request):
        self.mk_calls += 1
        assert input_ids.shape[0] <= 16, "the MK draft must not be traced at this size"
        return hidden, input_ids

    def forward(self, input_ids, positions, hidden, intermediate_tensors, inputs_embeds,
                spec_step_idx=0):
        return hidden

    def compute_logits(self, hidden):
        return torch.zeros(hidden.shape[0], 8)


@pytest.mark.parametrize("rows,mk", [(8192, False), (16, True)])
def test_fused_loop_uses_mk_only_within_rows(monkeypatch, rows, mk):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_LOOP", True)
    mtp = _StubMtp()
    loop = fl._MtpFusedLoop.__new__(fl._MtpFusedLoop)
    torch.nn.Module.__init__(loop)
    loop.mtp, loop.num_steps, loop.prefix = mtp, 2, ""
    ids = torch.zeros(rows, dtype=torch.int32)
    tokens = loop.forward(ids, torch.zeros(rows), torch.zeros(rows, 4), torch.zeros(rows))
    assert len(tokens) == 2
    assert (mtp.mk_calls > 0) == mk
