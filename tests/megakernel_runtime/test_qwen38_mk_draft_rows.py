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
"""VLLM_TPU_MK_DRAFT: the MK draft programs are compiled only at the kernel's rows.

za9e c4 j-c6b8b17f: the fused loop chose the MK step for every compile range and built
the draft kernel at 8192 rows (1872 MiB VMEM). za9g: a token-count branch inside the
traced loop froze the graph at the first bucket (ShapeSpecializationError at 8192).
The MK choice is now a construction-time attribute of a separate program whose
compile sizes stop at the kernel's rows; the caller picks it per step in Python.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from vllm_torchtpu.spec_decode import mtp_fused_loop as fl


def _inner(max_rows):
    return SimpleNamespace(mk_draft=SimpleNamespace(max_rows=max_rows))


def test_rows_gate():
    inner = _inner({4: 32, 1: 16})
    assert fl.mk_draft_rows_ok(inner, 16, 1) and fl.mk_draft_rows_ok(inner, 32, 4)
    assert not fl.mk_draft_rows_ok(inner, 8192, 1)
    assert not fl.mk_draft_rows_ok(inner, 17, 1)
    assert not fl.mk_draft_rows_ok(SimpleNamespace(mk_draft=None), 4, 1)
    assert not fl.mk_draft_rows_ok(_inner({}), 4, 1)


def test_mk_compile_config_drops_prefill_sized_buckets():
    cc = SimpleNamespace(
        compile_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 1024, 8192],
        inductor_compile_config={"pre_grad_custom_pass": object(), "keep": 1},
    )
    cfg = SimpleNamespace(compilation_config=cc)
    inner = _inner({4: 32, 1: 16})
    loop_cfg, loop_sizes = fl.mk_draft_compile_config(cfg, inner, 1)
    step0_cfg, step0_sizes = fl.mk_draft_compile_config(cfg, inner, 4)
    assert loop_sizes == [1, 2, 4, 8, 16] and loop_cfg.compilation_config.compile_sizes == loop_sizes
    assert step0_sizes == [1, 2, 4, 8, 16, 32]
    assert cc.compile_sizes[-1] == 8192  # the stock draft config is untouched
    # A fresh backend asserts the pre-grad pass absent: the MK copy drops it, the stock
    # config keeps it.
    assert loop_cfg.compilation_config.inductor_compile_config == {"keep": 1}
    assert "pre_grad_custom_pass" in cc.inductor_compile_config
    _, none = fl.mk_draft_compile_config(cfg, _inner({}), 1)
    assert none == []


class _StubMtp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _inner({4: 32, 1: 16})
        self.mk_calls = 0

    def mk_draft_step(self, input_ids, positions, hidden, rows_per_request):
        self.mk_calls += 1
        return hidden + 1, input_ids

    def forward(self, input_ids, positions, hidden, intermediate_tensors, inputs_embeds,
                spec_step_idx=0):
        return hidden * 2

    def compute_logits(self, hidden):
        return hidden


def _loop(use_mk):
    loop = fl._MtpFusedLoop.__new__(fl._MtpFusedLoop)
    torch.nn.Module.__init__(loop)
    loop.mtp, loop.num_steps, loop.prefix, loop.use_mk = _StubMtp(), 2, "", use_mk
    return loop


def _args(rows):
    return (torch.zeros(rows, dtype=torch.int32), torch.zeros(rows),
            torch.zeros(rows, 4), torch.zeros(rows))


def test_mk_choice_is_static():
    stock, mk = _loop(False), _loop(True)
    stock.forward(*_args(16))
    mk.forward(*_args(16))
    assert stock.mtp.mk_calls == 0 and mk.mtp.mk_calls == 2


def test_stock_loop_has_no_token_count_guard():
    """One dynamo trace serves 16 and 8192 rows: the loop must not branch on the token
    count (a shape guard would freeze every compile range at the first bucket)."""
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    loop = _loop(False)
    counter = CompileCounter()
    fn = torch.compile(lambda *a: loop.forward(*a), backend=counter, dynamic=True)
    a16, a8192 = _args(16), _args(8192)
    for t in (*a16, *a8192):
        torch._dynamo.mark_dynamic(t, 0)
    fn(*copy.deepcopy(a16))
    fn(*copy.deepcopy(a8192))
    assert counter.frame_count == 1, counter.frame_count
    assert loop.mtp.mk_calls == 0
