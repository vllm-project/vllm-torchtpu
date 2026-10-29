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
"""`wholecheck` bookkeeping on CPU: per-request agreement, the cut residual and the metadata of
the request that disagrees; the stock-vs-MK diff helper."""

import logging
from types import SimpleNamespace

import torch

from vllm_torchtpu.megakernel_runtime import qwen38_driver as qd


def _runner():
    return SimpleNamespace(model=SimpleNamespace(compute_logits=lambda h: h))


def test_per_request_log_names_the_request_that_disagrees(caplog):
    T, n, V = qd.T_VERIFY, 4, 16
    torch.manual_seed(0)
    stock = torch.randn(n * T, V)
    mk = stock.clone()
    mk[2 * T : 3 * T] = -mk[2 * T : 3 * T]  # request 2: every row's argmax flips
    pre = torch.randn(n * T, 32)
    seg = pre.clone()
    seg[2 * T : 3 * T] += 1.0
    kw = dict(context_lens=torch.tensor([100, 200, 300, 400]),
              query_start_loc=torch.tensor([0, 4, 8, 12, 16]),
              state_slots=torch.tensor([7, 8, 9, 10]))
    positions = torch.arange(n * T)
    drv = object.__new__(qd.Qwen38SegmentDriver)
    with caplog.at_level(logging.INFO):
        drv._log_per_request(_runner(), mk, stock, pre, seg, kw, positions, n * T, n)
    line = caplog.records[-1].getMessage()
    assert "agree=[1.0, 1.0, 0.0, 1.0]" in line, line
    assert "bad={2: {'ctx': 300, 'qsl': 8, 'slot': 9, 'pos': 8}}" in line, line
    rel = eval(line.split("pre_last_rel_vs_seg=")[1].split(" bad=")[0])
    assert rel[2] > 0.5 and max(rel[0], rel[1], rel[3]) == 0.0, rel


def test_vs_stock_counts_top1_agreement():
    stock = torch.eye(8)
    mk = stock.clone()
    mk[0] = torch.roll(mk[0], 1)
    rel, max_abs, agree = qd.Qwen38SegmentDriver._vs_stock(_runner(), (mk,), (stock,))
    assert agree == 7 / 8 and max_abs == 1.0 and rel > 0
