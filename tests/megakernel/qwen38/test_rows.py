# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The Qwen3.8 stage's row plan (CPU)."""

import numpy as np
import pytest

from vllm_torchtpu.megakernel.qwen38.rows import Span, plan_rows, rows_per_core


def test_flat_chunks_are_dealt_round_robin_to_the_cores():
    plan = plan_rows([Span(0, 300), Span(512, 700)], cores=4, chunk=64)
    assert plan.rows == rows_per_core(488, 4, 64) == 128
    assert plan.flat_rows == 512
    covered = np.concatenate([plan.core_rows(c) for c in range(4)])
    assert sorted(covered) == list(range(512))
    assert list(plan.core_rows(1)[:3]) == [64, 65, 66]
    assert list(plan.core_rows(1)[64:66]) == [320, 321]


def test_rows_keep_each_request_in_order_with_absolute_positions():
    plan = plan_rows([Span(0, 300), Span(512, 700)], cores=4, chunk=64)
    assert list(plan.flat_start) == [0, 300]
    assert list(plan.request[[0, 299, 300, 487, 488]]) == [0, 0, 1, 1, -1]
    assert list(plan.position[[0, 299, 300, 487, 488]]) == [0, 299, 512, 699, -1]


def test_a_plan_takes_a_larger_size_but_not_a_smaller_one():
    assert plan_rows([Span(0, 100)], cores=4, chunk=64, rows=256).rows == 256
    with pytest.raises(ValueError, match="cannot hold"):
        plan_rows([Span(0, 1000)], cores=4, chunk=64, rows=128)
