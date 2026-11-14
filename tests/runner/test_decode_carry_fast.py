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
"""The decode carry's cached mrope position view returns the same values as the
torch slice, follows in-place updates and refreshes on a replaced buffer."""

import types

import numpy as np
import torch


def test_carry_fast_positions_match():
    from vllm_torchtpu.runner import decode_carry as dc

    buf = torch.arange(3 * 16, dtype=torch.int64).reshape(3, 16)
    runner = types.SimpleNamespace(
        uses_mrope=True, mrope_positions=types.SimpleNamespace(cpu=buf)
    )
    ref = np.asarray(buf[:, :5].numpy(), dtype=np.int64)
    fast = dc._host_positions(runner, 5)
    assert fast.dtype == ref.dtype == np.int64 and np.array_equal(fast, ref)
    buf[1, 2] = 999  # the cached view sees in-place updates
    assert dc._host_positions(runner, 5)[1, 2] == 999
    new = torch.zeros(3, 16, dtype=torch.int64)
    runner.mrope_positions.cpu = new  # a replaced buffer refreshes the view
    assert dc._host_positions(runner, 5).sum() == 0
