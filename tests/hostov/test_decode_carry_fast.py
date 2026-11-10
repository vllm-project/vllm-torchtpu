# SPDX-License-Identifier: Apache-2.0
"""The decode carry's cached mrope position view returns the same values as the
torch slice, follows in-place updates and refreshes on a replaced buffer."""
import types

import numpy as np
import torch


def test_carry_fast_positions_match(monkeypatch):
    from vllm_torchtpu.runner import decode_carry as dc

    buf = torch.arange(3 * 16, dtype=torch.int64).reshape(3, 16)
    runner = types.SimpleNamespace(uses_mrope=True, mrope_positions=types.SimpleNamespace(cpu=buf))
    monkeypatch.setattr(dc, "_T3_FAST", False)
    ref = dc._host_positions(runner, 5)
    monkeypatch.setattr(dc, "_T3_FAST", True)
    fast = dc._host_positions(runner, 5)
    assert fast.dtype == ref.dtype == np.int64 and np.array_equal(fast, ref)
    buf[1, 2] = 999  # the cached view sees in-place updates
    assert dc._host_positions(runner, 5)[1, 2] == 999
    new = torch.zeros(3, 16, dtype=torch.int64)
    runner.mrope_positions.cpu = new  # a replaced buffer refreshes the view
    assert dc._host_positions(runner, 5).sum() == 0
