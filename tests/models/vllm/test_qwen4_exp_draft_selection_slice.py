# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DRAFT_SELECTION_SLICE: draft step 0 gathers its shared
selection rows from the sink's first T rows, bit-identical to gathering from
the whole sink (rows below T, padding -1 read as row 0) (CPU)."""

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.qwen4_exp import qsa


@pytest.mark.parametrize("tokens,num_rows", [(8, 32), (32, 32), (128, 32), (1, 4)])
def test_slice_gather_bitwise_equal(monkeypatch, tokens, num_rows):
    g = torch.Generator().manual_seed(tokens)
    sink = torch.randint(-1, 1 << 20, (8192, 2051), generator=g, dtype=torch.int32)
    real = min(tokens, num_rows)
    rows = torch.full((num_rows,), -1, dtype=torch.int64)
    rows[:real] = torch.randint(0, tokens, (real,), generator=g)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DRAFT_SELECTION_SLICE", False)
    want = qsa.shared_step0_rows(sink, rows, tokens)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DRAFT_SELECTION_SLICE", True)
    got = qsa.shared_step0_rows(sink, rows, tokens)
    assert got.shape == (num_rows, 2051) and got.dtype == torch.int32
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(got, sink[rows.clamp_min(0)], rtol=0, atol=0)


def test_rows_past_t_are_never_read(monkeypatch):
    # The sink's rows past T hold stale selections of an earlier, larger step.
    sink = torch.arange(64 * 3, dtype=torch.int32).reshape(64, 3)
    rows = torch.tensor([3, 0, -1, 7])
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DRAFT_SELECTION_SLICE", True)
    sink_poisoned = sink.clone()
    sink_poisoned[8:] = -7
    torch.testing.assert_close(
        qsa.shared_step0_rows(sink_poisoned, rows, 8), sink[rows.clamp_min(0)]
    )


def test_off_by_default():
    assert envs.environment_variables["VLLM_TPU_QSA_DRAFT_SELECTION_SLICE"]() is False
