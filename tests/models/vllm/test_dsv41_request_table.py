# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import request_table


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_per_request_matches_indexing(dtype):
    values = torch.tensor([0, 6, 12, 12, 18, 2**40 if dtype == torch.int64 else -7])
    values = values.to(dtype)
    requests = torch.tensor([[0, 1, 1, 5], [4, 3, 2, 0]])
    out = request_table.per_request(values, requests)
    assert out.dtype == dtype
    assert torch.equal(out, values[requests])
