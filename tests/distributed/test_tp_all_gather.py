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
"""`pallas_all_reduce.tp_all_gather`: the Pallas row gather of 32-bit words
(16-bit tensors viewed as words) rearranged into vLLM's all-gather along any
dim, with the per-rank blocks emulated on the host."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import pallas_all_reduce

N = 8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int32])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_matches_concatenation_in_rank_order(monkeypatch, dtype, dim):
    import vllm.distributed

    blocks = [
        (torch.arange(8 * 24) * 10 + r).reshape(8, 24).to(dtype) for r in range(N)
    ]
    rank = 3

    def fake_gather(ranks, words):
        assert words.element_size() == 4
        return torch.cat([b.contiguous().view(words.dtype) for b in blocks])

    monkeypatch.setattr(pallas_all_reduce, "all_gather_small", fake_gather)
    monkeypatch.setattr(
        vllm.distributed, "get_tp_group", lambda: SimpleNamespace(ranks=range(N))
    )
    got = pallas_all_reduce.tp_all_gather(blocks[rank], dim)
    assert torch.equal(got, torch.cat(blocks, dim=dim))


def test_falls_back_when_not_armed(monkeypatch):
    import vllm.distributed

    x = torch.ones(8, 6, dtype=torch.bfloat16)
    monkeypatch.setattr(pallas_all_reduce, "all_gather_small", lambda r, w: None)
    monkeypatch.setattr(
        vllm.distributed, "get_tp_group", lambda: SimpleNamespace(ranks=range(N))
    )
    monkeypatch.setattr(
        vllm.distributed,
        "tensor_model_parallel_all_gather",
        lambda t, dim: torch.cat([t] * N, dim=dim),
    )
    assert pallas_all_reduce.tp_all_gather(x, -1).shape == (8, 6 * N)
