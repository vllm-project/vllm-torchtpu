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
"""`isolated_all_gather` and the patch that installs it for data-parallel ViTs."""

import torch

import vllm_torchtpu.utils as tpu_utils
from vllm_torchtpu.distributed.collectives import isolated_all_gather


def test_isolated_all_gather_synchronizes_around_the_collective(monkeypatch):
    import vllm.distributed

    calls = []
    copy = torch.ones(2, 3)
    gathered = torch.ones(4, 3)

    def fresh_buffer(t):
        calls.append("fresh")
        return copy

    def synchronize_tensors(t):
        calls.append(("sync", id(t)))

    def all_gather(t, dim):
        calls.append(("gather", id(t), dim))
        return gathered

    monkeypatch.setattr(tpu_utils, "fresh_buffer", fresh_buffer)
    monkeypatch.setattr(tpu_utils, "synchronize_tensors", synchronize_tensors)
    monkeypatch.setattr(
        vllm.distributed, "tensor_model_parallel_all_gather", all_gather
    )

    assert isolated_all_gather(torch.zeros(2, 3), dim=0) is gathered
    assert calls == [
        "fresh",
        ("sync", id(copy)),
        ("gather", id(copy), 0),
        ("sync", id(gathered)),
    ]


def test_dp_vision_patch_installs_the_isolated_all_gather(monkeypatch):
    import vllm.model_executor.models.vision as vision

    from vllm_torchtpu import _patch_vllm_dp_vision_all_gather

    monkeypatch.setattr(
        vision,
        "tensor_model_parallel_all_gather",
        vision.tensor_model_parallel_all_gather,
    )
    _patch_vllm_dp_vision_all_gather()
    assert vision.tensor_model_parallel_all_gather is isolated_all_gather
