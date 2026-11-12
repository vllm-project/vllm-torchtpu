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
"""sampling_metadata_fixed_shape_copies: upstream InputBatch's sampling
metadata uploads write the whole device tensor, one shape for every request
count, and every row a reader sees holds the value upstream writes."""

import pytest
import torch
from vllm.v1.utils import copy_slice
from vllm.v1.worker import gpu_input_batch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    _fixed_shape_copy_slice,
    fixed_shape_sampling_copies_enabled,
    install_fixed_shape_sampling_copies,
)

MAX_REQS = 48


def test_key_parses_and_defaults_off():
    assert not fixed_shape_sampling_copies_enabled(None)
    assert not fixed_shape_sampling_copies_enabled({})
    assert fixed_shape_sampling_copies_enabled(
        {"sampling_metadata_fixed_shape_copies": True}
    )
    with pytest.raises(ValueError):
        fixed_shape_sampling_copies_enabled({"sampling_metadata_fixed_shape_copies": 1})


class _DeviceLike:
    """Stands in for a TPU tensor: records the shape of every in-place copy
    and keeps the values in a CPU tensor."""

    def __init__(self, n):
        self.data = torch.full((n,), -7.0)
        self.copies = []

    @property
    def device(self):
        return torch.device("meta")  # any non-CPU device

    @property
    def shape(self):
        return self.data.shape

    def copy_(self, src, non_blocking=False):
        self.copies.append(tuple(src.shape))
        self.data.copy_(src)
        return self

    def __getitem__(self, idx):
        return self.data[idx]


def test_one_copy_shape_for_every_request_count():
    cpu = torch.zeros(MAX_REQS)
    dev = _DeviceLike(MAX_REQS)
    ref = torch.full((MAX_REQS,), -7.0)
    for num_reqs in [1, 3, 17, 48, 5, 33, 2]:
        cpu.copy_(torch.rand(MAX_REQS))
        got = _fixed_shape_copy_slice(cpu, dev, num_reqs)
        want = copy_slice(cpu, ref, num_reqs)
        # Every row a reader of the returned view sees is upstream's value.
        assert torch.equal(got, want)
        assert got.shape == (num_reqs,)
    assert set(dev.copies) == {(MAX_REQS,)}


def test_cpu_target_keeps_upstream_copy():
    src, dst = torch.arange(6.0), torch.zeros(6)
    out = _fixed_shape_copy_slice(src, dst, 4)
    assert torch.equal(out, src[:4]) and bool((dst[4:] == 0).all())


def test_install_routes_input_batch_uploads(monkeypatch):
    monkeypatch.setattr(gpu_input_batch, "copy_slice", gpu_input_batch.copy_slice)
    install_fixed_shape_sampling_copies()
    assert gpu_input_batch.copy_slice is tpu_runner._fixed_shape_copy_slice
    install_fixed_shape_sampling_copies()
    assert gpu_input_batch.copy_slice is tpu_runner._fixed_shape_copy_slice
