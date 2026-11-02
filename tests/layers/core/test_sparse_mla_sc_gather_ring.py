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
"""TPU_SPARSE_MLA_SC_GATHER_RING: the sparse MLA SparseCore gather escape hatch."""

import inspect

import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.core import attention_interface

ENV = "TPU_SPARSE_MLA_SC_GATHER_RING"


@pytest.mark.parametrize("value", [None, "1", "true"])
def test_on_keeps_the_kernel_default_ring(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)

    # No kwarg: the kernel serves with its default DMA ring.
    assert attention_interface.sparse_mla_gather_kwargs() == {}


@pytest.mark.parametrize("value", ["0", "false"])
def test_off_stands_the_ring_down(monkeypatch, value):
    monkeypatch.setenv(ENV, value)

    assert attention_interface.sparse_mla_gather_kwargs() == {
        "sparsecore_gather_ring_slots": 0
    }


def test_kernel_entry_points_share_the_ring_default():
    def default(fn):
        return inspect.signature(fn).parameters["sparsecore_gather_ring_slots"].default

    assert default(sparse_mla_kernel.sparse_ragged_paged_attention) == default(
        sparse_mla_kernel.sparse_attention_chunk
    )
