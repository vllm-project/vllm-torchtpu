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
"""Size dispatch of the TP all-reduce kernel at hidden 6144 over 8 ranks: the
cap values that admit each decode bucket, and the libtpu flags the arm log
reports."""

import pytest

from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel

HIDDEN, N = 6144, 8


@pytest.mark.parametrize("rows", [8, 16, 24, 32, 48, 96, 128, 256])
def test_decode_buckets_need_no_padding(rows):
    padded, m, w = kernel.view_shape(rows, HIDDEN, N)
    assert padded == rows
    assert m * w == rows * HIDDEN


@pytest.mark.parametrize(
    "cap,largest", [(524288, 40), (589824, 48), (1572864, 128)]
)
def test_cap_admits_up_to(cap, largest):
    assert kernel.view_bytes(largest, HIDDEN, N) <= cap
    assert kernel.view_bytes(largest + 8, HIDDEN, N) > cap


def test_collective_flags_last_value_wins():
    par = pytest.importorskip("vllm_torchtpu.distributed.pallas_all_reduce")
    args = (
        " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
        " --xla_tpu_dvfs_p_state=7"
        " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
        " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=16777216"
    )
    got = par.collective_flags(args)
    assert got["xla_tpu_enable_sparse_core_collective_offload_reduce_scatter"] == "true"
    assert got["xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"] == "16777216"
    assert got["xla_tpu_enable_sparse_core_hierarchical_all_reduce"] == "default"
    assert set(par.collective_flags("")) == set(got)
