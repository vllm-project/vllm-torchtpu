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


@pytest.mark.parametrize(
    "rows,direct_rows,want",
    [(32, 16, True), (24, 16, False), (24, 8, True), (8, 8, True), (12, 4, False), (32, 0, False)],
)
def test_direct_layout_gate(rows, direct_rows, want):
    assert kernel.direct_ok(rows, HIDDEN, N, direct_rows) is want
    # 6144 / 8 = 768 lanes per rank chunk: whole lane tiles.
    assert not kernel.direct_ok(rows, 6144 + 128, N, 8)


@pytest.mark.parametrize(
    "rows,qar_min,want",
    [
        (8, 1024, "pallas"),  # decode bucket under the cap: Pallas, direct rows
        (512, 1024, "psum"),  # between the cap and the qar minimum
        (1024, 1024, "qar"),
        (8192, 1024, "qar"),
        (8192, 0, "psum"),  # qar off
    ],
)
def test_size_routing_with_qar_and_direct_rows(monkeypatch, rows, qar_min, want):
    """The op picks one path per static size: Pallas (keeping direct_rows),
    psum, or qar at >= TPU_TP_QAR_MIN_TOKENS rows."""
    par = pytest.importorskip("vllm_torchtpu.distributed.pallas_all_reduce")
    calls = []
    monkeypatch.setattr(
        par.kernel, "all_reduce", lambda x, r, n, d=0: calls.append(("pallas", d)) or x
    )
    monkeypatch.setattr(par.qar, "all_reduce_v2", lambda x, r, t: calls.append(("qar",)) or x)
    monkeypatch.setattr(par.lax, "psum", lambda x, a: calls.append(("psum",)) or x)
    tab = object() if qar_min else None
    body = par._body(N, 589824, "x", direct_rows=8, qar_min_rows=qar_min, qar_tab=tab)
    import numpy as np

    body(np.zeros((rows, HIDDEN), np.float32), None)
    assert [c[0] for c in calls] == [want]
    if want == "pallas":
        assert calls[0][1] == 8
