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
"""Size dispatch of the TP all-reduce kernel: which decode buckets its cap
admits, and that its VMEM request stays inside a tpu7x TensorCore."""

import pytest

from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel

HIDDEN, N = 6144, 8
TPU7X_VMEM = 64 << 20


@pytest.mark.parametrize("rows", [8, 16, 24, 40, 48, 64, 96, 128, 256])
def test_decode_buckets_fit_the_cap(rows):
    padded, m, w = kernel.view_shape(rows, HIDDEN, N)
    assert padded == rows  # multiples of 8 rows need no padding at hidden 6144
    assert m * w == rows * HIDDEN
    assert kernel.view_bytes(rows, HIDDEN, N) <= kernel.MAX_VIEW_BYTES


def test_cap_vmem_request_fits():
    assert 4 * kernel.MAX_VIEW_BYTES + (1 << 20) < TPU7X_VMEM


def test_cap_value_for_128_rows():
    # TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES=1572864 admits the 128-row bucket
    # of hidden 6144 and nothing larger.
    assert kernel.view_bytes(128, HIDDEN, N) == 1572864
    assert kernel.view_bytes(136, HIDDEN, N) > 1572864


def test_selftest_rows():
    par = pytest.importorskip("vllm_torchtpu.distributed.pallas_all_reduce")
    assert par._selftest_rows(N, HIDDEN, 524288) == par._SELFTEST_ROWS
    assert par._selftest_rows(N, HIDDEN, 0) == par._SELFTEST_ROWS
    rows = par._selftest_rows(N, HIDDEN, kernel.MAX_VIEW_BYTES)
    assert par._selftest_rows(N, HIDDEN, 1572864) == par._SELFTEST_ROWS + (128,)
    assert rows[:-1] == par._SELFTEST_ROWS
    big = rows[-1]
    assert big % 8 == 0 and big >= 256
    assert kernel.view_bytes(big, HIDDEN, N) <= kernel.MAX_VIEW_BYTES
    assert kernel.view_bytes(big + 8, HIDDEN, N) > kernel.MAX_VIEW_BYTES
