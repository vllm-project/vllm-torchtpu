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
"""Mocked static resource boundaries; these are not cross-chip hardware tests."""

from dataclasses import replace

import pytest

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
    HcHardware,
    full_mix_memory,
    supports_full_mix,
    supports_residual,
)


def hardware(capacity=64 << 20):
    return HcHardware("mock-layout", capacity, 128, 16, 8, 256)


@pytest.mark.parametrize("capacity,expected", [(16, False), (64, True), (128, True)])
def test_full_residency_and_capacity(capacity, expected):
    info = hardware(capacity << 20)
    # 320 lowrank rows, but up-weight minor320 pads to384 vector columns.
    estimate = full_mix_memory(4, 2560, 320, 256, info)
    assert estimate.resident_bytes == 10240 * 2 + (320 + 384) * 10240 * 2
    assert estimate.buffer_bytes == (4 * 256 * 10240 + 2 * 256 * 2560) * 2
    assert supports_full_mix(4, 2560, 320, 256, info) is expected
    # 512-row rings plus the same working policy exceed64MiB, without an OOM.
    assert not supports_full_mix(4, 2560, 320, 512, hardware())


@pytest.mark.parametrize(
    "branches,hidden,lowrank,tile",
    [
        (0, 128, 128, 128),
        (4, 0, 128, 128),
        (4, 129, 128, 128),
        (4, 128, 0, 128),
        (4, 128, 128, 0),
        (4, 128, 128, 64),
        (2**31, 128, 128, 128),
    ],
)
def test_static_invalid_or_overflow_geometry(branches, hidden, lowrank, tile):
    assert not supports_full_mix(branches, hidden, lowrank, tile, hardware())


def test_unknown_or_unqualified_layout():
    assert not supports_full_mix(4, 128, 128, 128, None)
    for info in [
        replace(hardware(), vmem_bytes=0),
        replace(hardware(), lanes=64),
        replace(hardware(), bf16_rows=0),
        replace(hardware(), mxu_columns=0),
    ]:
        assert not supports_full_mix(4, 128, 128, 128, info)


@pytest.mark.parametrize("tile", [16, 32, 64, 128])
def test_residual_alignment_is_dtype_derived(tile):
    assert supports_residual(3, 128, tile, hardware())
    assert supports_residual(4, 2560, tile, hardware())
    different_layout = replace(hardware(), bf16_rows=32, fp32_rows=16)
    assert supports_residual(3, 128, tile, different_layout) is (tile % 32 == 0)
    assert not supports_residual(3, 128, tile + 1, hardware())
    assert not supports_residual(3, 128, tile, None)
