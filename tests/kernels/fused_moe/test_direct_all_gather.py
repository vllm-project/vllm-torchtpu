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
from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import direct_all_gather as gather
from vllm_torchtpu.kernels.fused_moe.v2 import layer

DECODE_ROW_BYTES = 4096


@pytest.fixture
def tpu7x(monkeypatch):
    monkeypatch.setattr(gather.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        gather.pltpu, "get_tpu_info", lambda: SimpleNamespace(generation=7)
    )


def _supports(ep=8, local_bytes=DECODE_ROW_BYTES, single_host=True):
    return gather.supports_direct_all_gather(ep, local_bytes, single_host)


def test_serves_the_decode_exchange_on_8_shards(tpu7x):
    assert _supports()


def test_serves_the_largest_measured_routing_row(tpu7x):
    width = layer._exchange_layout(2048, 6, 384, True)[2]
    assert _supports(local_bytes=width * 4)


def test_needs_a_peer(tpu7x):
    assert not _supports(ep=1)


def test_stays_within_the_measured_shard_count(tpu7x):
    assert _supports(ep=gather.MAX_SHARDS)
    assert not _supports(ep=16)


def test_needs_every_shard_on_one_host(tpu7x):
    assert not _supports(single_host=False)


def test_stays_within_the_measured_row_size(tpu7x):
    assert _supports(local_bytes=gather.MAX_LOCAL_BYTES)
    assert not _supports(
        local_bytes=gather.MAX_LOCAL_BYTES + gather.LOCAL_BYTES_ALIGNMENT
    )


def test_needs_whole_128_lane_int32_rows(tpu7x):
    assert not _supports(local_bytes=DECODE_ROW_BYTES + 4)


def test_every_admitted_exchange_fits_the_shared_memory():
    assert gather.MAX_SHARDS * gather.MAX_LOCAL_BYTES <= gather.SHARED_MEMORY_BYTES


@pytest.mark.parametrize("generation", [6, 8])
def test_needs_a_tpu7x_sparse_core(tpu7x, monkeypatch, generation):
    monkeypatch.setattr(
        gather.pltpu, "get_tpu_info", lambda: SimpleNamespace(generation=generation)
    )
    assert not _supports()


def test_off_tpu_the_exchange_stays_on_xla(monkeypatch):
    monkeypatch.setattr(gather.jax, "default_backend", lambda: "cpu")
    assert not _supports()
