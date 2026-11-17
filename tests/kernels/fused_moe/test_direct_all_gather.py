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

DECODE_ROW_BYTES = 4096
SHARED_MEMORY_BYTES = 32 * 1024 * 1024


@pytest.fixture
def tpu7x(monkeypatch):
    monkeypatch.setattr(gather.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        gather.pltpu, "get_tpu_info", lambda: SimpleNamespace(generation=7)
    )


def test_serves_the_decode_exchange_on_8_shards(tpu7x):
    assert gather.supports_direct_all_gather(8, DECODE_ROW_BYTES)


def test_needs_a_peer(tpu7x):
    assert not gather.supports_direct_all_gather(1, DECODE_ROW_BYTES)


def test_every_row_fits_the_shared_memory(tpu7x):
    assert gather.supports_direct_all_gather(8, SHARED_MEMORY_BYTES // 8)
    assert not gather.supports_direct_all_gather(8, SHARED_MEMORY_BYTES // 8 + 4)


def test_needs_a_tpu7x_sparse_core(tpu7x, monkeypatch):
    monkeypatch.setattr(
        gather.pltpu, "get_tpu_info", lambda: SimpleNamespace(generation=6)
    )
    assert not gather.supports_direct_all_gather(8, DECODE_ROW_BYTES)


def test_off_tpu_the_exchange_stays_on_xla(monkeypatch):
    monkeypatch.setattr(gather.jax, "default_backend", lambda: "cpu")
    assert not gather.supports_direct_all_gather(8, DECODE_ROW_BYTES)
