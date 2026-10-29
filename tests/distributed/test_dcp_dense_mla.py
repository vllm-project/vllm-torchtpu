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

import pytest

from vllm_torchtpu.distributed import dcp_dense_mla


@pytest.fixture
def active(monkeypatch):
    monkeypatch.setattr(dcp_dense_mla, "_ACTIVE", None)
    dcp_dense_mla.enable(dcp_size=32, page_size=256, max_model_len=82432)


def test_inactive_changes_nothing(monkeypatch):
    monkeypatch.setattr(dcp_dense_mla, "_ACTIVE", None)
    dcp_dense_mla.check_prefill_pool([10**6, 10**6], [5, 5])


def test_prefill_pool_holds_one_whole_context(active):
    assert dcp_dense_mla.active().pool_pages == 11
    dcp_dense_mla.check_prefill_pool([82431 - 8192, 0, 70000], [8192, 4096, 1])
    with pytest.raises(RuntimeError):
        dcp_dense_mla.check_prefill_pool([50000, 50000], [100, 100])


def test_layers_must_agree(active):
    dcp_dense_mla.enable(dcp_size=32, page_size=256, max_model_len=82432)
    with pytest.raises(ValueError):
        dcp_dense_mla.enable(dcp_size=8, page_size=256, max_model_len=82432)
