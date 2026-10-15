# Copyright 2025 Google LLC
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
"""Stage-3 transfer groups: request-state pages and connector-defined groups."""

from types import SimpleNamespace

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPUConnector,
    _Stage3TransferGroup,
)


def test_request_state_group_moves_one_page_per_request():
    state = _Stage3TransferGroup(
        cache_group_index=1, page_tokens=512, request_state=True
    )
    paged = _Stage3TransferGroup(cache_group_index=0, page_tokens=512)
    assert [state.page_count(n) for n in (0, 1, 513, 4096)] == [0, 1, 1, 1]
    assert [paged.page_count(n) for n in (0, 1, 513, 4096)] == [0, 1, 2, 8]


def test_connector_subclass_defines_its_transfer_groups():
    groups = (
        _Stage3TransferGroup(cache_group_index=0, page_tokens=512),
        _Stage3TransferGroup(cache_group_index=3, page_tokens=512, request_state=True),
    )

    class ModelConnector(TPUConnector):
        @staticmethod
        def _build_stage3_transfer_groups(config):
            return groups

    assert ModelConnector._build_stage3_transfer_groups(SimpleNamespace()) is groups
    # The default keeps the shared rule.
    assert TPUConnector._build_stage3_transfer_groups is not (
        ModelConnector._build_stage3_transfer_groups
    )
