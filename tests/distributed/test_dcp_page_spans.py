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

"""Stage-3 transfer of a decode-context-parallel cache (each rank its own share)."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_dcp_page_spans,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorWorker,
)


def test_each_rank_sends_its_blocks_whole_to_its_own_ordinal():
    registration = lower_dcp_page_spans(
        tag="fa", transfer_rank=5, live_bytes_per_block=327680, block_ids=[7, 9, 11]
    )
    assert registration.block_ids == (7, 9, 11)
    assert registration.dst_space_version == 1
    assert registration.declared_bytes == 3 * 327680
    assert [
        (s.src_block_ordinal, s.dst_offset_bytes, s.size_bytes, s.dst_unit_ordinal)
        for s in registration.spans
    ] == [(j, j * 327680, 327680, 5) for j in range(3)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tag": ""},
        {"transfer_rank": -1},
        {"live_bytes_per_block": 0},
    ],
)
def test_rejects_malformed_arguments(kwargs):
    arguments = dict(tag="fa", transfer_rank=0, live_bytes_per_block=1, block_ids=[1])
    with pytest.raises(ValueError):
        lower_dcp_page_spans(**{**arguments, **kwargs})


@pytest.mark.parametrize("dcp, blocks", [(1, 512), (32, 16)])
def test_request_blocks_count_logical_blocks_under_dcp(dcp, blocks):
    # 131072 tokens at 256-token pages: 512 blocks, or 16 logical blocks of
    # 8192 tokens under DCP 32 (each rank stores 256 tokens of each).
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(block_size=256),
            model_config=SimpleNamespace(max_model_len=131072),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp,
                prefill_context_parallel_size=1,
            ),
        )
    )
    assert TPURaidenConnectorWorker._max_request_blocks(worker) == blocks
