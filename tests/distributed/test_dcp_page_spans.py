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
    lower_dcp_page_spans_to_dcp,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPURaidenConnectorWorker,
)

pytestmark = pytest.mark.cpu_test


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


def test_pcp_ranks_fan_stacked_pages_out_to_dcp_ranks():
    # K3 PP x PCP8 into DCP32: PCP rank r's block j holds the PCP chunk
    # 8j + r, four decode pages long; its quarter i is decode page
    # 32j + 4r + i, which DCP rank 4r + i keeps at its block j.
    page_bytes, blocks = 4, 3
    destination: dict[tuple[int, int], int] = {}
    for rank in range(8):
        source = [
            [4 * (8 * j + rank) + i for i in range(4) for _ in range(page_bytes)]
            for j in range(blocks)
        ]
        registration = lower_dcp_page_spans(
            tag="fa.l3",
            transfer_rank=rank,
            live_bytes_per_block=4 * page_bytes,
            block_ids=[90 - j for j in range(blocks)],
            sub_blocks=4,
        )
        assert registration.dst_space_version == 1
        assert registration.declared_bytes == blocks * 4 * page_bytes
        for span in registration.spans:
            chunk = source[span.src_block_ordinal][
                span.src_offset_bytes : span.src_offset_bytes + span.size_bytes
            ]
            assert span.size_bytes == page_bytes and len(set(chunk)) == 1
            key = (span.dst_unit_ordinal, span.dst_offset_bytes // page_bytes)
            assert key not in destination
            destination[key] = chunk[0]
    assert destination == {(q, j): 32 * j + q for q in range(32) for j in range(blocks)}


def test_pcp_stage_pages_reach_every_attention_dp_group():
    # K3 PP x PCP8 into an attention-DP2 decoder (two DCP16 groups, units
    # 0-15 and 16-31): decode page g = 32j + 4r + i goes to units g % 16 and
    # g % 16 + 16, at local page g // 16; a 40000-token request fills 5 of
    # each unit's local pages (8192 tokens each), so later pages stay home.
    page_bytes, blocks, tokens = 4, 3, 40000
    destination: dict[tuple[int, int], int] = {}
    for rank in range(8):
        source = [
            [32 * j + 4 * rank + i for i in range(4) for _ in range(page_bytes)]
            for j in range(blocks)
        ]
        registration = lower_dcp_page_spans_to_dcp(
            tag="fa.l3",
            transfer_rank=rank,
            live_bytes_per_block=4 * page_bytes,
            block_ids=[90 - j for j in range(blocks)],
            src_dcp_size=32,
            dst_dcp_size=16,
            dst_units=32,
            num_tokens=tokens,
            page_tokens=512,
            sub_blocks=4,
        )
        assert registration.declared_bytes == len(registration.spans) * page_bytes
        for span in registration.spans:
            chunk = source[span.src_block_ordinal][
                span.src_offset_bytes : span.src_offset_bytes + span.size_bytes
            ]
            assert span.size_bytes == page_bytes and len(set(chunk)) == 1
            key = (span.dst_unit_ordinal, span.dst_offset_bytes // page_bytes)
            assert key not in destination
            destination[key] = chunk[0]
    local_pages = -(-tokens // (512 * 16))
    assert destination == {
        (group * 16 + q, p): 16 * p + q
        for group in range(2)
        for q in range(16)
        for p in range(local_pages)
    }


def test_one_page_blocks_lower_as_before():
    registration = lower_dcp_page_spans_to_dcp(
        tag="fa",
        transfer_rank=21,
        live_bytes_per_block=8,
        block_ids=[3, 4],
        src_dcp_size=32,
        dst_dcp_size=16,
        dst_units=32,
        num_tokens=4 * 512 * 32,
        page_tokens=512,
    )
    assert [
        (
            s.src_block_ordinal,
            s.src_offset_bytes,
            s.dst_offset_bytes,
            s.dst_unit_ordinal,
        )
        for s in registration.spans
    ] == [(0, 0, 8, 5), (0, 0, 8, 21), (1, 0, 24, 5), (1, 0, 24, 21)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tag": ""},
        {"transfer_rank": -1},
        {"live_bytes_per_block": 0},
        {"sub_blocks": 0},
        {"sub_blocks": 3},
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
                prefill_context_parallel_size=1, decode_context_parallel_size=dcp
            ),
        )
    )
    assert TPURaidenConnectorWorker._max_request_blocks(worker) == blocks
