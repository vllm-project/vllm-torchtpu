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
"""Which K/V pages a pipelined prefill's helpers send the owner."""

import pytest

from vllm_torchtpu.distributed.dp_pipeline import kv_plan
from vllm_torchtpu.distributed.dp_pipeline.kv_plan import KvGroup

B = 256


def _gemma4_like_groups():
    """Global layers 5, 11, ..., 59 in one group; the sliding ones in groups
    of ten, in model order (vLLM's hybrid grouping)."""
    full = [i for i in range(60) if i % 6 == 5]
    sliding = [i for i in range(60) if i % 6 != 5]
    groups = [KvGroup(0, tuple((i, f"l{i}") for i in full), None)]
    for g in range(5):
        layers = sliding[10 * g : 10 * g + 10]
        groups.append(KvGroup(1 + g, tuple((i, f"l{i}") for i in layers), 1024))
    return tuple(groups)


def test_chunk_transfer_takes_the_full_attention_layers_of_the_range():
    groups = _gemma4_like_groups()
    pages = kv_plan.chunk_pages(16384, B)
    assert pages == 65
    t = kv_plan.chunk_transfer(groups, (0, 30), 7, 16384, 16384, B, pages)
    assert t.token == 7 and t.pages == 65
    (entry,) = t.entries
    assert entry.group_id == 0
    assert entry.layer_names == ("l5", "l11", "l17", "l23", "l29")
    assert entry.blocks == tuple(range(64, 128))
    assert t.rows == 5 * 65
    # The prompt's last chunk ends inside a block.
    t = kv_plan.chunk_transfer(groups, (30, 44), 7, 81920, 5228, B, pages)
    assert t.entries[0].layer_names == ("l35", "l41")
    assert t.entries[0].blocks == tuple(range(320, 341))
    # A range without a global layer sends nothing per chunk.
    assert kv_plan.chunk_transfer(groups, (0, 5), 7, 0, 100, B, pages) is None
    with pytest.raises(ValueError, match="spans more"):
        kv_plan.chunk_transfer(groups, (0, 30), 7, 100, 16384, B, 64)


@pytest.mark.parametrize(
    ("prompt_len", "blocks"),
    [
        (87148, range(336, 341)),
        # The window starts on a block boundary.
        (33791, range(128, 132)),
        (1000, range(0, 4)),
    ],
)
def test_window_transfer_takes_the_last_window(prompt_len, blocks):
    groups = _gemma4_like_groups()
    assert kv_plan.window_pages(groups, B) == 5
    t = kv_plan.window_transfer(groups, (0, 30), 9, prompt_len, B)
    assert [e.group_id for e in t.entries] == [1, 2, 3]
    assert [len(e.layer_names) for e in t.entries] == [10, 10, 5]
    assert all(e.blocks == tuple(blocks) for e in t.entries)
    assert t.pages == 5 and t.rows == 25 * 5


def test_ring_rows_cover_every_helper_range():
    groups = _gemma4_like_groups()
    ranges = [(0, 30), (0, 14), (14, 29), (29, 44)]
    # [0, 30) holds five global layers of 65 rows and 25 sliding ones of 5.
    assert kv_plan.max_ring_rows(groups, ranges, 65, B) == 325
