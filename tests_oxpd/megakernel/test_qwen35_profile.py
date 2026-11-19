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
"""Unit tests for the Qwen3.5 baseline tile rule (`models.qwen35_profile`)."""

from __future__ import annotations

import pytest

from vllm_torchtpu.megakernel.exec_spec import Tiles
from vllm_torchtpu.megakernel.models.qwen35_profile import (
    DECODE_TILES,
    baseline_tiles,
)


@pytest.mark.parametrize("R", [1, 8])
@pytest.mark.parametrize("moe_tp", [False, True])
def test_qwen35_baseline_tiles_ignore_moe_tp(R: int, moe_tp: bool):
    for B in range(1, 65):
        t = baseline_tiles(B, moe_tp=moe_tp, R=R)
        assert t == baseline_tiles(B, R=R)  # no moe_tp dependence
        gqa, moe = t.block("gqa"), t.block("moe")
        assert gqa.kv_sub == 0 and not moe.route_vec and not t.dense_acc_pair
        assert not moe.pipe_ep and moe.pipe  # PIPE stays on (TP single tile)
    assert Tiles(hier=1, mb8=2).with_blocks({"gqa": {"pack": 1}}) == DECODE_TILES


@pytest.mark.parametrize("R", [2, 4])
def test_qwen35_baseline_non_v7x8_ranks_use_flat_ring(R: int):
    for B in (2, 4, 8, 16, 32, 64):
        assert baseline_tiles(B, R=R).hier == 0
