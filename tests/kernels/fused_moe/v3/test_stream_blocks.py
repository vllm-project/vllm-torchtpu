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
"""The column blocks a streamed expert may take, on the host."""

import pytest

from vllm_torchtpu.kernels.fused_moe.v3.layer import stream_blocks_for

pytestmark = pytest.mark.cpu_test


def test_blocks_are_divisors_in_whole_activation_blocks():
    assert stream_blocks_for(3072, 512) == [1536, 1024, 512]
    assert stream_blocks_for(4096, 512) == [2048, 1024, 512]


def test_block_scaled_blocks_cover_whole_scale_tiles():
    # 3072 in weight blocks of 512 has 6 scale rows: no block below the
    # whole width covers a whole tile of 8, so nothing can stream.
    assert stream_blocks_for(3072, 512, 512) == []
    # 4096 in blocks of 128 has 32: blocks of 8 rows are 1024 columns.
    assert stream_blocks_for(4096, 512, 128) == [2048, 1024]
