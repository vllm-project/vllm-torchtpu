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
"""The ragged prefill indexer's chunk: 512 tokens when that divides the step."""

import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels import pooled_qsa


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (8192, 512),
        (4096, 512),
        (6400, None),  # the served 6400 / 3200 buckets run unchunked
        (3200, None),
    ],
)
def test_prefill_indexer_chunk(tokens, expected):
    assert pooled_qsa.prefill_indexer_chunk(tokens) == expected
