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
"""DCP sparse prefill tile choice: the query-block option and its limits."""

import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels import qsa_rpa


@pytest.fixture
def query_block_option():
    saved = qsa_rpa._OPTIONS["VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK"]

    def set_option(value):
        qsa_rpa.configure(VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK=value)

    yield set_option
    qsa_rpa.configure(VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK=saved)


@pytest.mark.parametrize(
    ("option", "table", "tokens", "heads", "expected"),
    [
        # No option: the table's tiles, capped.
        (0, (512, 128), 8192, 6, (512, 128)),
        (0, (1024, 256), 8192, 12, (256, None)),
        # The option replaces the table where it fits the row cap.
        (1024, (512, 128), 8192, 6, (1024, 256)),
        # It does not fit 12 heads: the table's tuned tiles stay.
        (1024, (512, 128), 8192, 12, (512, 128)),
        # Short steps halve it, like the table's shrink.
        (1024, (512, 128), 512, 6, (512, 256)),
        (1024, (512, 128), 100, 6, (256, None)),
    ],
)
def test_dcp_prefill_tiles(query_block_option, option, table, tokens, heads, expected):
    query_block_option(option)
    assert (
        qsa_rpa._dcp_prefill_tiles(*table, tokens=tokens, query_heads=heads) == expected
    )


@pytest.mark.parametrize("value", [1000, -128])
def test_unusable_query_block_option_raises(query_block_option, value):
    with pytest.raises(ValueError, match="VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK"):
        query_block_option(value)
