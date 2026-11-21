# SPDX-License-Identifier: Apache-2.0
"""The ragged prefill indexer's chunk: 512 tokens when that divides the step."""

import pytest

from vllm_torchtpu.models.qwen38_flash.experimental.kernels import pooled_qsa


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
