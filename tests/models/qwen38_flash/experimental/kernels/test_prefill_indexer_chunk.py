# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK picks the ragged prefill indexer's chunk."""

import pytest

from vllm_torchtpu.models.qwen38_flash.experimental.kernels import pooled_qsa


@pytest.mark.parametrize(
    "override,tokens,expected",
    [
        (0, 8192, 512),  # default: 512 divides the step
        (0, 4096, 512),
        (0, 6400, None),  # default: the served 6400 / 3200 buckets run unchunked
        (0, 3200, None),
        (640, 6400, 640),  # override divides the step
        (640, 3200, 640),
        (640, 8192, 512),  # does not divide: default
        (500, 6000, 512 if 6000 % 512 == 0 else None),  # not a multiple of 8
    ],
)
def test_prefill_indexer_chunk(monkeypatch, override, tokens, expected):
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", str(override))
    assert pooled_qsa.prefill_indexer_chunk(tokens) == expected
