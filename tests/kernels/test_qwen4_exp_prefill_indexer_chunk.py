# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK picks the ragged prefill indexer's chunk."""

import pytest

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa


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


@pytest.mark.parametrize(
    "small,override,tokens,expected",
    [
        (0, 0, 512, 512),  # off: default
        (0, 0, 256, None),
        (128, 0, 256, 128),  # replicated 256 bucket / SP 2K per rank
        (128, 0, 512, 128),  # replicated 512 bucket / SP 4K per rank
        (128, 0, 1024, 128),  # replicated 1024 bucket / SP 8K per rank
        (128, 0, 3072, 512),  # above the small-step bound: default
        (128, 0, 8192, 512),
        (128, 0, 128, None),  # not smaller than the step: default
        (128, 0, 96, None),
        (96, 0, 384, 96),
        (96, 0, 512, 512),  # does not divide: default
        (100, 0, 400, None),  # not a multiple of 8: default
        (128, 256, 512, 256),  # the explicit override wins
        (128, 640, 512, 128),  # override does not divide: small applies
    ],
)
def test_prefill_indexer_small_chunk(monkeypatch, small, override, tokens, expected):
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_INDEXER_SMALL_CHUNK", str(small))
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", str(override))
    assert pooled_qsa.prefill_indexer_chunk(tokens) == expected
