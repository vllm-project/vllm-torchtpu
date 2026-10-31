# SPDX-License-Identifier: Apache-2.0
"""Every tuning-table entry must pass its kernel's config checks on CPU."""

import pytest

from vllm_torchtpu.kernels.attention_preprocessing import (
    compress_rope,
    qa_rope,
    qb_rope,
)
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    COMPRESS_ROPE_TUNING,
    QA_ROPE_TUNING,
    QB_ROPE_TUNING,
)


@pytest.mark.parametrize("key", list(QA_ROPE_TUNING))
def test_qa_entries_are_valid(key):
    _, tokens, hidden, width = key
    config = QA_ROPE_TUNING[key]
    qa_rope.check_config(config, tokens, width)
    assert qa_rope.vmem_bytes(config, hidden, width, 64) <= config.vmem_limit_bytes


@pytest.mark.parametrize("key", list(QB_ROPE_TUNING))
def test_qb_entries_are_valid(key):
    _, tokens, rank, width = key
    config = QB_ROPE_TUNING[key]
    heads = width // 512
    segment = qb_rope._segment_tokens(config, tokens, rank, heads, 512)
    qb_rope.check_config(config, segment, heads)


@pytest.mark.parametrize("key", list(COMPRESS_ROPE_TUNING))
def test_compress_entries_are_valid(key):
    _, tokens, hidden, ratio = key
    config = COMPRESS_ROPE_TUNING[key]
    compress_rope.check_config(config, tokens, ratio * 512)
    assert compress_rope.vmem_bytes(config, hidden, ratio) <= config.vmem_limit_bytes
