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

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from vllm_torchtpu.kernels.mla.v2 import tuned_params
from vllm_torchtpu.kernels.mla.v2.tuned_params import TuningKey, get_tuned_params


def _k3_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=4096,
        actual_num_q_heads=12,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size_per_kv_packing=128,
        kv_packing=2,
        max_num_seqs=8,
        pages_per_seq=128,
    )


def _k3_tp32_fp8_8k_mixed_key() -> TuningKey:
    return TuningKey(
        case="mixed",
        max_num_tokens=8192,
        actual_num_q_heads=3,
        actual_lkv_dim=512,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        page_size_per_kv_packing=4,
        kv_packing=4,
        max_num_seqs=8,
        pages_per_seq=520,
    )


def _k3_tp32_fp8_8k_page256_mixed_key() -> TuningKey:
    return replace(
        _k3_tp32_fp8_8k_mixed_key(),
        page_size_per_kv_packing=64,
        pages_per_seq=40,
    )


def _k3_tp32_fp8_unified_pool_key(
    max_num_tokens: int, pages_per_seq: int
) -> TuningKey:
    # The unified pool page holds one KDA state slot: 77 SSM + 8 FP32 conv
    # packed rows, i.e. 85 rows x 4 = 340 FP8 tokens at TP32.
    return replace(
        _k3_tp32_fp8_8k_mixed_key(),
        max_num_tokens=max_num_tokens,
        page_size_per_kv_packing=85,
        pages_per_seq=pages_per_seq,
    )


@pytest.mark.parametrize(
    "max_num_tokens,pages_per_seq,kv_pages_per_block",
    [
        (8192, 32, 8),
        (1024, 32, 4),
        (2048, 32, 8),
        (4096, 32, 8),
        (1024, 64, 4),
        (2048, 64, 8),
        (4096, 64, 8),
        (8192, 64, 8),
        (8192, 3104, 8),
        (16384, 64, 8),
    ],
)
def test_k3_tp32_fp8_unified_pool_tuned_params(
    monkeypatch: pytest.MonkeyPatch,
    max_num_tokens: int,
    pages_per_seq: int,
    kv_pages_per_block: int,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(tuned_params, "logger", logger)

    tuned = get_tuned_params(
        _k3_tp32_fp8_unified_pool_key(max_num_tokens, pages_per_seq)
    )

    logger.warning.assert_not_called()
    assert tuned.num_kv_pages_per_block == kv_pages_per_block
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 2


def test_k3_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_mixed_key())

    assert tuned.num_kv_pages_per_block == 4
    assert tuned.num_queries_per_block == 256
    assert tuned.q_split == 1


def test_k3_tp32_fp8_8k_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_mixed_key())

    assert tuned.num_kv_pages_per_block == 256
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 4
    assert tuned.vmem_limit_bytes == 56 * 1024 * 1024


def test_k3_tp32_fp8_8k_page256_mixed_tuned_params() -> None:
    tuned = get_tuned_params(_k3_tp32_fp8_8k_page256_mixed_key())

    assert tuned.num_kv_pages_per_block == 16
    assert tuned.num_queries_per_block == 512
    assert tuned.q_split == 8
    assert tuned.vmem_limit_bytes == 52 * 1024 * 1024


def test_k3_mixed_tuned_params_require_matching_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    mismatched_key = replace(_k3_mixed_key(), page_size_per_kv_packing=64)

    tuned = get_tuned_params(mismatched_key)

    assert tuned.num_kv_pages_per_block == 1
    assert tuned.num_queries_per_block == 16
    assert tuned.q_split == 1
