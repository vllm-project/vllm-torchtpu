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


@pytest.mark.parametrize(
    "page,pages_per_seq,max_num_seqs,decode_kv_pages,q_split,mixed_kv_pages",
    [(4, 640, 32, 192, 8, 256), (85, 32, 8, 16, 2, 8), (85, 32, 32, 16, 2, 8)],
)
def test_k3_tp32_fp8_8k_layout_tuned_params(
    page: int,
    pages_per_seq: int,
    max_num_seqs: int,
    decode_kv_pages: int,
    q_split: int,
    mixed_kv_pages: int,
) -> None:
    mixed_key = replace(
        _k3_tp32_fp8_8k_mixed_key(),
        page_size_per_kv_packing=page,
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
    )

    mixed = get_tuned_params(mixed_key)
    assert mixed.q_split == q_split
    assert mixed.num_kv_pages_per_block == mixed_kv_pages

    for max_num_tokens in (1, 32, 8192):
        decode_key = replace(
            mixed_key, case="batched_decode", max_num_tokens=max_num_tokens
        )
        assert decode_key not in tuned_params.tuned_params_mapping
        decode = tuned_params.live_keyed_decode_params[decode_key]
        assert decode.decode_batch_size == 2
        assert decode.num_kv_pages_per_block == decode_kv_pages


def test_k3_mixed_tuned_params_require_matching_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tuned_params, "logger", MagicMock())
    mismatched_key = replace(_k3_mixed_key(), page_size_per_kv_packing=64)

    tuned = get_tuned_params(mismatched_key)

    assert tuned.num_kv_pages_per_block == 1
    assert tuned.num_queries_per_block == 16
    assert tuned.q_split == 1
