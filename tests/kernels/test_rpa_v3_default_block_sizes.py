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
"""Tests for RPA v3 default block sizes and divisibility validation (Issue #2662)."""

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase,
    get_default_block_sizes,
    validate_block_sizes,
)


def test_validate_block_sizes_valid():
    """Verify valid block sizes in tuple or dict form pass validation."""
    # Tuple format: (bq_sz, bkv_sz, bq_csz, bkv_csz)
    validate_block_sizes((128, 1568, 64, 784), page_size=392, prefix="test")
    validate_block_sizes((256, 1024, 128, 512), page_size=16, prefix="test")

    # Dict format
    validate_block_sizes(
        {"bq_sz": 128, "bkv_sz": 1568, "bq_csz": 64, "bkv_csz": 784},
        page_size=392,
        prefix="test",
    )
    # None passes (optional)
    validate_block_sizes(None, page_size=16)


@pytest.mark.parametrize(
    "block_sizes,page_size,expected_error_match",
    [
        # bkv_sz % bkv_csz != 0 (the exact bug in #2662: 1176 % 784 != 0)
        ((128, 1176, 64, 784), 392, "bkv_sz % bkv_csz == 0"),
        # bq_sz % bq_csz != 0
        ((128, 1568, 50, 784), 392, "bq_sz % bq_csz == 0"),
        # bkv_sz % page_size != 0
        ((128, 1024, 64, 512), 392, "divisible by page_size=392"),
        # bkv_csz % page_size != 0 (while bkv_sz % bkv_csz == 0)
        ((128, 1000, 64, 500), 392, "divisible by page_size=392"),
    ],
)
def test_validate_block_sizes_invalid(block_sizes, page_size, expected_error_match):
    """Verify invalid block sizes violate invariants and raise ValueError."""
    with pytest.raises(ValueError, match=expected_error_match):
        validate_block_sizes(block_sizes, page_size=page_size, prefix="invalid_test")


def test_get_default_block_sizes_regression_issue_2662():
    """Regression test for Issue #2662: Qwen3.5-27B at TP=8 on TPU v6e (page_size=392).

    Previously, get_default_block_sizes independently aligned bkv_sz to 392 (= 1176)
    and bkv_csz to 392 (= 784), resulting in 1176 % 784 = 392 != 0.
    With the fix, bkv_sz is aligned to bkv_csz (= 1568), satisfying 1568 % 784 == 0.
    """
    with patch(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel.get_tpu_version",
        return_value=6,
    ):
        res = get_default_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=6,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=392,
            max_num_tokens=131072,
            max_num_seqs=64,
            pages_per_seq=335,
            case=RpaCase.MIXED,
        )
        assert res["bkv_csz"] == 784
        assert res["bkv_sz"] == 1568
        assert res["bkv_sz"] % res["bkv_csz"] == 0
        assert res["bkv_sz"] % 392 == 0
        assert res["bkv_csz"] % 392 == 0
        assert res["bq_sz"] % res["bq_csz"] == 0


@pytest.mark.parametrize("tpu_version", [5, 6, 7])
@pytest.mark.parametrize(
    "page_size",
    [16, 32, 64, 128, 256, 392, 400, 448, 512, 784],
)
@pytest.mark.parametrize("case", [RpaCase.MIXED, RpaCase.PREFILL, RpaCase.DECODE])
def test_get_default_block_sizes_divisibility_invariants(tpu_version, page_size, case):
    """Ensure divisibility invariants hold across TPU versions, page sizes, cases."""
    with patch(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel.get_tpu_version",
        return_value=tpu_version,
    ):
        res = get_default_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=8,
            actual_num_kv_heads=2,
            head_dim=256,
            page_size=page_size,
            max_num_tokens=16384,
            max_num_seqs=32,
            pages_per_seq=500,
            case=case,
        )
        assert res["bkv_csz"] > 0
        assert res["bkv_sz"] % res["bkv_csz"] == 0
        assert res["bkv_sz"] % page_size == 0
        assert res["bkv_csz"] % page_size == 0
        assert res["bq_csz"] > 0
        assert res["bq_sz"] % res["bq_csz"] == 0
        assert res["bq_csz"] <= res["bq_sz"]


@pytest.mark.parametrize("tpu_version", [5, 6])
def test_get_default_block_sizes_power_of_two_parity(tpu_version):
    """Verify that power-of-two page sizes retain identical behavior to baseline."""
    with patch(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel.get_tpu_version",
        return_value=tpu_version,
    ):
        for ps in [16, 32, 64, 128, 256, 512]:
            res = get_default_block_sizes(
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.bfloat16,
                actual_num_q_heads=8,
                actual_num_kv_heads=2,
                head_dim=256,
                page_size=ps,
                max_num_tokens=16384,
                max_num_seqs=32,
                pages_per_seq=500,
                case=RpaCase.MIXED,
            )
            assert res["bkv_sz"] == 1024
            assert res["bkv_csz"] == 512


@pytest.mark.parametrize("page_size", [16, 32, 64, 128, 256, 392, 400, 448, 512, 784])
@pytest.mark.parametrize("kv_block_cap", [256, 512, 1024, 1176, 1200, 1500, 2048])
def test_speculative_decoding_draft_cap_divisibility(page_size, kv_block_cap):
    """Verify speculative decoding draft capping guarantees bkv % bkv_csz == 0."""
    with patch(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel.get_tpu_version",
        return_value=6,
    ):
        bs = get_default_block_sizes(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=8,
            actual_num_kv_heads=2,
            head_dim=256,
            page_size=page_size,
            max_num_tokens=16384,
            max_num_seqs=32,
            pages_per_seq=500,
            case=RpaCase.MIXED,
        )
        cap_tokens = max(page_size, (kv_block_cap // page_size) * page_size)
        bkv_csz = bs["bkv_csz"]
        if cap_tokens < bkv_csz:
            bkv_csz = cap_tokens
            bkv = cap_tokens
        else:
            bkv = min(bs["bkv_sz"], (cap_tokens // bkv_csz) * bkv_csz)

        # Validate capped block sizes pass static validation without raising ValueError
        validate_block_sizes(
            (bs["bq_sz"], bkv, bs["bq_csz"], bkv_csz), page_size=page_size
        )
        assert bkv % bkv_csz == 0
        assert bkv % page_size == 0
        assert bkv_csz % page_size == 0
