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
"""CPU unit tests for RPA v3 per-sequence 128-lane-aligned adaptive decode tiers."""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase,
    compute_decode_bkv_tier_bounds,
    get_default_block_sizes,
    get_rpa_block_size_stats,
    reset_rpa_block_size_stats,
    resolve_decode_bkv_csz_tiers,
)


@pytest.fixture(autouse=True)
def _clean_rpa_env(monkeypatch):
    monkeypatch.delenv("TPU_RPA_DECODE_BKV_PAGES", raising=False)
    monkeypatch.delenv("TPU_RPA_DECODE_BKV_COMPUTE_PAGES", raising=False)
    monkeypatch.delenv("TPU_RPA_DECODE_ADAPTIVE_BKV_COMPUTE", raising=False)
    reset_rpa_block_size_stats()
    yield
    reset_rpa_block_size_stats()


def test_rpa_decode_env_defaults_and_validation(monkeypatch):
    assert envs.TPU_RPA_DECODE_BKV_PAGES == 0
    assert envs.TPU_RPA_DECODE_BKV_COMPUTE_PAGES == 0
    assert envs.TPU_RPA_DECODE_ADAPTIVE_BKV_COMPUTE is False

    monkeypatch.setenv("TPU_RPA_DECODE_BKV_PAGES", "-1")
    with pytest.raises(ValueError, match="TPU_RPA_DECODE_BKV_PAGES"):
        _ = envs.TPU_RPA_DECODE_BKV_PAGES

    monkeypatch.setenv("TPU_RPA_DECODE_BKV_PAGES", "8")
    monkeypatch.setenv("TPU_RPA_DECODE_ADAPTIVE_BKV_COMPUTE", "invalid")
    with pytest.raises(ValueError, match="TPU_RPA_DECODE_ADAPTIVE_BKV_COMPUTE"):
        _ = envs.TPU_RPA_DECODE_ADAPTIVE_BKV_COMPUTE


def test_page_544_tiers_and_128_lane_bounds():
    """For hybrid unified-pool page_size=544 (4*128 + 32) and bkv_sz=4352 (8 pages),
    active kv_len in 513..544 must promote to tier 1088 to avoid the 32-lane tail."""
    tiers = resolve_decode_bkv_csz_tiers(
        bkv_sz=4352, bkv_csz=544, page_size=544, adaptive=True
    )
    assert tiers == (544, 1088, 2176, 4352)
    bounds = compute_decode_bkv_tier_bounds(tiers)
    assert bounds == (512, 1024, 2176, 4352)


@pytest.mark.parametrize("page_size", [16, 32, 64])
def test_small_page_size_clamps_minimum_tier_to_128_lanes(monkeypatch, page_size):
    """When page_size < 128, adaptive tiers must never emit <128-lane sub-tiles."""
    tiers = resolve_decode_bkv_csz_tiers(
        bkv_sz=4096, bkv_csz=page_size, page_size=page_size, adaptive=True
    )
    assert tiers == (128, 256, 512, 1024, 2048, 4096)
    assert compute_decode_bkv_tier_bounds(tiers) == (128, 256, 512, 1024, 2048, 4096)

    monkeypatch.setenv("TPU_RPA_DECODE_BKV_PAGES", str(4096 // page_size))
    monkeypatch.setenv("TPU_RPA_DECODE_ADAPTIVE_BKV_COMPUTE", "1")
    with patch(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel.get_tpu_version",
        return_value=7,
    ):
        bs = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=page_size,
            max_num_tokens=256,
            max_num_seqs=256,
            pages_per_seq=4096 // page_size,
            case=RpaCase.DECODE,
        )
    assert bs["bkv_sz"] == 4096
    assert bs["bkv_csz"] == 128
    stats = get_rpa_block_size_stats()
    entry = stats[f"d_tok256_seq256_p{page_size}"]
    assert entry["adaptive_bkv_compute"] is True
    assert entry["bkv_csz_tiers"] == [128, 256, 512, 1024, 2048, 4096]


def test_get_default_block_sizes_stock_vs_adaptive_and_prefill_invariance(monkeypatch):
    with patch(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel.get_tpu_version",
        return_value=7,
    ):
        stock_dec = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=544,
            max_num_tokens=256,
            max_num_seqs=256,
            pages_per_seq=8,
            case=RpaCase.DECODE,
        )
        stock_prefill = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=544,
            max_num_tokens=4096,
            max_num_seqs=256,
            pages_per_seq=8,
            case=RpaCase.PREFILL,
        )
        stock_mixed = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=544,
            max_num_tokens=4096,
            max_num_seqs=256,
            pages_per_seq=8,
            case=RpaCase.MIXED,
        )
        assert stock_dec == {"bq_sz": 1, "bkv_sz": 4352, "bq_csz": 1, "bkv_csz": 4352}

        monkeypatch.setenv("TPU_RPA_DECODE_BKV_PAGES", "8")
        monkeypatch.setenv("TPU_RPA_DECODE_ADAPTIVE_BKV_COMPUTE", "1")
        adapt_dec = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=544,
            max_num_tokens=256,
            max_num_seqs=256,
            pages_per_seq=8,
            case=RpaCase.DECODE,
        )
        adapt_prefill = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=544,
            max_num_tokens=4096,
            max_num_seqs=256,
            pages_per_seq=8,
            case=RpaCase.PREFILL,
        )
        adapt_mixed = get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=1,
            head_dim=256,
            page_size=544,
            max_num_tokens=4096,
            max_num_seqs=256,
            pages_per_seq=8,
            case=RpaCase.MIXED,
        )
        assert adapt_dec == {"bq_sz": 1, "bkv_sz": 4352, "bq_csz": 1, "bkv_csz": 544}
        assert adapt_prefill == stock_prefill
        assert adapt_mixed == stock_mixed
        entry = get_rpa_block_size_stats()["d_tok256_seq256_p544"]
        assert entry["adaptive_bkv_compute"] is True
        assert entry["bkv_csz_tiers"] == [544, 1088, 2176, 4352]

        # Non-divisible compute_pages=3 for fetch_pages=8 must fail closed
        monkeypatch.setenv("TPU_RPA_DECODE_BKV_COMPUTE_PAGES", "3")
        with pytest.raises(ValueError, match="positive multiple"):
            get_default_block_sizes(
                jnp.bfloat16,
                jnp.bfloat16,
                actual_num_q_heads=4,
                actual_num_kv_heads=1,
                head_dim=256,
                page_size=544,
                max_num_tokens=256,
                max_num_seqs=256,
                pages_per_seq=8,
                case=RpaCase.DECODE,
            )
