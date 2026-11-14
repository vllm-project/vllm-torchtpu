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
"""Tests for batched RPA config and block size calculations across TPU generations."""

import dataclasses
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _create_mock_tpu_info(
    generation: int,
    chip_version: str = "v8i",
    num_lanes: int = 128,
    mxu_column_size: int = 128,
    vmem_capacity_bytes: int = 64 * 1024 * 1024,
    fp8_ops_per_second: int = 2,
    bf16_ops_per_second: int = 1,
):
    mock = MagicMock()
    mock.generation = generation
    mock.chip_version = chip_version
    mock.num_lanes = num_lanes
    mock.mxu_column_size = mxu_column_size
    mock.vmem_capacity_bytes = vmem_capacity_bytes
    mock.fp8_ops_per_second = fp8_ops_per_second
    mock.bf16_ops_per_second = bf16_ops_per_second
    return mock


@pytest.mark.parametrize(
    "generation,chip_version,is_8bit,expected_bq_c_min",
    [
        (8, "v8i", True, 1),
        (8, "v8i", False, 1),
        (8, "v8t", True, 1),
        (8, "v8t", False, 1),
        (7, "v7x", True, 1),
        (7, "v7x", False, 1),
        (5, "v5p", False, 1),
    ],
)
def test_calculate_block_sizes_generation(
    generation, chip_version, is_8bit, expected_bq_c_min
):
    dtype = jnp.float8_e4m3fn if is_8bit else jnp.bfloat16
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=16,
        total_q_tokens=8,
        num_page_indices=128,
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_out=jnp.bfloat16,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(generation, chip_version=chip_version),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit
        )
        assert decode_blocks.bq_sz >= decode_blocks.bq_c_sz >= expected_bq_c_min
        assert prefill_blocks.bq_sz >= prefill_blocks.bq_c_sz >= expected_bq_c_min


def test_swa_decode_bkv_capped_at_half_window_prefill_uncapped():
    """Decode SWA caps bkv_sz at half the sliding window; prefill stays unchanged."""
    swa_cfg = configs.ModelConfigs(
        num_q_heads=16,
        num_kv_heads=1,
        head_dim=256,
        mask_value=-1e9,
        sliding_window=1024,
    )
    full_cfg = configs.ModelConfigs(
        num_q_heads=16,
        num_kv_heads=1,
        head_dim=256,
        mask_value=-1e9,
        sliding_window=None,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=128,
        total_q_tokens=2048,
        num_page_indices=64 * 8,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x", mxu_column_size=256),
    ):
        swa_dec, swa_pre = wrapper.calculate_block_sizes(swa_cfg, serve_cfg, vmem_limit)
        full_dec, full_pre = wrapper.calculate_block_sizes(
            full_cfg, serve_cfg, vmem_limit
        )
        # Window sizes without A/B data keep the VMEM-driven decode tile.
        other_w_dec, _ = wrapper.calculate_block_sizes(
            dataclasses.replace(swa_cfg, sliding_window=128), serve_cfg, vmem_limit
        )
    assert full_dec.bkv_sz > 512
    assert swa_dec.bkv_sz == 512
    assert swa_pre == full_pre
    assert other_w_dec == full_dec

    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(6, chip_version="v6e", mxu_column_size=256),
    ):
        v6e_swa_dec, _ = wrapper.calculate_block_sizes(swa_cfg, serve_cfg, vmem_limit)
        v6e_full_dec, _ = wrapper.calculate_block_sizes(full_cfg, serve_cfg, vmem_limit)
    assert v6e_swa_dec == v6e_full_dec


def test_head_along_sublane_large_page_size_aligns_bkv_default():
    """When page_size > mxu_column_size, bkv_default aligns up to page_size."""
    model_cfg = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=256,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=256,
        total_q_tokens=512,
        num_page_indices=32 * 8,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
    )
    vmem_limit = 16 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(6, chip_version="v6e", mxu_column_size=128),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit
        )
    assert decode_blocks.bkv_sz % 256 == 0 and decode_blocks.bkv_sz >= 256
    assert prefill_blocks.bkv_sz % 256 == 0 and prefill_blocks.bkv_sz >= 256
