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

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _create_mock_tpu_info(
    generation: int,
    chip_version: str = "v8i",
    num_lanes: int = 128,
    num_sublanes: int = 8,
    mxu_column_size: int = 128,
    vmem_capacity_bytes: int = 64 * 1024 * 1024,
    fp8_ops_per_second: int = 2,
    bf16_ops_per_second: int = 1,
):
    mock = MagicMock()
    mock.generation = generation
    mock.chip_version = chip_version
    mock.num_lanes = num_lanes
    mock.num_sublanes = num_sublanes
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


@pytest.mark.parametrize(
    (
        "num_q_heads,num_kv_heads,kv_layout,page_size,"
        "expected_bq,expected_bq_c,expected_bkv"
    ),
    [
        # Qwen3.5-397B PP8/DP8 & Qwen3.8-2.4T (TP1): 32q/2kv, d=256, fp8 KV
        (32, 2, configs.KVLayout.HEAD_ALONG_SUBLANE, 256, 256, 32, 512),
        (32, 2, configs.KVLayout.SEQ_ALONG_LANE, 128, 256, 32, 512),
        # Qwen3.5-397B PP4/TP2 & Qwen3.8 TP8DCP8: 16q/1kv, head_dim=256, fp8 KV
        (16, 1, configs.KVLayout.HEAD_ALONG_SUBLANE, 256, 512, 32, 1024),
        (16, 1, configs.KVLayout.SEQ_ALONG_LANE, 128, 512, 32, 1024),
        # Qwen3.5-35B TP1/DP8 (16q/2kv) & Qwen3.8-27B TP2/DP4 (12q/2kv): 512x512
        (16, 2, configs.KVLayout.HEAD_ALONG_SUBLANE, 256, 512, 64, 512),
        (12, 2, configs.KVLayout.HEAD_ALONG_SUBLANE, 256, 512, 64, 512),
        # Qwen3.8-27B PP8/TP1 (24q/4kv): decoupled 256x512
        (24, 4, configs.KVLayout.HEAD_ALONG_SUBLANE, 256, 256, 32, 512),
        # Lower head counts remain square power-of-2 (bq_sz == bkv_sz == 1024)
        (8, 1, configs.KVLayout.HEAD_ALONG_SUBLANE, 256, 1024, 64, 1024),
        (4, 1, configs.KVLayout.HEAD_ALONG_SUBLANE, 256, 1024, 128, 1024),
        # Non-power-of-2 page_size in SEQ_ALONG_LANE still selects power-of-2 bq_c_sz
        (16, 2, configs.KVLayout.SEQ_ALONG_LANE, 384, 384, 32, 768),
    ],
)
def test_prefill_block_sizes_single_cell_and_bkv_decoupling(
    num_q_heads,
    num_kv_heads,
    kv_layout,
    page_size,
    expected_bq,
    expected_bq_c,
    expected_bkv,
):
    model_cfg = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=page_size,
        total_q_tokens=4096,
        num_page_indices=8 * (32768 // page_size),
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        kv_layout=kv_layout,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x", mxu_column_size=256),
    ):
        _, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit
        )
        assert prefill_blocks.batch_size == 1
        assert prefill_blocks.bq_sz == expected_bq
        assert prefill_blocks.bq_c_sz == expected_bq_c
        assert prefill_blocks.bkv_sz == expected_bkv
        assert prefill_blocks.bkv_sz % prefill_blocks.bq_sz == 0
        assert prefill_blocks.bq_sz % prefill_blocks.bq_c_sz == 0
        assert (prefill_blocks.bq_c_sz & (prefill_blocks.bq_c_sz - 1)) == 0


@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [(2, 1), (4, 1), (8, 1), (16, 2), (32, 4), (64, 8)],
)
def test_prefill_bq_c_sz_is_power_of_two_divisor(num_q_heads, num_kv_heads):
    model_cfg = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=256,
        total_q_tokens=4096,
        num_page_indices=1024,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        return_lse=True,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x", mxu_column_size=256),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit, decode_query_size=4
        )
        assert decode_blocks.bq_sz == 4
        assert decode_blocks.bq_c_sz == 4
        assert prefill_blocks.bq_sz % prefill_blocks.bq_c_sz == 0
        assert (prefill_blocks.bq_c_sz & (prefill_blocks.bq_c_sz - 1)) == 0
