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
"""Unit tests for batched_rpa FP32 online-softmax state and VMEM scratch sizing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _mock_tpu_info(vmem_capacity_bytes: int = 64 * 1024 * 1024) -> MagicMock:
    mock = MagicMock()
    mock.generation = 7
    mock.chip_version = "v7x"
    mock.num_lanes = 128
    mock.num_sublanes = 8
    mock.mxu_column_size = 128
    mock.vmem_capacity_bytes = vmem_capacity_bytes
    mock.fp8_ops_per_second = 2
    mock.bf16_ops_per_second = 1
    return mock


def test_batched_rpa_dtype_acc_is_always_fp32() -> None:
    """ServingConfigs.dtype_acc stays float32 even when dtype_out is bfloat16."""
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=16,
        total_q_tokens=8,
        num_page_indices=128,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    assert serve_cfg.dtype_acc == jnp.float32


def test_batched_rpa_block_sizes_account_for_fp32_scratch_vmem() -> None:
    """calculate_block_sizes succeeds and respects tight VMEM with fp32 scratch."""
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=16,
        page_size=16,
        total_q_tokens=512,
        num_page_indices=2048,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    tpu_info = _mock_tpu_info(vmem_capacity_bytes=32 * 1024 * 1024)
    with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, 32 * 1024 * 1024
        )
    assert decode_blocks.bq_sz >= decode_blocks.bq_c_sz >= 1
    assert prefill_blocks.bq_sz >= prefill_blocks.bq_c_sz >= 1
    assert decode_blocks.bkv_sz >= 16
    assert prefill_blocks.bkv_sz >= 16
