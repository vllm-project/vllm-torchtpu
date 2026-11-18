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
from jax.experimental.pallas import tpu as pltpu

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


def _decode_cfgs():
    model_cfg = configs.ModelConfigs(
        num_q_heads=4,
        num_kv_heads=1,
        head_dim=256,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=256,
        total_q_tokens=64,
        num_page_indices=8 * 64,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
    )
    return model_cfg, serve_cfg


def test_default_decode_batch_size_is_eight():
    model_cfg, serve_cfg = _decode_cfgs()
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x"),
    ):
        default = wrapper.calculate_block_sizes(model_cfg, serve_cfg, vmem_limit)
        explicit = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit, decode_batch_size=8
        )
    assert default == explicit
    assert default[0].batch_size == 8


def test_decode_batch_size_changes_only_decode_blocks():
    model_cfg, serve_cfg = _decode_cfgs()
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x"),
    ):
        default_decode, default_prefill = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit
        )
        small_decode, small_prefill = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit, decode_batch_size=2
        )
    assert small_prefill == default_prefill
    assert small_decode.batch_size == 2
    # Fewer lanes leave VMEM for a KV tile at least as large.
    assert small_decode.bkv_sz >= default_decode.bkv_sz


class _BlockSizesCalled(Exception):
    pass


@pytest.mark.parametrize(("decode_batch_size", "expected"), [(None, 8), (2, 2)])
def test_ragged_paged_attention_passes_the_decode_batch_size(
    monkeypatch, decode_batch_size, expected
):
    seen = []

    def record(*args, **kwargs):
        seen.append(kwargs["decode_batch_size"])
        raise _BlockSizesCalled

    monkeypatch.setattr(wrapper, "calculate_block_sizes", record)
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    num_seqs, pages_per_seq, page_size, head_dim = 8, 4, 16, 128
    cache_shape = wrapper.get_kv_cache_shape(
        num_seqs * pages_per_seq, page_size, 1, head_dim, jnp.bfloat16
    )
    q = jnp.zeros((8, 4, head_dim), jnp.bfloat16)
    kv = jnp.zeros((8, 1, head_dim), jnp.bfloat16)
    with (
        patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info),
        pytest.raises(_BlockSizesCalled),
    ):
        wrapper.ragged_paged_attention(
            q,
            kv,
            kv,
            jnp.zeros(cache_shape, jnp.bfloat16),
            jnp.ones((num_seqs,), jnp.int32),
            jnp.zeros((num_seqs * pages_per_seq,), jnp.int32),
            jnp.arange(num_seqs + 1, dtype=jnp.int32),
            jnp.array([num_seqs, num_seqs, num_seqs], jnp.int32),
            decode_batch_size=decode_batch_size,
        )
    assert seen == [expected]
