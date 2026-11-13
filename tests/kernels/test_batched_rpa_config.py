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

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils, wrapper


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


# The cached-KV DMA schedule fetches block k from pages k * bkv_p with no
# intra-page offset, so bkv_sz must be a page multiple in every layout.
@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
@pytest.mark.parametrize(
    "kv_dtype,num_kv_heads",
    [
        (jnp.bfloat16, 1),
        (jnp.bfloat16, 2),
        (jnp.float8_e4m3fn, 1),
        (jnp.float8_e4m3fn, 2),
    ],
)
@pytest.mark.parametrize("page_size", [16, 128, 512])
def test_calculate_block_sizes_bkv_is_page_multiple(
    kv_layout, kv_dtype, num_kv_heads, page_size
):
    model_cfg = configs.ModelConfigs(
        num_q_heads=6 * num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=page_size,
        total_q_tokens=512,
        num_page_indices=8 * 17,
        dtype_q=jnp.bfloat16,
        dtype_kv=kv_dtype,
        dtype_out=jnp.bfloat16,
        kv_layout=kv_layout,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x", mxu_column_size=256),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, 64 * 1024 * 1024
        )
    for blocks in (decode_blocks, prefill_blocks):
        assert blocks.bkv_sz >= page_size
        assert blocks.bkv_sz % page_size == 0, (
            f"{kv_layout=} {kv_dtype=} {num_kv_heads=} {page_size=}:"
            f" bkv_sz={blocks.bkv_sz} is not a page multiple"
        )


@pytest.mark.parametrize(
    "kv_layout,expected_error",
    [
        (configs.KVLayout.HEAD_ALONG_SUBLANE, ValueError),
        (configs.KVLayout.SEQ_ALONG_LANE, NotImplementedError),
    ],
)
def test_validate_inputs_rejects_unaligned_bkv_sz(kv_layout, expected_error):
    page_size, head_dim, num_kv_heads, num_seqs, num_pages = 512, 128, 2, 8, 4
    dtype = jnp.bfloat16
    packing = utils.get_dtype_packing(dtype)
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=768, batch_size=8),
        model=configs.ModelConfigs(
            num_q_heads=6 * num_kv_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask_value=-1e9,
        ),
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=num_seqs,
            num_page_indices=num_seqs * num_pages,
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_out=dtype,
            kv_layout=kv_layout,
        ),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x", mxu_column_size=256),
    ):
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            kv_cache_shape = (
                num_pages,
                num_kv_heads * 2,
                cfgs.aligned_kv_head_dim // packing,
                packing,
                page_size,
            )
        else:
            kv_cache_shape = (
                num_pages,
                page_size,
                cfgs.aligned_num_kv_heads_x2 // packing,
                packing,
                cfgs.aligned_kv_head_dim,
            )
        with pytest.raises(expected_error, match="multiple of"):
            cfgs.validate_inputs(
                q=jnp.zeros((num_seqs, 6 * num_kv_heads, head_dim), dtype),
                k=jnp.zeros((num_seqs, num_kv_heads, head_dim), dtype),
                v=jnp.zeros((num_seqs, num_kv_heads, head_dim), dtype),
                kv_cache=jnp.zeros(kv_cache_shape, dtype),
                kv_lens=jnp.zeros((num_seqs,), jnp.int32),
                page_indices=jnp.zeros((num_seqs * num_pages,), jnp.int32),
                cu_q_lens=jnp.zeros((num_seqs + 1,), jnp.int32),
                distribution=jnp.zeros((3,), jnp.int32),
            )
