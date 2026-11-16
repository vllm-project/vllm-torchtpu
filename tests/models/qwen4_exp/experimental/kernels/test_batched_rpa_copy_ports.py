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
"""CPU tests for the fixes ported from main into the model-owned batched RPA copy.

Covers main's #3000 (bkv_sz is a multiple of page_size in both KV layouts)
and #2454 (mask_invalid_v defaults), plus the served Qwen3.8 QSA decode
geometry (HEAD_ALONG_SUBLANE, 4-token virtual pages, bkv_sz 256 or 512),
which must keep passing the alignment check.
"""

import dataclasses
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.batched_rpa import (
    configs,
    wrapper,
)


def _tpu_info(generation: int, chip_version: str, mxu_column_size: int):
    mock = MagicMock()
    mock.generation = generation
    mock.chip_version = chip_version
    mock.num_lanes = 128
    mock.num_sublanes = 8
    mock.mxu_column_size = mxu_column_size
    mock.vmem_capacity_bytes = 64 * 1024 * 1024
    mock.fp8_ops_per_second = 2
    mock.bf16_ops_per_second = 1
    return mock


def _serve(page_size: int, kv_layout: configs.KVLayout) -> configs.ServingConfigs:
    return configs.ServingConfigs(
        num_seqs=4,
        page_size=page_size,
        total_q_tokens=4,
        num_page_indices=32,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
        kv_layout=kv_layout,
    )


def _validate(bkv_sz: int, page_size: int, kv_layout: configs.KVLayout) -> None:
    model_cfg = configs.ModelConfigs(
        num_q_heads=8, num_kv_heads=2, head_dim=128, mask_value=-1e9
    )
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv_sz, batch_size=4),
        model=model_cfg,
        serve=_serve(page_size, kv_layout),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    q = jnp.zeros((4, 8, 128), dtype=jnp.bfloat16)
    kv = jnp.zeros((4, 2, 128), dtype=jnp.bfloat16)
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_tpu_info(7, "v7x", 256),
    ):
        kv_cache = jnp.zeros(
            wrapper.get_kv_cache_shape(
                8, page_size, 2, 128, jnp.bfloat16, kv_layout=kv_layout
            ),
            dtype=jnp.bfloat16,
        )
        cfgs.validate_inputs(
            q,
            kv,
            kv,
            kv_cache,
            jnp.ones((4,), dtype=jnp.int32),
            jnp.arange(32, dtype=jnp.int32),
            jnp.arange(5, dtype=jnp.int32),
            jnp.array([4, 4, 4], dtype=jnp.int32),
        )


@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
def test_rejects_bkv_sz_not_aligned_to_page_size(kv_layout):
    with pytest.raises(NotImplementedError, match="aligned to page_size"):
        _validate(bkv_sz=320, page_size=128, kv_layout=kv_layout)


@pytest.mark.parametrize("bkv_sz", [256, 512])
def test_served_qsa_decode_geometry_passes_alignment(bkv_sz):
    # qsa_rpa presents the cache as 4-token virtual pages (compress ratio 4).
    _validate(bkv_sz=bkv_sz, page_size=4, kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE)


def test_default_bkv_aligns_to_large_page_size():
    model_cfg = configs.ModelConfigs(
        num_q_heads=8, num_kv_heads=2, head_dim=256, mask_value=-1e9
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
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_tpu_info(6, "v6e", 128),
    ):
        decode, prefill = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, 16 * 1024 * 1024
        )
    assert decode.bkv_sz % 256 == 0 and decode.bkv_sz >= 256
    assert prefill.bkv_sz % 256 == 0 and prefill.bkv_sz >= 256


def test_mask_invalid_v_defaults_by_layout():
    assert _serve(128, configs.KVLayout.SEQ_ALONG_LANE).mask_invalid_v is True
    # The served layout keeps the unmasked V load.
    assert _serve(128, configs.KVLayout.HEAD_ALONG_SUBLANE).mask_invalid_v is False
    explicit = dataclasses.replace(
        _serve(128, configs.KVLayout.HEAD_ALONG_SUBLANE), mask_invalid_v=True
    )
    assert explicit.mask_invalid_v is True
