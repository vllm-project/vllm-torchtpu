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
"""Tests for PCP (Prefill Context Parallelism) batched RPA attention adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.experimental.batched_rpa import configs
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    PCPMetadata,
)


def test_make_cu_cache():
    # 2 virtual sequences per request, e.g. [0, 128, 256] -> merged request [0, 256, 256]
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    cu_cache = cp_attention._make_cu_cache(cu, num_seqs=2)
    assert cu_cache.tolist() == [0, 256, 256]


def test_per_request_helpers():
    num_seqs = 4
    # 1D seq_lens
    seq_lens = torch.tensor([100, 100, 200, 200], dtype=torch.int32)
    req_lens = cp_attention._per_request_1d(seq_lens, num_seqs)
    assert req_lens.shape == (num_seqs,)
    assert req_lens.tolist() == [100, 200, 0, 0]

    # 2D block_tables flattened
    bt = torch.tensor(
        [
            [1, 2],
            [1, 2],
            [3, 4],
            [3, 4],
        ],
        dtype=torch.int32,
    ).reshape(-1)
    req_bt = cp_attention._per_request_pages(bt, num_seqs)
    req_bt_2d = req_bt.reshape(num_seqs, 2)
    assert req_bt_2d[0].tolist() == [1, 2]
    assert req_bt_2d[1].tolist() == [3, 4]
    assert req_bt_2d[2].tolist() == [0, 0]
    assert req_bt_2d[3].tolist() == [0, 0]


def test_lse_weighted_combine():
    T, H, D = 4, 2, 64
    out_a = torch.ones((T, H, D), dtype=torch.float32)
    lse_a = torch.zeros((T, H), dtype=torch.float32)  # exp(0) = 1

    out_b = torch.full((T, H, D), 3.0, dtype=torch.float32)
    lse_b = torch.zeros((T, H), dtype=torch.float32)  # exp(0) = 1

    # Equal weights -> average = (1 + 3) / 2 = 2.0
    combined_out, combined_lse = cp_attention.lse_weighted_combine(
        out_a, lse_a, out_b, lse_b
    )
    assert torch.allclose(combined_out, torch.full((T, H, D), 2.0))
    # log(exp(0) + exp(0)) = log(2)
    assert torch.allclose(combined_lse, torch.full((T, H), float(np.log(2.0))))


def test_build_pcp_kernels_mock():
    with patch(
        "vllm_torchtpu.layers.adapter.cp_attention.get_or_create_pcp_mesh",
        return_value=None,
    ), patch(
        "torch_tpu._internal.pallas.jax_op"
    ) as mock_jax_op:
        mock_op = MagicMock()
        mock_jax_op.return_value = mock_op

        cache_k, new_k = cp_attention.build_pcp_kernels(
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            q_scale=None,
            k_scale=None,
            v_scale=None,
            cp_group_size=2,
            cp_rank=0,
            use_causal_mask=True,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        )
        assert callable(cache_k)
        assert callable(new_k)


def test_pcp_forward_dispatch():
    backend = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=4,
        head_size=128,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )

    query = torch.ones(8, 4, 128)
    key = torch.ones(8, 2, 128)
    value = torch.ones(8, 2, 128)
    kv_cache = torch.zeros(4, 16, 2, 1, 128)

    md = AttentionMetadata(
        input_positions=torch.arange(8, dtype=torch.int32),
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        pcp=PCPMetadata(
            query_start_loc=torch.tensor([[0, 4, 8], [0, 4, 8]], dtype=torch.int32),
            kv_cache_lens=torch.tensor([0, 0], dtype=torch.int32),
            q_pos_offsets=torch.tensor([[0, 4], [2, 6]], dtype=torch.int32),
            cache_pages=0,
            new_kv_page_indices=torch.zeros(8, dtype=torch.int32),
            update_kv_cache=torch.tensor([[False, True], [False, True]]),
        ),
    )

    layer = MagicMock()
    layer._k_scale_float = None
    layer._v_scale_float = None

    expected_out = torch.ones(8, 4, 128)
    with patch.object(backend, "_run_pcp_forward", return_value=expected_out) as mock_pcp:
        out = backend.forward(layer, query, key, value, kv_cache, md)
        mock_pcp.assert_called_once()
        assert torch.allclose(out, expected_out)
