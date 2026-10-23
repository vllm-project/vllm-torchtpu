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
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.kernels.experimental.batched_rpa import configs
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    PCPMetadata,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with (
        set_current_vllm_config(config),
        set_vllm_model_wrapper_context(mesh=None, vllm_config=config),
    ):
        yield


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
    cp_attention._PCP_KERNEL_REGISTRY.clear()
    with patch("torch_tpu._internal.pallas.jax_op") as mock_jax_op:
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
        assert mock_jax_op.call_count == 2
        for call in mock_jax_op.call_args_list:
            assert call.kwargs["mesh"] is None
            assert call.kwargs["input_partition_specs"] is None


def test_pcp_forward_dispatch(cpu_vllm_config_context):
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
    with patch.object(
        backend, "_run_pcp_forward", return_value=expected_out
    ) as mock_pcp:
        out = backend.forward(layer, query, key, value, kv_cache, md)
        mock_pcp.assert_called_once()
        assert torch.allclose(out, expected_out)


def test_forward_with_pcp_unaligned_prefix_hit_cache_pass(monkeypatch):
    """Regression test for unaligned prefix-hit prefill across PCP ranks.

    At P=2, page_size=128 with 13,056 cached tokens and 3,328 new tokens:
      rank 0 owns 1,536 valid tokens -> cu_cache = [0, 1536, 1536]
      rank 1 owns 1,792 valid tokens -> cu_cache = [0, 1792, 1792]
    Rank 0 must pass rank 1's cu_cache ([0, 1792, 1792]) when evaluating rank 1's
    Q chunk against rank 0's KV shard, and exchange partials via all-to-all so
    rank 1's rows [1536, 1792) include rank 0's cached KV contribution.
    """
    import threading

    from vllm_torchtpu.layers.core.pcp_sequence_layout import (
        pcp_zigzag_seq_arrays,
        pcp_zigzag_token_layout,
    )

    pcp_size = 2
    page_size = 128
    num_tokens_padded = 2048
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64

    cached_tokens = 13056
    new_tokens = 3328
    chunk, off, _ = pcp_zigzag_token_layout(
        [new_tokens], pcp_size, block_size=page_size
    )
    cu_q_lens, q_pos_offsets, update_kv = pcp_zigzag_seq_arrays(
        chunk, off, pcp_size, n_off=2, counts=[new_tokens]
    )
    assert cu_q_lens[0].tolist() == [0, 896, 1536]
    assert cu_q_lens[1].tolist() == [0, 896, 1792]

    barrier = threading.Barrier(pcp_size)
    ag_slots: dict[int, torch.Tensor] = {}
    a2a_slots: dict[int, torch.Tensor] = {}
    recorded_cu_src: dict[int, list[list[int]]] = {0: [], 1: []}
    results: dict[int, torch.Tensor] = {}
    errors: list[BaseException] = []

    tls = threading.local()

    def fake_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        r = tls.rank
        ag_slots[r] = x.clone()
        barrier.wait()
        out = torch.cat([ag_slots[i] for i in range(pcp_size)], dim=dim)
        barrier.wait()
        return out

    def fake_all_to_all(x: torch.Tensor) -> torch.Tensor:
        r = tls.rank
        a2a_slots[r] = x.clone()
        barrier.wait()
        recv_chunks = [
            a2a_slots[src].chunk(pcp_size, dim=0)[r] for src in range(pcp_size)
        ]
        out = torch.cat(recv_chunks, dim=0)
        barrier.wait()
        return out

    monkeypatch.setattr(cp_attention, "get_pcp_group", lambda: object())
    monkeypatch.setattr(cp_attention, "all_gather_equal_tokens", fake_all_gather)
    monkeypatch.setattr(cp_attention, "all_to_all_single", fake_all_to_all)
    monkeypatch.setattr(cp_attention, "synchronize_tensors", lambda *a, **k: None)

    def make_kernels_for_rank(rank: int):
        shard_val = 1.0 if rank == 0 else 3.0

        def fake_cache_kernel(
            kv_cache,
            q_chunk,
            key,
            value,
            seq_lens_req,
            block_tables_req,
            cu_src,
            dist_cache,
            kv_new_lens_req,
        ):
            recorded_cu_src[rank].append(cu_src.tolist())
            valid_end = int(cu_src[-1].item())
            out = torch.zeros_like(q_chunk, dtype=torch.float32)
            lse = torch.full(
                (q_chunk.shape[0], q_chunk.shape[1]),
                float("-inf"),
                dtype=torch.float32,
            )
            out[:valid_end] = shard_val
            lse[:valid_end] = 0.0
            return out, lse

        def fake_new_kernel(kv_cache, query, *args):
            valid_end = int(cu_q_lens[rank, -1])
            out = torch.zeros_like(query, dtype=torch.float32)
            lse = torch.full(
                (query.shape[0], query.shape[1]),
                float("-inf"),
                dtype=torch.float32,
            )
            out[:valid_end] = 5.0
            lse[:valid_end] = 0.0
            return out, lse

        return fake_cache_kernel, fake_new_kernel

    monkeypatch.setattr(
        cp_attention,
        "build_pcp_kernels",
        lambda **kw: make_kernels_for_rank(kw["cp_rank"]),
    )

    def run_rank(rank: int) -> None:
        try:
            tls.rank = rank
            query = torch.ones((num_tokens_padded, num_heads, head_dim))
            key = torch.ones((num_tokens_padded, num_kv_heads, head_dim))
            value = torch.ones((num_tokens_padded, num_kv_heads, head_dim))
            kv_cache = torch.zeros((8, 4, 1, 64, page_size))
            md = AttentionMetadata(
                input_positions=torch.arange(num_tokens_padded, dtype=torch.int32),
                block_tables=torch.zeros((2, 4), dtype=torch.int32),
                seq_lens=torch.tensor(
                    [cached_tokens + new_tokens] * 2, dtype=torch.int32
                ),
                query_start_loc=torch.from_numpy(cu_q_lens[rank]),
                request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
                pcp=PCPMetadata(
                    query_start_loc=torch.from_numpy(cu_q_lens),
                    kv_cache_lens=torch.tensor([cached_tokens] * 2, dtype=torch.int32),
                    q_pos_offsets=torch.from_numpy(q_pos_offsets),
                    cache_pages=102,
                    new_kv_page_indices=torch.zeros((2, 32), dtype=torch.int32),
                    update_kv_cache=torch.from_numpy(update_kv),
                    is_decode=False,
                    page_size=page_size,
                ),
            )
            layer = SimpleNamespace(_k_scale_float=None, _v_scale_float=None)
            results[rank] = cp_attention.forward_with_pcp(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=md,
                sliding_window=None,
                sm_scale=1.0,
                logits_soft_cap=None,
                kv_cache_quantized_dtype=None,
                pcp_world_size=pcp_size,
                pcp_rank=rank,
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=run_rank, args=(r,)) for r in range(pcp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[0]
    assert recorded_cu_src[0] == [[0, 1536, 1536], [0, 1792, 1792]]
    assert recorded_cu_src[1] == [[0, 1536, 1536], [0, 1792, 1792]]

    # Equal LSE weights across {rank0_cache=1, rank1_cache=3, new=5} -> 3.0.
    # On the unfixed bug, rank 1 rows [1536, 1792) dropped rank 0's cache -> 4.0.
    torch.testing.assert_close(
        results[0][:1536], torch.full((1536, num_heads, head_dim), 3.0)
    )
    torch.testing.assert_close(
        results[0][1536:], torch.zeros((num_tokens_padded - 1536, num_heads, head_dim))
    )
    torch.testing.assert_close(
        results[1][:1792], torch.full((1792, num_heads, head_dim), 3.0)
    )
    torch.testing.assert_close(
        results[1][1536:1792], torch.full((256, num_heads, head_dim), 3.0)
    )
    torch.testing.assert_close(
        results[1][1792:], torch.zeros((num_tokens_padded - 1792, num_heads, head_dim))
    )


def test_forward_with_pcp_validation_guards(monkeypatch):
    query = torch.ones((256, 4, 64))
    key = torch.ones((256, 2, 64))
    value = torch.ones((256, 2, 64))
    kv_cache_hnd = torch.zeros((4, 4, 1, 64, 128))

    def make_md(*, is_decode: bool, page_size: int):
        return AttentionMetadata(
            input_positions=torch.arange(256, dtype=torch.int32),
            block_tables=torch.zeros((2, 4), dtype=torch.int32),
            seq_lens=torch.tensor([256, 256], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 128, 256], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
            pcp=PCPMetadata(
                query_start_loc=torch.tensor(
                    [[0, 128, 256], [0, 128, 256]], dtype=torch.int32
                ),
                kv_cache_lens=torch.tensor([0, 0], dtype=torch.int32),
                q_pos_offsets=torch.tensor([[0, 128], [64, 192]], dtype=torch.int32),
                cache_pages=0,
                new_kv_page_indices=torch.zeros((2, 4), dtype=torch.int32),
                update_kv_cache=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
                is_decode=is_decode,
                page_size=page_size,
            ),
        )

    # 1. FP8 zero scale rejected in forward_with_pcp.
    zero_scale_layer = SimpleNamespace(_k_scale_float=0.0, _v_scale_float=1.0)
    with pytest.raises(ValueError, match="must be non-zero"):
        cp_attention.forward_with_pcp(
            layer=zero_scale_layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache_hnd,
            attn_metadata=make_md(is_decode=False, page_size=128),
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=torch.float8_e4m3fn,
            pcp_world_size=2,
            pcp_rank=0,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

    # 2. Planner page_size mismatch against kv_cache rejected.
    valid_layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    with pytest.raises(ValueError, match="does not match kv_cache page dimension"):
        cp_attention.forward_with_pcp(
            layer=valid_layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache_hnd,
            attn_metadata=make_md(is_decode=False, page_size=256),
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=None,
            pcp_world_size=2,
            pcp_rank=0,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

    # 3. is_decode=True with query.shape[0] == 256 (>= 2 * 128) rejected.
    with pytest.raises(ValueError, match="disagrees with query bucket shape"):
        cp_attention.forward_with_pcp(
            layer=valid_layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache_hnd,
            attn_metadata=make_md(is_decode=True, page_size=128),
            sliding_window=None,
            sm_scale=1.0,
            logits_soft_cap=None,
            kv_cache_quantized_dtype=None,
            pcp_world_size=2,
            pcp_rank=0,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

    # 4. initialize_kernel rejects zero FP8 scales on zigzag PCP.
    monkeypatch.setenv("VLLM_TPU_PCP_LAYOUT", "zigzag")
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.parallel_config.prefill_context_parallel_size = 2
    cfg.parallel_config.cp_kv_cache_interleave_size = 128
    resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])
    with (
        set_current_vllm_config(cfg),
        set_vllm_model_wrapper_context(mesh=None, vllm_config=cfg),
    ):
        backend = attention.PallasBatchedRPAAttentionBackendImpl(
            num_heads=4,
            head_size=128,
            scale=1.0,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_e4m3",
        )
        with pytest.raises(ValueError, match="must be non-zero"):
            backend.initialize_kernel(zero_scale_layer)
