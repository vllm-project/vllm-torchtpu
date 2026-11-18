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
"""Spec tests for split absorb_q wiring and searchsorted compare_all fast path."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.kernels.mla import dcp_token_parallel
from vllm_torchtpu.layers.adapter import attention as attention_adapter
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op


def _make_layout(dcp_size: int = 4, split_rows: int = 8):
    return dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp_size,
        interleave_size=1,
        page_size=16,
        split_rows=split_rows,
        max_pages=4,
    )


class _FakeLayer:
    def __init__(
        self, num_heads: int = 4, kv_lora_rank: int = 32, v_head_dim: int = 16
    ) -> None:
        self.layer_name = "model.layers.0.self_attn.attn"
        self.use_sparse = True
        self.kv_cache_quantized_dtype = None
        self.scale = 0.125
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.W_UV = torch.zeros(
            (num_heads, kv_lora_rank, v_head_dim), dtype=torch.bfloat16
        )
        self.W_UV_tp_all = torch.zeros(
            (num_heads * 4, kv_lora_rank, v_head_dim), dtype=torch.bfloat16
        )


def _make_fake_layer(num_heads: int = 4, kv_lora_rank: int = 32, v_head_dim: int = 16):
    return _FakeLayer(num_heads, kv_lora_rank, v_head_dim)


def _capture_built_ops(monkeypatch, *, q_replicated: bool, local_oproj: bool):
    captured: list[dict] = []

    def fake_pcp_op(name, fn, **kwargs):
        record = {"name": name, "fn": fn, "kwargs": kwargs, "calls": [], "fake": None}

        def op_call(*args):
            record["calls"].append(args)
            return record["fake"](*args)

        def register_fake(fake_fn):
            record["fake"] = fake_fn

        op_call.register_fake = register_fake
        captured.append(record)
        return op_call

    monkeypatch.setattr(dcp_token_parallel_op, "pcp_streaming_jax_op", fake_pcp_op)
    monkeypatch.setattr(
        dcp_token_parallel_op,
        "_sharded_rank_ids",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.distributed.dcp.dcp_rank_operand",
        lambda: torch.zeros((1,), dtype=torch.int32),
    )
    monkeypatch.setattr(
        dcp_token_parallel_op.envs, "TPU_DCP_TP_PREFILL_LOCAL_OPROJ", local_oproj
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.cp_mla_attention._retain_rank",
        lambda x: x,
    )

    layout = _make_layout(dcp_size=4, split_rows=8)
    layer = _make_fake_layer()
    impl = SimpleNamespace(
        _dcp_mesh=f"mesh_{q_replicated}_{local_oproj}_{id(captured)}",
        dcp_size=4,
        dcp_interleave_size=1,
    )
    tp_op = dcp_token_parallel_op.build_attention_op(
        layer, impl, layout, q_replicated=q_replicated
    )
    layer.sparse_mla_tp_op = tp_op
    return layer, tp_op, captured


def test_should_accept_split_q_tuple_and_preserve_fake_shapes(monkeypatch):
    layer, tp_op, captured = _capture_built_ops(
        monkeypatch, q_replicated=True, local_oproj=True
    )
    assert tp_op.split_q_inputs is True
    assert len(captured) == 2  # base op + oproj_op

    num_tokens, s, rows = 64, 8, 16
    nope_cache = torch.zeros((4, 16, 32), dtype=torch.bfloat16)
    rope_cache = torch.zeros((4, 16, 8), dtype=torch.bfloat16)
    q_head = torch.zeros((s, 16, 32), dtype=torch.bfloat16)
    q_tail = torch.zeros((rows, 16, 32), dtype=torch.bfloat16)
    q_pe = torch.zeros((s + rows, 16, 8), dtype=torch.bfloat16)
    kv_c = torch.zeros((num_tokens, 32), dtype=torch.bfloat16)
    k_pe = torch.zeros((num_tokens, 8), dtype=torch.bfloat16)
    topk = torch.zeros((s + rows, 4), dtype=torch.int32)
    md = SimpleNamespace(
        seq_lens=torch.zeros((2,), dtype=torch.int32),
        block_tables=torch.zeros((4,), dtype=torch.int32),
        query_start_loc=torch.zeros((3,), dtype=torch.int32),
        request_distribution=torch.zeros((3,), dtype=torch.int32),
    )

    out_base = dcp_token_parallel_op.attention(
        layer,
        (nope_cache, rope_cache),
        (q_head, q_tail),
        q_pe,
        kv_c,
        k_pe,
        topk,
        md,
        q_replicated=True,
    )
    assert tuple(out_base.shape) == (num_tokens, layer.num_heads, layer.v_head_dim)
    assert captured[0]["calls"][0][2] is q_head
    assert captured[0]["calls"][0][3] is q_tail

    hidden = 48
    w_o = (
        torch.zeros((layer.num_heads * layer.v_head_dim, hidden), dtype=torch.bfloat16),
        torch.ones((hidden,), dtype=torch.float32),
    )
    out_oproj = dcp_token_parallel_op.attention(
        layer,
        (nope_cache, rope_cache),
        (q_head, q_tail),
        q_pe,
        kv_c,
        k_pe,
        topk,
        md,
        q_replicated=True,
        w_o=w_o,
    )
    assert tuple(out_oproj.shape) == (num_tokens // 4, hidden)
    assert captured[1]["calls"][0][2] is q_head
    assert captured[1]["calls"][0][3] is q_tail

    # Legacy contiguous tensor input still splits automatically.
    q_contig = torch.cat([q_head, q_tail], dim=0)
    out_contig = dcp_token_parallel_op.attention(
        layer,
        (nope_cache, rope_cache),
        q_contig,
        q_pe,
        kv_c,
        k_pe,
        topk,
        md,
        q_replicated=True,
    )
    assert tuple(out_contig.shape) == (num_tokens, layer.num_heads, layer.v_head_dim)
    assert tuple(captured[0]["calls"][1][2].shape) == (s, 16, 32)
    assert tuple(captured[0]["calls"][1][3].shape) == (rows, 16, 32)


def test_should_reject_mismatched_split_q_row_counts(monkeypatch):
    layer, _, _ = _capture_built_ops(monkeypatch, q_replicated=True, local_oproj=False)
    num_tokens = 64
    nope_cache = torch.zeros((4, 16, 32), dtype=torch.bfloat16)
    rope_cache = torch.zeros((4, 16, 8), dtype=torch.bfloat16)
    bad_head = torch.zeros((7, 16, 32), dtype=torch.bfloat16)
    good_tail = torch.zeros((16, 16, 32), dtype=torch.bfloat16)
    q_pe = torch.zeros((24, 16, 8), dtype=torch.bfloat16)
    kv_c = torch.zeros((num_tokens, 32), dtype=torch.bfloat16)
    k_pe = torch.zeros((num_tokens, 8), dtype=torch.bfloat16)
    topk = torch.zeros((24, 4), dtype=torch.int32)
    md = SimpleNamespace(
        seq_lens=torch.zeros((2,), dtype=torch.int32),
        block_tables=torch.zeros((4,), dtype=torch.int32),
        query_start_loc=torch.zeros((3,), dtype=torch.int32),
        request_distribution=torch.zeros((3,), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="split q has"):
        dcp_token_parallel_op.attention(
            layer,
            (nope_cache, rope_cache),
            (bad_head, good_tail),
            q_pe,
            kv_c,
            k_pe,
            topk,
            md,
            q_replicated=True,
        )


def test_should_compute_exact_fake_shapes_for_absorb_q_split():
    q_nope = torch.empty((1088, 64, 192), dtype=torch.bfloat16)
    w_uk_t = torch.empty((64, 192, 512), dtype=torch.bfloat16)
    head, tail = attention_adapter._fake_absorb_q_split(q_nope, w_uk_t, 64)  # noqa: SLF001 -- verifying registered torch fake kernel shape contract directly on CPU
    assert tuple(head.shape) == (64, 64, 512)
    assert tuple(tail.shape) == (1024, 64, 512)
