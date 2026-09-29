# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Unit tests for standalone layers in llm_tpu_core."""

import torch

from vllm_torchtpu.llm_tpu_core.layers.moe import MoERouter, rowmax_topk
from vllm_torchtpu.llm_tpu_core.layers.rms_norm import RMSNorm
from vllm_torchtpu.llm_tpu_core.layers.rotary import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from vllm_torchtpu.llm_tpu_core.layers.swiglu import SwiGLU


def test_rmsnorm_forward_and_residual():
    hidden_size = 64
    norm = RMSNorm(hidden_size=hidden_size, eps=1e-5)
    x = torch.randn(2, 8, hidden_size, dtype=torch.bfloat16)

    # Standard forward
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16

    # Forward with residual
    res_in = torch.randn_like(x)
    out, res_out = norm(x, residual=res_in)
    assert out.shape == x.shape
    assert torch.allclose(res_out, x + res_in)

    # Numerical verification against explicit float32 RMS formula
    x_f32 = (x + res_in).float()
    expected_variance = x_f32.pow(2).mean(-1, keepdim=True)
    expected_norm = (x_f32 * torch.rsqrt(expected_variance + 1e-5)).bfloat16()
    assert torch.allclose(out, expected_norm, atol=1e-2, rtol=1e-2)


def test_swiglu_forward():
    swiglu = SwiGLU()
    # SwiGLU splits the last dimension in half: input [B, S, 2 * H] -> output [B, S, H]
    hidden_dim = 128
    x = torch.randn(4, 16, 2 * hidden_dim, dtype=torch.bfloat16)
    out = swiglu(x)

    assert out.shape == (4, 16, hidden_dim)
    gate, val = x.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * val
    assert torch.allclose(out, expected)


def test_rotary_embedding():
    head_size = 64
    rope = RotaryEmbedding(head_size=head_size, max_position_embeddings=512)

    batch_size = 2
    seq_len = 16
    num_heads = 4
    positions = torch.arange(seq_len)
    q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16)

    q_out, k_out = rope(positions, q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == torch.bfloat16


def test_rotary_embedding_partial_and_styles():
    head_size = 128
    rotary_dim = 64
    seq_len = 8
    positions = torch.arange(seq_len)
    q = torch.randn(2, seq_len, 4, head_size, dtype=torch.bfloat16)
    k = torch.randn(2, seq_len, 4, head_size, dtype=torch.bfloat16)

    # NeoX style with partial rotary dim
    rope_neox = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=512,
        is_neox_style=True,
    )
    q_out_neox, k_out_neox = rope_neox(positions, q, k)
    assert q_out_neox.shape == q.shape
    assert k_out_neox.shape == k.shape
    # Pass-through channels should match exactly
    assert torch.equal(q_out_neox[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_out_neox[..., rotary_dim:], k[..., rotary_dim:])

    # GPT-J style with partial rotary dim
    rope_gptj = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=512,
        is_neox_style=False,
    )
    q_out_gptj, k_out_gptj = rope_gptj(positions, q, k)
    assert q_out_gptj.shape == q.shape
    assert k_out_gptj.shape == k.shape
    assert torch.equal(q_out_gptj[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_out_gptj[..., rotary_dim:], k[..., rotary_dim:])

    # Free function helper apply_rotary_pos_emb
    cos = rope_neox.cos_cached[:seq_len]
    sin = rope_neox.sin_cached[:seq_len]
    q_free, k_free = apply_rotary_pos_emb(q, k, cos, sin, is_neox_style=True)
    assert q_free.shape == q.shape
    assert torch.equal(q_free[..., rotary_dim:], q[..., rotary_dim:])


def test_moe_router():
    hidden_size = 128
    num_experts = 8
    top_k = 2
    router = MoERouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)

    x = torch.randn(10, hidden_size, dtype=torch.float32)
    result = router(x)

    assert result.topk_weights.shape == (10, top_k)
    assert result.topk_ids.shape == (10, top_k)
    # Weights should sum to 1.0 along the top-k dimension
    sums = result.topk_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Expert IDs should be within valid range [0, num_experts)
    assert (result.topk_ids >= 0).all() and (result.topk_ids < num_experts).all()


def test_rowmax_topk_and_router_integration():
    scores = torch.tensor(
        [
            [1.0, 5.0, 3.0, 2.0],
            [10.0, 2.0, 8.0, 4.0],
        ],
        dtype=torch.float32,
    )
    weights, ids = rowmax_topk(scores, k=2)
    assert ids.shape == (2, 2)
    assert torch.equal(ids[0], torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(ids[1], torch.tensor([0, 2], dtype=torch.int32))

    # Test MoERouter with use_rowmax=True
    router = MoERouter(hidden_size=4, num_experts=4, top_k=2, use_rowmax=True)
    res = router(torch.randn(5, 4))
    assert res.topk_weights.shape == (5, 2)
    assert res.topk_ids.shape == (5, 2)
    assert torch.allclose(res.topk_weights.sum(dim=-1), torch.ones(5), atol=1e-5)
