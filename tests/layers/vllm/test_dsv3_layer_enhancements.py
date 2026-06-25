"""Unit tests for DeepSeek V3 layer enhancements (Attention kwargs/reshape, Grouped Top-K MoE, Unaligned FP8)."""

from unittest.mock import MagicMock, patch

import torch

from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackendImpl
from vllm_torchtpu.layers.vllm.moe_routing import select_experts
from vllm_torchtpu.layers.vllm.quantization.fp8 import _dequantize_fp8_linear


def test_pallas_attention_init_kwargs():
    """Test that PallasAttentionBackendImpl accepts arbitrary kwargs like q_lora_rank without raising TypeError."""
    backend = PallasAttentionBackendImpl(
        num_heads=16,
        head_size=256,
        scale=1.0,
        num_kv_heads=16,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bf16",
        kv_cache_quantized_dtype=None,
        q_lora_rank=512,  # Additional kwarg
        custom_v3_flag=True,
    )
    assert backend.num_heads == 16
    assert backend.head_size == 256


@patch("vllm_torchtpu.layers.vllm.attention._pallas_rpa_kernel_default")
def test_pallas_attention_forward_2d_reshape(mock_pallas_rpa):
    """Test that PallasAttentionBackendImpl correctly reshapes 2D query/key tensors to 3D and restores 2D output."""
    q_len = 10
    num_heads = 4
    head_size = 64

    backend = PallasAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=num_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bf16",
        kv_cache_quantized_dtype=None,
    )

    # Directly mock rpa_kernel to return 3D tensor
    backend.rpa_kernel = MagicMock(
        return_value=torch.ones(q_len, num_heads, head_size))

    # 2D query/key/value tensors [q_len, num_heads * head_size]
    query = torch.ones(q_len, num_heads * head_size)
    key = torch.ones(q_len, num_heads * head_size)
    value = torch.ones(q_len, num_heads * head_size)
    kv_cache = torch.ones(1, 1, 1, 1, 1)

    attn_metadata = MagicMock()
    layer_mock = MagicMock()
    layer_mock._k_scale_float = 1.0
    layer_mock._v_scale_float = 1.0

    # Execute forward pass
    out = backend.forward(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        layer=layer_mock,
    )

    # Verify output is restored to 2D [q_len, num_heads * head_size]
    assert out.dim() == 2
    assert out.shape == query.shape


@patch("vllm_torchtpu.layers.vllm.moe_routing.torch.topk")
def test_moe_routing_select_experts_grouped_topk(mock_topk):
    """Test that select_experts correctly delegates to grouped_topk when use_grouped_topk is True."""
    hidden_states = torch.ones(2, 1024, dtype=torch.bfloat16)
    router_logits = torch.ones(2, 64, dtype=torch.float32)

    # 1. Standard Top-K Path
    mock_topk.return_value = (torch.ones(2, 4),
                              torch.zeros(2, 4, dtype=torch.int64))
    weights_std, ids_std = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk=4,
        renormalize=False,
        scoring_fn="softmax",
        layer=None,
    )
    assert mock_topk.called
    assert weights_std.dtype == hidden_states.dtype
    assert ids_std.dtype == torch.int32

    # 2. Grouped Top-K Path
    layer_mock = MagicMock()
    layer_mock.use_grouped_topk = True
    layer_mock.num_expert_group = 8
    layer_mock.topk_group = 2
    layer_mock.routed_scaling_factor = 1.0
    layer_mock.e_score_correction_bias = None

    with patch(
            "vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk"
    ) as mock_grouped:
        mock_grouped.return_value = (torch.ones(2, 4),
                                     torch.zeros(2, 4, dtype=torch.int64))
        weights_grp, ids_grp = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk=4,
            renormalize=False,
            scoring_fn="softmax",
            layer=layer_mock,
        )
        assert mock_grouped.called
        assert weights_grp.dtype == hidden_states.dtype
        assert ids_grp.dtype == torch.int32


def test_dequantize_fp8_linear_unaligned_dims():
    """Test that _dequantize_fp8_linear correctly handles unaligned weight tensor dimensions via repeat_interleave."""
    # Create unaligned weight tensor (e.g., 10x10 with block size 4x4)
    weight = torch.ones(10, 10, dtype=torch.float32)
    weight_scale_inv = torch.ones(
        3, 3, dtype=torch.float32) * 2.0  # 3x3 blocks cover 12x12
    weight_block_size = (4, 4)

    dequant = _dequantize_fp8_linear(
        weight=weight,
        weight_scale=None,
        weight_scale_inv=weight_scale_inv,
        block_quant=True,
        weight_block_size=weight_block_size,
        out_dtype=torch.float32,
    )

    assert dequant.shape == (10, 10)
    assert torch.all(dequant == 2.0)
