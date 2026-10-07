# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytestmark = [pytest.mark.omni]


class TestWanRoPEOnTpu:
    """Verify Wan pair-swap RoPE bit-exactness against forward_native on TPU."""

    def test_wan_rotary_embedding_pair_swap_bit_exact_on_tpu(self) -> None:
        from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan

        from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

        apply_omni_model_specific_patches()

        torch.manual_seed(42)
        rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
        x_tpu = torch.randn(2, 128, 12, 128, dtype=torch.bfloat16, device="tpu")
        cos_tpu = torch.randn(1, 128, 1, 64, dtype=torch.bfloat16, device="tpu")
        sin_tpu = torch.randn(1, 128, 1, 64, dtype=torch.bfloat16, device="tpu")

        out = rope(x_tpu, cos_tpu, sin_tpu)
        native_out = rope.forward_native(x_tpu, cos_tpu, sin_tpu)

        assert out.device.type == "tpu"
        torch.testing.assert_close(out.cpu(), native_out.cpu(), rtol=0, atol=0)


class TestWanFusedQKNormRoPEOnTpu:
    """Verify fused q/k RMSNorm + RoPE on the real WanSelfAttention on TPU."""

    def test_wan_self_attention_fused_matches_unpatched_on_tpu(self) -> None:
        from unittest.mock import patch

        from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
            WanSelfAttention,
        )

        from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches
        from vllm_torchtpu.omni.rope import can_fuse_qk_norm_rope

        apply_omni_model_specific_patches()

        torch.manual_seed(42)
        with (
            patch(
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                return_value=0,
            ),
        ):
            attn = WanSelfAttention(dim=256, num_heads=2, head_dim=128).to(
                device="tpu", dtype=torch.bfloat16
            )
        with torch.no_grad():
            attn.norm_q.weight.copy_(
                torch.linspace(0.5, 1.5, 256, device="tpu", dtype=torch.bfloat16)
            )
            attn.norm_k.weight.copy_(
                torch.linspace(1.5, 0.5, 256, device="tpu", dtype=torch.bfloat16)
            )

        hidden = torch.randn(2, 128, 256, device="tpu", dtype=torch.bfloat16)
        cos = torch.randn(1, 128, 1, 64, device="tpu", dtype=torch.bfloat16)
        sin = torch.randn(1, 128, 1, 64, device="tpu", dtype=torch.bfloat16)
        rotary_emb = (cos, sin)

        assert can_fuse_qk_norm_rope(attn, hidden, rotary_emb)
        fused_out = attn(hidden, rotary_emb)
        unpatched_out = WanSelfAttention._tpu_orig_forward(attn, hidden, rotary_emb)

        assert fused_out.device.type == "tpu"
        torch.testing.assert_close(
            fused_out.cpu().float(),
            unpatched_out.cpu().float(),
            rtol=1e-2,
            atol=1e-2,
        )
