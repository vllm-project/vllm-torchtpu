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
