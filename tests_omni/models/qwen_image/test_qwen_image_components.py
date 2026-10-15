# SPDX-License-Identifier: Apache-2.0
"""In-process TPU component tests for Qwen-Image-Edit.

Tested components:
- QwenEmbedRope and QwenEmbedLayer3DRope: TPU RoPE frequency generation parity
  vs CPU under torch.inference_mode() (including the txt_freqs clone patch).
- RotaryEmbedding: Cross-attention q/k rotation parity vs float64 CPU reference.
- TpuSDPAImpl: Unmasked joint attention parity (the production path for
  batch-1 requests) vs float64 CPU reference.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenEmbedLayer3DRope,
    QwenEmbedRope,
)

from tests_omni.omni_test_utils import (
    NUMERIC_MISMATCH_PATTERN,
    assert_numerical_parity,
)
from vllm_torchtpu.omni.attention import TpuSDPAImpl
from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

pytestmark = [pytest.mark.omni]


class TestQwenEmbedRopeOnTpu:
    """Verify QwenEmbedRope and QwenEmbedLayer3DRope on TPU under inference_mode."""

    def test_torch_tpu_still_fails_on_pos_freqs_slice(self) -> None:
        """Canary for the bug behind _patch_qwen_image_rope_txt_freqs_clone.

        When this fails, torch_tpu handles the slice and the patch can be removed.
        """
        with torch.device("tpu"):
            embedder = QwenEmbedRope(
                theta=10000, axes_dim=[16, 56, 56], scale_rope=True
            )

        with torch.inference_mode():
            txt_slice = embedder.pos_freqs[16:80, ...]
            try:
                torch.real(txt_slice)
            except RuntimeError as e:
                assert "Cannot set version_counter for inference tensor" in str(e)
            else:
                pytest.fail(
                    "torch.real on a slice of pos_freqs now works under "
                    "torch.inference_mode() on TPU. Remove "
                    "_patch_qwen_image_rope_txt_freqs_clone and this canary."
                )

    @pytest.mark.parametrize(
        "rope_cls",
        [QwenEmbedRope, QwenEmbedLayer3DRope],
        ids=["rope_2d", "rope_3d"],
    )
    def test_qwen_image_rope_txt_freqs_under_inference_mode_on_tpu(
        self, rope_cls: type[torch.nn.Module]
    ) -> None:
        apply_omni_model_specific_patches()

        with torch.device("tpu"):
            tpu_embedder = rope_cls(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        cpu_embedder = rope_cls(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        video_fhw = [(1, 32, 32)]
        txt_seq_lens = [64]

        # Pass pos_freqs' own device, as the transformer does through
        # hidden_states.device, so forward skips its .to(device) branch.
        device = tpu_embedder.pos_freqs.device
        with torch.inference_mode():
            tpu_vid, tpu_txt = tpu_embedder(video_fhw, txt_seq_lens, device=device)
            txt_real = torch.real(tpu_txt)
            txt_imag = torch.imag(tpu_txt)
            vid_real = torch.real(tpu_vid)
            vid_imag = torch.imag(tpu_vid)

        # pos_freqs must still be the tensor built outside inference mode;
        # otherwise this test would pass without the patch.
        assert not tpu_embedder.pos_freqs.is_inference()
        assert txt_real.device.type == "tpu"
        assert txt_imag.device.type == "tpu"

        ref_vid, ref_txt = cpu_embedder(
            video_fhw, txt_seq_lens, device=torch.device("cpu")
        )
        torch.testing.assert_close(
            txt_real.cpu(), torch.real(ref_txt), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            txt_imag.cpu(), torch.imag(ref_txt), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            vid_real.cpu(), torch.real(ref_vid), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            vid_imag.cpu(), torch.imag(ref_vid), atol=1e-5, rtol=1e-5
        )


def _complex_rotary_reference(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """CPU math in float64 representing _apply_qwen_image_rotary_emb."""
    x_f64 = x.to(torch.float64)
    freqs_f64 = freqs.to(torch.complex128)
    paired = torch.view_as_complex(x_f64.reshape(*x_f64.shape[:-1], -1, 2))
    return torch.view_as_real(paired * freqs_f64.unsqueeze(1)).flatten(3)


class TestQwenRotaryKernelApplication:
    """Test QwenImageCrossAttention q/k rotation on TPU vs float64 CPU reference."""

    def test_apply_qwen_rope_numeric_parity_on_tpu(self):
        """Check the rotation kernel on TPU against the float64 CPU reference.

        Both frequency tables are computed on CPU and copied to TPU: the embedder
        is built on CPU, and _compute_video_freqs is lru_cached without the
        device in its key, so the second call returns the cached CPU tensor. The
        test runs outside torch.inference_mode(); TestQwenEmbedRopeOnTpu covers
        the TPU-built frequencies and the inference-mode path.
        """
        torch.manual_seed(42)
        batch_size = 1
        num_heads = 24
        head_size = 128

        video_fhw = [(1, 32, 32)]
        txt_seq_len = 256
        vid_seq_len = 32 * 32

        rope = RotaryEmbedding(is_neox_style=False)
        embedder = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        vid_freqs_cpu, txt_freqs_cpu = embedder(
            video_fhw, [txt_seq_len] * batch_size, device=torch.device("cpu")
        )
        vid_freqs_tpu, txt_freqs_tpu = embedder(
            video_fhw, [txt_seq_len] * batch_size, device=torch.device("tpu")
        )

        q_vid_cpu = torch.randn(
            batch_size, vid_seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        q_txt_cpu = torch.randn(
            batch_size, txt_seq_len, num_heads, head_size, dtype=torch.bfloat16
        )

        q_vid_tpu = q_vid_cpu.to(device="tpu")
        q_txt_tpu = q_txt_cpu.to(device="tpu")

        ref_q_vid = _complex_rotary_reference(
            q_vid_cpu.double(), vid_freqs_cpu.to(torch.complex128)
        )
        ref_q_txt = _complex_rotary_reference(
            q_txt_cpu.double(), txt_freqs_cpu.to(torch.complex128)
        )

        vid_cos = torch.real(vid_freqs_tpu).to(torch.bfloat16)
        vid_sin = torch.imag(vid_freqs_tpu).to(torch.bfloat16)
        txt_cos = torch.real(txt_freqs_tpu).to(torch.bfloat16)
        txt_sin = torch.imag(txt_freqs_tpu).to(torch.bfloat16)

        tpu_q_vid = rope(q_vid_tpu, vid_cos, vid_sin)
        tpu_q_txt = rope(q_txt_tpu, txt_cos, txt_sin)

        assert_numerical_parity(
            tpu_q_vid.cpu(), ref_q_vid, atol=5e-2, rtol=0.0
        )  # Measured max abs error: 0.030761
        assert_numerical_parity(
            tpu_q_txt.cpu(), ref_q_txt, atol=5e-2, rtol=0.0
        )  # Measured max abs error: 0.025155

        # Must-fail control: Neox rotation (chunked halves) instead of interleaved pairs
        bad_rope = RotaryEmbedding(is_neox_style=True)
        bad_tpu_q_vid = bad_rope(q_vid_tpu, vid_cos, vid_sin)
        with pytest.raises(AssertionError, match=NUMERIC_MISMATCH_PATTERN):
            assert_numerical_parity(bad_tpu_q_vid.cpu(), ref_q_vid, atol=5e-2, rtol=0.0)


def _unmasked_sdpa_reference(q, k, v, scale):
    batch, seq, heads, d = q.shape
    out = torch.zeros_like(q)

    for h in range(heads):
        qh = q[:, :, h, :]
        kh = k[:, :, h, :]
        vh = v[:, :, h, :]

        scores = torch.bmm(qh, kh.transpose(1, 2)) * scale
        probs = F.softmax(scores, dim=-1)
        out[:, :, h, :] = torch.bmm(probs, vh)

    return out


class TestQwenJointAttention:
    """Test TpuSDPAImpl with Qwen joint text+image attention configuration."""

    def test_qwen_joint_attention_numeric_parity_on_tpu(self):
        torch.manual_seed(42)
        batch_size = 1
        num_heads = 24
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        # Production scale: text=256, target=1024, condition=4096 => total 5376
        seq_len = 5376

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        q_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        k_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        v_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )

        # Unmasked is the production path for batch size 1.
        attn_metadata = None

        ref_out = _unmasked_sdpa_reference(
            q_cpu.double(), k_cpu.double(), v_cpu.double(), scale
        )

        q_tpu = q_cpu.to(device="tpu")
        k_tpu = k_cpu.to(device="tpu")
        v_tpu = v_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu, attn_metadata=attn_metadata)

        tpu_out_cpu = tpu_out.cpu()

        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert torch.all(torch.isfinite(tpu_out_cpu))

        assert_numerical_parity(
            tpu_out_cpu, ref_out, atol=1e-3, rtol=1e-2
        )  # Measured max abs error: 0.000482

        # Bitwise replay test
        tpu_out_2 = impl.forward(q_tpu, k_tpu, v_tpu, attn_metadata=attn_metadata)
        tpu_out_2_cpu = tpu_out_2.cpu()
        torch.testing.assert_close(tpu_out_cpu, tpu_out_2_cpu, atol=0.0, rtol=0.0)

        # Must-fail control: doubling the softmax scale
        impl_bad = TpuSDPAImpl(
            num_heads=num_heads, head_size=head_size, softmax_scale=scale * 2.0
        )
        tpu_out_bad = impl_bad.forward(q_tpu, k_tpu, v_tpu, attn_metadata=attn_metadata)
        with pytest.raises(AssertionError, match=NUMERIC_MISMATCH_PATTERN):
            assert_numerical_parity(tpu_out_bad.cpu(), ref_out, atol=1e-3, rtol=1e-2)
