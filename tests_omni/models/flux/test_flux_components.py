# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness and component-wise tests for FLUX models on TPU hardware.

Tested FLUX Components:
=======================
1. TpuSDPAImpl (vllm_torchtpu): TPU SDPA diffusion attention parity for standard,
   asymmetric joint text/image, and GQA head expansion against CPU ground truth.
2. FluxPosEmbed / Flux2PosEmbed (vllm_omni): RoPE frequencies computed on TPU (float64,
   as in production) against the same module on CPU.
3. RotaryEmbedding / apply_rope_to_qk (vllm_omni): RoPE rotation kernel application
   on TPU vs CPU ground truth with bitwise replay determinism.
"""

import math

import pytest
import torch
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk
from vllm_omni.diffusion.models.flux.flux_transformer import FluxPosEmbed
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2PosEmbed

from vllm_torchtpu.omni.attention import TpuSDPAImpl

pytestmark = [pytest.mark.omni]


def _assert_numerical_parity(
    actual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    min_cos_sim: float = 0.999,
):
    """Assert cosine similarity and floating-point tolerance bounds."""
    a_flat = actual.flatten().float()
    r_flat = reference.flatten().float()

    norm_a = torch.linalg.norm(a_flat)
    norm_r = torch.linalg.norm(r_flat)
    if norm_a > 0 and norm_r > 0:
        cos_sim = float(torch.dot(a_flat, r_flat) / (norm_a * norm_r))
        assert cos_sim >= min_cos_sim, (
            f"Cosine similarity {cos_sim:.5f} fell below minimum required {min_cos_sim}"
        )

    torch.testing.assert_close(
        actual.float(),
        reference.float(),
        atol=atol,
        rtol=rtol,
        msg="TPU kernel output diverged from CPU baseline",
    )


def _sdpa_reference(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor:
    """softmax(Q K^T * scale) V in float64 on CPU, for [B, S, H, D] inputs.

    KV heads are expanded with repeat_interleave: query head h attends with KV
    head h // (num_heads // num_kv_heads).
    """
    q, k, v = (t.cpu().double() for t in (query, key, value))
    group = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)
    probs = (torch.einsum("bqhd,bkhd->bhqk", q, k) * scale).softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


class TestFluxTpuSDPAAttention:
    """Numerical correctness and parity tests for TPU SDPA attention backend.

    vllm_omni delegates all FLUX diffusion attention to this backend on TPU.
    We test standard, asymmetric cross-attention, and GQA configurations directly
    against CPU execution.

    Architectural Note on Why We Do Not Call vllm_omni Attention() directly:
    -----------------------------------------------------------------------
    `vllm_omni.diffusion.attention.layer.Attention` queries the active platform via
    `current_omni_platform.get_diffusion_attn_backend_cls()`. On CPU environments,
    current_omni_platform is `UnspecifiedOmniPlatform`, which has no attention backend
    registered and raises `NotImplementedError: No forward implementation for
    platform: <UnspecifiedOmniPlatform>`.
    `TpuSDPAImpl` is the hardware implementation that vllm-torchtpu registers into
    vllm_omni, so we test its forward pass directly. The reference is an independent
    float64 softmax(Q K^T * scale) V on CPU from the same bf16 inputs, so a
    [B, S, H, D] layout or GQA head-mapping mistake cannot cancel out (measured max
    abs error 0.0043 against atol 1e-2).
    """

    def test_flux_standard_attention_numeric_parity_on_tpu(self):
        """Test FLUX.1 standard attention numeric parity against CPU baseline."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 64
        num_heads = 24
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        query_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        key_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        value_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )

        ref_out = _sdpa_reference(query_cpu, key_cpu, value_cpu, scale)

        # TPU execution in bfloat16
        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert tpu_out.dtype == torch.bfloat16
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        # Bitwise determinism on TPU (exact replay parity)
        tpu_out_replay = impl.forward(q_tpu, k_tpu, v_tpu)
        torch.testing.assert_close(
            tpu_out.cpu(),
            tpu_out_replay.cpu(),
            atol=0.0,
            rtol=0.0,
            msg="TPU execution must be bitwise identical for identical inputs",
        )

        # Numerical parity against CPU baseline (CosSim >= 0.999, bfloat16 tolerance)
        _assert_numerical_parity(tpu_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)

    def test_flux_joint_attention_asymmetry_numeric_parity_on_tpu(self):
        """Test asymmetric joint text/image cross-attention numeric parity on TPU."""
        torch.manual_seed(123)
        batch_size = 2
        q_len = 128
        kv_len = 64
        num_heads = 24
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        query_cpu = torch.randn(
            batch_size, q_len, num_heads, head_size, dtype=torch.bfloat16
        )
        key_cpu = torch.randn(
            batch_size, kv_len, num_heads, head_size, dtype=torch.bfloat16
        )
        value_cpu = torch.randn(
            batch_size, kv_len, num_heads, head_size, dtype=torch.bfloat16
        )

        ref_out = _sdpa_reference(query_cpu, key_cpu, value_cpu, scale)

        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, q_len, num_heads, head_size)
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        _assert_numerical_parity(tpu_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)

    def test_flux_gqa_repeat_interleave_numeric_parity_on_tpu(self):
        """Test grouped-query attention repeat-interleave numeric parity on TPU."""
        torch.manual_seed(456)
        batch_size = 1
        seq_len = 32
        num_heads = 24
        num_kv_heads = 8
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = TpuSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
            num_kv_heads=num_kv_heads,
        )

        query_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.bfloat16
        )
        key_cpu = torch.randn(
            batch_size, seq_len, num_kv_heads, head_size, dtype=torch.bfloat16
        )
        value_cpu = torch.randn(
            batch_size, seq_len, num_kv_heads, head_size, dtype=torch.bfloat16
        )

        ref_out = _sdpa_reference(query_cpu, key_cpu, value_cpu, scale)

        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        _assert_numerical_parity(tpu_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)


class TestFluxRoPEPositionEmbeddingOnTpu:
    """RoPE frequencies computed on the TPU, as in production.

    FluxTransformer2DModel.forward calls pos_embed(ids) on the transformer's
    device (only NPU moves the ids to CPU first), and FluxPosEmbed/Flux2PosEmbed
    compute in float64 on every device except MPS and NPU.
    """

    @pytest.mark.parametrize(
        "embedder",
        [
            FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56]),
            Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32]),
        ],
        ids=["flux1", "flux2"],
    )
    def test_pos_embed_on_tpu_matches_cpu(self, embedder):
        torch.manual_seed(0)
        # 1536 tokens: 512 text tokens plus the 32x32 patches of a 512x512 image.
        ids = torch.randint(0, 512, (1536, len(embedder.axes_dim))).float()
        ref_cos, ref_sin = embedder(ids)
        cos, sin = embedder(ids.to("tpu"))

        assert cos.device.type == "tpu" and sin.device.type == "tpu"
        assert cos.dtype == sin.dtype == torch.float64
        assert cos.shape == sin.shape == (ids.shape[0], sum(embedder.axes_dim) // 2)
        # Default float64 tolerances (atol and rtol 1e-7).
        torch.testing.assert_close(cos.cpu(), ref_cos)
        torch.testing.assert_close(sin.cpu(), ref_sin)


class TestFluxRoPEKernelApplication:
    """Numerical correctness and parity tests for RoPE kernel application.

    Directly imports RotaryEmbedding and apply_rope_to_qk from
    vllm_omni.diffusion.layers.rope, which is the exact module and function
    invoked inside FluxAttention.forward() to rotate Q and K tensors before attention.
    """

    def test_apply_rope_to_qk_numeric_parity_on_tpu(self):
        """Test apply_rope_to_qk parity between CPU baseline and TPU hardware."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 32
        num_heads = 24
        head_size = 128
        axes_dim = [16, 56, 56]

        rope = RotaryEmbedding(is_neox_style=False)
        pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dim)

        ids_cpu = torch.stack(
            [
                torch.zeros(seq_len),
                torch.arange(seq_len),
                torch.arange(seq_len),
            ],
            dim=-1,
        )
        cos_cpu, sin_cpu = pos_embed(ids_cpu)

        q_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.float32
        )
        k_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.float32
        )

        ref_q, ref_k = apply_rope_to_qk(rope, q_cpu, k_cpu, (cos_cpu, sin_cpu))

        q_tpu = q_cpu.to(device="tpu")
        k_tpu = k_cpu.to(device="tpu")
        cos_tpu = cos_cpu.to(device="tpu")
        sin_tpu = sin_cpu.to(device="tpu")

        tpu_q, tpu_k = apply_rope_to_qk(rope, q_tpu, k_tpu, (cos_tpu, sin_tpu))

        assert tpu_q.device.type == "tpu"
        assert tpu_k.device.type == "tpu"
        assert tpu_q.shape == q_cpu.shape
        assert tpu_k.shape == k_cpu.shape
        assert torch.all(torch.isfinite(tpu_q.cpu()))
        assert torch.all(torch.isfinite(tpu_k.cpu()))

        # Bitwise replay determinism
        tpu_q_replay, tpu_k_replay = apply_rope_to_qk(
            rope, q_tpu, k_tpu, (cos_tpu, sin_tpu)
        )
        torch.testing.assert_close(tpu_q.cpu(), tpu_q_replay.cpu(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(tpu_k.cpu(), tpu_k_replay.cpu(), atol=0.0, rtol=0.0)

        # Numerical parity against CPU ground truth
        _assert_numerical_parity(tpu_q.cpu(), ref_q, atol=1e-4, rtol=1e-4)
        _assert_numerical_parity(tpu_k.cpu(), ref_k, atol=1e-4, rtol=1e-4)
