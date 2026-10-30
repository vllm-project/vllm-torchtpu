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
"""Numerical correctness and component-wise tests for FLUX models on TPU hardware.

Tested FLUX Components:
=======================
1. TpuSDPAImpl / TpuFlashAttentionImpl (vllm_torchtpu): TPU diffusion attention parity
   for standard, asymmetric joint text/image, GQA head expansion and masked calls
   against CPU ground truth, in fp32 and bf16 (bf16 runs the Pallas kernel, fp32
   and masked calls the SDPA fallback).
2. FluxPosEmbed / Flux2PosEmbed (vllm_omni): RoPE frequencies computed on TPU (float64,
   as in production) against the same module on CPU.
3. RotaryEmbedding / apply_rope_to_qk (vllm_omni): RoPE rotation kernel application
   on TPU in fp32 and bf16 vs CPU ground truth with bitwise replay determinism.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk
from vllm_omni.diffusion.models.flux.flux_transformer import FluxPosEmbed
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2PosEmbed

from tests_omni.omni_test_utils import assert_parity
from vllm_torchtpu.omni.attention import TpuFlashAttentionImpl, TpuSDPAImpl

pytestmark = [pytest.mark.omni]

# (TPU fp32, TPU bf16) limits vs the float64 CPU reference, ~3% above the worst
# observed value across the parametrized attention impls. Production runs every
# component in the configured dtype (--dtype bfloat16 or float32).
STANDARD_ATTN_MAX_REL_L2 = (3.2e-3, 4.4e-3)
STANDARD_ATTN_MAX_ABS = (7.6e-3, 8.6e-3)
JOINT_ATTN_MAX_REL_L2 = (3.2e-3, 4.5e-3)
JOINT_ATTN_MAX_ABS = (7.7e-3, 1.4e-2)
GQA_ATTN_MAX_REL_L2 = (3.1e-3, 4.1e-3)
GQA_ATTN_MAX_ABS = (7.2e-3, 1.3e-2)
LONG_SEQ_ATTN_MAX_REL_L2 = (3.5e-3, 5.4e-3)
LONG_SEQ_ATTN_MAX_ABS = (6.9e-4, 2.2e-3)
# FIXME: masked calls take the TPU SDPA path, which currently returns wrong values
# (fp32 rel_l2 0.60, bf16 NaN), so the masked test is a strict xfail. These reuse
# the standard-attention limits; recalibrate them when the fix removes the xfail.
MASKED_ATTN_MAX_REL_L2 = STANDARD_ATTN_MAX_REL_L2
MASKED_ATTN_MAX_ABS = STANDARD_ATTN_MAX_ABS
ROPE_MAX_REL_L2 = (3.9e-8, 2.3e-3)
ROPE_MAX_ABS = (4.0e-7, 2.8e-2)
# The RoPE tables are float64 on TPU whatever the configured dtype.
POS_EMBED_MAX_REL_L2 = 2.9e-13
POS_EMBED_MAX_ABS = 1.7e-12

# Pairs each dtype with its index into the (TPU fp32, TPU bf16) limit tuples.
DTYPES = pytest.mark.parametrize(
    "dtype, limit_index",
    [(torch.float32, 0), (torch.bfloat16, 1)],
    ids=["fp32", "bf16"],
)


def _sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """softmax(Q K^T * scale) V in float64 on CPU, for [B, S, H, D] inputs.

    KV heads are expanded with repeat_interleave: query head h attends with KV
    head h // (num_heads // num_kv_heads). key_mask is a [B, S_kv] bool mask of
    the keys to attend to.
    """
    q, k, v = (t.cpu().double() for t in (query, key, value))
    group = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask.cpu()[:, None, None, :], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v)


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
    float64 softmax(Q K^T * scale) V on CPU from the same fp32 or bf16 inputs, so a
    [B, S, H, D] layout or GQA head-mapping mistake cannot cancel out (limits in
    the *_ATTN_MAX_REL_L2 / *_ATTN_MAX_ABS constants).
    """

    @DTYPES
    @pytest.mark.parametrize("impl_cls", [TpuSDPAImpl, TpuFlashAttentionImpl])
    def test_flux_standard_attention_numeric_parity_on_tpu(
        self, impl_cls, dtype, limit_index
    ):
        """Test FLUX.1 standard attention numeric parity against CPU baseline."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 64
        num_heads = 24
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        query_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        key_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        value_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)

        ref_out = _sdpa_reference(query_cpu, key_cpu, value_cpu, scale)

        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert tpu_out.dtype == dtype
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

        # Numerical parity against CPU baseline
        assert_parity(
            tpu_out.cpu(),
            ref_out,
            name=f"{impl_cls.__name__} {dtype} standard attention",
            max_rel_l2=STANDARD_ATTN_MAX_REL_L2[limit_index],
            max_abs=STANDARD_ATTN_MAX_ABS[limit_index],
        )

    @DTYPES
    @pytest.mark.parametrize("impl_cls", [TpuSDPAImpl, TpuFlashAttentionImpl])
    def test_flux_joint_attention_asymmetry_numeric_parity_on_tpu(
        self, impl_cls, dtype, limit_index
    ):
        """Test asymmetric joint text/image cross-attention numeric parity on TPU."""
        torch.manual_seed(123)
        batch_size = 2
        q_len = 128
        kv_len = 64
        num_heads = 24
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        query_cpu = torch.randn(batch_size, q_len, num_heads, head_size, dtype=dtype)
        key_cpu = torch.randn(batch_size, kv_len, num_heads, head_size, dtype=dtype)
        value_cpu = torch.randn(batch_size, kv_len, num_heads, head_size, dtype=dtype)

        ref_out = _sdpa_reference(query_cpu, key_cpu, value_cpu, scale)

        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, q_len, num_heads, head_size)
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        assert_parity(
            tpu_out.cpu(),
            ref_out,
            name=f"{impl_cls.__name__} {dtype} joint attention",
            max_rel_l2=JOINT_ATTN_MAX_REL_L2[limit_index],
            max_abs=JOINT_ATTN_MAX_ABS[limit_index],
        )

    @DTYPES
    @pytest.mark.parametrize("impl_cls", [TpuSDPAImpl, TpuFlashAttentionImpl])
    def test_flux_gqa_repeat_interleave_numeric_parity_on_tpu(
        self, impl_cls, dtype, limit_index
    ):
        """Test grouped-query attention repeat-interleave numeric parity on TPU."""
        torch.manual_seed(456)
        batch_size = 1
        seq_len = 32
        num_heads = 24
        num_kv_heads = 8
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
            num_kv_heads=num_kv_heads,
        )

        query_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        key_cpu = torch.randn(batch_size, seq_len, num_kv_heads, head_size, dtype=dtype)
        value_cpu = torch.randn(
            batch_size, seq_len, num_kv_heads, head_size, dtype=dtype
        )

        ref_out = _sdpa_reference(query_cpu, key_cpu, value_cpu, scale)

        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert torch.all(torch.isfinite(tpu_out.cpu()))

        assert_parity(
            tpu_out.cpu(),
            ref_out,
            name=f"{impl_cls.__name__} {dtype} GQA attention",
            max_rel_l2=GQA_ATTN_MAX_REL_L2[limit_index],
            max_abs=GQA_ATTN_MAX_ABS[limit_index],
        )

    @DTYPES
    @pytest.mark.parametrize("impl_cls", [TpuSDPAImpl, TpuFlashAttentionImpl])
    def test_flux_long_sequence_fixed_m_numeric_parity_on_tpu(
        self, impl_cls, dtype, limit_index
    ):
        """Test multi-block fixed-m flash attention parity at seq_len > 2048 on TPU."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 4133
        num_heads = 4
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=scale,
        )

        query_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        key_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        value_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)

        ref_out = _sdpa_reference(query_cpu, key_cpu, value_cpu, scale)

        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")

        tpu_out = impl.forward(q_tpu, k_tpu, v_tpu)
        assert_parity(
            tpu_out.cpu(),
            ref_out,
            name=f"{impl_cls.__name__} {dtype} long-sequence attention",
            max_rel_l2=LONG_SEQ_ATTN_MAX_REL_L2[limit_index],
            max_abs=LONG_SEQ_ATTN_MAX_ABS[limit_index],
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Masked SDPA on TPU returns wrong values (fp32 rel_l2 0.60, bf16 NaN)",
    )
    @DTYPES
    def test_flux_masked_attention_fallback_matches_sdpa_on_tpu(
        self, monkeypatch, dtype, limit_index
    ):
        """Verify masked calls fall back from TpuFlashAttentionImpl to TpuSDPAImpl."""
        from unittest.mock import MagicMock

        torch.manual_seed(42)
        batch_size = 1
        seq_len = 64
        num_heads = 2
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        mask_cpu = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask_cpu[:, -16:] = False

        flash_impl = TpuFlashAttentionImpl(
            num_heads=num_heads, head_size=head_size, softmax_scale=scale
        )

        query_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        key_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        value_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)

        q_tpu = query_cpu.to(device="tpu")
        k_tpu = key_cpu.to(device="tpu")
        v_tpu = value_cpu.to(device="tpu")
        mask_tpu = mask_cpu.to(device="tpu")
        attn_metadata = SimpleNamespace(attn_mask=mask_tpu)

        orig_sdpa_forward = TpuSDPAImpl.forward
        spy_calls = []

        def _spy_sdpa(self_obj, q_in, k_in, v_in, meta=None):
            spy_calls.append((q_in, k_in, v_in, meta))
            return orig_sdpa_forward(self_obj, q_in, k_in, v_in, meta)

        mock_warn = MagicMock()
        monkeypatch.setattr(TpuSDPAImpl, "forward", _spy_sdpa)
        monkeypatch.setattr(
            "vllm_torchtpu.omni.attention.logger.warning_once", mock_warn
        )

        masked_out = flash_impl.forward(
            q_tpu, k_tpu, v_tpu, attn_metadata=attn_metadata
        )
        assert len(spy_calls) == 1
        assert spy_calls[0][3] is attn_metadata
        assert masked_out.device.type == "tpu"
        assert masked_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert masked_out.dtype == dtype
        mock_warn.assert_called_once_with(
            "TpuFlashAttentionImpl falling back to PyTorch SDPA (%s).",
            "attn_mask is not None",
        )

        ref_out = _sdpa_reference(
            query_cpu, key_cpu, value_cpu, scale, key_mask=mask_cpu
        )
        assert_parity(
            masked_out.cpu(),
            ref_out,
            name=f"masked fallback {dtype}",
            max_rel_l2=MASKED_ATTN_MAX_REL_L2[limit_index],
            max_abs=MASKED_ATTN_MAX_ABS[limit_index],
        )


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
        for table, actual, reference in (("cos", cos, ref_cos), ("sin", sin, ref_sin)):
            assert_parity(
                actual,
                reference,
                name=f"{type(embedder).__name__} {table}",
                max_rel_l2=POS_EMBED_MAX_REL_L2,
                max_abs=POS_EMBED_MAX_ABS,
            )


class TestFluxRoPEKernelApplication:
    """Numerical correctness and parity tests for RoPE kernel application.

    Directly imports RotaryEmbedding and apply_rope_to_qk from
    vllm_omni.diffusion.layers.rope, which is the exact module and function
    invoked inside FluxAttention.forward() to rotate Q and K tensors before attention.
    """

    @DTYPES
    def test_apply_rope_to_qk_numeric_parity_on_tpu(self, dtype, limit_index):
        """Test apply_rope_to_qk on TPU against a float64 CPU reference."""
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

        q_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
        k_cpu = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)

        ref_q, ref_k = apply_rope_to_qk(
            rope, q_cpu.double(), k_cpu.double(), (cos_cpu, sin_cpu)
        )

        q_tpu = q_cpu.to(device="tpu")
        k_tpu = k_cpu.to(device="tpu")
        cos_tpu = cos_cpu.to(device="tpu")
        sin_tpu = sin_cpu.to(device="tpu")

        tpu_q, tpu_k = apply_rope_to_qk(rope, q_tpu, k_tpu, (cos_tpu, sin_tpu))

        assert tpu_q.device.type == "tpu"
        assert tpu_k.device.type == "tpu"
        assert tpu_q.shape == q_cpu.shape
        assert tpu_k.shape == k_cpu.shape
        assert tpu_q.dtype == dtype
        assert tpu_k.dtype == dtype
        assert torch.all(torch.isfinite(tpu_q.cpu()))
        assert torch.all(torch.isfinite(tpu_k.cpu()))

        # Bitwise replay determinism
        tpu_q_replay, tpu_k_replay = apply_rope_to_qk(
            rope, q_tpu, k_tpu, (cos_tpu, sin_tpu)
        )
        torch.testing.assert_close(tpu_q.cpu(), tpu_q_replay.cpu(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(tpu_k.cpu(), tpu_k_replay.cpu(), atol=0.0, rtol=0.0)

        # Numerical parity against the float64 CPU reference
        for which, tpu_out, ref_out in (("q", tpu_q, ref_q), ("k", tpu_k, ref_k)):
            assert_parity(
                tpu_out.cpu(),
                ref_out,
                name=f"RoPE {which} {dtype}",
                max_rel_l2=ROPE_MAX_REL_L2[limit_index],
                max_abs=ROPE_MAX_ABS[limit_index],
            )
