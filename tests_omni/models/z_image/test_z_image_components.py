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
"""Component-level numerical correctness tests for Z-Image on TPU hardware.

Tested Z-Image Components:
==========================
1. RopeEmbedder (vllm_omni) + RotaryEmbedding / apply_rope_to_qk: 3D RoPE
   (theta=256, axes_dims=[32, 48, 48]) and Q/K rotation parity on TPU vs CPU.
2. TpuSDPAImpl / TpuFlashAttentionImpl (vllm_torchtpu): diffusion attention
   parity at Z-Image head geometry (30 heads x 128 head_dim) against float64
   CPU ground truth.
3. _sp_plan["unified_prepare"] auto_pad=True canary and
   OmniTpuPlatform.init_diffusion_model_runner_runtime on DiffusionParallelConfig.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    RopeEmbedder,
    ZImageTransformer2DModel,
)

from tests_omni.omni_test_utils import assert_parity
from vllm_torchtpu.omni.attention import TpuFlashAttentionImpl, TpuSDPAImpl
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [pytest.mark.omni]


class TestZImageComponentsOnTpu:
    """Component-level numerical parity tests for Z-Image on TPU."""

    def test_z_image_rope_embedder_parity_on_tpu(self):
        """Test Z-Image 3D RopeEmbedder (theta=256, axes_dims=[32, 48, 48]) on TPU."""
        torch.manual_seed(42)
        embedder = RopeEmbedder(
            theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512]
        )

        ids_cpu = torch.stack(
            [
                torch.arange(128) % 64,
                torch.arange(128) // 16,
                torch.arange(128) % 16,
            ],
            dim=-1,
        )
        cos_cpu, sin_cpu = embedder(ids_cpu)
        cos_tpu, sin_tpu = embedder(ids_cpu.to("tpu"))

        assert cos_tpu.device.type == "tpu" and sin_tpu.device.type == "tpu"
        assert cos_tpu.shape == (128, 64) and sin_tpu.shape == (128, 64)
        assert_parity(cos_tpu, cos_cpu, name="rope_cos", max_abs=1e-4)
        assert_parity(sin_tpu, sin_cpu, name="rope_sin", max_abs=1e-4)

    def test_z_image_rotary_qk_parity_on_tpu(self):
        """Test interleaved RoPE Q/K rotation at Z-Image head geometry (30x128)."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 128
        num_heads = 30
        head_size = 128

        embedder = RopeEmbedder(
            theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512]
        )
        ids_cpu = torch.stack(
            [
                torch.arange(seq_len) % 64,
                torch.arange(seq_len) // 16,
                torch.arange(seq_len) % 16,
            ],
            dim=-1,
        )
        cos_cpu, sin_cpu = embedder(ids_cpu)
        cos_tpu, sin_tpu = embedder(ids_cpu.to("tpu"))

        rope = RotaryEmbedding(is_neox_style=False)
        q_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.float32
        )
        k_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_size, dtype=torch.float32
        )

        ref_q, ref_k = apply_rope_to_qk(
            rope, q_cpu, k_cpu, (cos_cpu.unsqueeze(0), sin_cpu.unsqueeze(0))
        )
        tpu_q, tpu_k = apply_rope_to_qk(
            rope,
            q_cpu.to("tpu"),
            k_cpu.to("tpu"),
            (cos_tpu.unsqueeze(0), sin_tpu.unsqueeze(0)),
        )

        assert tpu_q.device.type == "tpu" and tpu_k.device.type == "tpu"
        assert_parity(tpu_q, ref_q, name="rotary_q", max_abs=1e-4)
        assert_parity(tpu_k, ref_k, name="rotary_k", max_abs=1e-4)

    # Measured bf16 errors vs float64 CPU reference on v6e-8 (seed=42,
    # shape=(1, 256, 30, 128)); limits sit ~3-5% above the observed values:
    #   TpuSDPAImpl:           rel_l2=4.85e-3, max_abs=1.03e-2
    #   TpuFlashAttentionImpl: rel_l2=2.20e-3, max_abs=2.06e-3
    @pytest.mark.parametrize(
        ("impl_cls", "max_rel_l2", "max_abs"),
        [
            (TpuSDPAImpl, 5.0e-3, 1.1e-2),
            (TpuFlashAttentionImpl, 2.3e-3, 2.2e-3),
        ],
        ids=["sdpa", "flash"],
    )
    def test_z_image_attention_shape_parity_on_tpu(
        self, impl_cls, max_rel_l2: float, max_abs: float
    ):
        """Test TPU attention impls at Z-Image head geometry (30 heads x 128)."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 256
        num_heads = 30
        head_size = 128
        scale = 1.0 / math.sqrt(head_size)

        impl = impl_cls(
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

        probs = (
            torch.einsum("bqhd,bkhd->bhqk", q_cpu.double(), k_cpu.double()) * scale
        ).softmax(dim=-1)
        ref_out = torch.einsum("bhqk,bkhd->bqhd", probs, v_cpu.double())

        tpu_out = impl.forward(q_cpu.to("tpu"), k_cpu.to("tpu"), v_cpu.to("tpu"))
        assert tpu_out.device.type == "tpu"
        assert tpu_out.shape == (batch_size, seq_len, num_heads, head_size)
        assert_parity(
            tpu_out,
            ref_out,
            name=f"z_image_{impl_cls.__name__}",
            max_rel_l2=max_rel_l2,
            max_abs=max_abs,
        )

    def test_z_image_unified_prepare_auto_pad_canary(self, monkeypatch):
        """Verify ``_patch_z_image_transformer_block_flush`` enables ``auto_pad``.

        Once ``apply_omni_model_specific_patches()`` installs per-block async
        flushing, ``unified_prepare``'s SP plan is updated to ``auto_pad=True``
        so ``UlyssesParallelAttention`` skips its per-layer host sync.
        """
        from vllm_omni.diffusion.models.z_image.z_image_transformer import (
            ZImageTransformerBlock,
        )

        from vllm_torchtpu.omni.patches import _patch_z_image_transformer_block_flush

        monkeypatch.setattr(
            ZImageTransformer2DModel,
            "_sp_plan",
            dict(ZImageTransformer2DModel._sp_plan),
        )
        monkeypatch.setattr(
            ZImageTransformer2DModel, "forward", ZImageTransformer2DModel.forward
        )
        monkeypatch.setattr(
            ZImageTransformerBlock, "forward", ZImageTransformerBlock.forward
        )
        _patch_z_image_transformer_block_flush()
        unified_prepare_plan = ZImageTransformer2DModel._sp_plan["unified_prepare"]
        assert isinstance(unified_prepare_plan, dict)
        assert unified_prepare_plan, "unified_prepare SP plan unexpectedly empty"
        for sp_input in unified_prepare_plan.values():
            assert sp_input.auto_pad is True

    @pytest.mark.parametrize(
        ("ulysses_degree", "expected_mode"),
        [(8, "advanced_uaa"), (4, "advanced_uaa"), (2, "strict"), (1, "strict")],
    )
    def test_ulysses_mode_promotion_on_real_parallel_config(
        self, ulysses_degree: int, expected_mode: str
    ):
        """Run the platform hook against vLLM-Omni's real ``DiffusionParallelConfig``.

        Guards the attribute names the hook relies on (``ulysses_degree``,
        ``ulysses_mode``) against upstream renames: with the real dataclass a
        rename would surface here instead of silently disabling the promotion.
        """
        parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree)
        assert parallel_config.ulysses_mode == "strict", "upstream default changed"
        od_config = SimpleNamespace(
            model_class_name="ZImagePipeline", parallel_config=parallel_config
        )
        OmniTpuPlatform.init_diffusion_model_runner_runtime(
            vllm_config=None, od_config=od_config, device=torch.device("cpu")
        )
        assert parallel_config.ulysses_mode == expected_mode
