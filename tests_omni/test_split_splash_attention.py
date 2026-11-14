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
"""Tests for Pallas Split-Splash attention and QwenImage prefix-mask wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer

from tests_omni.omni_test_utils import assert_parity
from vllm_torchtpu import envs
from vllm_torchtpu.omni.attention import (
    TpuFlashAttentionImpl,
    TpuSDPAImpl,
    _prefix_mask_len_for_split_splash,
)
from vllm_torchtpu.omni.patches import (
    _QWEN_IMAGE_PREFIX_MASK_PATCHED,
    _patch_qwen_image_prefix_mask,
    _patch_sdpa_fallback_dispatch,
)

pytestmark = [pytest.mark.omni]

NUMERIC_MISMATCH_PATTERN = r"relative L2 error|max absolute error"
SPLIT_SPLASH_MAX_REL_L2 = 5.5e-3
SPLIT_SPLASH_MAX_ABS = 5.0e-3

_UPSTREAM_SDPA_FORWARD = SDPAImpl.forward
_UPSTREAM_QWEN_ATTN_FORWARD = qwen_image_transformer.QwenImageCrossAttention.forward


def _make_qwen_cross_attention(
    dim: int = 64, heads: int = 2, head_dim: int = 32
) -> qwen_image_transformer.QwenImageCrossAttention:
    attn_cls = qwen_image_transformer.QwenImageCrossAttention
    attn = object.__new__(attn_cls)
    torch.nn.Module.__init__(attn)
    attn.query_num_heads = attn.kv_num_heads = heads
    attn.add_query_num_heads = attn.add_kv_num_heads = heads
    attn.head_dim = head_dim
    attn.qk_norm = True
    attn.eps = 1e-6
    attn.reference_precision = False
    attn.parallel_config = None
    width = heads * head_dim
    to_qkv = torch.nn.Linear(dim, 3 * width)
    add_kv = torch.nn.Linear(dim, 3 * width)
    attn.to_qkv = lambda x: (to_qkv(x), None)
    attn.add_kv_proj = lambda x: (add_kv(x), None)
    attn.norm_q = RMSNorm(head_dim, eps=1e-6)
    attn.norm_k = RMSNorm(head_dim, eps=1e-6)
    attn.norm_added_q = RMSNorm(head_dim, eps=1e-6)
    attn.norm_added_k = RMSNorm(head_dim, eps=1e-6)
    attn.rope = lambda x, freqs, use_real=False: x
    attn.to_out = torch.nn.Linear(width, dim)
    attn.to_add_out = torch.nn.Linear(width, dim)
    return attn


class TestSplitSplashAttention:
    """Verify Split-Splash opt-in gating, numerical parity, and dispatch."""

    def test_disabled_by_default_without_env_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VLLM_TPU_OMNI_SPLIT_SPLASH", raising=False)
        assert envs.VLLM_TPU_OMNI_SPLIT_SPLASH is False

        attn_cls = qwen_image_transformer.QwenImageCrossAttention
        monkeypatch.setattr(attn_cls, "forward", _UPSTREAM_QWEN_ATTN_FORWARD)
        _patch_qwen_image_prefix_mask()
        assert not getattr(attn_cls.forward, _QWEN_IMAGE_PREFIX_MASK_PATCHED, False)

    @pytest.mark.parametrize(
        ("batch", "prefix_len", "img_len", "scale_mult", "mask_mode"),
        [
            (1, 32, 4096, 1.0, "prefix_partial"),
            (1, 600, 2304, 1.0, "prefix_partial"),
            (2, 32, 512, 0.5, "prefix_all_false"),
        ],
        ids=[
            "split_4128_prefix_partial",
            "non_1024_img_2304_long_prefix_600",
            "batched_544_custom_scale_all_false_prefix",
        ],
    )
    def test_split_splash_matches_reference_and_respects_mask(
        self,
        monkeypatch: pytest.MonkeyPatch,
        batch: int,
        prefix_len: int,
        img_len: int,
        scale_mult: float,
        mask_mode: str,
    ) -> None:
        monkeypatch.setattr(envs, "VLLM_TPU_OMNI_SPLIT_SPLASH", True)
        torch.manual_seed(0)
        seq_len = prefix_len + img_len
        h, d = 4, 128
        scale = (d**-0.5) * scale_mult

        q = torch.randn((batch, seq_len, h, d), dtype=torch.bfloat16)
        k = torch.randn((batch, seq_len, h, d), dtype=torch.bfloat16)
        v = torch.randn((batch, seq_len, h, d), dtype=torch.bfloat16)

        mask_cpu = torch.ones((batch, seq_len), dtype=torch.bool)
        if mask_mode == "prefix_partial":
            mask_cpu[:, prefix_len // 2 : prefix_len] = False
        elif mask_mode == "prefix_all_false":
            mask_cpu[:, :prefix_len] = False

        ref = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3).double(),
            k.permute(0, 2, 1, 3).double(),
            v.permute(0, 2, 1, 3).double(),
            attn_mask=mask_cpu[:, None, None, :],
            scale=scale,
        ).permute(0, 2, 1, 3)

        tpu = torch.device("tpu")
        impl = TpuSDPAImpl(num_heads=h, head_size=d, softmax_scale=scale)
        monkeypatch.setattr(
            impl,
            "_forward_impl",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Expected Split-Splash path, not _forward_impl")
            ),
        )
        qt, kt, vt = q.to(tpu), k.to(tpu), v.to(tpu)
        meta = AttentionMetadata(
            attn_mask=mask_cpu.to(tpu),
            extra={"prefix_mask_len": prefix_len},
        )
        all_true_meta = AttentionMetadata(
            attn_mask=torch.ones_like(mask_cpu, device=tpu),
            extra={"prefix_mask_len": prefix_len},
        )
        with torch.inference_mode():
            out = impl.forward(qt, kt, vt, meta)
            out_all_true = impl.forward(qt, kt, vt, all_true_meta)

        assert_parity(
            out,
            ref,
            name=f"split_splash_{seq_len}_{mask_mode}",
            max_rel_l2=SPLIT_SPLASH_MAX_REL_L2,
            max_abs=SPLIT_SPLASH_MAX_ABS,
        )
        # Must-fail control: ignoring attn_mask (all-True prefix) fails the same gate.
        with pytest.raises(AssertionError, match=NUMERIC_MISMATCH_PATTERN):
            assert_parity(
                out_all_true,
                ref,
                name=f"split_splash_{seq_len}_{mask_mode}_unmasked_control",
                max_rel_l2=SPLIT_SPLASH_MAX_REL_L2,
                max_abs=SPLIT_SPLASH_MAX_ABS,
            )

    def test_unmasked_sdpa_routes_to_flash_attention_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "VLLM_TPU_OMNI_SPLIT_SPLASH", True)
        called: list[float] = []

        def _fake_flash(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
        ) -> torch.Tensor:
            called.append(scale)
            return q.clone()

        monkeypatch.setattr(
            "vllm_torchtpu.omni.attention._flash_attention_op",
            lambda: _fake_flash,
        )
        monkeypatch.setattr(
            "vllm_torchtpu.omni.attention._get_split_splash_op",
            lambda: MagicMock(),
        )
        tpu = torch.device("tpu")
        impl = TpuSDPAImpl(num_heads=4, head_size=128, softmax_scale=0.125)
        q = torch.randn((1, 4128, 4, 128), dtype=torch.bfloat16, device=tpu)
        with torch.inference_mode():
            out = impl.forward(q, q, q, None)
        assert out.shape == q.shape
        assert called == [0.125]

    def test_flash_impl_routes_prefix_masked_to_split_splash_without_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "VLLM_TPU_OMNI_SPLIT_SPLASH", True)
        mock_warn = MagicMock()
        monkeypatch.setattr(
            "vllm_torchtpu.omni.attention.logger.warning_once", mock_warn
        )
        mock_split = MagicMock(side_effect=lambda q, k, v, m, p, s: q.clone())
        monkeypatch.setattr(
            "vllm_torchtpu.omni.attention._get_split_splash_op",
            lambda: mock_split,
        )
        tpu = torch.device("tpu")
        impl = TpuFlashAttentionImpl(
            num_heads=4, head_size=128, softmax_scale=128**-0.5
        )
        q = torch.randn((1, 4128, 4, 128), dtype=torch.bfloat16, device=tpu)
        mask = torch.ones((1, 4128), dtype=torch.bool, device=tpu)
        meta = AttentionMetadata(attn_mask=mask, extra={"prefix_mask_len": 32})
        with torch.inference_mode():
            out = impl.forward(q, q, q, meta)
        assert out.shape == q.shape
        mock_split.assert_called_once()
        mock_warn.assert_not_called()

    @pytest.mark.parametrize(
        ("seq_len", "dtype", "extra"),
        [
            (4128, torch.bfloat16, {}),
            (4128, torch.bfloat16, {"prefix_mask_len": 0}),
            (600, torch.bfloat16, {"prefix_mask_len": 32}),
            (4128, torch.float16, {"prefix_mask_len": 32}),
        ],
        ids=[
            "missing_prefix_mask_len",
            "invalid_zero_prefix_len",
            "unaligned_img_seq_len",
            "fp16_dtype",
        ],
    )
    def test_unsupported_inputs_fall_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        seq_len: int,
        dtype: torch.dtype,
        extra: dict[str, int],
    ) -> None:
        monkeypatch.setattr(envs, "VLLM_TPU_OMNI_SPLIT_SPLASH", True)
        tpu = torch.device("tpu")
        q = torch.randn((1, seq_len, 4, 128), dtype=dtype, device=tpu)
        mask = torch.ones((1, seq_len), dtype=torch.bool, device=tpu)
        meta = AttentionMetadata(attn_mask=mask, extra=extra)
        assert _prefix_mask_len_for_split_splash(q, q, q, meta, causal=False) is None

    def test_patched_sdpa_impl_dispatches_to_split_splash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "VLLM_TPU_OMNI_SPLIT_SPLASH", True)
        monkeypatch.setattr(SDPAImpl, "forward", _UPSTREAM_SDPA_FORWARD)
        _patch_sdpa_fallback_dispatch()

        tpu = torch.device("tpu")
        sdpa = SDPAImpl(num_heads=4, head_size=128, softmax_scale=128**-0.5)
        monkeypatch.setattr(
            sdpa,
            "_forward_impl",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Expected patched SDPAImpl to run Split-Splash")
            ),
        )
        q = torch.randn((1, 4128, 4, 128), dtype=torch.bfloat16, device=tpu)
        mask = torch.ones((1, 4128), dtype=torch.bool, device=tpu)
        mask[:, 20:32] = False
        meta = AttentionMetadata(attn_mask=mask, extra={"prefix_mask_len": 32})
        with torch.inference_mode():
            out = sdpa.forward(q, q, q, meta)
        assert out.shape == q.shape

    def test_qwen_image_cross_attention_prefix_mask_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "VLLM_TPU_OMNI_SPLIT_SPLASH", True)
        attn_cls = qwen_image_transformer.QwenImageCrossAttention
        monkeypatch.setattr(attn_cls, "forward", _UPSTREAM_QWEN_ATTN_FORWARD)
        _patch_qwen_image_prefix_mask()
        _patch_qwen_image_prefix_mask()
        assert getattr(attn_cls.forward, _QWEN_IMAGE_PREFIX_MASK_PATCHED, False)

        torch.manual_seed(0)
        attn = _make_qwen_cross_attention()
        seen_meta: list[AttentionMetadata | None] = []

        def _stub_attn(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_metadata: AttentionMetadata | None = None,
        ) -> torch.Tensor:
            seen_meta.append(attn_metadata)
            return q * 0.5 + k * 0.3 + v * 0.2

        attn.attn = _stub_attn
        hs = torch.randn(2, 16, 64)
        enc = torch.randn(2, 7, 64)
        enc_mask = torch.tensor(
            [[True] * 7, [True] * 4 + [False] * 3], dtype=torch.bool
        )
        vid_freqs = torch.ones(16, 16, dtype=torch.complex64)
        txt_freqs = torch.ones(7, 16, dtype=torch.complex64)

        out_patched = attn(
            hs, enc, vid_freqs, txt_freqs, encoder_hidden_states_mask=enc_mask
        )
        assert len(seen_meta) == 1
        assert seen_meta[0] is not None
        assert seen_meta[0].extra == {"prefix_mask_len": 7}

        out_orig = attn_cls._tpu_orig_forward(
            attn,
            hs,
            enc,
            vid_freqs,
            txt_freqs,
            encoder_hidden_states_mask=enc_mask,
        )
        torch.testing.assert_close(out_patched[0], out_orig[0], rtol=0, atol=0)
        torch.testing.assert_close(out_patched[1], out_orig[1], rtol=0, atol=0)

        # When hidden_states_mask is also set, delegates to _tpu_orig_forward.
        seen_meta.clear()
        hs_mask = torch.ones(2, 16, dtype=torch.bool)
        _ = attn(
            hs,
            enc,
            vid_freqs,
            txt_freqs,
            hidden_states_mask=hs_mask,
            encoder_hidden_states_mask=enc_mask,
        )
        assert seen_meta[0] is not None
        assert seen_meta[0].extra == {}
