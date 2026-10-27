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
import math
import os
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from .mock_omni import (  # isort: split
    _mock_apply_split_rotary_emb,
    _mock_ltx2_rope_cls,
    _MockRMSNorm,
    _MockRotaryEmbeddingWan,
    vllm_omni_mock,
)

import vllm_omni
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.models.ltx2 import ltx2_transformer
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer

from vllm_torchtpu.omni import register_omni_tpu_platform
from vllm_torchtpu.omni import rope as omni_rope
from vllm_torchtpu.omni.attention import (
    TpuFlashAttentionBackend,
    TpuFlashAttentionImpl,
    TpuSDPABackend,
    TpuSDPAImpl,
)
from vllm_torchtpu.omni.patches import (
    _BWE_FILTER_CPU_RESTORE_PATCHED,
    _D2H_STREAM_PATCHED,
    _SDPA_FALLBACK_PATCHED,
    _VOCODER_CPU_INIT_PATCHED,
    _patch_diffusion_startup_warmup_sync,
    _patch_diffusion_worker_d2h_stream,
    _patch_ltx2_split_rope,
    _patch_ltx2_vocoder_cpu_init,
    _patch_sdpa_fallback_dispatch,
    apply_omni_model_specific_patches,
    apply_omni_tpu_patches,
)
from vllm_torchtpu.omni.platform import OmniTpuPlatform
from vllm_torchtpu.omni.rope import (
    _pair_swap_matrix,
    apply_rotary_pair_swap,
    can_fuse_qk_norm_rope,
    rotary_embedding_wan_forward,
    split_rope_natural_tables,
    wan_self_attention_fused,
)

pytestmark = pytest.mark.omni


@pytest.fixture(scope="module", autouse=True)
def cleanup_mock_omni():
    yield
    for mod_name in list(sys.modules.keys()):
        if mod_name == "vllm_omni" or mod_name.startswith("vllm_omni."):
            del sys.modules[mod_name]


def test_register_omni_tpu_platform(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 0)
    assert register_omni_tpu_platform() is None

    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 8)
    assert register_omni_tpu_platform() == "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def test_platform_attributes():
    platform = OmniTpuPlatform()
    assert platform.device_name == "tpu"
    assert platform.device_type == "tpu"
    assert platform.dist_backend == "gloo"
    assert platform.supports_torch_inductor() is False
    assert platform.supports_diffusion_dense_flash_attention() is False
    assert (
        platform.get_diffusion_attn_backend_cls(None, 64)
        == "vllm_torchtpu.omni.attention.TpuFlashAttentionBackend"
    )
    for name in ("SDPA", "TPU_SDPA", "torch_sdpa"):
        assert (
            platform.get_diffusion_attn_backend_cls(name, 64)
            == "vllm_torchtpu.omni.attention.TpuSDPABackend"
        )
    with patch("torch.device") as mock_dev:
        platform.get_torch_device()
        mock_dev.assert_called_once_with("tpu")


def test_rope_pair_swap_matches_native():
    torch.manual_seed(0)
    rope = _MockRotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    x = torch.randn(1, 24, 3, 16)
    cos = torch.randn(1, 24, 1, 8)
    sin = torch.randn(1, 24, 1, 8)
    torch.testing.assert_close(
        apply_rotary_pair_swap(x, cos, sin), rope.forward_native(x, cos, sin)
    )
    x_bf16, cos_bf16, sin_bf16 = (
        x.bfloat16(),
        cos.bfloat16(),
        sin.bfloat16(),
    )
    torch.testing.assert_close(
        apply_rotary_pair_swap(x_bf16, cos_bf16, sin_bf16),
        rope.forward_native(x_bf16, cos_bf16, sin_bf16),
        rtol=0,
        atol=0,
    )


def test_rope_patch_installed():
    from unittest.mock import MagicMock

    apply_omni_model_specific_patches()
    assert _MockRotaryEmbeddingWan.forward is rotary_embedding_wan_forward
    # Interleaved half-dim calls use apply_rotary_pair_swap (bit-exact in bf16).
    rope = _MockRotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    x = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    cos = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    sin = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(
        rope(x, cos, sin), rope.forward_native(x, cos, sin), rtol=0, atol=0
    )

    # Non-half-dim or Neox calls delegate to _forward_method.
    rope_fallback = _MockRotaryEmbeddingWan(is_neox_style=False, half_head_dim=False)
    spy = MagicMock(wraps=rope_fallback.forward_native)
    rope_fallback._forward_method = spy
    out_fb = rope_fallback(x, cos, sin)
    spy.assert_called_once_with(x, cos, sin)
    torch.testing.assert_close(
        out_fb, rope_fallback.forward_native(x, cos, sin), rtol=0, atol=0
    )

    # f32 calls delegate to _forward_method (precision fallback).
    spy_f32 = MagicMock(wraps=rope.forward_native)
    rope._forward_method = spy_f32
    x_f32, cos_f32, sin_f32 = x.float(), cos.float(), sin.float()
    out_f32 = rope(x_f32, cos_f32, sin_f32)
    spy_f32.assert_called_once_with(x_f32, cos_f32, sin_f32)
    torch.testing.assert_close(
        out_f32, rope.forward_native(x_f32, cos_f32, sin_f32), rtol=0, atol=0
    )


def test_rope_patch_accepts_token_major_tables():
    # vLLM-Omni 0.32 (vllm-omni#8112): LingBot-World passes (tokens, D/2) tables.
    rope = _MockRotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    x = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    cos = torch.randn(4, 4, dtype=torch.bfloat16)
    sin = torch.randn(4, 4, dtype=torch.bfloat16)
    expected = rope.forward_native(x, cos[None, :, None], sin[None, :, None])
    torch.testing.assert_close(
        rotary_embedding_wan_forward(rope, x, cos, sin), expected, rtol=0, atol=0
    )


def test_tpu_flash_attention_backend(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    assert TpuFlashAttentionBackend.get_name() == "TPU_FLASH_ATTN"
    assert TpuFlashAttentionBackend.get_impl_cls() is TpuFlashAttentionImpl

    mock_op = MagicMock(side_effect=lambda q, k, v, scale: q.clone())
    monkeypatch.setattr(
        "vllm_torchtpu.omni.attention._flash_attention_op", lambda: mock_op
    )
    mock_warn = MagicMock()
    monkeypatch.setattr("vllm_torchtpu.omni.attention.logger.warning_once", mock_warn)

    q = torch.randn(1, 16, 6, 16, dtype=torch.bfloat16)
    k = torch.randn(1, 16, 6, 16, dtype=torch.bfloat16)
    v = torch.randn(1, 16, 6, 16, dtype=torch.bfloat16)

    impl = TpuFlashAttentionImpl(num_heads=6, head_size=16, softmax_scale=0.25)
    out = impl.forward(q, k, v)
    assert out.shape == (1, 16, 6, 16)
    mock_op.assert_called_once()
    mock_warn.assert_not_called()

    # 1. causal=True falls back to PyTorch SDPA with warning_once
    mock_op.reset_mock()
    impl_causal = TpuFlashAttentionImpl(
        num_heads=6, head_size=16, softmax_scale=0.25, causal=True
    )
    out_causal = impl_causal.forward(q, k, v)
    assert out_causal.shape == (1, 16, 6, 16)
    mock_op.assert_not_called()
    mock_warn.assert_called_once_with(
        "TpuFlashAttentionImpl falling back to PyTorch SDPA (%s).", "causal=True"
    )

    # 2. attn_mask is not None falls back
    mock_warn.reset_mock()
    meta = SimpleNamespace(attn_mask=torch.ones(1, 16, dtype=torch.bool))
    impl.forward(q, k, v, attn_metadata=meta)
    mock_op.assert_not_called()
    mock_warn.assert_called_once_with(
        "TpuFlashAttentionImpl falling back to PyTorch SDPA (%s).",
        "attn_mask is not None",
    )

    # 3. non-bfloat16 falls back
    mock_warn.reset_mock()
    impl.forward(q.float(), k.float(), v.float())
    mock_op.assert_not_called()
    mock_warn.assert_called_once_with(
        "TpuFlashAttentionImpl falling back to PyTorch SDPA (%s).",
        "query.dtype=torch.float32 (requires torch.bfloat16)",
    )

    # 4. uneven GQA raises, like vLLM-Omni's SDPA
    mock_warn.reset_mock()
    impl_uneven = TpuFlashAttentionImpl(
        num_heads=6, head_size=16, softmax_scale=0.25, num_kv_heads=4
    )
    k4 = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)
    v4 = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"num_heads \(6\) must be a multiple"):
        impl_uneven.forward(q, k4, v4)
    mock_op.assert_not_called()
    mock_warn.assert_not_called()


def test_wan_ffn_gelu_fence(monkeypatch):
    # The Wan FFN returns the GELU output passed through the XLA
    # optimization barrier (its result must be used, or XLA drops it).
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        ColumnParallelGELU,
    )

    from vllm_torchtpu.omni import patches

    apply_omni_model_specific_patches()
    ffn = ColumnParallelGELU(8, 16)
    x = torch.randn(3, 8)
    want = torch.nn.functional.gelu(ffn.proj(x), approximate="tanh")
    fenced = []
    monkeypatch.setattr(
        patches,
        "_optimization_barrier_op",
        lambda: lambda t: fenced.append(t) or t * 2,
    )
    torch.testing.assert_close(ffn(x), want * 2, rtol=0, atol=0)
    assert len(fenced) == 1
    torch.testing.assert_close(fenced[0], want, rtol=0, atol=0)


def test_device_control_env_var():
    assert OmniTpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"


def test_shared_slicebuilder_addresses(monkeypatch):
    # Every worker derives the same list from the shared MASTER_PORT.
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "localhost:1")
    OmniTpuPlatform._set_shared_slicebuilder_addresses(2)
    assert (
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"]
        == "localhost:29501,localhost:29502"
    )
    # A correctly sized inherited list (e.g. another stage's) is replaced too.
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "a:1,b:2")
    OmniTpuPlatform._set_shared_slicebuilder_addresses(2)
    assert (
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"]
        == "localhost:29501,localhost:29502"
    )


def test_offload_dense_all_to_all(monkeypatch):
    flag = "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all"
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_foo=1")
    OmniTpuPlatform._offload_dense_all_to_all()
    OmniTpuPlatform._offload_dense_all_to_all()
    assert os.environ["LIBTPU_INIT_ARGS"] == f"--xla_foo=1 {flag}=true"
    # An explicit user setting wins.
    monkeypatch.setenv("LIBTPU_INIT_ARGS", f"{flag}=false")
    OmniTpuPlatform._offload_dense_all_to_all()
    assert os.environ["LIBTPU_INIT_ARGS"] == f"{flag}=false"
    monkeypatch.delenv("LIBTPU_INIT_ARGS")
    OmniTpuPlatform._offload_dense_all_to_all()
    assert os.environ["LIBTPU_INIT_ARGS"] == f"{flag}=true"


def test_dist_backend_after_world_size_pop(monkeypatch):
    # Building the worker's VllmConfig pops WORLD_SIZE after set_device; the
    # size set_device recorded must keep multi-rank groups on tpu_dist.
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "a:1,b:2")
    monkeypatch.setattr(OmniTpuPlatform, "_worker_world_size", 1)
    monkeypatch.setattr(
        OmniTpuPlatform,
        "_prepare_singlehost_tpu_env",
        staticmethod(lambda world_size: None),
    )
    monkeypatch.setattr(torch, "accelerator", MagicMock(), raising=False)
    with patch("torch.empty"), patch("torch.device"):
        OmniTpuPlatform.set_device(None)
    monkeypatch.delenv("WORLD_SIZE")
    assert OmniTpuPlatform().dist_backend == "tpu_dist"


@pytest.mark.parametrize("chips, warns", [(None, False), ("0,1", False), ("1,2", True)])
def test_set_device_warns_on_pinned_multi_rank_stage(monkeypatch, chips, warns):
    # Ranks bind chips 0..N-1 by local rank, so a stage pinned to other chips
    # (e.g. a diffusion stage in a multi-stage pipeline) is flagged.
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "localhost:1")
    if chips is None:
        monkeypatch.delenv("TPU_VISIBLE_CHIPS", raising=False)
    else:
        monkeypatch.setenv("TPU_VISIBLE_CHIPS", chips)
    monkeypatch.setattr(OmniTpuPlatform, "_worker_world_size", 1)
    monkeypatch.setattr(
        OmniTpuPlatform,
        "_prepare_singlehost_tpu_env",
        staticmethod(lambda world_size: None),
    )
    monkeypatch.setattr(torch, "accelerator", MagicMock(), raising=False)
    with (
        patch("vllm_torchtpu.omni.platform.logger.warning") as mock_warning,
        patch("torch.empty"),
        patch("torch.device"),
    ):
        OmniTpuPlatform.set_device(None)
    assert mock_warning.called == warns


def test_memory_methods(monkeypatch):
    class MockAccelerator:
        @staticmethod
        def get_memory_info(device=None):
            return (16 * 1024**3, 32 * 1024**3)

        @staticmethod
        def max_memory_reserved(device=None):
            return 16 * 1024**3

        @staticmethod
        def max_memory_allocated(device=None):
            return 16 * 1024**3

    monkeypatch.setattr(torch, "accelerator", MockAccelerator(), raising=False)
    with patch(
        "torch_tpu._internal.utils.hardware.get_hbm_bytes_per_device",
        return_value=32 * 1024**3,
    ):
        assert OmniTpuPlatform.get_device_total_memory() == 32 * 1024**3
        assert OmniTpuPlatform.get_free_memory() == 16 * 1024**3
        assert OmniTpuPlatform.get_device_memory() == (16 * 1024**3, 32 * 1024**3)
        assert OmniTpuPlatform.max_memory_reserved() == 16 * 1024**3
        assert OmniTpuPlatform.max_memory_allocated() == 16 * 1024**3


def test_tpu_sdpa_attention_backend():
    assert TpuSDPABackend.get_name() == "TPU_SDPA"
    assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl
    assert TpuSDPABackend.supports_attention_mask() is True

    impl = TpuSDPAImpl(num_heads=4, head_size=16, softmax_scale=0.25)
    query = torch.randn(2, 8, 4, 16)
    key = torch.randn(2, 8, 4, 16)
    value = torch.randn(2, 8, 4, 16)

    res = impl.forward(query, key, value)
    assert res.shape == (2, 8, 4, 16)

    # GQA
    impl_gqa = TpuSDPAImpl(
        num_heads=8, head_size=16, softmax_scale=0.25, num_kv_heads=2
    )
    query_gqa = torch.randn(1, 16, 8, 16)
    key_gqa = torch.randn(1, 16, 2, 16)
    value_gqa = torch.randn(1, 16, 2, 16)
    res_gqa = impl_gqa.forward(query_gqa, key_gqa, value_gqa)
    assert res_gqa.shape == (1, 16, 8, 16)


def test_vllm_omni_is_mocked():
    assert sys.modules["vllm_omni"] is vllm_omni_mock
    assert not hasattr(vllm_omni, "__file__") or vllm_omni.__file__ is None


@pytest.mark.parametrize(
    "rope_cls_name",
    ["QwenEmbedRope", "QwenEmbedLayer3DRope"],
)
def test_qwen_image_rope_txt_freqs_clone(rope_cls_name):
    apply_omni_model_specific_patches()

    rope_cls = getattr(qwen_image_transformer, rope_cls_name)
    rope = rope_cls()

    vid_freqs, txt_freqs = rope.forward([(1, 2, 2)], [3], torch.device("cpu"))

    assert not txt_freqs._is_view()
    assert (
        txt_freqs.untyped_storage().data_ptr()
        != rope.pos_freqs.untyped_storage().data_ptr()
    )
    assert torch.equal(txt_freqs, rope.pos_freqs[2:5])
    assert txt_freqs.dtype == torch.complex64
    assert torch.equal(vid_freqs, rope.pos_freqs[:2])


def test_safe_pack_tensor_if_large():
    from vllm_omni.diffusion import ipc

    orig_pack = MagicMock()
    ipc._pack_tensor_if_large = orig_pack

    apply_omni_model_specific_patches.cache_clear()
    apply_omni_model_specific_patches()

    cpu_tensor = torch.empty(1, device="cpu")
    tpu_tensor = MagicMock()
    tpu_tensor.cpu.return_value = cpu_tensor
    created = []

    res = ipc._pack_tensor_if_large(
        tpu_tensor, "stream", created, enable_registered_shm=True
    )

    tpu_tensor.cpu.assert_called_once_with()
    orig_pack.assert_called_once_with(
        cpu_tensor, "stream", created, enable_registered_shm=True
    )
    assert res is orig_pack.return_value


def test_omni_tpu_patches_configuration():
    # Call the undecorated body so the run-once cache is left alone, and
    # restore the process-global state it edits (env var).
    orig = torch.nn.functional.interpolate
    with patch.dict(os.environ):
        os.environ.pop("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", None)
        apply_omni_tpu_patches.__wrapped__()
        assert os.environ["VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT"] == "1800.0"
        assert torch.nn.functional.interpolate is orig


def test_startup_warmup_device_sync(monkeypatch):
    # The startup warmup must be followed by a device sync so that its pending
    # deferred TPU ops are not fused into the first request.
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    calls = []

    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda device=None: calls.append("sync")
    )
    monkeypatch.setattr(
        DiffusionEngine, "run_startup_warmup", lambda self: calls.append("warmup")
    )
    _patch_diffusion_startup_warmup_sync()

    DiffusionEngine.run_startup_warmup(object())
    assert calls == ["warmup", "sync"]


def _rms_norm(x, w, eps):
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * out).to(x.dtype)


def _reference_qk_norm_rope(qkv, wq, wk, cos, sin, eps, head_dim):
    # Semantics of the fused kernel: (T, 3C) in, repeated-pair tables.
    dim = qkv.shape[1] // 3
    swap = _pair_swap_matrix(head_dim, qkv.dtype, qkv.device)
    outs = []
    for j, w in ((0, wq), (1, wk)):
        y = _rms_norm(qkv[:, j * dim : (j + 1) * dim], w, eps)
        y = y.unflatten(1, (-1, head_dim))
        out = y * cos[:, None] + (y @ swap) * sin[:, None]
        outs.append(out.flatten(1))
    return tuple(outs)


class _FakeWanSelfAttention(torch.nn.Module):
    def __init__(self, dim=64, heads=2, head_dim=128):
        super().__init__()
        self.num_heads = self.num_kv_heads = heads
        self.head_dim = head_dim
        self.qkv = torch.nn.Linear(dim, 3 * heads * head_dim)
        self.norm_q = _MockRMSNorm(heads * head_dim)
        self.norm_k = _MockRMSNorm(heads * head_dim)
        with torch.no_grad():
            self.norm_q.weight.uniform_(0.5, 1.5)
            self.norm_k.weight.uniform_(0.5, 1.5)
        self.to_gate_compress = None
        self.to_out = torch.nn.Identity()
        self.dropout = torch.nn.Identity()

    def to_qkv(self, x):
        return self.qkv(x), None

    def attn(self, q, k, v, attn_metadata=None):
        return q * 3 + k * 2 + v


def test_wan_self_attention_fused_matches_native(monkeypatch):
    # The fused forward's glue (flattening, tables, value slice) around the
    # kernel reproduces vLLM-Omni's norm -> unflatten -> RoPE -> attention.
    torch.manual_seed(0)
    attn = _FakeWanSelfAttention()
    batch, seq, half = 2, 5, attn.head_dim // 2
    x = torch.randn(batch, seq, 64)
    cos, sin = torch.randn(1, seq, 1, half), torch.randn(1, seq, 1, half)
    monkeypatch.setattr(omni_rope, "_qk_norm_rope_op", lambda: _reference_qk_norm_rope)
    out = wan_self_attention_fused(attn, x, (cos, sin))

    qkv, _ = attn.to_qkv(x)
    q, k, v = qkv.split(attn.num_heads * attn.head_dim, dim=-1)
    shape = (attn.num_heads, attn.head_dim)
    q = _rms_norm(q, attn.norm_q.weight, 1e-6).unflatten(2, shape)
    k = _rms_norm(k, attn.norm_k.weight, 1e-6).unflatten(2, shape)
    q = apply_rotary_pair_swap(q, cos, sin)
    k = apply_rotary_pair_swap(k, cos, sin)
    want = attn.attn(q, k, v.unflatten(2, shape)).flatten(2, 3)
    torch.testing.assert_close(out, want)


def test_can_fuse_qk_norm_rope():
    attn = _FakeWanSelfAttention()
    rot = (torch.ones(1, 3, 1, 64), torch.ones(1, 3, 1, 64))
    bf16_x = torch.ones(1, 3, 64, dtype=torch.bfloat16)
    assert can_fuse_qk_norm_rope(attn, bf16_x, rot)
    assert not can_fuse_qk_norm_rope(attn, torch.ones(1, 3, 64), rot)
    assert not can_fuse_qk_norm_rope(attn, bf16_x, None)
    attn.norm_q = torch.nn.LayerNorm(8)  # e.g. DistributedRMSNorm under TP
    assert not can_fuse_qk_norm_rope(attn, bf16_x, rot)


def test_wan_self_attention_patch_falls_back_on_f32():
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanSelfAttention,
    )

    apply_omni_model_specific_patches()
    assert WanSelfAttention()(torch.ones(1, 3, 8)) == "native"


def test_sdpa_fallback_routes_to_tpu_sdpa(monkeypatch):
    # Upstream AttentionImpl.forward raises on out-of-tree platforms; after
    # the patch, sdpa_fallback (a plain SDPAImpl) runs PyTorch SDPA.
    def _raise_oot(self, query, key, value, attn_metadata=None):
        raise NotImplementedError("No forward implementation for platform: oot")

    monkeypatch.setattr(SDPAImpl, "forward", _raise_oot)
    fallback = SDPAImpl(num_heads=2, head_size=8, softmax_scale=0.3)
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 5, 2, 8) for _ in range(3))
    with pytest.raises(NotImplementedError):
        fallback.forward(q, k, v)

    apply_omni_model_specific_patches.cache_clear()
    apply_omni_model_specific_patches()
    assert getattr(SDPAImpl.forward, _SDPA_FALLBACK_PATCHED, False)
    assert SDPAImpl.forward is TpuSDPAImpl.forward

    keep = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    for mask in (None, keep.bool(), keep.float()):
        meta = None if mask is None else SimpleNamespace(attn_mask=mask)
        scores = torch.einsum("bqhd,bkhd->bhqk", q.double(), k.double()) * 0.3
        if mask is not None:
            scores = scores.masked_fill(~keep.bool()[:, None, None, :], -torch.inf)
        want = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.double())
        torch.testing.assert_close(
            fallback.forward(q, k, v, meta), want.float(), rtol=1e-5, atol=1e-5
        )

    # Installing again returns at the flag check, before the info log.
    infos = []
    monkeypatch.setattr("vllm_torchtpu.omni.patches.logger.info", infos.append)
    _patch_sdpa_fallback_dispatch()
    assert infos == []
    assert SDPAImpl.forward is TpuSDPAImpl.forward


_DIFFUSION_WORKER = "vllm_omni.diffusion.worker.diffusion_worker"


class _FakeWorkerProc:
    """Builds its stream device from self.gpu_id like the upstream loop."""

    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.queue = "queue"

    def _async_output_loop(self):
        self.stream_device = torch.device("cpu", self.gpu_id)
        self.seen_queue = self.queue


def _install_worker_proc(monkeypatch, cls):
    module = types.ModuleType(_DIFFUSION_WORKER)
    module.WorkerProc = cls
    monkeypatch.setitem(sys.modules, _DIFFUSION_WORKER, module)


@pytest.fixture
def worker_proc(monkeypatch):
    # A fresh subclass per test, so the patch never edits a shared class.
    cls = type("WorkerProc", (_FakeWorkerProc,), {})
    _install_worker_proc(monkeypatch, cls)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 1)
    return cls


def test_d2h_stream_uses_current_device_index(worker_proc):
    _patch_diffusion_worker_d2h_stream()

    worker = worker_proc(gpu_id=3)
    worker._async_output_loop()

    assert worker.stream_device == torch.device("cpu", 1)
    assert worker.seen_queue == "queue"
    assert worker.gpu_id == 3


def test_d2h_stream_patch_is_idempotent(worker_proc):
    orig_loop = worker_proc._async_output_loop

    _patch_diffusion_worker_d2h_stream()
    patched_loop = worker_proc._async_output_loop
    _patch_diffusion_worker_d2h_stream()

    assert patched_loop is not orig_loop
    assert getattr(patched_loop, _D2H_STREAM_PATCHED, False)
    assert worker_proc._async_output_loop is patched_loop


def test_d2h_stream_patch_targets_real_worker_proc():
    # Runs in a fresh interpreter because tests/omni mock vllm_omni in
    # sys.modules. Skipped when no importable vLLM-Omni is on the path.
    code = textwrap.dedent(
        """
        import sys

        try:
            from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
        except ImportError as e:
            print(e)
            sys.exit(77)

        from vllm_torchtpu.omni.patches import (
            _D2H_STREAM_PATCHED,
            _patch_diffusion_worker_d2h_stream,
        )

        orig_loop = WorkerProc._async_output_loop
        _patch_diffusion_worker_d2h_stream()
        assert WorkerProc._async_output_loop is not orig_loop
        assert getattr(WorkerProc._async_output_loop, _D2H_STREAM_PATCHED, False)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 77:
        pytest.skip(f"vllm_omni is not importable: {result.stdout.strip()}")
    assert result.returncode == 0, result.stdout + result.stderr


def test_ltx2_vocoder_cpu_init_patch(monkeypatch):
    calls = []

    def _load_component(
        component_cls, model, subfolder, *, local_files_only, dtype, revision
    ):
        calls.append((subfolder, local_files_only, dtype, revision))
        return torch.empty(0).device.type

    # A real module: the MagicMock-backed ltx2 package reports every
    # attribute as present, which would trip the idempotency guard.
    ltx2_components = types.ModuleType("ltx2_components")
    ltx2_components._load_component = _load_component
    monkeypatch.setattr(
        vllm_omni_mock.diffusion.models.ltx2, "ltx2_components", ltx2_components
    )

    apply_omni_model_specific_patches.cache_clear()
    apply_omni_model_specific_patches()
    patched = ltx2_components._load_component
    assert patched is not _load_component
    assert getattr(patched, _VOCODER_CPU_INIT_PATCHED, False)

    load_kwargs = {"local_files_only": True, "dtype": torch.float32, "revision": None}
    with torch.device("meta"):
        assert patched(object, "m", "vocoder", **load_kwargs) == "cpu"
        assert patched(object, "m", "vae", **load_kwargs) == "meta"
    assert calls == [
        ("vocoder", True, torch.float32, None),
        ("vae", True, torch.float32, None),
    ]

    _patch_ltx2_vocoder_cpu_init()
    assert ltx2_components._load_component is patched


def _upstream_restore_ltx_bwe_resampler_filter(vocoder):
    # Copy of vllm-omni v0.32.0rc1 ltx2_runtime._restore_ltx_bwe_resampler_filter
    # (ltx2_runtime.py:81-110); vllm_omni is mocked here, so the real module
    # cannot be imported.
    resampler = getattr(vocoder, "resampler", None)
    current_filter = getattr(resampler, "filter", None)
    ratio = getattr(resampler, "ratio", None)
    kernel_size = getattr(resampler, "kernel_size", None)
    if (
        not isinstance(current_filter, torch.Tensor)
        or not isinstance(ratio, int)
        or not isinstance(kernel_size, int)
    ):
        return
    if current_filter is getattr(resampler, "_ltx_restored_filter", None):
        return
    rolloff = 0.99
    lowpass_filter_width = 6
    width = math.ceil(lowpass_filter_width / rolloff)
    if kernel_size != 2 * width * ratio + 1:
        return
    device = current_filter.device
    time_axis = torch.arange(kernel_size, device=device, dtype=torch.float32)
    time_axis = (time_axis / ratio - width) * rolloff
    time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2).square()
    filter_value = (torch.sinc(time_axis) * window * rolloff / ratio).reshape(1, 1, -1)
    resampler.filter = filter_value
    resampler._ltx_restored_filter = filter_value


class _FakeBweResampler(torch.nn.Module):
    def __init__(self, device, kernel_size=29):
        super().__init__()
        self.ratio = 2
        self.kernel_size = kernel_size
        bf16 = torch.empty(1, 1, kernel_size, dtype=torch.bfloat16, device=device)
        self.register_buffer("filter", bf16, persistent=False)


def test_ltx2_bwe_filter_cpu_restore_patch(monkeypatch):
    calls = []  # (filter device seen by upstream, filter after upstream)

    def _restore(vocoder):
        device = vocoder.resampler.filter.device.type
        _upstream_restore_ltx_bwe_resampler_filter(vocoder)
        calls.append((device, vocoder.resampler.filter))

    ltx2_runtime = types.ModuleType("ltx2_runtime")
    ltx2_runtime._restore_ltx_bwe_resampler_filter = _restore
    ltx2_components = types.ModuleType("ltx2_components")
    ltx2_components._load_component = lambda *args, **kwargs: None
    ltx2 = vllm_omni_mock.diffusion.models.ltx2
    monkeypatch.setattr(ltx2, "ltx2_runtime", ltx2_runtime)
    monkeypatch.setattr(ltx2, "ltx2_components", ltx2_components)

    _patch_ltx2_vocoder_cpu_init()
    patched = ltx2_runtime._restore_ltx_bwe_resampler_filter
    assert patched is not _restore
    assert getattr(patched, _BWE_FILTER_CPU_RESTORE_PATCHED, False)
    _patch_ltx2_vocoder_cpu_init()
    assert ltx2_runtime._restore_ltx_bwe_resampler_filter is patched

    # A CPU filter goes straight to upstream.
    cpu_vocoder = types.SimpleNamespace(resampler=_FakeBweResampler("cpu"))
    patched(cpu_vocoder)
    want = cpu_vocoder.resampler.filter
    assert calls == [("cpu", want)] and want.dtype == torch.float32

    # Off CPU ("meta" stands in for TPU), upstream rebuilds on CPU and the
    # FP32 result moves to the filter's device as the restored filter.
    resampler = _FakeBweResampler("meta")
    vocoder = types.SimpleNamespace(resampler=resampler)
    patched(vocoder)
    assert calls[-1][0] == "cpu" and torch.equal(calls[-1][1], want)
    restored = resampler.filter
    assert restored.device.type == "meta" and restored.dtype == torch.float32
    assert restored.shape == want.shape
    assert resampler._buffers["filter"] is restored
    assert resampler._ltx_restored_filter is restored

    # The next decode hits upstream's identity check: nothing is rebuilt.
    patched(vocoder)
    assert calls[-1] == ("meta", restored)
    assert resampler.filter is restored
    assert resampler._ltx_restored_filter is restored

    # Upstream's early return (unexpected kernel size) leaves the filter alone.
    other = _FakeBweResampler("meta", kernel_size=31)
    original = other.filter
    patched(types.SimpleNamespace(resampler=other))
    assert calls[-1][0] == "cpu"
    assert other.filter is original
    assert not hasattr(other, "_ltx_restored_filter")


@pytest.fixture
def table_syncs(monkeypatch):
    """Records the patch's synchronize_tensors calls instead of syncing."""
    syncs = []

    def record(tensors, wait):
        syncs.append(tensors)

    monkeypatch.setattr("vllm_torchtpu.utils.synchronize_tensors", record)
    return syncs


@pytest.fixture
def ltx2(monkeypatch, table_syncs):
    """The mocked LTX-2 transformer module with _patch_ltx2_split_rope applied.

    The patch is installed on a fresh rope class; the module attributes it
    replaces are restored after the test.
    """
    for name, value in (
        ("LTX2AudioVideoRotaryPosEmbed", _mock_ltx2_rope_cls()),
        ("apply_split_rotary_emb", _mock_apply_split_rotary_emb),
        ("get_tensor_model_parallel_world_size", lambda: 1),
        ("get_tensor_model_parallel_rank", lambda: 0),
    ):
        monkeypatch.setattr(ltx2_transformer, name, value)
    _patch_ltx2_split_rope()
    return ltx2_transformer


def _ltx2_coords(batch=2, tokens=5):
    # (B, 3, T, 2) patch bounds, the input of LTX2AudioVideoRotaryPosEmbed.
    return 10 * torch.rand(batch, 3, tokens, 2)


def _upstream_rope_forward(rope, coords, device=None, out_dtype=None):
    # The unpatched forward: _mock_ltx2_rope_cls() builds a new class per call.
    return _mock_ltx2_rope_cls().forward(rope, coords, device, out_dtype)


@pytest.mark.parametrize(
    "x_dtype,table_dtype,tol",
    [
        (torch.float32, torch.float32, 1e-6),
        (torch.bfloat16, torch.bfloat16, None),
        (torch.bfloat16, None, None),  # f32 tables (no out_dtype)
    ],
)
def test_ltx2_split_rope_matches_upstream(ltx2, x_dtype, table_dtype, tol):
    # The patched rope and apply_split_rotary_emb give upstream's rotation
    # with its per-head (B, H, T, D/2) tables, in x.dtype.
    torch.manual_seed(0)
    rope = ltx2.LTX2AudioVideoRotaryPosEmbed(heads=3, head_dim=8)
    coords = _ltx2_coords()
    tables = rope(coords, out_dtype=table_dtype)
    assert tables[0].shape == (2, 1, 5, 3 * 8)
    x = torch.randn(2, 5, 3 * 8, dtype=x_dtype)
    out = ltx2.apply_split_rotary_emb(x, tables, head_dim=8)

    upstream_tables = _upstream_rope_forward(rope, coords, None, table_dtype)
    assert upstream_tables[0].shape == (2, 3, 5, 8 // 2)  # (B, H, T, D/2)
    want = _mock_apply_split_rotary_emb(x, upstream_tables, head_dim=8)
    assert out.dtype == x_dtype
    torch.testing.assert_close(out, want.to(x_dtype), rtol=tol, atol=tol)


def test_ltx2_split_rope_tensor_parallel_lanes(ltx2, monkeypatch):
    # _slice_rope_for_tp leaves the (B, 1, T, H * D) tables whole; rank 1 of
    # 2 rotates its two local heads with heads 2 and 3 of the tables.
    torch.manual_seed(0)
    rope = ltx2.LTX2AudioVideoRotaryPosEmbed(heads=4, head_dim=8)
    coords = _ltx2_coords()
    tables = rope(coords)
    cos, sin = _upstream_rope_forward(rope, coords)
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_rank", lambda: 1)
    x = torch.randn(2, 5, 2 * 8)
    out = ltx2.apply_split_rotary_emb(x, tables, head_dim=8)
    want = _mock_apply_split_rotary_emb(x, (cos[:, 2:], sin[:, 2:]), head_dim=8)
    torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-6)

    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_world_size", lambda: 3)
    with pytest.raises(ValueError, match="tensor-parallel"):
        ltx2.apply_split_rotary_emb(x, tables, head_dim=8)


def test_ltx2_split_rope_passes_other_tables_through(ltx2):
    torch.manual_seed(0)
    coords = _ltx2_coords()
    rope = ltx2.LTX2AudioVideoRotaryPosEmbed(rope_type="interleaved")
    upstream = _upstream_rope_forward(rope, coords)
    for got, want in zip(rope(coords), upstream, strict=True):
        assert torch.equal(got, want)

    # (B, H, T, D/2) tables, also with a single head, use upstream's function.
    for heads, head_dim in ((3, 8), (1, 24)):
        rope = ltx2.LTX2AudioVideoRotaryPosEmbed(heads=heads, head_dim=head_dim)
        tables = _upstream_rope_forward(rope, coords)
        x = torch.randn(2, 5, heads * head_dim)
        torch.testing.assert_close(
            ltx2.apply_split_rotary_emb(x, tables, head_dim=head_dim),
            _mock_apply_split_rotary_emb(x, tables, head_dim=head_dim),
            rtol=0,
            atol=0,
        )

    # Installing again keeps the single wrapper.
    forward = ltx2.LTX2AudioVideoRotaryPosEmbed.forward
    apply_split_rotary_emb = ltx2.apply_split_rotary_emb
    _patch_ltx2_split_rope()
    assert ltx2.LTX2AudioVideoRotaryPosEmbed.forward is forward
    assert ltx2.apply_split_rotary_emb is apply_split_rotary_emb


def test_ltx2_split_rope_tables_reused(ltx2, table_syncs):
    # The original forward runs, and the tables are synced, once per coords
    # tensor, coords version, device and out_dtype.
    torch.manual_seed(0)
    rope = ltx2.LTX2AudioVideoRotaryPosEmbed()
    coords = _ltx2_coords()
    cpu, bf16, f32 = torch.device("cpu"), torch.bfloat16, torch.float32
    tables = rope(coords, device=cpu, out_dtype=bf16)
    assert rope(coords, device=cpu, out_dtype=bf16) is tables
    assert (rope.calls, len(table_syncs)) == (1, 1)
    rope(coords.clone(), device=cpu, out_dtype=bf16)
    rope(coords, device=cpu, out_dtype=f32)
    stale = rope(coords, out_dtype=f32)
    assert rope(coords, out_dtype=f32) is stale
    assert (rope.calls, len(table_syncs)) == (4, 4)

    coords.add_(1.0)  # bumps the version counter
    fresh = rope(coords, out_dtype=f32)
    assert rope.calls == 5
    want = split_rope_natural_tables(*_upstream_rope_forward(rope, coords, None, f32))
    assert torch.equal(fresh[0], want[0]) and torch.equal(fresh[1], want[1])
    assert not torch.equal(fresh[0], stale[0])

    # Inference tensors have no version counter; they are keyed on identity.
    with torch.inference_mode():
        coords = _ltx2_coords()
    tables = rope(coords)
    assert rope(coords) is tables

    rope = ltx2.LTX2AudioVideoRotaryPosEmbed(rope_type="interleaved")
    rope(coords)
    rope(coords)
    assert rope.calls == 2


@pytest.mark.parametrize("inference", [False, True], ids=["grad", "inference"])
def test_ltx2_split_rope_tables_reused_for_coords_views(ltx2, table_syncs, inference):
    # Cross-attention ropes pass a fresh coords[:, 0:1, :] view every step,
    # in vLLM-Omni under torch.inference_mode(), where views have no _base.
    # Views of the same storage with the same layout and dtype hit; another
    # storage offset, shape or dtype, or another tensor with equal values,
    # rebuilds.
    torch.manual_seed(0)
    rope = ltx2.LTX2AudioVideoRotaryPosEmbed()
    ref = ltx2.LTX2AudioVideoRotaryPosEmbed()  # counts the reference calls
    ref.freqs = rope.freqs
    f32 = torch.float32

    def call(view):
        tables = rope(view, out_dtype=f32)
        want = split_rope_natural_tables(*_upstream_rope_forward(ref, view, None, f32))
        assert torch.equal(tables[0], want[0]) and torch.equal(tables[1], want[1])
        return tables

    with torch.inference_mode(inference):
        coords = _ltx2_coords()
        assert (coords[:, 0:1, :]._base is None) == inference
        tables = call(coords[:, 0:1, :])
        assert call(coords[:, 0:1, :]) is tables
        assert (rope.calls, len(table_syncs)) == (1, 1)

        other = coords.clone()  # equal values and layout, another storage
        for view in (
            coords[:, 1:2, :],  # another storage offset
            coords[:, 0:2, :],  # another shape at the same offset
            coords[:, 0:1, :].view(torch.int32),  # another dtype, same bytes
            other[:, 0:1, :],
        ):
            call(coords[:, 0:1, :])
            calls = rope.calls
            call(view)
            assert (rope.calls, len(table_syncs)) == (calls + 1, calls + 1)

    if not inference:
        # An in-place edit bumps the version counter that views share.
        stale = call(coords[:, 0:1, :])
        calls = rope.calls
        coords.add_(1.0)
        fresh = call(coords[:, 0:1, :])
        assert (rope.calls, len(table_syncs)) == (calls + 1, calls + 1)
        assert not torch.equal(fresh[0], stale[0])
