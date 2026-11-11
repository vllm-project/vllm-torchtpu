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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.pipelines.ltx2.vocoder import UpSample1d
from vllm_omni.diffusion import ipc
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan
from vllm_omni.diffusion.models.ltx2 import (
    ltx2_components,
    ltx2_runtime,
    ltx2_transformer,
)
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    ColumnParallelGELU,
    WanSelfAttention,
)
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

from vllm_torchtpu.omni import patches, register_omni_tpu_platform
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
    _SPLIT_ROPE_PATCHED,
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
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

pytestmark = pytest.mark.omni

# The patches rebind these upstream functions in place, and the first
# apply_omni_model_specific_patches() call in the session installs them for
# good. Tests that reinstall a patch start from the functions captured here,
# at collection, before any test has run.
_UPSTREAM_SDPA_FORWARD = SDPAImpl.forward
_UPSTREAM_ROPE_FORWARD = ltx2_transformer.LTX2AudioVideoRotaryPosEmbed.forward
_UPSTREAM_APPLY_SPLIT_ROTARY_EMB = ltx2_transformer.apply_split_rotary_emb
_UPSTREAM_RESTORE_BWE_FILTER = ltx2_runtime._restore_ltx_bwe_resampler_filter


def test_register_omni_tpu_platform(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 0)
    assert register_omni_tpu_platform() is None

    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 8)
    assert register_omni_tpu_platform() == "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def test_vllm_platform_accepts_omni_knobs():
    # vLLM-Omni's stages run EngineArgs.create_engine_config, which validates
    # the environment through vLLM's current_platform: TpuPlatform here, while
    # OmniTpuPlatform is only current_omni_platform.
    from vllm.platforms import current_platform

    assert isinstance(current_platform, TpuPlatform)
    assert not isinstance(current_platform, OmniTpuPlatform)
    with patch.dict(os.environ, {"VLLM_OMNI_FOO": "1"}, clear=True):
        current_platform.validate_environ(hard_fail=True)


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
    rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
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
    apply_omni_model_specific_patches()
    assert RotaryEmbeddingWan.forward is rotary_embedding_wan_forward
    # Interleaved half-dim calls use apply_rotary_pair_swap (bit-exact in bf16).
    rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    x = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    cos = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    sin = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(
        rope(x, cos, sin), rope.forward_native(x, cos, sin), rtol=0, atol=0
    )

    # Non-half-dim or Neox calls delegate to _forward_method.
    rope_fallback = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=False)
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
    rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    x = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    cos = torch.randn(4, 4, dtype=torch.bfloat16)
    sin = torch.randn(4, 4, dtype=torch.bfloat16)
    expected = rope.forward_native(x, cos[None, :, None], sin[None, :, None])
    torch.testing.assert_close(
        rotary_embedding_wan_forward(rope, x, cos, sin), expected, rtol=0, atol=0
    )


def test_tpu_flash_attention_backend(monkeypatch):
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
    apply_omni_model_specific_patches()
    # Skip __init__ (ColumnParallelLinear requires an initialized TP group).
    ffn = object.__new__(ColumnParallelGELU)
    torch.nn.Module.__init__(ffn)
    ffn.proj = torch.nn.Linear(8, 16)
    ffn.approximate = "tanh"
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


@pytest.mark.parametrize(
    "rope_cls_name",
    ["QwenEmbedRope", "QwenEmbedLayer3DRope"],
)
def test_qwen_image_rope_txt_freqs_clone(rope_cls_name):
    apply_omni_model_specific_patches()

    rope_cls = getattr(qwen_image_transformer, rope_cls_name)
    rope = rope_cls(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

    vid_freqs, txt_freqs = rope.forward([(1, 2, 2)], [3], torch.device("cpu"))

    assert not txt_freqs._is_view()
    assert (
        txt_freqs.untyped_storage().data_ptr()
        != rope.pos_freqs.untyped_storage().data_ptr()
    )
    # With scale_rope, text positions start at max(h // 2, w // 2) = 1.
    assert torch.equal(txt_freqs, rope.pos_freqs[1:4])
    assert txt_freqs.dtype == torch.complex64
    assert vid_freqs.shape == (4, 64)


@pytest.mark.parametrize(
    "nbytes",
    [1, 1 << 20],
    ids=["lt_1mb", "gt_1mb"],
)
def test_safe_pack_tensor_if_large(monkeypatch, nbytes):
    apply_omni_model_specific_patches()
    # Return the tensor that would be copied to SHM, so that the test sees
    # what the original function received for tensors above the threshold.
    monkeypatch.setattr(ipc, "_tensor_to_shm", lambda val, **kwargs: val)

    cpu_tensor = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    tpu_tensor = MagicMock()
    tpu_tensor.cpu.return_value = cpu_tensor
    created = []

    res = ipc._pack_tensor_if_large(
        tpu_tensor, "stream", created, enable_registered_shm=True
    )

    tpu_tensor.cpu.assert_called_once_with()
    assert res is cpu_tensor


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


def _make_wan_self_attention(dim=64, heads=2, head_dim=128) -> WanSelfAttention:
    attn = object.__new__(WanSelfAttention)
    torch.nn.Module.__init__(attn)
    attn.num_heads = attn.num_kv_heads = heads
    attn.head_dim = head_dim
    qkv = torch.nn.Linear(dim, 3 * heads * head_dim)
    attn.to_qkv = lambda x: (qkv(x), None)
    attn.norm_q = RMSNorm(heads * head_dim)
    attn.norm_k = RMSNorm(heads * head_dim)
    with torch.no_grad():
        attn.norm_q.weight.uniform_(0.5, 1.5)
        attn.norm_k.weight.uniform_(0.5, 1.5)
    attn.rotary_embedding = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    attn.to_gate_compress = None
    attn.to_out = torch.nn.Identity()
    attn.dropout = torch.nn.Identity()
    attn.attn = lambda q, k, v, attn_metadata=None: q * 3 + k * 2 + v
    return attn


def test_wan_self_attention_fused_matches_native(monkeypatch):
    # The fused forward's glue (flattening, tables, value slice) around the
    # kernel reproduces vLLM-Omni's norm -> unflatten -> RoPE -> attention.
    apply_omni_model_specific_patches()
    torch.manual_seed(0)
    attn = _make_wan_self_attention()
    batch, seq, half = 2, 5, attn.head_dim // 2
    x = torch.randn(batch, seq, 64)
    cos, sin = torch.randn(1, seq, 1, half), torch.randn(1, seq, 1, half)
    monkeypatch.setattr(omni_rope, "_qk_norm_rope_op", lambda: _reference_qk_norm_rope)
    out = wan_self_attention_fused(attn, x, (cos, sin))
    want = WanSelfAttention._tpu_orig_forward(attn, x, (cos, sin))
    torch.testing.assert_close(out, want)


def test_can_fuse_qk_norm_rope():
    attn = _make_wan_self_attention()
    rot = (torch.ones(1, 3, 1, 64), torch.ones(1, 3, 1, 64))
    bf16_x = torch.ones(1, 3, 64, dtype=torch.bfloat16)
    assert can_fuse_qk_norm_rope(attn, bf16_x, rot)
    assert not can_fuse_qk_norm_rope(attn, torch.ones(1, 3, 64), rot)
    assert not can_fuse_qk_norm_rope(attn, bf16_x, None)
    attn.norm_q = torch.nn.LayerNorm(8)  # e.g. DistributedRMSNorm under TP
    assert not can_fuse_qk_norm_rope(attn, bf16_x, rot)


def test_wan_self_attention_patch_falls_back_on_f32():
    apply_omni_model_specific_patches()
    attn = _make_wan_self_attention()
    x = torch.ones(1, 3, 64)
    rot = (torch.ones(1, 3, 1, 64), torch.ones(1, 3, 1, 64))
    torch.testing.assert_close(
        attn(x, rot),
        WanSelfAttention._tpu_orig_forward(attn, x, rot),
        rtol=0,
        atol=0,
    )


def test_sdpa_fallback_routes_to_tpu_sdpa(monkeypatch):
    # Upstream AttentionImpl.forward raises on out-of-tree platforms; after
    # the patch, sdpa_fallback (a plain SDPAImpl) runs PyTorch SDPA.
    monkeypatch.setattr(SDPAImpl, "forward", _UPSTREAM_SDPA_FORWARD)
    fallback = SDPAImpl(num_heads=2, head_size=8, softmax_scale=0.3)
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 5, 2, 8) for _ in range(3))
    with pytest.raises(NotImplementedError):
        fallback.forward(q, k, v)

    _patch_sdpa_fallback_dispatch()
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


@pytest.fixture
def output_loop_calls(monkeypatch):
    """Replaces the upstream WorkerProc._async_output_loop with a recorder.

    The upstream loop serves its output queue until shutdown; the recorder
    keeps only what the patch changes, the gpu_id the loop reads.
    """
    calls = []

    def _async_output_loop(self):
        calls.append((self.gpu_id, self.queue))

    monkeypatch.setattr(WorkerProc, "_async_output_loop", _async_output_loop)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 1)
    return calls


def _make_worker_proc(gpu_id):
    worker = object.__new__(WorkerProc)  # skips __init__, which builds the worker
    worker.gpu_id = gpu_id
    worker.queue = "queue"
    return worker


def test_d2h_stream_uses_current_device_index(output_loop_calls):
    _patch_diffusion_worker_d2h_stream()

    worker = _make_worker_proc(gpu_id=3)
    worker._async_output_loop()

    assert output_loop_calls == [(1, "queue")]
    assert worker.gpu_id == 3


def test_d2h_stream_patch_is_idempotent(output_loop_calls):
    orig_loop = WorkerProc._async_output_loop

    _patch_diffusion_worker_d2h_stream()
    patched_loop = WorkerProc._async_output_loop
    _patch_diffusion_worker_d2h_stream()

    assert patched_loop is not orig_loop
    assert getattr(patched_loop, _D2H_STREAM_PATCHED, False)
    assert WorkerProc._async_output_loop is patched_loop


def test_ltx2_vocoder_cpu_init_patch(monkeypatch):
    calls = []

    def _load_component(
        component_cls, model, subfolder, *, local_files_only, dtype, revision
    ):
        calls.append((subfolder, local_files_only, dtype, revision))
        return torch.empty(0).device.type

    # Upstream _load_component loads weights from the checkpoint. Both module
    # attributes the patch rebinds are restored after the test.
    monkeypatch.setattr(ltx2_components, "_load_component", _load_component)
    monkeypatch.setattr(
        ltx2_runtime,
        "_restore_ltx_bwe_resampler_filter",
        _UPSTREAM_RESTORE_BWE_FILTER,
    )

    _patch_ltx2_vocoder_cpu_init()
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


def _make_bwe_vocoder(device, kernel_size=None):
    # LTX2VocoderWithBWE's resampler, with its filter cast to BF16 on device
    # as at decode. kernel_size builds a Kaiser filter, which upstream's
    # restore skips.
    if kernel_size is None:
        resampler = UpSample1d(ratio=2, window_type="hann", persistent=False)
    else:
        resampler = UpSample1d(ratio=2, kernel_size=kernel_size, persistent=False)
    return SimpleNamespace(resampler=resampler.to(device, torch.bfloat16))


def test_ltx2_bwe_filter_cpu_restore_patch(monkeypatch):
    calls = []  # (filter device seen by upstream, filter after upstream)

    def _restore(vocoder):
        device = vocoder.resampler.filter.device.type
        _UPSTREAM_RESTORE_BWE_FILTER(vocoder)
        calls.append((device, vocoder.resampler.filter))

    monkeypatch.setattr(ltx2_runtime, "_restore_ltx_bwe_resampler_filter", _restore)
    monkeypatch.setattr(
        ltx2_components,
        "_load_component",
        ltx2_components._load_component,
    )

    _patch_ltx2_vocoder_cpu_init()
    patched = ltx2_runtime._restore_ltx_bwe_resampler_filter
    assert patched is not _restore
    assert getattr(patched, _BWE_FILTER_CPU_RESTORE_PATCHED, False)
    _patch_ltx2_vocoder_cpu_init()
    assert ltx2_runtime._restore_ltx_bwe_resampler_filter is patched

    # A CPU filter goes straight to upstream.
    cpu_vocoder = _make_bwe_vocoder("cpu")
    patched(cpu_vocoder)
    want = cpu_vocoder.resampler.filter
    assert calls == [("cpu", want)] and want.dtype == torch.float32

    # Off CPU ("meta" stands in for TPU), upstream rebuilds on CPU and the
    # FP32 result moves to the filter's device as the restored filter.
    vocoder = _make_bwe_vocoder("meta")
    resampler = vocoder.resampler
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
    other = _make_bwe_vocoder("meta", kernel_size=31).resampler
    original = other.filter
    patched(SimpleNamespace(resampler=other))
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
    """The LTX-2 transformer module with _patch_ltx2_split_rope reinstalled.

    The patch wraps the upstream functions again; the attributes it rebinds
    are restored after the test.
    """
    monkeypatch.setattr(
        ltx2_transformer.LTX2AudioVideoRotaryPosEmbed, "forward", _UPSTREAM_ROPE_FORWARD
    )
    for name, value in (
        ("apply_split_rotary_emb", _UPSTREAM_APPLY_SPLIT_ROTARY_EMB),
        ("get_tensor_model_parallel_world_size", lambda: 1),
        ("get_tensor_model_parallel_rank", lambda: 0),
    ):
        monkeypatch.setattr(ltx2_transformer, name, value)
    _patch_ltx2_split_rope()
    assert getattr(
        ltx2_transformer.LTX2AudioVideoRotaryPosEmbed.forward, _SPLIT_ROPE_PATCHED
    )
    return ltx2_transformer


def _ltx2_rope(rope_type="split", heads=3, head_dim=8):
    return ltx2_transformer.LTX2AudioVideoRotaryPosEmbed(
        dim=heads * head_dim, rope_type=rope_type, num_attention_heads=heads
    )


def _ltx2_coords(batch=2, tokens=5):
    # (B, 3, T, 2) patch bounds, the input of LTX2AudioVideoRotaryPosEmbed.
    return 10 * torch.rand(batch, 3, tokens, 2)


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
    rope = _ltx2_rope(heads=3, head_dim=8)
    coords = _ltx2_coords()
    tables = rope(coords, out_dtype=table_dtype)
    assert tables[0].shape == (2, 1, 5, 3 * 8)
    x = torch.randn(2, 5, 3 * 8, dtype=x_dtype)
    out = ltx2.apply_split_rotary_emb(x, tables, head_dim=8)

    upstream_tables = _UPSTREAM_ROPE_FORWARD(rope, coords, None, table_dtype)
    assert upstream_tables[0].shape == (2, 3, 5, 8 // 2)  # (B, H, T, D/2)
    want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, upstream_tables, head_dim=8)
    assert out.dtype == x_dtype
    torch.testing.assert_close(out, want.to(x_dtype), rtol=tol, atol=tol)


def test_ltx2_split_rope_tensor_parallel_lanes(ltx2, monkeypatch):
    # _slice_rope_for_tp leaves the (B, 1, T, H * D) tables whole; rank 1 of
    # 2 rotates its two local heads with heads 2 and 3 of the tables.
    torch.manual_seed(0)
    rope = _ltx2_rope(heads=4, head_dim=8)
    coords = _ltx2_coords()
    tables = rope(coords)
    cos, sin = _UPSTREAM_ROPE_FORWARD(rope, coords)
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_rank", lambda: 1)
    x = torch.randn(2, 5, 2 * 8)
    out = ltx2.apply_split_rotary_emb(x, tables, head_dim=8)
    want = _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, (cos[:, 2:], sin[:, 2:]), head_dim=8)
    torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-6)

    monkeypatch.setattr(ltx2, "get_tensor_model_parallel_world_size", lambda: 3)
    with pytest.raises(ValueError, match="tensor-parallel"):
        ltx2.apply_split_rotary_emb(x, tables, head_dim=8)


def test_ltx2_split_rope_passes_other_tables_through(ltx2):
    torch.manual_seed(0)
    coords = _ltx2_coords()
    rope = _ltx2_rope(rope_type="interleaved")
    upstream = _UPSTREAM_ROPE_FORWARD(rope, coords)
    for got, want in zip(rope(coords), upstream, strict=True):
        assert torch.equal(got, want)

    # (B, H, T, D/2) tables, also with a single head, use upstream's function.
    for heads, head_dim in ((3, 8), (1, 24)):
        rope = _ltx2_rope(heads=heads, head_dim=head_dim)
        tables = _UPSTREAM_ROPE_FORWARD(rope, coords)
        x = torch.randn(2, 5, heads * head_dim)
        torch.testing.assert_close(
            ltx2.apply_split_rotary_emb(x, tables, head_dim=head_dim),
            _UPSTREAM_APPLY_SPLIT_ROTARY_EMB(x, tables, head_dim=head_dim),
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
    rope = _ltx2_rope()
    coords = _ltx2_coords()
    cpu, bf16, f32 = torch.device("cpu"), torch.bfloat16, torch.float32
    tables = rope(coords, device=cpu, out_dtype=bf16)
    assert rope(coords, device=cpu, out_dtype=bf16) is tables
    assert len(table_syncs) == 1
    rope(coords.clone(), device=cpu, out_dtype=bf16)
    rope(coords, device=cpu, out_dtype=f32)
    stale = rope(coords, out_dtype=f32)
    assert rope(coords, out_dtype=f32) is stale
    assert len(table_syncs) == 4

    coords.add_(1.0)  # bumps the version counter
    fresh = rope(coords, out_dtype=f32)
    assert len(table_syncs) == 5
    want = split_rope_natural_tables(*_UPSTREAM_ROPE_FORWARD(rope, coords, None, f32))
    assert torch.equal(fresh[0], want[0]) and torch.equal(fresh[1], want[1])
    assert not torch.equal(fresh[0], stale[0])

    # Inference tensors have no version counter; they are keyed on identity.
    with torch.inference_mode():
        coords = _ltx2_coords()
    tables = rope(coords)
    assert rope(coords) is tables

    # Interleaved RoPE is not cached.
    rope = _ltx2_rope(rope_type="interleaved")
    assert rope(coords)[0] is not rope(coords)[0]


@pytest.mark.parametrize("inference", [False, True], ids=["grad", "inference"])
def test_ltx2_split_rope_tables_reused_for_coords_views(ltx2, table_syncs, inference):
    # Cross-attention ropes pass a fresh coords[:, 0:1, :] view every step,
    # in vLLM-Omni under torch.inference_mode(), where views have no _base.
    # Views of the same storage with the same layout and dtype hit; another
    # storage offset, shape or dtype, or another tensor with equal values,
    # rebuilds.
    torch.manual_seed(0)
    rope = _ltx2_rope()
    f32 = torch.float32

    def call(view):
        tables = rope(view, out_dtype=f32)
        want = split_rope_natural_tables(*_UPSTREAM_ROPE_FORWARD(rope, view, None, f32))
        assert torch.equal(tables[0], want[0]) and torch.equal(tables[1], want[1])
        return tables

    with torch.inference_mode(inference):
        coords = _ltx2_coords()
        assert (coords[:, 0:1, :]._base is None) == inference
        tables = call(coords[:, 0:1, :])
        assert call(coords[:, 0:1, :]) is tables
        assert len(table_syncs) == 1

        other = coords.clone()  # equal values and layout, another storage
        for view in (
            coords[:, 1:2, :],  # another storage offset
            coords[:, 0:2, :],  # another shape at the same offset
            coords[:, 0:1, :].view(torch.int32),  # another dtype, same bytes
            other[:, 0:1, :],
        ):
            call(coords[:, 0:1, :])
            syncs = len(table_syncs)
            call(view)
            assert len(table_syncs) == syncs + 1

    if not inference:
        # An in-place edit bumps the version counter that views share.
        stale = call(coords[:, 0:1, :])
        syncs = len(table_syncs)
        coords.add_(1.0)
        fresh = call(coords[:, 0:1, :])
        assert len(table_syncs) == syncs + 1
        assert not torch.equal(fresh[0], stale[0])


def test_supports_talker_mtp_graph_capture_disabled():
    assert OmniTpuPlatform.supports_talker_mtp_graph_capture() is False
    assert OmniTpuPlatform().supports_talker_mtp_graph_capture() is False


def _non_ring_reference_bshd(q, k, v, scale):
    """Single-device f32 softmax attention over (B, S, H, D) inputs."""
    import jax
    import jax.numpy as jnp

    heads_per_kv = q.shape[2] // k.shape[2]
    k_rep = jnp.repeat(k, heads_per_kv, axis=2).astype(jnp.float32)
    v_rep = jnp.repeat(v, heads_per_kv, axis=2).astype(jnp.float32)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k_rep) * scale
    probs = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, v_rep).astype(q.dtype)


_HOST_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
# Tests whose cases share one CPU-only child pytest run.
_CPU_CHILD_TESTS = (
    "test_ring_flash_attention_bhsd_matches_non_ring",
    "test_ring_ulysses_2d_matches_non_ring_reference",
)
_cpu_child_run: subprocess.CompletedProcess[str] | None = None


def _cpu_devices_or_rerun(request, count):
    """Returns ``count`` JAX CPU devices, or None once the child run passed.

    XLA reads the host device count only when JAX starts, so this process
    cannot raise it after the fact. With too few CPU devices, every case of
    ``_CPU_CHILD_TESTS`` runs once in a single child pytest whose JAX uses 8
    CPU devices only; each case here then checks that its own copy passed.
    The child skips conftest.py (``--noconftest``): its autouse fixtures would
    open the TPU this process already holds. The env of this process is not
    changed.
    """
    global _cpu_child_run
    import jax

    devices = jax.devices("cpu")
    if len(devices) >= count:
        return devices[:count]
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if (
        os.environ.get("JAX_PLATFORMS") == "cpu"
        and _HOST_DEVICE_COUNT_FLAG in xla_flags
    ):
        pytest.skip(f"Requires {count} CPU devices")
    if _cpu_child_run is None:
        kept = [
            t for t in xla_flags.split() if not t.startswith(_HOST_DEVICE_COUNT_FLAG)
        ]
        env = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": " ".join([*kept, f"{_HOST_DEVICE_COUNT_FLAG}=8"]),
        }
        _cpu_child_run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "--color=no",
                "-p",
                "no:cacheprovider",
                "-q",
                "-rA",
                str(request.node.path),
                "-k",
                " or ".join(_CPU_CHILD_TESTS),
            ],
            cwd=request.config.rootpath,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    proc = _cpu_child_run
    assert proc.returncode == 0 and f"PASSED {request.node.nodeid}" in proc.stdout, (
        f"{proc.stdout}\n{proc.stderr}"
    )
    return None


def test_ring_lse_helpers_match_single_pass():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.diffusion_attention.kernel import (
        RING_LSE_INIT,
        merge_partials,
        partial_to_lse,
        residual_reference_bhsd,
    )

    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)
    b, h_q, h_kv, s_q, s_kv, d = 2, 4, 2, 32, 64, 16
    scale = 1.0 / math.sqrt(d)
    q = jax.random.normal(kq, (b, h_q, s_q, d), jnp.bfloat16)
    k = jax.random.normal(kk, (b, h_kv, s_kv, d), jnp.bfloat16)
    v = jax.random.normal(kv, (b, h_kv, s_kv, d), jnp.bfloat16)

    o_full_un, m_full, l_full = residual_reference_bhsd(q, k, v, scale=scale)
    o_full, lse_full = partial_to_lse(o_full_un, m_full, l_full)

    # Merge 4 KV chunks and compare against single-pass attention.
    o_acc = jnp.zeros_like(o_full)
    lse_acc = jnp.full_like(lse_full, RING_LSE_INIT)
    for k_chunk, v_chunk in zip(jnp.split(k, 4, axis=2), jnp.split(v, 4, axis=2)):
        o_un, m_c, l_c = residual_reference_bhsd(q, k_chunk, v_chunk, scale=scale)
        o_c, lse_c = partial_to_lse(o_un, m_c, l_c)
        o_acc, lse_acc = merge_partials(o_acc, lse_acc, o_c, lse_c)

    np.testing.assert_allclose(
        np.asarray(o_acc), np.asarray(o_full), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(lse_acc), np.asarray(lse_full), rtol=1e-5, atol=1e-5
    )

    # Merging with an empty partial (l == 0) leaves the accumulator unchanged.
    o_empty, lse_empty = partial_to_lse(
        jnp.zeros_like(o_full), jnp.zeros_like(m_full), jnp.zeros_like(l_full)
    )
    o_after, lse_after = merge_partials(o_full, lse_full, o_empty, lse_empty)
    np.testing.assert_allclose(
        np.asarray(o_after), np.asarray(o_full), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(lse_after), np.asarray(lse_full), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("ring_size,h_q,h_kv", [(1, 4, 4), (2, 4, 4), (4, 4, 2)])
def test_ring_flash_attention_bhsd_matches_non_ring(request, ring_size, h_q, h_kv):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec

    from vllm_torchtpu.kernels.diffusion_attention.kernel import (
        residual_reference_bhsd,
        ring_flash_attention_bhsd,
    )

    cpu_devices = _cpu_devices_or_rerun(request, ring_size)
    if cpu_devices is None:
        return
    mesh = Mesh(np.array(cpu_devices), ("ring",))
    b, s_per_rank, d = 2, 16, 16
    s_total = s_per_rank * ring_size
    scale = 1.0 / math.sqrt(d)

    kq, kk, kv = jax.random.split(jax.random.PRNGKey(10 + ring_size), 3)
    q_full = jax.random.normal(kq, (b, s_total, h_q, d), jnp.bfloat16)
    k_full = jax.random.normal(kk, (b, s_total, h_kv, d), jnp.bfloat16)
    v_full = jax.random.normal(kv, (b, s_total, h_kv, d), jnp.bfloat16)
    want = _non_ring_reference_bshd(q_full, k_full, v_full, scale)

    q_sharded = q_full.reshape(b, ring_size, s_per_rank, h_q, d).transpose(
        1, 0, 3, 2, 4
    )
    k_sharded = k_full.reshape(b, ring_size, s_per_rank, h_kv, d).transpose(
        1, 0, 3, 2, 4
    )
    v_sharded = v_full.reshape(b, ring_size, s_per_rank, h_kv, d).transpose(
        1, 0, 3, 2, 4
    )
    spec = PartitionSpec("ring", None, None, None, None)

    def _run(q_s, k_s, v_s):
        out = ring_flash_attention_bhsd(
            q_s[0],
            k_s[0],
            v_s[0],
            ring_axis="ring",
            scale=scale,
            residual_fn=residual_reference_bhsd,
        )
        return out[None]

    got_sharded = jax.jit(
        jax.shard_map(
            _run,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
    )(q_sharded, k_sharded, v_sharded)
    got = got_sharded.transpose(1, 0, 3, 2, 4).reshape(b, s_total, h_q, d)
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    "cfg_size,ring_size,uly_size,h_q,h_kv",
    [
        (2, 2, 2, 4, 4),
        (2, 2, 2, 4, 2),
        (2, 4, 1, 4, 2),
        (1, 2, 2, 4, 4),
        (2, 2, 1, 4, 2),
    ],
)
def test_ring_ulysses_2d_matches_non_ring_reference(
    request, cfg_size, ring_size, uly_size, h_q, h_kv
):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec

    from vllm_torchtpu.kernels.diffusion_attention.kernel import (
        residual_reference_bhsd,
        ring_ulysses_flash_attention_bshd,
    )

    n_dev = cfg_size * ring_size * uly_size
    cpu_devices = _cpu_devices_or_rerun(request, n_dev)
    if cpu_devices is None:
        return
    mesh = Mesh(
        np.array(cpu_devices).reshape(cfg_size, ring_size, uly_size),
        ("batch", "ring", "uly"),
    )
    b, s_per_sp, d = 1, 12, 16
    sp_size = ring_size * uly_size
    s_total = s_per_sp * sp_size
    scale = 1.0 / math.sqrt(d)

    kq, kk, kv = jax.random.split(jax.random.PRNGKey(42), 3)
    q_full = jax.random.normal(kq, (cfg_size, b, s_total, h_q, d), jnp.bfloat16)
    k_full = jax.random.normal(kk, (cfg_size, b, s_total, h_kv, d), jnp.bfloat16)
    v_full = jax.random.normal(kv, (cfg_size, b, s_total, h_kv, d), jnp.bfloat16)

    want_branches = [
        _non_ring_reference_bshd(q_full[c], k_full[c], v_full[c], scale)
        for c in range(cfg_size)
    ]
    want = jnp.stack(want_branches, axis=0)

    def _shard_sp(x):
        return x.reshape(
            cfg_size, b, ring_size, uly_size, s_per_sp, x.shape[3], d
        ).transpose(0, 2, 3, 1, 4, 5, 6)

    q_in = _shard_sp(q_full)
    k_in = _shard_sp(k_full)
    v_in = _shard_sp(v_full)
    spec = PartitionSpec("batch", "ring", "uly", None, None, None, None)

    def _body(q_loc, k_loc, v_loc):
        out = ring_ulysses_flash_attention_bshd(
            q_loc[0, 0, 0],
            k_loc[0, 0, 0],
            v_loc[0, 0, 0],
            ring_axis="ring",
            uly_axis="uly" if uly_size > 1 else None,
            scale=scale,
            residual_fn=residual_reference_bhsd,
        )
        return out[None, None, None]

    got_sharded = jax.jit(
        jax.shard_map(
            _body,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
    )(q_in, k_in, v_in)
    got = got_sharded.transpose(0, 3, 1, 2, 4, 5, 6).reshape(
        cfg_size, b, s_total, h_q, d
    )
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_tpu_ring_ulysses_pallas_stablehlo_export():
    import jax
    import jax.export
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec
    from torch_tpu._internal.pallas.pallas import _PallasDevice

    from vllm_torchtpu.kernels.diffusion_attention.kernel import (
        _RING_MEASURED_BLOCK_OVERRIDES,
        ring_ulysses_flash_attention_bshd,
    )

    assert _RING_MEASURED_BLOCK_OVERRIDES[(7, 37800, 37800)] == {
        "block_q": 6400,
        "block_kv": 2048,
        "block_kv_compute": 2048,
    }
    assert _RING_MEASURED_BLOCK_OVERRIDES[(7, 18900, 18900)] == {
        "block_q": 9472,
        "block_kv": 1024,
        "block_kv_compute": 1024,
    }
    assert _RING_MEASURED_BLOCK_OVERRIDES[(7, 55800, 55800)] == {
        "block_q": 6400,
        "block_kv": 2048,
        "block_kv_compute": 2048,
    }

    devs = [_PallasDevice(id=i, process_index=i, device_kind="TPU7x") for i in range(8)]
    mesh = Mesh(np.array(devs).reshape(2, 2, 2), ("batch", "ring", "uly"))
    spec = PartitionSpec("batch", "ring", "uly", None, None, None, None)
    scale = 1.0 / math.sqrt(128)

    def _body(q, k, v):
        out = ring_ulysses_flash_attention_bshd(
            q[0, 0, 0],
            k[0, 0, 0],
            v[0, 0, 0],
            ring_axis="ring",
            uly_axis="uly",
            scale=scale,
        )
        return out[None, None, None]

    fn = jax.jit(
        jax.shard_map(
            _body,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
    )
    shape = jax.ShapeDtypeStruct((2, 2, 2, 1, 1024, 4, 128), jnp.bfloat16)
    exported = jax.export.export(
        fn,
        platforms=["tpu"],
        disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
    )(shape, shape, shape)
    mlir = str(exported.mlir_module())
    assert "stablehlo.collective_permute" in mlir
    assert "stablehlo.all_to_all" in mlir
    assert "stablehlo.all_gather" not in mlir
    assert "tpu_custom_call" in mlir

    from torch_tpu._internal.pallas import pallas as pallas_impl

    from vllm_torchtpu.omni.attention import _ring_ulysses_attention_op

    _ring_ulysses_attention_op.cache_clear()
    orig_tp = pallas_impl.torch_placeholder
    registered_names: list[str] = []
    with (
        patch("vllm_torchtpu.omni.attention.pallas.get_pallas_mesh", return_value=mesh),
        patch.object(pallas_impl, "_validate_pallas_mesh_world_size"),
        patch.object(
            pallas_impl.tpu_torch_pallas, "lookup_custom_kernel", return_value=False
        ),
        patch.object(
            pallas_impl.tpu_torch_pallas,
            "register_custom_kernel",
            side_effect=lambda name, *args, **kw: registered_names.append(name),
        ),
        patch.object(
            pallas_impl.tpu_torch_pallas,
            "call_custom_kernel",
            side_effect=lambda name, fp, inputs, output_shapes, **kw: output_shapes,
        ),
        patch.object(
            pallas_impl,
            "torch_placeholder",
            side_effect=lambda aval, mesh=None, partition_spec=None, device="cpu": (
                orig_tp(aval, mesh=mesh, partition_spec=partition_spec, device="cpu")
            ),
        ),
    ):
        op = _ring_ulysses_attention_op(2, 2, 2, True)
        q = torch.empty((1, 1, 1, 1, 1024, 4, 128), dtype=torch.bfloat16)
        out = op(q, q, q, scale)
        assert out.shape == q.shape
    assert registered_names == ["pallas::diffusion_ring_ulysses_attn_2_2_2"]
    _ring_ulysses_attention_op.cache_clear()


def _make_ring_attention_layer(
    ring_runner=None, strategy=None, softmax_scale=0.25, causal=False
):
    layer = object.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.use_ring = ring_runner is not None
    layer.ring_runner = ring_runner
    layer._no_parallel_strategy = object()
    layer.parallel_strategy = strategy or layer._no_parallel_strategy
    layer.softmax_scale = softmax_scale
    layer.causal = causal
    layer._explicit_cross_contract = False
    layer.attention = None
    layer._disable_kv_quant = True
    layer._kv_cache_dtype = None
    layer._kv_cache_skip_steps = None
    layer._kv_cache_skip_layers = None
    layer.paged_kv_cache_role = None
    return layer


def test_ring_attention_patch_and_dispatch(monkeypatch):
    from vllm_torchtpu.omni.attention import tpu_ring_flash_attention
    from vllm_torchtpu.omni.patches import _patch_ring_attention

    def _orig_run(
        self, query, key, value, attn_metadata=None, softmax_scale=None, causal=False
    ):
        return "native_ring"

    def _orig_forward_impl(self, query, key, value, attn_metadata=None):
        if self.use_ring and self.ring_runner is not None:
            return self.ring_runner.run_attention(
                query,
                key,
                value,
                attn_metadata,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )
        return "native_attn"

    monkeypatch.setattr(RingParallelAttention, "run_attention", _orig_run)
    monkeypatch.setattr(Attention, "_forward_impl", _orig_forward_impl)
    _patch_ring_attention()
    calls = []

    def _fake_op(outer_size, ring_size, uly_size, fuse_ulysses):
        def _run(q, k, v, scale):
            calls.append((outer_size, ring_size, uly_size, fuse_ulysses, scale))
            return q.clone()

        return _run

    monkeypatch.setattr(
        "vllm_torchtpu.omni.attention._ring_ulysses_attention_op", _fake_op
    )

    sp_group = SimpleNamespace(ring_world_size=2, ulysses_world_size=2)
    runner = RingParallelAttention(sp_group, attn_backend_pref="TPU_FLASH_ATTN")
    strategy = SimpleNamespace(name="ulysses", _scatter_idx=2, _gather_idx=1)
    layer = _make_ring_attention_layer(
        ring_runner=runner,
        strategy=strategy,
        softmax_scale=0.125,
    )
    monkeypatch.setattr(
        layer, "_get_active_parallel_strategy", lambda: layer.parallel_strategy
    )
    monkeypatch.setattr(layer, "is_paged_kv_active", lambda: False)

    q = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)

    # 1. Hybrid Ring+Ulysses Attention._forward_impl fuses Ulysses all-to-all.
    out = layer._forward_impl(q, k, v)
    assert out.shape == q.shape
    assert calls == [(1, 2, 2, True, 0.125)]

    # 2. Direct RingParallelAttention.run_attention runs 1D ring (fuse_ulysses=False).
    calls.clear()
    out_ring = runner.run_attention(q, k, v, softmax_scale=0.25)
    assert out_ring.shape == q.shape
    assert calls == [(1, 2, 2, False, 0.25)]

    # 3. Calls that do not fit the TPU ring kernel (causal=True, float32,
    # attn_mask, joint_query, or SDPA pref) raise at ring_world_size > 1: the
    # vLLM-Omni P2P ring fallback is not supported on TPU.
    calls.clear()
    with patch.object(patches.logger, "warning_once") as mock_warn:
        with pytest.raises(NotImplementedError, match="causal=True"):
            runner.run_attention(q, k, v, causal=True)
        with pytest.raises(NotImplementedError, match="dtype=torch.float32"):
            runner.run_attention(q.float(), k.float(), v.float())
        masked_meta = SimpleNamespace(attn_mask=torch.ones(1, 16, dtype=torch.bool))
        with pytest.raises(NotImplementedError, match="metadata=SimpleNamespace"):
            runner.run_attention(q, k, v, attn_metadata=masked_meta)
        # joint_meta: the fused path logs why it cannot fuse Ulysses, then the
        # 1-D ring runner raises.
        joint_meta = SimpleNamespace(joint_query=q, extra={})
        with pytest.raises(NotImplementedError, match="not supported on TPU"):
            layer._forward_impl(q, k, v, attn_metadata=joint_meta)
        sdpa_runner = RingParallelAttention(sp_group, attn_backend_pref="TPU_SDPA")
        with pytest.raises(NotImplementedError, match="attn_backend_pref="):
            sdpa_runner.run_attention(q, k, v)
    assert calls == []
    warned = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert not any("does not fit the TPU Pallas ring kernel" in w for w in warned)
    assert sum("cannot fuse Ulysses" in w for w in warned) == 1

    # 3b. With ring_world_size == 1 a call that does not fit the kernel still
    # logs once and runs upstream (no P2P ring exchange happens).
    single_runner = RingParallelAttention(
        SimpleNamespace(ring_world_size=1, ulysses_world_size=2),
        attn_backend_pref="TPU_FLASH_ATTN",
    )
    with patch.object(patches.logger, "warning_once") as mock_warn:
        assert single_runner.run_attention(q, k, v, causal=True) == "native_ring"
    assert calls == []
    warned = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert sum("does not fit the TPU Pallas ring kernel" in w for w in warned) == 1

    # 3c. Pure ring (ulysses_world_size == 1): the ring runner is the TPU kernel
    # path, so no "cannot fuse Ulysses" warning, whatever the strategy name.
    pure_runner = RingParallelAttention(
        SimpleNamespace(ring_world_size=2, ulysses_world_size=1),
        attn_backend_pref="TPU_FLASH_ATTN",
    )
    with patch.object(patches.logger, "warning_once") as mock_warn:
        for pure_strategy in (strategy, SimpleNamespace(name="ring")):
            pure_layer = _make_ring_attention_layer(
                ring_runner=pure_runner,
                strategy=pure_strategy,
                softmax_scale=0.125,
            )
            monkeypatch.setattr(
                pure_layer,
                "_get_active_parallel_strategy",
                lambda pure_layer=pure_layer: pure_layer.parallel_strategy,
            )
            monkeypatch.setattr(pure_layer, "is_paged_kv_active", lambda: False)
            assert pure_layer._forward_impl(q, k, v).shape == q.shape
    assert calls == [(1, 2, 1, False, 0.125)] * 2
    warned = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert not any("cannot fuse Ulysses" in w for w in warned)
    calls.clear()

    # 4. Non-divisible GQA heads raise ValueError in tpu_ring_flash_attention.
    with pytest.raises(ValueError, match="num_heads"):
        tpu_ring_flash_attention(
            q, torch.randn(1, 16, 3, 16, dtype=torch.bfloat16), v, 0.25, ring_size=2
        )

    # 5. Re-applying _patch_ring_attention is idempotent.
    run_fn = RingParallelAttention.run_attention
    fwd_fn = Attention._forward_impl
    _patch_ring_attention()
    assert RingParallelAttention.run_attention is run_fn
    assert Attention._forward_impl is fwd_fn


def test_device_id_rank_layout_c2r2u2_and_c2r4u1(
    monkeypatch, _restore_parallel_state_device_order
):
    import numpy as np
    from torch_tpu._internal.pallas.pallas import _PallasDevice
    from vllm_omni.diffusion.distributed import group_coordinator, parallel_state

    import vllm_torchtpu.omni.patches as omni_patches
    from vllm_torchtpu.omni.attention import (
        _get_device_ordered_pallas_mesh,
        compute_device_id_group_layout,
        ranks_ordered_by_device_id,
        validate_ring_cfg_lanes,
    )

    device_ids = [0, 1, 4, 5, 6, 7, 2, 3]
    assert ranks_ordered_by_device_id(device_ids) == [0, 1, 6, 7, 2, 3, 4, 5]

    c2r2u2 = compute_device_id_group_layout(
        device_ids, cfg_size=2, ring_size=2, uly_size=2
    )
    assert c2r2u2["cfg_groups"] == [[0, 2], [1, 3], [6, 4], [7, 5]]
    assert c2r2u2["sp_groups"] == [[0, 1, 6, 7], [2, 3, 4, 5]]
    assert c2r2u2["ulysses_groups"] == [[0, 1], [6, 7], [2, 3], [4, 5]]
    assert c2r2u2["ring_groups"] == [[0, 6], [1, 7], [2, 4], [3, 5]]
    assert [[device_ids[r] for r in g] for g in c2r2u2["ring_groups"]] == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]
    assert [[device_ids[r] for r in g] for g in c2r2u2["ulysses_groups"]] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]

    c2r4u1 = compute_device_id_group_layout(
        device_ids, cfg_size=2, ring_size=4, uly_size=1
    )
    assert c2r4u1["cfg_groups"] == [[0, 2], [1, 3], [6, 4], [7, 5]]
    assert c2r4u1["sp_groups"] == [[0, 1, 6, 7], [2, 3, 4, 5]]
    assert c2r4u1["ulysses_groups"] == [[0], [1], [6], [7], [2], [3], [4], [5]]
    assert c2r4u1["ring_groups"] == [[0, 1, 6, 7], [2, 3, 4, 5]]
    assert [[device_ids[r] for r in g] for g in c2r4u1["ring_groups"]] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]

    # Verify patched RankGenerator.get_ranks applies the device-ID permutation.
    omni_patches._patch_parallel_state_device_order()
    monkeypatch.setattr(omni_patches, "_DEVICE_IDS_BY_RANK_OVERRIDE", device_ids)
    rg = parallel_state.RankGenerator(1, 4, 1, 2, 1, order="tp-sp-pp-cfg-dp")
    assert rg.get_ranks("sp") == [[0, 1, 6, 7], [2, 3, 4, 5]]
    assert rg.get_ranks("cfg") == [[0, 2], [1, 3], [6, 4], [7, 5]]

    # Verify _get_device_ordered_pallas_mesh sorts devices into 0..7.
    import jax.sharding

    rank_ordered_devs = [
        _PallasDevice(id=dev_id, process_index=r, device_kind="TPU7x")
        for r, dev_id in enumerate(device_ids)
    ]
    raw_mesh = jax.sharding.Mesh(
        np.array(rank_ordered_devs, dtype=object).reshape(2, 2, 2),
        ("batch", "ring", "uly"),
    )
    with patch(
        "vllm_torchtpu.omni.attention.pallas.get_pallas_mesh", return_value=raw_mesh
    ):
        ordered_mesh = _get_device_ordered_pallas_mesh(
            ("batch", "ring", "uly"), (2, 2, 2)
        )
    assert [int(d.id) for d in ordered_mesh.devices.flat] == list(range(8))

    validate_ring_cfg_lanes(
        cfg_groups=c2r2u2["cfg_groups"],
        ring_groups=c2r2u2["ring_groups"],
        cfg_lane_by_rank=c2r2u2["cfg_lane_by_rank"],
        device_ids_by_rank=device_ids,
        mesh_device_ids=ordered_mesh,
        outer_size=2,
        ring_size=2,
        uly_size=2,
    )
    validate_ring_cfg_lanes(
        cfg_groups=c2r4u1["cfg_groups"],
        ring_groups=c2r4u1["ring_groups"],
        cfg_lane_by_rank=c2r4u1["cfg_lane_by_rank"],
        device_ids_by_rank=device_ids,
        mesh_device_ids=list(range(8)),
        outer_size=2,
        ring_size=4,
        uly_size=1,
    )

    # Verify GroupCoordinator.all_gather reorders unsorted group [6, 4].
    t_r4 = torch.tensor([40.0, 41.0])
    t_r6 = torch.tensor([60.0, 61.0])

    def _fake_orig_ag(self, input_, dim=0, separate_tensors=False, group=None):
        assert separate_tensors is True
        return [t_r4, t_r6]

    fake_gc = SimpleNamespace(ranks=[6, 4], device_group="cfg_pg")
    with (
        patch.object(
            group_coordinator.GroupCoordinator,
            "all_gather",
            wraps=group_coordinator.GroupCoordinator.all_gather,
        ),
        patch("torch.distributed.get_world_size", return_value=2),
        patch(
            "torch.distributed.all_gather_into_tensor",
            side_effect=lambda out, inp, group=None: out.copy_(
                torch.cat([t_r4, t_r6], dim=0)
            ),
        ),
    ):
        sep = group_coordinator.GroupCoordinator.all_gather(
            fake_gc, t_r6, dim=0, separate_tensors=True
        )
        assert torch.equal(sep[0], t_r6)
        assert torch.equal(sep[1], t_r4)
        cat_out = group_coordinator.GroupCoordinator.all_gather(
            fake_gc, t_r6, dim=0, separate_tensors=False
        )
        assert torch.equal(cat_out, torch.cat([t_r6, t_r4], dim=0))


def test_same_cfg_lane_check_raises():
    from vllm_torchtpu.omni.attention import (
        RingCfgLaneMismatchError,
        clear_ring_cfg_validation_state,
        compute_device_id_group_layout,
        set_ring_cfg_validation_state,
        tpu_ring_flash_attention,
        validate_ring_cfg_lanes,
    )

    device_ids = [0, 1, 4, 5, 6, 7, 2, 3]
    layout = compute_device_id_group_layout(
        device_ids, cfg_size=2, ring_size=2, uly_size=2
    )

    # 1. Direct ring group containing partners from different CFG lanes raises.
    with pytest.raises(RingCfgLaneMismatchError, match="CFG lane"):
        validate_ring_cfg_lanes(
            cfg_groups=layout["cfg_groups"],
            ring_groups=[[0, 2], [1, 3], [6, 4], [7, 5]],
        )

    # 2. Device-ordered mesh [0..7] paired with unpatched rank-ordered vllm-omni groups
    # pairs device 0 (rank 0, cfg=0) with device 2 (rank 6, cfg=1) along mesh ring axis.
    unpatched_cfg = [[0, 4], [1, 5], [2, 6], [3, 7]]
    unpatched_ring = [[0, 2], [1, 3], [4, 6], [5, 7]]
    with pytest.raises(RingCfgLaneMismatchError, match="CFG lane"):
        validate_ring_cfg_lanes(
            cfg_groups=unpatched_cfg,
            ring_groups=unpatched_ring,
            device_ids_by_rank=device_ids,
            mesh_device_ids=list(range(8)),
            outer_size=2,
            ring_size=2,
            uly_size=2,
        )

    # 3. tpu_ring_flash_attention enforces the active runtime validation state.
    set_ring_cfg_validation_state(
        {
            "cfg_groups": unpatched_cfg,
            "ring_groups": unpatched_ring,
            "cfg_lane_by_rank": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1},
            "device_ids_by_rank": device_ids,
        }
    )
    try:
        q = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)
        with pytest.raises(RingCfgLaneMismatchError, match="CFG lane"):
            tpu_ring_flash_attention(
                q, q, q, 0.25, ring_size=2, uly_size=2, outer_size=2, fuse_ulysses=True
            )
    finally:
        clear_ring_cfg_validation_state()


def test_resolve_device_ids_by_rank_raises_on_failed_query(
    monkeypatch, _restore_parallel_state_device_order
):
    """A rank-local device-ID failure enters all_gather_object, then raises."""
    from typing import Any

    from vllm_omni.diffusion.distributed import parallel_state

    import vllm_torchtpu.omni.patches as omni_patches

    omni_patches._patch_parallel_state_device_order()
    monkeypatch.setattr(omni_patches, "_DEVICE_IDS_BY_RANK_OVERRIDE", None)
    monkeypatch.setattr(omni_patches, "_CACHED_DEVICE_ID_ORDER", None)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    fake_world_group = SimpleNamespace(cpu_group="fake_cpu_pg")
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: fake_world_group)

    # 1. Rank 2 fails its local TPU device-ID query; all 4 ranks still enter
    # all_gather_object, then every rank raises instead of falling back to
    # rank-ordered groups.
    per_rank_attrs: dict[int, Any] = {
        0: {"id": 0},
        1: {"id": 2},
        2: RuntimeError("device query failed on rank 2"),
        3: {"id": 3},
    }
    current_rank = {"val": 0}
    gather_participating_ranks: list[tuple[int, int | None, Any]] = []

    def _fake_get_attrs():
        entry = per_rank_attrs[current_rank["val"]]
        if isinstance(entry, Exception):
            raise entry
        return entry

    fake_tpu = SimpleNamespace(get_local_device_attributes=_fake_get_attrs)
    monkeypatch.setattr(torch, "tpu", fake_tpu, raising=False)

    expected_gathered: list[int | None] = [0, 2, None, 3]

    def _fake_all_gather_object(object_list, obj, group=None):
        gather_participating_ranks.append((current_rank["val"], obj, group))
        object_list[:] = expected_gathered

    monkeypatch.setattr("torch.distributed.all_gather_object", _fake_all_gather_object)

    rg = parallel_state.RankGenerator(1, 2, 1, 2, 1, order="tp-sp-pp-cfg-dp")
    for r in range(4):
        current_rank["val"] = r
        with pytest.raises(RuntimeError, match="get_local_device_attributes") as exc:
            rg.get_ranks("sp")
        assert "[0, 2, None, 3]" in str(exc.value)
        if r == 2:
            assert "device query failed on rank 2" in str(exc.value.__cause__)
        else:
            assert exc.value.__cause__ is None

    assert gather_participating_ranks == [
        (0, 0, "fake_cpu_pg"),
        (1, 2, "fake_cpu_pg"),
        (2, None, "fake_cpu_pg"),
        (3, 3, "fake_cpu_pg"),
    ]
    assert omni_patches._CACHED_DEVICE_ID_ORDER is None

    # 2. Duplicate (non-permutation) device IDs also raise on every rank.
    expected_gathered = [0, 1, 1, 3]
    per_rank_attrs[2] = {"id": 1}
    gather_participating_ranks.clear()
    for r in range(4):
        current_rank["val"] = r
        with pytest.raises(RuntimeError, match="not a permutation") as dup_exc:
            omni_patches._resolve_device_ids_by_rank(4)
        assert "[0, 1, 1, 3]" in str(dup_exc.value)
    assert len(gather_participating_ranks) == 4
    assert omni_patches._CACHED_DEVICE_ID_ORDER is None

    # 3. Valid permutation resolves and caches device-ordered ranks across ranks.
    expected_gathered = [0, 2, 1, 3]
    gather_participating_ranks.clear()
    current_rank["val"] = 0
    resolved = omni_patches._resolve_device_ids_by_rank(4)
    assert resolved == ([0, 2, 1, 3], [0, 2, 1, 3])
    assert len(gather_participating_ranks) == 1
    assert rg.get_ranks("sp") == [[0, 2], [1, 3]]
    assert len(gather_participating_ranks) == 1
    parallel_state.destroy_model_parallel()
    assert omni_patches._CACHED_DEVICE_ID_ORDER is None


@pytest.fixture
def _restore_parallel_state_device_order():
    from vllm_omni.diffusion.distributed import group_coordinator, parallel_state

    import vllm_torchtpu.omni.patches as omni_patches
    from vllm_torchtpu.omni.attention import (
        get_ring_cfg_validation_state,
        set_ring_cfg_validation_state,
    )

    orig_get_ranks = parallel_state.RankGenerator.get_ranks
    orig_init_mp = parallel_state._initialize_model_parallel
    orig_public_init_mp = parallel_state.initialize_model_parallel
    orig_destroy_mp = parallel_state.destroy_model_parallel
    orig_all_gather = group_coordinator.GroupCoordinator.all_gather
    orig_gather = group_coordinator.GroupCoordinator.gather
    orig_sp_init = group_coordinator.SequenceParallelGroupCoordinator.__init__
    orig_cached = omni_patches._CACHED_DEVICE_ID_ORDER
    orig_override = omni_patches._DEVICE_IDS_BY_RANK_OVERRIDE
    orig_active = omni_patches._ACTIVE_INIT_MP_DEVICE_ORDER
    orig_cfg_state = get_ring_cfg_validation_state()
    try:
        yield parallel_state, group_coordinator
    finally:
        parallel_state.RankGenerator.get_ranks = orig_get_ranks
        parallel_state._initialize_model_parallel = orig_init_mp
        parallel_state.initialize_model_parallel = orig_public_init_mp
        parallel_state.destroy_model_parallel = orig_destroy_mp
        group_coordinator.GroupCoordinator.all_gather = orig_all_gather
        group_coordinator.GroupCoordinator.gather = orig_gather
        group_coordinator.SequenceParallelGroupCoordinator.__init__ = orig_sp_init
        omni_patches._CACHED_DEVICE_ID_ORDER = orig_cached
        omni_patches._DEVICE_IDS_BY_RANK_OVERRIDE = orig_override
        omni_patches._ACTIVE_INIT_MP_DEVICE_ORDER = orig_active
        set_ring_cfg_validation_state(orig_cfg_state)


@pytest.fixture
def _restore_diffusion_ring_degree_state(_restore_parallel_state_device_order):
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    orig_ring_degree = OmniTpuPlatform._diffusion_ring_degree
    orig_init_device = DiffusionWorker.init_device
    try:
        yield DiffusionWorker, orig_init_device
    finally:
        OmniTpuPlatform._diffusion_ring_degree = orig_ring_degree
        DiffusionWorker.init_device = orig_init_device


def _call_set_device_multi_rank(monkeypatch, world_size: int = 4) -> None:
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.delenv("TPU_VISIBLE_CHIPS", raising=False)
    monkeypatch.setattr(OmniTpuPlatform, "_worker_world_size", 1)
    monkeypatch.setattr(
        OmniTpuPlatform,
        "_prepare_singlehost_tpu_env",
        staticmethod(lambda ws: None),
    )
    monkeypatch.setattr(torch, "accelerator", MagicMock(), raising=False)
    with patch("torch.empty"), patch("torch.device"):
        OmniTpuPlatform.set_device(None)


@pytest.mark.parametrize("ring_degree", [1, 2])
def test_set_device_offload_dense_all_to_all_any_ring_degree(
    monkeypatch, _restore_diffusion_ring_degree_state, ring_degree
):
    flag = "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all"
    OmniTpuPlatform._diffusion_ring_degree = ring_degree

    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_foo=1")
    _call_set_device_multi_rank(monkeypatch)
    assert os.environ["LIBTPU_INIT_ARGS"] == f"--xla_foo=1 {flag}=true"

    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    _call_set_device_multi_rank(monkeypatch)
    assert os.environ["LIBTPU_INIT_ARGS"] == f"{flag}=true"


@pytest.mark.parametrize("explicit_val", ["true", "false"])
def test_set_device_preserves_explicit_offload_flag_for_ring_degree_two(
    monkeypatch, _restore_diffusion_ring_degree_state, explicit_val
):
    flag = "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all"
    OmniTpuPlatform._diffusion_ring_degree = 2

    expected = f"--xla_foo=1 {flag}={explicit_val}"
    monkeypatch.setenv("LIBTPU_INIT_ARGS", expected)
    _call_set_device_multi_rank(monkeypatch)
    assert os.environ["LIBTPU_INIT_ARGS"] == expected


def test_get_diffusion_worker_cls_wraps_init_device_idempotently(
    monkeypatch, _restore_diffusion_ring_degree_state
):
    from vllm_omni.platforms.interface import OmniPlatform

    flag = "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all"
    diffusion_worker_cls, _ = _restore_diffusion_ring_degree_state
    observed_in_orig: list[int] = []

    def _fake_orig_init_device(self):
        observed_in_orig.append(OmniTpuPlatform._diffusion_ring_degree)
        _call_set_device_multi_rank(monkeypatch)

    diffusion_worker_cls.init_device = _fake_orig_init_device

    parent_ret = OmniPlatform.get_diffusion_worker_cls()
    ret1 = OmniTpuPlatform.get_diffusion_worker_cls()
    wrapped_first = diffusion_worker_cls.init_device
    ret2 = OmniTpuPlatform.get_diffusion_worker_cls()
    wrapped_second = diffusion_worker_cls.init_device

    assert ret1 == parent_ret
    assert ret2 == parent_ret
    assert wrapped_first is not _fake_orig_init_device
    assert wrapped_second is wrapped_first

    # Records ring_degree=2 before calling original init_device once.
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_foo=1")
    fake_worker_ring2 = SimpleNamespace(
        od_config=SimpleNamespace(parallel_config=SimpleNamespace(ring_degree=2))
    )
    diffusion_worker_cls.init_device(fake_worker_ring2)
    assert OmniTpuPlatform._diffusion_ring_degree == 2
    assert observed_in_orig == [2]
    assert os.environ["LIBTPU_INIT_ARGS"] == f"--xla_foo=1 {flag}=true"

    # Defensive fallbacks: None, 0, "0", missing configs -> 1.
    for fake_worker in (
        SimpleNamespace(
            od_config=SimpleNamespace(parallel_config=SimpleNamespace(ring_degree=None))
        ),
        SimpleNamespace(
            od_config=SimpleNamespace(parallel_config=SimpleNamespace(ring_degree=0))
        ),
        SimpleNamespace(
            od_config=SimpleNamespace(parallel_config=SimpleNamespace(ring_degree="0"))
        ),
        SimpleNamespace(od_config=SimpleNamespace(parallel_config=None)),
        SimpleNamespace(od_config=None),
        SimpleNamespace(),
    ):
        OmniTpuPlatform._diffusion_ring_degree = 4
        monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_foo=1")
        diffusion_worker_cls.init_device(fake_worker)
        assert OmniTpuPlatform._diffusion_ring_degree == 1
        assert os.environ["LIBTPU_INIT_ARGS"] == f"--xla_foo=1 {flag}=true"

    assert len(observed_in_orig) == 7


def test_init_device_ring_degree_one_leaves_parallel_state_unpatched(
    _restore_diffusion_ring_degree_state,
    _restore_parallel_state_device_order,
):
    import vllm_torchtpu.omni.patches as omni_patches
    from vllm_torchtpu.omni.attention import get_ring_cfg_validation_state

    diffusion_worker_cls, _ = _restore_diffusion_ring_degree_state
    parallel_state, group_coordinator = _restore_parallel_state_device_order
    marker = omni_patches._PARALLEL_DEVICE_ORDER_PATCHED

    assert omni_patches._DEVICE_IDS_BY_RANK_OVERRIDE is None

    orig_calls: list[int] = []
    diffusion_worker_cls.init_device = lambda self: orig_calls.append(
        OmniTpuPlatform._diffusion_ring_degree
    )
    OmniTpuPlatform.get_diffusion_worker_cls()

    fake_worker_ring1 = SimpleNamespace(
        od_config=SimpleNamespace(parallel_config=SimpleNamespace(ring_degree=1))
    )
    diffusion_worker_cls.init_device(fake_worker_ring1)
    assert orig_calls == [1]
    assert omni_patches._DEVICE_IDS_BY_RANK_OVERRIDE is None

    for fn in (
        parallel_state.RankGenerator.get_ranks,
        parallel_state._initialize_model_parallel,
        parallel_state.initialize_model_parallel,
        parallel_state.destroy_model_parallel,
        group_coordinator.GroupCoordinator.all_gather,
        group_coordinator.GroupCoordinator.gather,
        group_coordinator.SequenceParallelGroupCoordinator.__init__,
    ):
        assert not getattr(fn, marker, False)

    rg = parallel_state.RankGenerator(1, 4, 1, 2, 1, order="tp-sp-pp-cfg-dp")
    assert rg.get_ranks("sp") == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert rg.get_ranks("cfg") == [[0, 4], [1, 5], [2, 6], [3, 7]]

    # Even if the patch was installed earlier in the process with cached device IDs,
    # a subsequent ring_degree == 1 initialize_model_parallel call keeps default layout.
    observed_groups: dict[str, object] = {}

    def _fake_orig_sp_init(
        self, group_ranks, local_rank, torch_distributed_backend, **kwargs
    ):
        self.rank = 6
        self.ranks = [6, 7, 4, 5]
        self.ulysses_world_size = 4
        self.ring_world_size = 1
        self.allgather_world_size = 1
        self.ulysses_rank = 2
        self.ring_rank = 0

    def _fake_orig_init_mp(**kwargs):
        gen = parallel_state.RankGenerator(
            kwargs["tensor_parallel_size"],
            kwargs["ulysses_degree"] * kwargs["ring_degree"],
            kwargs["pipeline_parallel_size"],
            kwargs["cfg_parallel_size"],
            1,
            order="tp-sp-pp-cfg-dp",
        )
        observed_groups["sp"] = gen.get_ranks("sp")
        observed_groups["cfg"] = gen.get_ranks("cfg")
        fake_sp = object.__new__(group_coordinator.SequenceParallelGroupCoordinator)
        group_coordinator.SequenceParallelGroupCoordinator.__init__(
            fake_sp,
            observed_groups["sp"],
            0,
            "gloo",
            ulysses_group="u",
            ring_group="r",
            allgather_group="ag",
        )
        observed_groups["ulysses_rank"] = fake_sp.ulysses_rank
        observed_groups["ring_rank"] = fake_sp.ring_rank

    parallel_state._initialize_model_parallel = _fake_orig_init_mp
    group_coordinator.SequenceParallelGroupCoordinator.__init__ = _fake_orig_sp_init
    omni_patches._patch_parallel_state_device_order()
    omni_patches._CACHED_DEVICE_ID_ORDER = (
        8,
        [0, 1, 4, 5, 6, 7, 2, 3],
        [0, 1, 6, 7, 2, 3, 4, 5],
    )
    assert omni_patches._DEVICE_IDS_BY_RANK_OVERRIDE is None

    parallel_state.initialize_model_parallel(
        cfg_parallel_size=2, ulysses_degree=4, ring_degree=1
    )
    assert observed_groups["sp"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert observed_groups["cfg"] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert observed_groups["ulysses_rank"] == 2
    assert observed_groups["ring_rank"] == 0
    assert get_ring_cfg_validation_state() is None


def test_sp_init_ring_degree_two_uses_device_ordered_sp_group(
    monkeypatch,
    _restore_parallel_state_device_order,
):
    import vllm_torchtpu.omni.patches as omni_patches
    from vllm_torchtpu.omni.attention import compute_device_id_group_layout

    parallel_state, group_coordinator = _restore_parallel_state_device_order
    device_ids = [0, 1, 4, 5, 6, 7, 2, 3]
    layout = compute_device_id_group_layout(
        device_ids, cfg_size=2, ring_size=2, uly_size=2
    )
    current_rank = {"val": 0}

    def _fake_orig_sp_init(
        self, group_ranks, local_rank, torch_distributed_backend, **kwargs
    ):
        # Upstream reads the c10d sub-group ranks, which follow sorted global
        # ranks; poison them so the test sees whether the patch rewrote them.
        self.rank = current_rank["val"]
        self.ranks = next(g for g in group_ranks if self.rank in g)
        self.ulysses_world_size = 2
        self.ring_world_size = 2
        self.allgather_world_size = 1
        self.ulysses_rank = -1
        self.ring_rank = -1

    observed: dict[int, tuple[list[int], list[int], int, int]] = {}

    def _fake_orig_init_mp(**kwargs):
        gen = parallel_state.RankGenerator(
            kwargs["tensor_parallel_size"],
            kwargs["ulysses_degree"] * kwargs["ring_degree"],
            kwargs["pipeline_parallel_size"],
            kwargs["cfg_parallel_size"],
            1,
            order="tp-sp-pp-cfg-dp",
        )
        sp_groups = gen.get_ranks("sp")
        assert sp_groups == layout["sp_groups"] == [[0, 1, 6, 7], [2, 3, 4, 5]]
        for rank in range(8):
            current_rank["val"] = rank
            fake_sp = object.__new__(group_coordinator.SequenceParallelGroupCoordinator)
            group_coordinator.SequenceParallelGroupCoordinator.__init__(
                fake_sp,
                sp_groups,
                0,
                "gloo",
                ulysses_group="u",
                ring_group="r",
                allgather_group="ag",
            )
            observed[rank] = (
                fake_sp.ulysses_ranks,
                fake_sp.ring_ranks,
                fake_sp.ulysses_rank,
                fake_sp.ring_rank,
            )

    parallel_state._initialize_model_parallel = _fake_orig_init_mp
    group_coordinator.SequenceParallelGroupCoordinator.__init__ = _fake_orig_sp_init
    omni_patches._patch_parallel_state_device_order()
    monkeypatch.setattr(omni_patches, "_DEVICE_IDS_BY_RANK_OVERRIDE", device_ids)

    parallel_state.initialize_model_parallel(
        cfg_parallel_size=2, ring_degree=2, ulysses_degree=2
    )

    # SP group [0, 1, 6, 7]: Ulysses pairs are consecutive, ring pairs stride 2.
    assert observed[0] == ([0, 1], [0, 6], 0, 0)
    assert observed[1] == ([0, 1], [1, 7], 1, 0)
    assert observed[6] == ([6, 7], [0, 6], 0, 1)
    assert observed[7] == ([6, 7], [1, 7], 1, 1)
    assert observed[4] == ([4, 5], [2, 4], 0, 1)
    for rank, (uly_ranks, ring_ranks, uly_idx, ring_idx) in observed.items():
        assert uly_ranks in layout["ulysses_groups"]
        assert ring_ranks in layout["ring_groups"]
        assert uly_ranks[uly_idx] == rank
        assert ring_ranks[ring_idx] == rank


def test_init_device_ring_degree_two_installs_device_order_patch_once(
    _restore_diffusion_ring_degree_state,
    _restore_parallel_state_device_order,
):
    from vllm_omni.diffusion.worker import diffusion_worker

    import vllm_torchtpu.omni.patches as omni_patches

    diffusion_worker_cls, _ = _restore_diffusion_ring_degree_state
    parallel_state, group_coordinator = _restore_parallel_state_device_order
    marker = omni_patches._PARALLEL_DEVICE_ORDER_PATCHED

    patched_during_orig: list[tuple[bool, bool, bool]] = []
    mp_calls: list[int] = []

    def _fake_orig_init_mp(**kwargs):
        mp_calls.append(kwargs["ring_degree"])

    parallel_state._initialize_model_parallel = _fake_orig_init_mp

    def _fake_orig_init_device(self):
        patched_during_orig.append(
            (
                getattr(parallel_state.RankGenerator.get_ranks, marker, False),
                getattr(parallel_state._initialize_model_parallel, marker, False),
                getattr(group_coordinator.GroupCoordinator.gather, marker, False),
            )
        )
        # Exercise the exact by-name binding imported by diffusion_worker.py.
        diffusion_worker.initialize_model_parallel(
            cfg_parallel_size=2,
            ulysses_degree=2,
            ring_degree=self.od_config.parallel_config.ring_degree,
        )

    diffusion_worker_cls.init_device = _fake_orig_init_device
    OmniTpuPlatform.get_diffusion_worker_cls()

    fake_worker_ring2 = SimpleNamespace(
        od_config=SimpleNamespace(parallel_config=SimpleNamespace(ring_degree=2))
    )
    diffusion_worker_cls.init_device(fake_worker_ring2)

    first_refs = (
        parallel_state.RankGenerator.get_ranks,
        parallel_state._initialize_model_parallel,
        parallel_state.destroy_model_parallel,
        group_coordinator.GroupCoordinator.all_gather,
        group_coordinator.GroupCoordinator.gather,
        group_coordinator.SequenceParallelGroupCoordinator.__init__,
    )
    assert all(getattr(fn, marker, False) for fn in first_refs)
    assert patched_during_orig == [(True, True, True)]
    assert mp_calls == [2]

    OmniTpuPlatform.get_diffusion_worker_cls()
    diffusion_worker_cls.init_device(fake_worker_ring2)

    second_refs = (
        parallel_state.RankGenerator.get_ranks,
        parallel_state._initialize_model_parallel,
        parallel_state.destroy_model_parallel,
        group_coordinator.GroupCoordinator.all_gather,
        group_coordinator.GroupCoordinator.gather,
        group_coordinator.SequenceParallelGroupCoordinator.__init__,
    )
    for fn1, fn2 in zip(first_refs, second_refs):
        assert fn1 is fn2
    assert patched_during_orig == [(True, True, True), (True, True, True)]
    assert mp_calls == [2, 2]


def test_non_ascending_group_raises_early_and_valid_host_layouts_pass(
    monkeypatch,
    _restore_parallel_state_device_order,
):
    import vllm_torchtpu.omni.patches as omni_patches

    parallel_state, _ = _restore_parallel_state_device_order
    orig_calls: list[dict[str, object]] = []
    parallel_state._initialize_model_parallel = lambda **kw: orig_calls.append(kw)
    omni_patches._patch_parallel_state_device_order()

    # 1. Non-ascending ulysses group [1, 0] raises before orig_init_mp.
    bad_dev_ids = [1, 0, 4, 5, 6, 7, 2, 3]
    monkeypatch.setattr(omni_patches, "_DEVICE_IDS_BY_RANK_OVERRIDE", bad_dev_ids)
    with pytest.raises(RuntimeError) as exc_info:
        parallel_state.initialize_model_parallel(
            cfg_parallel_size=2, ring_degree=2, ulysses_degree=2
        )
    msg = str(exc_info.value)
    assert "[1, 0, 4, 5, 6, 7, 2, 3]" in msg
    assert "[1, 0, 6, 7, 2, 3, 4, 5]" in msg
    assert "[1, 0]" in msg
    assert orig_calls == []

    # 2. Non-ascending tp / pp groups also raise before orig_init_mp.
    with pytest.raises(RuntimeError) as tp_exc:
        parallel_state.initialize_model_parallel(
            cfg_parallel_size=2,
            ring_degree=2,
            ulysses_degree=1,
            tensor_parallel_size=2,
        )
    assert "[1, 0]" in str(tp_exc.value)
    assert "[1, 0, 4, 5, 6, 7, 2, 3]" in str(tp_exc.value)
    assert orig_calls == []

    host_dev_ids = [0, 1, 4, 5, 6, 7, 2, 3]
    monkeypatch.setattr(omni_patches, "_DEVICE_IDS_BY_RANK_OVERRIDE", host_dev_ids)
    with pytest.raises(RuntimeError) as pp_exc:
        parallel_state.initialize_model_parallel(
            cfg_parallel_size=1,
            ring_degree=2,
            ulysses_degree=2,
            pipeline_parallel_size=2,
        )
    assert "[6, 4]" in str(pp_exc.value)
    assert "[0, 1, 4, 5, 6, 7, 2, 3]" in str(pp_exc.value)
    assert orig_calls == []

    # 3. Host mapping [0, 1, 4, 5, 6, 7, 2, 3] with c2r2u2, c2r4u1, and c1r2u4 passes.
    parallel_state.initialize_model_parallel(
        cfg_parallel_size=2, ring_degree=2, ulysses_degree=2
    )
    parallel_state.initialize_model_parallel(
        cfg_parallel_size=2, ring_degree=4, ulysses_degree=1
    )
    parallel_state.initialize_model_parallel(
        cfg_parallel_size=1, ring_degree=2, ulysses_degree=4
    )
    assert len(orig_calls) == 3


def test_group_coordinator_gather_reorders_unsorted_group(
    _restore_parallel_state_device_order,
):
    import vllm_torchtpu.omni.patches as omni_patches

    _, group_coordinator = _restore_parallel_state_device_order
    omni_patches._patch_parallel_state_device_order()

    # Use a 3-cycle [6, 2, 4] (sorted [2, 4, 6]) so order != inverse(order).
    t_r2 = torch.tensor([[20.0, 21.0]])
    t_r4 = torch.tensor([[40.0, 41.0]])
    t_r6 = torch.tensor([[60.0, 61.0]])
    observed_dsts: list[int] = []

    def _fake_dist_gather(tensor, gather_list=None, dst=0, group=None):
        observed_dsts.append(dst)
        if gather_list is not None:
            for slot, src_t in zip(gather_list, (t_r2, t_r4, t_r6)):
                slot.copy_(src_t)

    with patch("torch.distributed.gather", side_effect=_fake_dist_gather):
        gc_dst0 = SimpleNamespace(
            ranks=[6, 2, 4],
            world_size=3,
            rank_in_group=0,
            device_group="cfg_pg",
        )
        out_dim_neg1 = group_coordinator.GroupCoordinator.gather(
            gc_dst0, t_r6, dst=0, dim=-1
        )
        assert out_dim_neg1 is not None
        assert torch.equal(out_dim_neg1, torch.cat([t_r6, t_r2, t_r4], dim=-1))

        out_dim0 = group_coordinator.GroupCoordinator.gather(
            gc_dst0, t_r6, dst=0, dim=0
        )
        assert out_dim0 is not None
        assert torch.equal(out_dim0, torch.cat([t_r6, t_r2, t_r4], dim=0))

        gc_dst2 = SimpleNamespace(
            ranks=[6, 2, 4],
            world_size=3,
            rank_in_group=2,
            device_group="cfg_pg",
        )
        out_dst2 = group_coordinator.GroupCoordinator.gather(
            gc_dst2, t_r4, dst=2, dim=0
        )
        assert out_dst2 is not None
        assert torch.equal(out_dst2, torch.cat([t_r6, t_r2, t_r4], dim=0))

        gc_non_dst = SimpleNamespace(
            ranks=[6, 2, 4],
            world_size=3,
            rank_in_group=1,
            device_group="cfg_pg",
        )
        assert (
            group_coordinator.GroupCoordinator.gather(gc_non_dst, t_r2, dst=0, dim=0)
            is None
        )
        assert observed_dsts == [6, 6, 4, 6]

        # Sorted group passes through unchanged without reordering.
        gc_sorted = SimpleNamespace(
            ranks=[2, 4, 6],
            world_size=3,
            rank_in_group=0,
            device_group="cfg_pg",
        )
        out_sorted = group_coordinator.GroupCoordinator.gather(
            gc_sorted, t_r2, dst=0, dim=0
        )
        assert out_sorted is not None
        assert torch.equal(out_sorted, torch.cat([t_r2, t_r4, t_r6], dim=0))

    # Empty chunk dimension does not fail on split(0).
    with patch(
        "torch.distributed.gather",
        side_effect=lambda tensor, gather_list=None, dst=0, group=None: None,
    ):
        empty_in = torch.empty((2, 0), dtype=torch.float32)
        out_empty = group_coordinator.GroupCoordinator.gather(
            gc_dst0, empty_in, dst=0, dim=-1
        )
        assert out_empty is not None
        assert out_empty.shape == (2, 0)


def test_can_fuse_tpu_ring_ulysses_requires_strict_ulysses_mode(monkeypatch):
    from vllm_omni.diffusion.forward_context import set_forward_context

    from vllm_torchtpu.omni.attention import can_fuse_tpu_ring_ulysses
    from vllm_torchtpu.omni.patches import _patch_ring_attention

    monkeypatch.setattr(
        Attention, "_forward_impl", lambda self, q, k, v, attn_metadata=None: "fallback"
    )
    _patch_ring_attention()

    sp_group = SimpleNamespace(ring_world_size=2, ulysses_world_size=2)
    runner = RingParallelAttention(sp_group, attn_backend_pref="TPU_FLASH_ATTN")
    strategy = SimpleNamespace(name="ulysses", _scatter_idx=2, _gather_idx=1)
    layer = _make_ring_attention_layer(ring_runner=runner, strategy=strategy)
    monkeypatch.setattr(
        layer, "_get_active_parallel_strategy", lambda: layer.parallel_strategy
    )
    monkeypatch.setattr(layer, "is_paged_kv_active", lambda: False)

    q = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)
    assert can_fuse_tpu_ring_ulysses(layer, strategy, q, q, q, None) is True

    strict_cfg = SimpleNamespace(parallel_config=SimpleNamespace(ulysses_mode="strict"))
    with set_forward_context(vllm_config=None, omni_diffusion_config=strict_cfg):
        assert can_fuse_tpu_ring_ulysses(layer, strategy, q, q, q, None) is True

    uaa_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(ulysses_mode="advanced_uaa")
    )
    with set_forward_context(vllm_config=None, omni_diffusion_config=uaa_cfg):
        assert can_fuse_tpu_ring_ulysses(layer, strategy, q, q, q, None) is False
        assert layer._forward_impl(q, q, q) == "fallback"


def test_non_ring_attention_layer_skips_ring_dispatch_work(monkeypatch):
    """Layers without a ring runner reach upstream before any ring helper runs."""
    import vllm_torchtpu.omni.attention as omni_attention
    from vllm_torchtpu.omni.patches import _patch_ring_attention

    def _fail(*args, **kwargs):
        raise AssertionError("non-ring layer ran ring dispatch work")

    monkeypatch.setattr(omni_attention, "can_fuse_tpu_ring_ulysses", _fail)
    monkeypatch.setattr(omni_attention, "tpu_ring_flash_attention", _fail)
    monkeypatch.setattr(
        Attention, "_forward_impl", lambda self, q, k, v, attn_metadata=None: "fallback"
    )
    _patch_ring_attention()

    sp_group = SimpleNamespace(ring_world_size=2, ulysses_world_size=2)
    runner = RingParallelAttention(sp_group, attn_backend_pref="TPU_FLASH_ATTN")
    no_runner = _make_ring_attention_layer(ring_runner=None)
    flag_without_runner = _make_ring_attention_layer(ring_runner=None)
    flag_without_runner.use_ring = True
    runner_without_flag = _make_ring_attention_layer(ring_runner=runner)
    runner_without_flag.use_ring = False

    q = torch.randn(1, 16, 4, 16, dtype=torch.bfloat16)
    for layer in (no_runner, flag_without_runner, runner_without_flag):
        monkeypatch.setattr(layer, "_get_active_parallel_strategy", _fail)
        monkeypatch.setattr(layer, "_with_kv_cache_dtype", _fail)
        assert layer._forward_impl(q, q, q) == "fallback"


def test_group_coordinator_methods_are_audited_for_reorder():
    """Fail when vLLM-Omni adds a GroupCoordinator method nobody audited.

    With ring_degree > 1 the cfg/dp groups can list ranks out of ascending order
    (for example [6, 4]) while torch.distributed.new_group sorts them. Every
    method in ``order_safe`` maps a group-local src/dst through ``self.ranks`` or
    does not depend on member order. ``all_gather`` and ``gather`` return data in
    sorted c10d order and are reordered by ``_patch_parallel_state_device_order``.
    """
    from vllm_omni.diffusion.distributed import group_coordinator

    order_safe = {
        "first_rank",
        "last_rank",
        "is_first_rank",
        "is_last_rank",
        "next_rank",
        "prev_rank",
        "group_next_rank",
        "group_prev_rank",
        "skip_rank",
        "group_skip_rank",
        "all_reduce",
        "broadcast",
        "broadcast_object",
        "broadcast_object_list",
        "send_object",
        "recv_object",
        "broadcast_tensor_dict",
        "isend_tensor_dict",
        "irecv_tensor_dict",
        "send_tensor_dict",
        "recv_tensor_dict",
        "barrier",
        "send",
        "recv",
        "destroy",
    }
    reordered = {"all_gather", "gather"}
    members = {
        name
        for name, attr in vars(group_coordinator.GroupCoordinator).items()
        if not name.startswith("__") and (callable(attr) or isinstance(attr, property))
    }
    assert members == order_safe | reordered
