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
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from .mock_omni import (  # isort: split
    _MockRMSNorm,
    _MockRotaryEmbeddingWan,
    vllm_omni_mock,
)

import vllm_omni
from vllm.config import ProfilerConfig
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
    _patch_diffusion_startup_warmup_sync,
    apply_omni_model_specific_patches,
    apply_omni_tpu_patches,
)
from vllm_torchtpu.omni.platform import OmniTpuPlatform
from vllm_torchtpu.omni.profiler import TpuOmniTorchProfilerWrapper
from vllm_torchtpu.omni.rope import (
    _pair_swap_matrix,
    apply_rotary_pair_swap,
    can_fuse_qk_norm_rope,
    rotary_embedding_wan_forward,
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
    assert (
        platform.get_profiler_cls()
        == "vllm_torchtpu.omni.profiler.TpuOmniTorchProfilerWrapper"
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


_PROFILER_TS = "2026_01_01_12_00_00"


def _profiler_config(tmp_path, with_stack=False):
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_with_stack=with_stack,
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=False,
        torch_profiler_with_flops=True,
    )


@pytest.fixture
def tpu_profiler(monkeypatch, tmp_path):
    """Stubs torch.profiler and the TPU profiler config for the wrapper tests.

    torch.distributed starts out uninitialized and profiler_trace is the real
    module. ``fake_rank(rank, world_size)`` fakes a distributed run, stubs the
    profiler_trace helpers and returns those stubs.
    """
    mocks = SimpleNamespace(profile=MagicMock(), tpu_config=MagicMock())
    monkeypatch.setattr("torch.profiler.profile", mocks.profile)
    monkeypatch.setattr(
        "vllm_torchtpu.omni.profiler.TpuProfilerConfig", mocks.tpu_config
    )
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)

    def fake_rank(rank, world_size):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda: world_size)
        trace = SimpleNamespace(
            resolve=MagicMock(return_value=_PROFILER_TS),
            capture=MagicMock(return_value=str(tmp_path / f"rank_{rank}")),
            merge=MagicMock(),
            clear=MagicMock(),
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.resolve_canonical_dst_ts", trace.resolve
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.rank_capture_dir", trace.capture
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.merge_rank_capture", trace.merge
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.clear_canonical_ts_marker", trace.clear
        )
        return trace

    mocks.fake_rank = fake_rank
    return mocks


def test_tpu_omni_profiler_init(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        wrapper = TpuOmniTorchProfilerWrapper(
            _profiler_config(tmp_path, with_stack=True), "worker_0", 0
        )

    assert wrapper._activities == ["CPU", "TPU"]
    assert getattr(wrapper, "dump_cpu_time_total", False) is False
    assert getattr(wrapper, "profiler", None) is None
    assert "Deferring torch.profiler construction" in caplog.text
    assert "enforcing torch.profiler with_stack=False" in caplog.text
    # The base config always gets with_stack=False.
    assert wrapper.profiler_config.torch_profiler_with_stack is False


def test_tpu_omni_profiler_start_stop_merges_rank(tmp_path, tpu_profiler):
    wrapper = TpuOmniTorchProfilerWrapper(
        _profiler_config(tmp_path, with_stack=True), "worker_0", 0
    )
    trace = tpu_profiler.fake_rank(2, world_size=4)

    wrapper._start()
    assert trace.resolve.called
    assert trace.capture.called
    tpu_profiler.profile.assert_called_once()
    # with_stack=True records Python stacks through the TPU python tracer.
    _, tpu_kwargs = tpu_profiler.tpu_config.call_args
    assert tpu_kwargs.get("python_tracer_level") == 1

    wrapper._stop()
    tpu_profiler.profile.return_value.stop.assert_called_once()
    trace.merge.assert_called_once_with(
        str(tmp_path / "rank_2"),
        str(tmp_path / "worker_0"),
        _PROFILER_TS,
        2,
        world_size=4,
    )
    # Only rank 0 clears the canonical timestamp marker.
    trace.clear.assert_not_called()


def test_tpu_omni_profiler_shared_run_root(tmp_path, tpu_profiler):
    # Same trace filename on two workers in one process: one shared run root.
    cfg = _profiler_config(tmp_path)
    w1 = TpuOmniTorchProfilerWrapper(cfg, "worker_1", 0)
    w1.set_trace_filename("my_trace")
    w2 = TpuOmniTorchProfilerWrapper(cfg, "worker_2", 0)
    w2.set_trace_filename("my_trace")

    w1._start()
    w2._start()

    assert w1._capture_state[1] == w2._capture_state[1]
    assert w1._capture_state[1] == os.path.join(str(tmp_path), "my_trace")

    # Omni's chrome-trace session dir and the XSpace run root are
    # siblings under torch_profiler_dir, not nested in each other.
    session_dir = w1._ensure_session_dir()
    run_root = w1._capture_state[1]
    assert os.path.dirname(session_dir) == str(tmp_path)
    assert os.path.dirname(run_root) == str(tmp_path)
    assert os.path.basename(session_dir).endswith("_my_trace")
    assert session_dir != run_root
    assert not session_dir.startswith(run_root + os.sep)
    assert not run_root.startswith(session_dir + os.sep)

    # with_stack=False turns the TPU python tracer off.
    _, tpu_kwargs = tpu_profiler.tpu_config.call_args
    assert tpu_kwargs.get("python_tracer_level") == 0

    w1._stop()
    w2._stop()


def test_tpu_omni_profiler_stop_error_still_merges(tmp_path, tpu_profiler):
    wrapper = TpuOmniTorchProfilerWrapper(_profiler_config(tmp_path), "w3", 0)
    trace = tpu_profiler.fake_rank(0, world_size=1)
    wrapper._start()
    wrapper.profiler.stop.side_effect = ValueError("Intentional error")

    with pytest.raises(ValueError, match="Intentional error"):
        wrapper._stop()

    trace.merge.assert_called_once()
    trace.clear.assert_called_once()


def test_tpu_omni_profiler_canonical_ts_sync(tmp_path):
    from vllm.config import ProfilerConfig

    from vllm_torchtpu.omni.profiler import TpuOmniTorchProfilerWrapper

    cfg = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_with_stack=False,
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=False,
        torch_profiler_with_flops=True,
    )

    w0 = TpuOmniTorchProfilerWrapper(cfg, "w0", 0)
    w0.set_trace_filename("my_trace")
    w1 = TpuOmniTorchProfilerWrapper(cfg, "w1", 0)
    w1.set_trace_filename("my_trace")

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.profiler.profile"),
        patch("vllm_torchtpu.omni.profiler.TpuProfilerConfig"),
    ):
        with patch("torch.distributed.get_rank", return_value=0):
            w0._start()
        with patch("torch.distributed.get_rank", return_value=1):
            w1._start()

        assert w0._capture_state[2] == w1._capture_state[2]
