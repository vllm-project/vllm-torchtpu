# Copyright 2025 Google LLC
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

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from vllm.config.multimodal import MultiModalConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu.layers.adapter.attention_backend import (
    ensure_flywheel_enum_registered,
)
from vllm_torchtpu.layers.adapter.vision_attention import TpuMMEncoderAttention
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")


def make_attention(
    num_heads: int = 2,
    head_size: int = 4,
    scale: float = 1.0,
    num_kv_heads: int | None = None,
    attn_backend: AttentionBackendEnum = AttentionBackendEnum.FLASH_ATTN,
) -> TpuMMEncoderAttention:
    attn = TpuMMEncoderAttention.__new__(TpuMMEncoderAttention)
    torch.nn.Module.__init__(attn)
    attn.num_heads = num_heads
    attn.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    attn.head_size = head_size
    attn.scale = scale
    attn.attn_backend = attn_backend
    attn._forward_method = attn.forward_oot
    return attn


def _reference_varlen_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Reference float32 non-causal attention for [B, S, H, D] inputs."""
    bsz, seq_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    enable_gqa = num_heads > num_kv_heads
    qf = q.float()
    kf = k.float()
    vf = v.float()

    if cu_seqlens is None:
        out = F.scaled_dot_product_attention(
            qf.permute(0, 2, 1, 3),
            kf.permute(0, 2, 1, 3),
            vf.permute(0, 2, 1, 3),
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out.permute(0, 2, 1, 3).to(q.dtype)

    flat_q = qf.reshape(bsz * seq_len, num_heads, head_dim)
    flat_k = kf.reshape(bsz * seq_len, num_kv_heads, head_dim)
    flat_v = vf.reshape(bsz * seq_len, num_kv_heads, head_dim)
    out_flat = torch.zeros_like(flat_q)
    bounds = cu_seqlens.cpu().tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end <= start:
            continue
        qi = flat_q[start:end].permute(1, 0, 2).unsqueeze(0)
        ki = flat_k[start:end].permute(1, 0, 2).unsqueeze(0)
        vi = flat_v[start:end].permute(1, 0, 2).unsqueeze(0)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, scale=scale, enable_gqa=enable_gqa
        )
        out_flat[start:end] = oi.squeeze(0).permute(1, 0, 2)
    return out_flat.reshape(bsz, seq_len, num_heads, head_dim).to(q.dtype)


def test_vision_flash_attention_pads_and_builds_segments(monkeypatch):
    calls = {}

    def fake_vision_op(q, k, v, q_seg, kv_seg):
        calls["q"] = q
        calls["k"] = k
        calls["v"] = v
        calls["q_seg"] = q_seg
        calls["kv_seg"] = kv_seg
        return q

    monkeypatch.setattr(TpuMMEncoderAttention, "_vision_op", fake_vision_op)

    attn = make_attention(scale=0.5)
    q = torch.arange(1 * 5 * 2 * 4, dtype=torch.float32).reshape(1, 5, 2, 4)
    k = q + 100
    v = q + 200
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    out = attn._flash_attention(q, k, v, cu_seqlens)

    assert out.shape == q.shape
    assert calls["q"].shape == (1, 2, 128, 4)
    assert calls["k"].shape == (1, 2, 128, 4)
    assert calls["v"].shape == (1, 2, 128, 4)
    assert calls["q_seg"].shape == (1, 128)
    assert calls["kv_seg"].shape == (1, 128)
    expected = [1, 1, 2, 2, 2] + [3] * (128 - 5)
    assert calls["q_seg"][0].tolist() == expected
    assert calls["kv_seg"][0].tolist() == expected


def test_vision_attention_falls_back_for_cross_attention(monkeypatch):
    def fail_vision_op(*args, **kwargs):
        raise AssertionError("flash path should not handle q_len != kv_len")

    monkeypatch.setattr(TpuMMEncoderAttention, "_vision_op", fail_vision_op)

    attn = make_attention()
    query = torch.zeros(1, 1, 8)
    key = torch.zeros(1, 4, 8)
    value = torch.ones(1, 4, 8)

    called = {}

    def fake_native(query, key, value, cu_seqlens, max_seqlen, sequence_lengths):
        called["args"] = (query, key, value, cu_seqlens, max_seqlen, sequence_lengths)
        return torch.full_like(query, 7.0)

    monkeypatch.setattr(attn, "forward_native", fake_native)

    out = attn.forward_oot(query, key, value)

    assert torch.equal(out, torch.full_like(query, 7.0))
    assert called["args"][0] is query
    assert called["args"][1] is key
    assert called["args"][2] is value


def test_platform_vit_attn_backend_defaults_and_overrides():
    flywheel = ensure_flywheel_enum_registered()
    supported = TpuPlatform.get_supported_vit_attn_backends()
    assert supported == [
        AttentionBackendEnum.FLASH_ATTN,
        flywheel,
        AttentionBackendEnum.TORCH_SDPA,
    ]
    assert (
        TpuPlatform.get_vit_attn_backend(head_size=64, dtype=torch.bfloat16)
        == AttentionBackendEnum.FLASH_ATTN
    )
    assert (
        TpuPlatform.get_vit_attn_backend(
            head_size=64, dtype=torch.bfloat16, backend=flywheel
        )
        == flywheel
    )
    assert (
        TpuPlatform.get_vit_attn_backend(
            head_size=64,
            dtype=torch.bfloat16,
            backend=AttentionBackendEnum.TORCH_SDPA,
        )
        == AttentionBackendEnum.TORCH_SDPA
    )
    with pytest.raises(ValueError, match="not supported on TPU"):
        TpuPlatform.get_vit_attn_backend(
            head_size=64,
            dtype=torch.bfloat16,
            backend=AttentionBackendEnum.TRITON_ATTN,
        )
    with pytest.raises(ValueError, match="head_size <= 256"):
        TpuPlatform.get_vit_attn_backend(
            head_size=320,
            dtype=torch.bfloat16,
            backend=flywheel,
        )


def test_flywheel_enum_unpickles_in_spawn_child_without_eager_import():
    flywheel = ensure_flywheel_enum_registered()
    cfg = MultiModalConfig(mm_encoder_attn_backend="flywheel")
    assert cfg.mm_encoder_attn_backend is flywheel
    payload = pickle.dumps(cfg)

    env = dict(
        os.environ,
        PYTHONPATH=f"{_SRC_DIR}:{os.environ.get('PYTHONPATH', '')}",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
    )
    script = """
import pickle
import sitecustomize
import sys

cfg = pickle.loads(sys.stdin.buffer.read())
assert cfg.mm_encoder_attn_backend is not None
assert cfg.mm_encoder_attn_backend.name == "FLYWHEEL"
assert "flywheel_tpu" not in sys.modules

from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.adapter.attention_backend import FLYWHEEL_BACKEND_CLASS_PATH
register_layers()
assert (
    sitecustomize._FlywheelAttnEnumFinder._FLYWHEEL_BACKEND_PATH
    == FLYWHEEL_BACKEND_CLASS_PATH
)
assert "flywheel_tpu" not in sys.modules
"""
    res = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr.decode()


def test_flywheel_enum_get_class_and_decoder_override():
    from vllm_torchtpu.layers.adapter import attention_backend as ab
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend

    flywheel = ensure_flywheel_enum_registered()
    assert not ab.is_registered_attention_backend(flywheel)
    assert flywheel.get_class() is PallasAttentionBackend

    custom_path = (
        "vllm_torchtpu.layers.adapter.attention.PallasBatchedRPAAttentionBackend"
    )
    try:
        ab.register_attention_backend("flywheel", vllm_backend=custom_path)
        assert ab.is_registered_attention_backend(flywheel)
        assert flywheel.get_path() == custom_path
        with pytest.raises(ValueError, match="already has an attention backend"):
            ab.register_attention_backend("flywheel", vllm_backend="some.Other")
    finally:
        ab._REGISTERED_BACKENDS.discard("flywheel")
        flywheel.clear_override()


def test_vision_attention_routes_torch_sdpa_to_forward_native(monkeypatch):
    def fail_op(*args, **kwargs):
        raise AssertionError("Pallas kernel should not run for TORCH_SDPA")

    monkeypatch.setattr(TpuMMEncoderAttention, "_vision_op", fail_op)
    monkeypatch.setattr(TpuMMEncoderAttention, "_flywheel_token_major_op", fail_op)
    monkeypatch.setattr(TpuMMEncoderAttention, "_flywheel_head_major_op", fail_op)

    attn = make_attention(attn_backend=AttentionBackendEnum.TORCH_SDPA)
    query = torch.ones(1, 4, 2, 4)
    monkeypatch.setattr(
        attn,
        "forward_native",
        lambda q, k, v, cu, ms, sl: torch.full_like(q, 3.0),
    )
    out = attn.forward_oot(query, query, query)
    assert torch.equal(out, torch.full_like(query, 3.0))


def test_vision_flywheel_d64_even_mha_uses_unpadded_token_major(monkeypatch):
    flywheel = ensure_flywheel_enum_registered()
    captured = {}

    def fake_tm_op(q, k, v, cu):
        captured["q"] = q
        captured["k"] = k
        captured["v"] = v
        captured["cu"] = cu
        return q + 1.0

    def fail_hm_op(*args, **kwargs):
        raise AssertionError("even MHA d=64 should use token_major fast path")

    monkeypatch.setattr(TpuMMEncoderAttention, "_flywheel_token_major_op", fake_tm_op)
    monkeypatch.setattr(TpuMMEncoderAttention, "_flywheel_head_major_op", fail_hm_op)

    attn = make_attention(
        num_heads=12,
        head_size=64,
        scale=0.5,
        attn_backend=flywheel,
    )
    q = torch.full((2, 10, 12, 64), 2.0, dtype=torch.float32)
    k = torch.ones_like(q)
    v = torch.ones_like(q)

    out = attn.forward_oot(q, k, v, cu_seqlens=None)
    assert out.shape == q.shape
    assert out.dtype == torch.float32
    assert captured["q"].shape == (20, 768)
    assert captured["q"].dtype == torch.bfloat16
    assert torch.all(captured["q"] == 1.0)
    assert captured["cu"].tolist() == [0, 10, 20]


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_size"),
    [
        (12, 4, 64),
        (16, 16, 72),
        (16, 16, 80),
        (8, 8, 128),
    ],
)
def test_vision_flywheel_general_and_gqa_pads_head_dim_to_128(
    monkeypatch, num_heads, num_kv_heads, head_size
):
    flywheel = ensure_flywheel_enum_registered()
    captured = {}

    def fake_hm_op(q, k, v, cu):
        captured["q"] = q
        captured["k"] = k
        captured["v"] = v
        captured["cu"] = cu
        return q

    monkeypatch.setattr(TpuMMEncoderAttention, "_flywheel_head_major_op", fake_hm_op)

    attn = make_attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        scale=0.25,
        attn_backend=flywheel,
    )
    q = torch.ones((1, 7, num_heads, head_size), dtype=torch.bfloat16)
    k = torch.ones((1, 7, num_kv_heads, head_size), dtype=torch.bfloat16)
    v = torch.ones((1, 7, num_kv_heads, head_size), dtype=torch.bfloat16)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)

    out = attn.forward_oot(q, k, v, cu_seqlens=cu)
    assert out.shape == q.shape
    assert captured["q"].shape == (num_heads, 7, 128)
    assert captured["k"].shape == (num_kv_heads, 7, 128)
    if head_size < 128:
        assert torch.all(captured["q"][:, :, head_size:] == 0)
        assert torch.all(captured["k"][:, :, head_size:] == 0)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_dim", "scale"),
    [
        (12, 12, 64, 1.0),
        (12, 4, 64, 0.125),
        (16, 16, 72, None),
        (16, 16, 80, None),
        (8, 8, 128, None),
    ],
)
@pytest.mark.parametrize("packed", [True, False])
def test_tpu_vision_flywheel_numerical_parity(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float | None,
    packed: bool,
):
    flywheel = ensure_flywheel_enum_registered()
    sm_scale = (head_dim**-0.5) if scale is None else scale

    fw_attn = make_attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        scale=sm_scale,
        attn_backend=flywheel,
    )
    if TpuMMEncoderAttention._flywheel_head_major_op is None:
        from vllm_torchtpu.layers.adapter.vision_attention import _build_flywheel_ops

        tm_op, hm_op = _build_flywheel_ops()
        TpuMMEncoderAttention._flywheel_token_major_op = tm_op
        TpuMMEncoderAttention._flywheel_head_major_op = hm_op

    device = torch.device("tpu:0")
    torch.manual_seed(42)
    if packed:
        bsz, seq_len = 1, 300
        # Includes a zero-length segment (196 -> 196) to verify padded cu_seqlens.
        cu_seqlens = torch.tensor(
            [0, 100, 196, 196, 300], dtype=torch.int32, device=device
        )
    else:
        bsz, seq_len = 2, 144
        cu_seqlens = None

    qk_std = (head_dim * (sm_scale**2)) ** -0.25
    q_cpu = (
        torch.randn(bsz, seq_len, num_heads, head_dim, dtype=torch.float32) * qk_std
    ).to(torch.bfloat16)
    k_cpu = (
        torch.randn(bsz, seq_len, num_kv_heads, head_dim, dtype=torch.float32) * qk_std
    ).to(torch.bfloat16)
    v_cpu = torch.randn(bsz, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16)

    ref_cpu = _reference_varlen_sdpa(
        q_cpu,
        k_cpu,
        v_cpu,
        cu_seqlens.cpu() if cu_seqlens is not None else None,
        sm_scale,
    )

    out_fw = fw_attn.forward_oot(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        cu_seqlens=cu_seqlens,
    ).cpu()

    torch.testing.assert_close(out_fw, ref_cpu, atol=2e-2, rtol=2e-2)

    if num_heads == num_kv_heads:
        if TpuMMEncoderAttention._vision_op is None:
            from torch_tpu._internal import pallas

            from vllm_torchtpu.layers.adapter.vision_attention import _vision_flash_core

            op = pallas.jax_op("pallas::vision_flash_attention", _vision_flash_core)
            op.register_fake(lambda q, k, v, q_seg, kv_seg: torch.empty_like(q))
            TpuMMEncoderAttention._vision_op = op

        flash_attn = make_attention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            scale=sm_scale,
            attn_backend=AttentionBackendEnum.FLASH_ATTN,
        )
        out_flash = flash_attn.forward_oot(
            q_cpu.to(device),
            k_cpu.to(device),
            v_cpu.to(device),
            cu_seqlens=cu_seqlens,
        ).cpu()
        torch.testing.assert_close(out_fw, out_flash, atol=2e-2, rtol=2e-2)


@pytest.mark.tpu_test
def test_tpu_vision_flywheel_fullgraph_compile():
    flywheel = ensure_flywheel_enum_registered()
    if TpuMMEncoderAttention._flywheel_head_major_op is None:
        from vllm_torchtpu.layers.adapter.vision_attention import _build_flywheel_ops

        tm_op, hm_op = _build_flywheel_ops()
        TpuMMEncoderAttention._flywheel_token_major_op = tm_op
        TpuMMEncoderAttention._flywheel_head_major_op = hm_op

    fw_attn = make_attention(
        num_heads=12,
        num_kv_heads=12,
        head_size=64,
        scale=1.0,
        attn_backend=flywheel,
    )

    def encoder_step(q, k, v, cu):
        bsz, seq_len = q.shape[:2]
        total = bsz * seq_len
        return fw_attn.forward_oot(
            q.reshape(1, total, 12, 64),
            k.reshape(1, total, 12, 64),
            v.reshape(1, total, 12, 64),
            cu_seqlens=cu,
        ).reshape(bsz, seq_len, 12, 64)

    compiled = torch.compile(encoder_step, backend="tpu", fullgraph=True, dynamic=False)

    device = torch.device("tpu:0")
    torch.manual_seed(7)
    bsz, seq_len = 2, 128
    qk_std = 64**-0.25
    q_cpu = (torch.randn(bsz, seq_len, 12, 64, dtype=torch.float32) * qk_std).to(
        torch.bfloat16
    )
    k_cpu = (torch.randn(bsz, seq_len, 12, 64, dtype=torch.float32) * qk_std).to(
        torch.bfloat16
    )
    v_cpu = torch.randn(bsz, seq_len, 12, 64, dtype=torch.bfloat16)
    cu_cpu = torch.tensor([0, 96, 128, 256], dtype=torch.int32)

    out_tpu = compiled(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        cu_cpu.to(device),
    ).cpu()

    ref = _reference_varlen_sdpa(
        q_cpu.reshape(1, bsz * seq_len, 12, 64),
        k_cpu.reshape(1, bsz * seq_len, 12, 64),
        v_cpu.reshape(1, bsz * seq_len, 12, 64),
        cu_cpu,
        1.0,
    ).reshape(bsz, seq_len, 12, 64)

    torch.testing.assert_close(out_tpu, ref, atol=2e-2, rtol=2e-2)
