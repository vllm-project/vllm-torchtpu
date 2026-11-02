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
"""Out-of-tree FP32 rotary embedding replaces vLLM's BF16 reference on TPU."""

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers import rotary_embedding as upstream_rope
from vllm.model_executor.layers.rotary_embedding import (
    DynamicNTKScalingRotaryEmbedding,
    FourierRotaryEmbedding,
    Llama3RotaryEmbedding,
    MRotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
    get_rope,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_torchtpu.layers.adapter import rotary_embedding as tpu_rope

HEAD = 128
MAX_POS = 4096


@pytest.fixture(autouse=True)
def _vllm_config(monkeypatch):
    # CustomOp.__init__ reads the compilation config to pick its forward.
    # get_rope memoizes instances in _ROPE_DICT; an entry built by an earlier
    # test (before this module registered the OOT classes) must not leak in.
    monkeypatch.setattr(upstream_rope, "_ROPE_DICT", {})
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _cuda_rope_reference(
    positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """``apply_token_rotary_embedding`` (NeoX): load BF16 x and BF16 cache
    entries, rotate in FP32, store once in BF16."""
    cos, sin = cos_sin_cache.to(x.dtype)[positions].chunk(2, dim=-1)
    xf = x.view(positions.numel(), -1, HEAD).float()
    x1, x2 = xf.chunk(2, dim=-1)
    c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    out = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    return out.to(x.dtype).reshape(x.shape)


def test_oot_classes_registered():
    assert op_registry_oot["ApplyRotaryEmb"] is tpu_rope.TpuApplyRotaryEmb
    for name, cls in tpu_rope.TPU_ROPE_CLASSES.items():
        assert op_registry_oot[name] is cls
        assert cls.__name__ == f"Tpu{name}"
    assert "RotaryEmbedding" in tpu_rope.TPU_ROPE_CLASSES
    assert "YaRNScalingRotaryEmbedding" in tpu_rope.TPU_ROPE_CLASSES


def test_discovery_covers_every_static_path_subclass():
    """Every upstream RotaryEmbedding subclass that still runs
    RotaryEmbedding.forward_static must have a TPU override; classes with
    their own forward_native must not (they are served via TpuApplyRotaryEmb
    or need a dedicated override)."""
    covered = set(tpu_rope.TPU_ROPE_CLASSES)
    for cls in (Llama3RotaryEmbedding, DynamicNTKScalingRotaryEmbedding):
        assert "forward_native" not in cls.__dict__
        assert cls.__name__ in covered
    for cls in (MRotaryEmbedding, FourierRotaryEmbedding):
        assert "forward_native" in cls.__dict__
        assert cls.__name__ not in covered
    # No upstream subclass slipped through.
    for cls in tpu_rope._all_subclasses(RotaryEmbedding):
        is_upstream = cls.__module__.startswith("vllm.")
        if is_upstream and tpu_rope._uses_static_rotary_forward(cls):
            assert cls.__name__ in covered, cls


def test_apply_rotary_emb_forces_fp32_for_positional_and_keyword_args():
    assert ApplyRotaryEmb().enable_fp32_compute is True
    kw = ApplyRotaryEmb(is_neox_style=False, enable_fp32_compute=False)
    assert kw.enable_fp32_compute is True
    # upstream signature: (enforce_enable, is_neox_style, enable_fp32_compute)
    emb = ApplyRotaryEmb(False, False, False)
    assert emb.enable_fp32_compute is True
    assert emb.is_neox_style is False


def test_upstream_constructors_yield_tpu_classes():
    rope = RotaryEmbedding(HEAD, HEAD, MAX_POS, 10000.0, True, torch.bfloat16)
    assert type(rope) is tpu_rope.TpuRotaryEmbedding
    assert type(rope.apply_rotary_emb) is tpu_rope.TpuApplyRotaryEmb
    assert rope.apply_rotary_emb.enable_fp32_compute is True

    yarn = YaRNScalingRotaryEmbedding(
        HEAD, HEAD, 1024, 10000.0, True, 4.0, torch.bfloat16
    )
    assert type(yarn) is tpu_rope.TPU_ROPE_CLASSES["YaRNScalingRotaryEmbedding"]

    # get_rope is what model code calls; the OOT swap must survive it.
    via_get_rope = get_rope(HEAD, MAX_POS, dtype=torch.bfloat16)
    assert isinstance(via_get_rope, tpu_rope.TpuRotaryEmbedding)


def test_mrope_apply_rotary_emb_is_fp32():
    mrope = MRotaryEmbedding(
        HEAD, HEAD, MAX_POS, 10000.0, True, torch.bfloat16, mrope_section=[16, 24, 24]
    )
    assert type(mrope.apply_rotary_emb) is tpu_rope.TpuApplyRotaryEmb
    assert mrope.apply_rotary_emb.enable_fp32_compute is True


def test_rotary_embedding_matches_cuda_reference():
    rope = RotaryEmbedding(HEAD, HEAD, MAX_POS, 10000.0, True, torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    positions = torch.randint(0, MAX_POS, (32,), generator=g)
    q = torch.randn(32, 4 * HEAD, generator=g).to(torch.bfloat16)
    k = torch.randn(32, 2 * HEAD, generator=g).to(torch.bfloat16)

    out_q, out_k = rope.forward_native(positions, q.clone(), k.clone())
    cache = rope.cos_sin_cache
    torch.testing.assert_close(
        out_q, _cuda_rope_reference(positions, q, cache), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_k, _cuda_rope_reference(positions, k, cache), rtol=0, atol=0
    )
    # forward_oot/forward_tpu/forward are the same implementation.
    oot_q, _ = rope.forward_oot(positions, q.clone(), k.clone())
    fwd_q, _ = rope(positions, q.clone(), k.clone())
    torch.testing.assert_close(oot_q, out_q, rtol=0, atol=0)
    torch.testing.assert_close(fwd_q, out_q, rtol=0, atol=0)


def test_fp32_rotation_is_closer_to_fp64_than_upstream_bf16_path():
    """The upstream static path (enable_fp32_compute=False) rotates in BF16.
    Ours must never be further from the exact rotation than that."""
    rope = RotaryEmbedding(HEAD, HEAD, MAX_POS, 10000.0, True, torch.bfloat16)
    g = torch.Generator().manual_seed(1)
    positions = torch.randint(0, MAX_POS, (256,), generator=g)
    q = torch.randn(256, 8 * HEAD, generator=g).to(torch.bfloat16)
    cache = rope._match_cos_sin_cache_dtype(q)

    tpu_q, _ = rope.forward_native(positions, q.clone(), None)
    upstream_q, _ = RotaryEmbedding.forward_static(
        positions, q.clone(), None, HEAD, HEAD, cache, True
    )
    cos, sin = cache.double()[positions].chunk(2, dim=-1)
    x1, x2 = q.view(256, -1, HEAD).double().chunk(2, dim=-1)
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    exact = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).reshape(q.shape)

    err_tpu = (tpu_q.double() - exact).abs().mean()
    err_upstream = (upstream_q.double() - exact).abs().mean()
    assert not torch.equal(tpu_q, upstream_q)
    assert err_tpu < err_upstream


def test_apply_rotary_emb_instance_forward_is_fp32():
    emb = ApplyRotaryEmb(is_neox_style=True)
    assert type(emb) is tpu_rope.TpuApplyRotaryEmb
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 16, 4, HEAD, generator=g).to(torch.bfloat16)
    cos = torch.randn(16, HEAD // 2, generator=g).to(torch.bfloat16)
    sin = torch.randn(16, HEAD // 2, generator=g).to(torch.bfloat16)
    ref = ApplyRotaryEmb.forward_static(x, cos, sin, True, enable_fp32_compute=True)
    torch.testing.assert_close(emb(x, cos, sin), ref, rtol=0, atol=0)
    torch.testing.assert_close(emb.forward_native(x, cos, sin), ref, rtol=0, atol=0)
