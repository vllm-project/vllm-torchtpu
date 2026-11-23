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
    positions: torch.Tensor,
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int | None = None,
) -> torch.Tensor:
    """``apply_token_rotary_embedding`` (NeoX): load BF16 x and BF16 cache
    entries, rotate in FP32, store once in BF16."""
    rotary_dim = cos_sin_cache.shape[-1]
    head_size = head_size or rotary_dim
    cos, sin = cos_sin_cache.to(x.dtype)[positions].chunk(2, dim=-1)
    x_3d = x.view(positions.numel(), -1, head_size)
    xf = x_3d[..., :rotary_dim].float()
    x1, x2 = xf.chunk(2, dim=-1)
    c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    rot = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)
    return torch.cat((rot, x_3d[..., rotary_dim:]), dim=-1).reshape(x.shape)


def test_oot_classes_registered():
    assert op_registry_oot["ApplyRotaryEmb"] is tpu_rope.TpuApplyRotaryEmb
    for name, cls in tpu_rope.TPU_ROPE_CLASSES.items():
        assert op_registry_oot[name] is cls
        assert cls.__name__ == f"Tpu{name}"
    assert "RotaryEmbedding" in tpu_rope.TPU_ROPE_CLASSES
    assert "YaRNScalingRotaryEmbedding" in tpu_rope.TPU_ROPE_CLASSES
    assert "Gemma4RotaryEmbedding" in tpu_rope.TPU_ROPE_CLASSES


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


@pytest.mark.parametrize(
    ("head_size", "partial_rotary_factor", "with_key"),
    [
        (128, 0.25, True),
        (256, 0.5, True),
        (512, 0.25, True),
        (512, 0.25, False),
    ],
)
def test_proportional_neox_rope_compacts_cache_and_matches_reference(
    head_size: int,
    partial_rotary_factor: float,
    with_key: bool,
):
    """Proportional NeoX RoPE compacts cos_sin_cache and matches CUDA FP32 bit-exact."""
    rope = get_rope(
        head_size,
        max_position=MAX_POS,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "proportional",
            "partial_rotary_factor": partial_rotary_factor,
            "rope_theta": 1_000_000.0,
        },
        dtype=torch.bfloat16,
    )
    assert type(rope) is tpu_rope.TPU_ROPE_CLASSES["Gemma4RotaryEmbedding"]
    rot_angles = int(partial_rotary_factor * head_size) // 2
    assert rope.cos_sin_cache.shape == (MAX_POS, 2 * rot_angles)
    assert dict(rope.named_buffers()).keys() == {"cos_sin_cache"}

    full_cache = rope._compute_cos_sin_cache().to(torch.bfloat16)
    assert full_cache.shape == (MAX_POS, head_size)

    g = torch.Generator().manual_seed(3)
    positions = torch.randint(0, MAX_POS, (24,), generator=g)
    q = torch.randn(24, 4 * head_size, generator=g).to(torch.bfloat16)
    k = (
        torch.randn(24, 2 * head_size, generator=g).to(torch.bfloat16)
        if with_key
        else None
    )

    out_q, out_k = rope.forward_oot(
        positions, q.clone(), k.clone() if k is not None else None
    )
    assert torch.equal(out_q, _cuda_rope_reference(positions, q, full_cache))

    half = head_size // 2
    q_3d = q.view(24, 4, head_size)
    out_q_3d = out_q.view(24, 4, head_size)
    assert torch.equal(out_q_3d[..., rot_angles:half], q_3d[..., rot_angles:half])
    assert torch.equal(
        out_q_3d[..., half + rot_angles :], q_3d[..., half + rot_angles :]
    )

    if k is not None:
        assert out_k is not None
        assert torch.equal(out_k, _cuda_rope_reference(positions, k, full_cache))
    else:
        assert out_k is None

    # Restoring the full-width cache exercises the uncompacted static path.
    rope.register_buffer("cos_sin_cache", full_cache.clone(), persistent=False)
    fallback_q, fallback_k = rope.forward_oot(
        positions, q.clone(), k.clone() if k is not None else None
    )
    assert torch.equal(fallback_q, out_q)
    if k is not None:
        assert fallback_k is not None and out_k is not None
        assert torch.equal(fallback_k, out_k)
    else:
        assert fallback_k is None


@pytest.mark.parametrize("rotary_dim", [128, 96])
def test_generic_neox_rope_compacts_trailing_identity_pairs(rotary_dim: int):
    """Any NeoX RotaryEmbedding with trailing zero inv_freq compacts cleanly."""

    def _zero_padded_inv_freq(self: RotaryEmbedding, base: float) -> torch.Tensor:
        active = 1.0 / (base ** (torch.arange(0, 32, 2, dtype=torch.float) / 32.0))
        zeros = torch.zeros(self.rotary_dim // 2 - 16, dtype=torch.float)
        return torch.cat((active, zeros))

    zero_padded_cls = type(
        "_ZeroPaddedNeoXRope",
        (tpu_rope.TpuRotaryEmbedding,),
        {"_compute_inv_freq": _zero_padded_inv_freq},
    )
    rope = zero_padded_cls(HEAD, rotary_dim, MAX_POS, 10000.0, True, torch.bfloat16)
    assert not isinstance(rope, upstream_rope.Gemma4RotaryEmbedding)
    assert rope.cos_sin_cache.shape == (MAX_POS, 32)

    full_cache = rope._compute_cos_sin_cache().to(torch.bfloat16)
    assert full_cache.shape == (MAX_POS, rotary_dim)

    g = torch.Generator().manual_seed(5)
    positions = torch.randint(0, MAX_POS, (20,), generator=g)
    q = torch.randn(20, 4 * HEAD, generator=g).to(torch.bfloat16)
    k = torch.randn(20, 2 * HEAD, generator=g).to(torch.bfloat16)

    out_q, out_k = rope.forward_oot(positions, q.clone(), k.clone())
    assert out_k is not None
    assert torch.equal(
        out_q, _cuda_rope_reference(positions, q, full_cache, head_size=HEAD)
    )
    assert torch.equal(
        out_k, _cuda_rope_reference(positions, k, full_cache, head_size=HEAD)
    )


def test_compacted_neox_rope_compiles_fullgraph():
    """Compacted NeoX RoPE traces cleanly under torch.compile(..., fullgraph=True)."""
    rope = get_rope(
        512,
        max_position=MAX_POS,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
        dtype=torch.bfloat16,
    )
    g = torch.Generator().manual_seed(4)
    positions = torch.randint(0, MAX_POS, (16,), generator=g)
    q = torch.randn(16, 4, 512, generator=g).to(torch.bfloat16)
    k = torch.randn(16, 2, 512, generator=g).to(torch.bfloat16)

    eager_q, eager_k = rope.forward_oot(positions, q.clone(), k.clone())
    compiled = torch.compile(rope.forward_oot, backend="eager", fullgraph=True)
    comp_q, comp_k = compiled(positions, q.clone(), k.clone())
    assert comp_k is not None and eager_k is not None
    assert torch.equal(comp_q, eager_q)
    assert torch.equal(comp_k, eager_k)


def test_neox_rope_skips_compaction_when_guards_fail():
    """Full-frequency, GPT-J, mscale != 1, and non-identity caches do not compact."""
    full_factor = get_rope(
        HEAD,
        max_position=MAX_POS,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "proportional",
            "partial_rotary_factor": 1.0,
            "rope_theta": 1_000_000.0,
        },
        dtype=torch.bfloat16,
    )
    assert full_factor.cos_sin_cache.shape == (MAX_POS, HEAD)

    gptj = get_rope(
        HEAD,
        max_position=MAX_POS,
        is_neox_style=False,
        rope_parameters={
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
        dtype=torch.bfloat16,
    )
    assert gptj.cos_sin_cache.shape == (MAX_POS, HEAD)

    # Trailing zero inv_freq with mscale != 1 produces cos = mscale != 1, which
    # scales trailing channels rather than leaving them as identity.
    def _zero_padded_yarn_inv_freq(
        self: YaRNScalingRotaryEmbedding, scaling_factor: float
    ) -> torch.Tensor:
        active = YaRNScalingRotaryEmbedding._compute_inv_freq(self, scaling_factor)[:16]
        return torch.cat((active, torch.zeros(48, dtype=torch.float)))

    zero_padded_yarn_cls = type(
        "_ZeroPaddedYaRNRope",
        (tpu_rope.TPU_ROPE_CLASSES["YaRNScalingRotaryEmbedding"],),
        {"_compute_inv_freq": _zero_padded_yarn_inv_freq},
    )
    mscaled = zero_padded_yarn_cls(
        HEAD, HEAD, 1024, 10000.0, True, 4.0, torch.bfloat16, attention_factor=1.25
    )
    assert mscaled.cos_sin_cache.shape == (4096, HEAD)
    q_ones = torch.ones(4, HEAD, dtype=torch.bfloat16)
    pos = torch.tensor([1, 2, 3, 4])
    mscaled_q, _ = mscaled.forward_oot(pos, q_ones.clone(), None)
    assert torch.equal(
        mscaled_q, _cuda_rope_reference(pos, q_ones, mscaled.cos_sin_cache)
    )
    assert not torch.equal(mscaled_q[:, 16:64], q_ones[:, 16:64])

    # Perturbing the last cos column prevents compaction from dropping active pairs.
    perturbed = get_rope(
        HEAD,
        max_position=MAX_POS,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
        dtype=torch.bfloat16,
    )
    corrupted = perturbed._compute_cos_sin_cache().to(torch.bfloat16)
    corrupted[:, 63] = 0.5
    perturbed.register_buffer("cos_sin_cache", corrupted, persistent=False)
    tpu_rope._maybe_compact_neox_rope_cache(perturbed)
    assert perturbed.cos_sin_cache.shape == (MAX_POS, HEAD)
