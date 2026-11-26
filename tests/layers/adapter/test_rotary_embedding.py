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
from vllm.config import (
    DeviceConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
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

from vllm_torchtpu import rope_in_graph_flag
from vllm_torchtpu.layers.adapter import rotary_embedding as tpu_rope

HEAD = 128
MAX_POS = 4096


@pytest.fixture(autouse=True)
def _vllm_config(monkeypatch):
    # CustomOp.__init__ reads the compilation config to pick its forward.
    # get_rope memoizes instances in _ROPE_DICT; an entry built by an earlier
    # test (before this module registered the OOT classes) must not leak in.
    monkeypatch.setattr(upstream_rope, "_ROPE_DICT", {})
    # Table-path tests must not depend on the `tpu_rope_in_graph` default;
    # in-graph tests opt in explicitly.
    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        additional_config={rope_in_graph_flag.KEY: False},
    )
    with set_current_vllm_config(cfg):
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


def test_vllm_rotary_subclass_exhaustiveness_guard():
    """Every upstream vLLM rotary class is mapped in `rope_class_registry`."""
    import importlib
    import pkgutil

    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase

    from vllm_torchtpu import rope_in_graph
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    # Import every submodule (including lazy ones like ernie45_vl_rope) so
    # __subclasses__() sees the full upstream inventory.
    for mod_info in pkgutil.iter_modules(upstream_rope.__path__):
        importlib.import_module(f"{upstream_rope.__name__}.{mod_info.name}")

    upstream_rotary_base_subclasses = {
        cls
        for cls in tpu_rope._all_subclasses(RotaryEmbeddingBase)
        if cls.__module__.startswith("vllm.")
    }

    registry = rope_in_graph.rope_class_registry()
    registry_rotary_base = {
        cls for cls in registry if issubclass(cls, RotaryEmbeddingBase)
    }
    assert upstream_rotary_base_subclasses == registry_rotary_base, (
        "Unclassified RotaryEmbeddingBase subclass(es): "
        f"{upstream_rotary_base_subclasses ^ registry_rotary_base}"
    )

    for cls, entry in registry.items():
        if entry.tier1 is not None:
            tpu_cls = op_registry_oot[cls.__name__]
            assert rope_in_graph.Tier1InGraphRopeMixin in tpu_cls.__bases__
        elif entry.build_spec is not None:
            assert not issubclass(cls, RotaryEmbeddingBase)
        else:
            assert rope_in_graph.spec_from_module(cls.__new__(cls)) is None


@pytest.mark.parametrize(
    "case",
    [
        "default",
        "gemma4_proportional",
        "llama3",
        "ntk",
        "linear_single",
        "linear_multi",
        "dynamic_ntk",
        "dynamic_ntk_alpha",
        "yarn",
        "deepseek_scaling",
        "mrope_contiguous",
        "mrope_yarn",
        "mrope_interleaved",
        "ernie45_vl",
        "bailing_mrope",
        "phi3_long_rope",
    ],
)
def test_in_scope_rotary_classes_match_upstream_cpu_table_and_table_forward(
    case: str,
):
    """In-graph rows match the upstream CPU table, forwards the flag-off forward."""
    from vllm.model_executor.layers.rotary_embedding import (
        BailingMRotaryEmbedding,
        DeepseekScalingRotaryEmbedding,
        DynamicNTKAlphaRotaryEmbedding,
        LinearScalingRotaryEmbedding,
        Llama3RotaryEmbedding,
        MRotaryEmbeddingInterleaved,
        NTKScalingRotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
        Ernie4_5_VLRotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.gemma4_rope import (
        Gemma4RotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope import (
        Phi3LongRoPEScaledRotaryEmbedding,
    )

    from vllm_torchtpu import rope_in_graph, rope_patch
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    def _construct() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor | None]:
        pos_1d = torch.tensor([0, 1, 7, 63, 128, 255], dtype=torch.int64)
        pos_3d = torch.tensor(
            [
                [0, 1, 5, 19, 37, 63],
                [0, 2, 11, 23, 41, 61],
                [0, 3, 13, 29, 43, 59],
            ],
            dtype=torch.int64,
        )
        if case == "default":
            return (
                RotaryEmbedding(64, 64, 256, 10000.0, True, torch.float32),
                pos_1d,
                None,
            )
        if case == "gemma4_proportional":
            return (
                Gemma4RotaryEmbedding(64, 16, 256, 1000000.0, True, torch.float32),
                pos_1d,
                None,
            )
        if case == "llama3":
            return (
                Llama3RotaryEmbedding(
                    64, 64, 256, 500000.0, True, torch.float32, 8.0, 1.0, 4.0, 8192
                ),
                pos_1d,
                None,
            )
        if case == "ntk":
            return (
                NTKScalingRotaryEmbedding(
                    64, 32, 256, 10000.0, False, 4.0, torch.float32
                ),
                pos_1d,
                None,
            )
        if case == "linear_single":
            return (
                LinearScalingRotaryEmbedding(
                    64, 64, 128, 10000.0, True, 2.0, torch.float32
                ),
                pos_1d,
                None,
            )
        if case == "linear_multi":
            mod = LinearScalingRotaryEmbedding(
                64, 64, 128, 10000.0, True, [1.0, 2.0, 4.0], torch.float32
            )
            offs = torch.tensor([0, 0, 128, 128, 384, 384], dtype=torch.int64)
            return mod, torch.tensor([0, 7, 5, 120, 9, 250], dtype=torch.int64), offs
        if case == "dynamic_ntk":
            return (
                DynamicNTKScalingRotaryEmbedding(
                    64, 64, 512, 128, 10000.0, True, 4.0, torch.float32
                ),
                pos_1d,
                None,
            )
        if case == "dynamic_ntk_alpha":
            return (
                DynamicNTKAlphaRotaryEmbedding(
                    64, 64, 512, 10000.0, True, 4.0, torch.float32
                ),
                pos_1d,
                None,
            )
        if case == "yarn":
            return (
                YaRNScalingRotaryEmbedding(
                    64,
                    64,
                    128,
                    10000.0,
                    True,
                    4.0,
                    torch.float32,
                    attention_factor=1.25,
                ),
                pos_1d,
                None,
            )
        if case == "deepseek_scaling":
            return (
                DeepseekScalingRotaryEmbedding(
                    64,
                    64,
                    128,
                    10000.0,
                    False,
                    8.0,
                    torch.float32,
                    beta_fast=32,
                    beta_slow=1,
                    mscale=0.707,
                    mscale_all_dim=0.707,
                ),
                pos_1d,
                None,
            )
        if case == "mrope_contiguous":
            return (
                MRotaryEmbedding(
                    64,
                    64,
                    128,
                    10000.0,
                    True,
                    torch.float32,
                    mrope_section=[8, 12, 12],
                    mrope_interleaved=False,
                ),
                pos_3d,
                None,
            )
        if case == "mrope_yarn":
            return (
                MRotaryEmbedding(
                    64,
                    64,
                    128,
                    10000.0,
                    True,
                    torch.float32,
                    mrope_section=[8, 12, 12],
                    mrope_interleaved=True,
                    scaling_factor=4.0,
                ),
                pos_3d,
                None,
            )
        if case == "mrope_interleaved":
            return (
                MRotaryEmbeddingInterleaved(
                    64,
                    64,
                    128,
                    10000.0,
                    True,
                    torch.float32,
                    mrope_section=[11, 11, 10],
                ),
                pos_3d,
                None,
            )
        if case == "ernie45_vl":
            return (
                Ernie4_5_VLRotaryEmbedding(
                    64,
                    64,
                    128,
                    10000.0,
                    False,
                    torch.float32,
                    mrope_section=[10, 10, 12],
                ),
                pos_3d,
                None,
            )
        if case == "bailing_mrope":
            return (
                BailingMRotaryEmbedding(
                    64,
                    64,
                    128,
                    10000.0,
                    True,
                    torch.float32,
                    mrope_section=[12, 10, 10],
                ),
                pos_3d,
                None,
            )
        return (
            Phi3LongRoPEScaledRotaryEmbedding(
                64,
                48,
                512,
                128,
                10000.0,
                True,
                torch.float32,
                short_factor=[1.0 + 0.05 * i for i in range(24)],
                long_factor=[2.0 + 0.25 * i for i in range(24)],
            ),
            torch.tensor([0, 1, 63, 127, 128, 400], dtype=torch.int64),
            None,
        )

    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        additional_config={rope_in_graph_flag.KEY: False},
    )
    cfg.model_config = type("MC", (), {"max_model_len": 512})()  # type: ignore[assignment]
    with set_current_vllm_config(cfg):
        ref, positions, offsets = _construct()
        assert sum(b.numel() for b in ref.buffers()) > 0

        cfg.additional_config[rope_in_graph_flag.KEY] = True
        uut, _, _ = _construct()

    cache_attr = (
        "long_short_cos_sin_cache" if case == "phi3_long_rope" else "cos_sin_cache"
    )
    ref_table = getattr(ref, cache_attr)
    assert isinstance(ref_table, torch.Tensor)
    if case != "phi3_long_rope":
        # `_maybe_compact_neox_rope_cache` may narrow the table-path buffer;
        # in-graph RoPE always serves the full upstream width.
        ref_table = ref._compute_cos_sin_cache().to(ref_table.dtype)

    if case == "phi3_long_rope":
        # Tier 2 converts plain-nn.Module Phi3LongRoPE post-load.
        assert (
            rope_patch.patch_in_graph_rotary_embedding(uut, allow_table_recovery=False)
            == 1
        )
    else:
        # Tier 1 allocates 0 buffer elements at __init__ BEFORE Tier 2 runs.
        assert sum(b.numel() for b in uut.buffers()) == 0
        assert (
            rope_patch.patch_in_graph_rotary_embedding(uut, allow_table_recovery=False)
            == 0
        )

    assert sum(b.numel() for b in uut.buffers()) == 0
    uut_cache = getattr(uut, cache_attr)
    assert isinstance(uut_cache, rope_in_graph.InGraphCosSinCache)
    assert uut_cache.shape == ref_table.shape

    sample_idx = torch.tensor(
        [0, 1, 7, ref_table.shape[0] // 2, ref_table.shape[0] - 1], dtype=torch.int64
    )
    assert torch.equal(uut_cache[sample_idx], ref_table[sample_idx])

    g = torch.Generator().manual_seed(42)
    n_tok = positions.shape[-1]
    if case == "deepseek_scaling":
        q = torch.randn(n_tok, 4, 64, generator=g, dtype=torch.float32)
        k = torch.randn(n_tok, 1, 64, generator=g, dtype=torch.float32)
    else:
        q = torch.randn(n_tok, 4 * 64, generator=g, dtype=torch.float32)
        k = torch.randn(n_tok, 2 * 64, generator=g, dtype=torch.float32)

    if offsets is not None:
        ref_q, ref_k = ref(positions, q.clone(), k.clone(), offsets=offsets)
        out_q, out_k = uut(positions, q.clone(), k.clone(), offsets=offsets)
    else:
        ref_q, ref_k = ref(positions, q.clone(), k.clone())
        out_q, out_k = uut(positions, q.clone(), k.clone())

    torch.testing.assert_close(out_q, ref_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-6, atol=1e-6)


def test_tier1_meta_to_empty_tracks_device():
    """Tier-1 `InGraphCosSinCache` tracks `to_empty()` across rotary families."""
    from vllm.model_executor.layers.rotary_embedding import (
        DeepseekScalingRotaryEmbedding,
    )

    from vllm_torchtpu import rope_in_graph
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    get_current_vllm_config().additional_config[rope_in_graph_flag.KEY] = True
    with torch.device("meta"):
        rope = RotaryEmbedding(64, 64, 128, 10000.0, True, torch.bfloat16)
        mrope = MRotaryEmbedding(
            64, 64, 128, 10000.0, True, torch.bfloat16, mrope_section=[8, 12, 12]
        )
        ds_rope = DeepseekScalingRotaryEmbedding(
            64, 64, 128, 10000.0, False, 4.0, torch.bfloat16
        )

    model = torch.nn.ModuleDict({"r": rope, "m": mrope, "d": ds_rope})
    for mod in (rope, mrope, ds_rope):
        assert isinstance(mod.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
        assert mod.cos_sin_cache.device.type == "meta"
        assert sum(b.numel() for b in mod.buffers()) == 0

    model.to_empty(device="cpu")
    for mod in (rope, mrope, ds_rope):
        assert mod.cos_sin_cache.device == torch.device("cpu")

    cos, sin = rope.get_cos_sin(16)
    assert cos.device == torch.device("cpu") and cos.shape == (16, 32)
    assert sin.device == torch.device("cpu") and sin.shape == (16, 32)


def test_dynamo_fullgraph_compile_across_tier1_and_tier2_paths():
    """`torch.compile(..., fullgraph=True)` traces Tier-1 & Tier-2 with 0 breaks."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.rotary_embedding import (
        BailingMRotaryEmbedding,
        DeepseekScalingRotaryEmbedding,
        LinearScalingRotaryEmbedding,
        MRotaryEmbeddingInterleaved,
    )
    from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
        Ernie4_5_VLRotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope import (
        Phi3LongRoPEScaledRotaryEmbedding,
    )

    from vllm_torchtpu import rope_in_graph, rope_patch
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        additional_config={rope_in_graph_flag.KEY: True},
    )
    cfg.model_config = SimpleNamespace(max_model_len=512)  # type: ignore[assignment]
    with set_current_vllm_config(cfg):
        r_std = RotaryEmbedding(64, 64, 256, 10000.0, True, torch.bfloat16)
        r_lin = LinearScalingRotaryEmbedding(
            64, 64, 128, 10000.0, True, [1.0, 2.0], torch.bfloat16
        )
        r_ds = DeepseekScalingRotaryEmbedding(
            64,
            64,
            128,
            10000.0,
            False,
            4.0,
            torch.bfloat16,
            mscale=0.707,
            mscale_all_dim=0.707,
        )
        r_mint = MRotaryEmbeddingInterleaved(
            64, 64, 128, 10000.0, True, torch.bfloat16, mrope_section=[11, 11, 10]
        )
        r_ernie = Ernie4_5_VLRotaryEmbedding(
            64, 64, 128, 10000.0, False, torch.bfloat16, mrope_section=[10, 10, 12]
        )
        r_bail = BailingMRotaryEmbedding(
            64, 64, 128, 10000.0, True, torch.bfloat16, mrope_section=[12, 10, 10]
        )
        r_phi = Phi3LongRoPEScaledRotaryEmbedding(
            64,
            48,
            512,
            128,
            10000.0,
            True,
            torch.bfloat16,
            short_factor=[1.0] * 24,
            long_factor=[2.0] * 24,
        )
        rope_patch.patch_in_graph_rotary_embedding(r_phi, allow_table_recovery=False)

    for tier1_mod in (r_std, r_lin, r_ds, r_mint, r_ernie, r_bail):
        assert isinstance(tier1_mod.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
        assert "cos_sin_cache" not in tier1_mod._buffers
    assert isinstance(r_phi.long_short_cos_sin_cache, rope_in_graph.InGraphCosSinCache)

    pos1 = torch.tensor([0, 3, 17, 129], dtype=torch.int64)
    offs = torch.tensor([0, 0, 128, 128], dtype=torch.int64)
    pos3 = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int64)
    q2 = torch.randn(4, 256, dtype=torch.bfloat16)
    k2 = torch.randn(4, 128, dtype=torch.bfloat16)
    q3 = torch.randn(4, 4, 64, dtype=torch.bfloat16)
    k3 = torch.randn(4, 1, 64, dtype=torch.bfloat16)

    for mod, args in (
        (r_std, (pos1, q2, k2)),
        (r_lin, (pos1, q2, k2, offs)),
        (r_ds, (pos1, q3, k3)),
        (r_mint, (pos3, q2, k2)),
        (r_ernie, (pos3, q2, k2)),
        (r_bail, (pos3, q2, k2)),
        (r_phi, (pos1, q2, k2)),
    ):
        torch._dynamo.reset()
        eq, ek = mod(*args)
        cq, ck = torch.compile(mod, backend="aot_eager", fullgraph=True)(*args)
        assert torch.equal(eq, cq)
        assert ek is not None and ck is not None and torch.equal(ek, ck)

    torch._dynamo.reset()
    c1, s1 = r_std.get_cos_sin(16)
    c2, s2 = torch.compile(
        lambda: r_std.get_cos_sin(16), backend="aot_eager", fullgraph=True
    )()
    assert torch.equal(c1, c2) and torch.equal(s1, s2)


def test_linear_scaling_power_of_two_bit_exact_on_cpu():
    """Power-of-two linear scaling matches upstream bit-exact in bf16."""
    from vllm.model_executor.layers.rotary_embedding import LinearScalingRotaryEmbedding

    from vllm_torchtpu import rope_in_graph

    rotary_dim = 64
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        cfg = get_current_vllm_config()
        cfg.additional_config[rope_in_graph_flag.KEY] = False
        ref = LinearScalingRotaryEmbedding(
            rotary_dim, rotary_dim, 256, 10000.0, True, [1.0, 2.0, 4.0], torch.bfloat16
        )
        ref_table = ref.cos_sin_cache
        assert isinstance(ref_table, torch.Tensor)

        cfg.additional_config[rope_in_graph_flag.KEY] = True
        uut = LinearScalingRotaryEmbedding(
            rotary_dim, rotary_dim, 256, 10000.0, True, [1.0, 2.0, 4.0], torch.bfloat16
        )
        assert sum(b.numel() for b in uut.buffers()) == 0
        assert isinstance(uut.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
        assert uut.scaling_factor_to_offset == ref.scaling_factor_to_offset

        all_pos = torch.arange(ref_table.shape[0], dtype=torch.int64)
        assert torch.equal(uut.cos_sin_cache[all_pos], ref_table)
        assert torch.equal(
            rope_in_graph.eval_spec_cos_sin(
                uut._in_graph_spec, all_pos, torch.bfloat16
            ),
            ref_table,
        )
    finally:
        torch.set_num_threads(prev_threads)


def test_linear_scaling_non_power_of_two_falls_back_to_table():
    """Non-power-of-two linear scaling (e.g. factor=3.0) falls back to table cleanly."""
    from vllm.model_executor.layers.rotary_embedding import LinearScalingRotaryEmbedding

    from vllm_torchtpu import rope_patch

    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        cfg = get_current_vllm_config()
        for factors in (3.0, [1.0, 3.0]):
            cfg.additional_config[rope_in_graph_flag.KEY] = False
            ref = LinearScalingRotaryEmbedding(
                64, 64, 128, 10000.0, True, factors, torch.bfloat16
            )
            cfg.additional_config[rope_in_graph_flag.KEY] = True
            uut = LinearScalingRotaryEmbedding(
                64, 64, 128, 10000.0, True, factors, torch.bfloat16
            )

            assert isinstance(uut.cos_sin_cache, torch.Tensor)
            assert "cos_sin_cache" in uut._buffers
            assert uut._in_graph_spec is None
            assert not hasattr(uut, "_tpu_in_graph_tier1")
            assert uut.scaling_factor_to_offset == ref.scaling_factor_to_offset
            assert torch.equal(uut.cos_sin_cache, ref.cos_sin_cache)
            assert (
                rope_patch.patch_in_graph_rotary_embedding(
                    uut, allow_table_recovery=False
                )
                == 0
            )

            g = torch.Generator().manual_seed(11)
            pos = torch.tensor([0, 7, 63, 127], dtype=torch.int64)
            q = torch.randn(4, 4 * 64, generator=g, dtype=torch.bfloat16)
            k = torch.randn(4, 2 * 64, generator=g, dtype=torch.bfloat16)
            ref_q, ref_k = ref(pos, q.clone(), k.clone())
            out_q, out_k = uut(pos, q.clone(), k.clone())
            assert torch.equal(out_q, ref_q)
            assert ref_k is not None and out_k is not None
            assert torch.equal(out_k, ref_k)
    finally:
        torch.set_num_threads(prev_threads)


def test_tier1_graceful_fallback_for_unsupported_mrope_and_subclasses():
    """Unsupported Tier-1 configs and subclasses fall back to the table buffer."""
    from vllm_torchtpu import rope_patch

    get_current_vllm_config().additional_config[rope_in_graph_flag.KEY] = True

    # 4-entry mrope_section has no 3-axis spec -> falls back cleanly instead of raising.
    four_sec = MRotaryEmbedding(
        64,
        64,
        128,
        10000.0,
        True,
        torch.bfloat16,
        mrope_section=[8, 8, 8, 8],
    )
    assert isinstance(four_sec.cos_sin_cache, torch.Tensor)
    assert "cos_sin_cache" in four_sec._buffers
    assert four_sec._in_graph_spec is None
    assert (
        rope_patch.patch_in_graph_rotary_embedding(four_sec, allow_table_recovery=False)
        == 0
    )

    # Subclasses of TpuRotaryEmbedding do not directly mix in Tier1InGraphRopeMixin,
    # so they keep the precomputed table (and NeoX compaction) even when flag is on.
    def _zero_padded_inv_freq(self: RotaryEmbedding, base: float) -> torch.Tensor:
        active = 1.0 / (base ** (torch.arange(0, 16, 2, dtype=torch.float) / 16.0))
        zeros = torch.zeros(self.rotary_dim // 2 - 8, dtype=torch.float)
        return torch.cat((active, zeros))

    sub_cls = type(
        "_CustomTpuRopeSubclass",
        (tpu_rope.TpuRotaryEmbedding,),
        {"_compute_inv_freq": _zero_padded_inv_freq},
    )
    sub = sub_cls(64, 64, 128, 10000.0, True, torch.bfloat16)
    assert isinstance(sub.cos_sin_cache, torch.Tensor)
    assert "cos_sin_cache" in sub._buffers
    assert sub._in_graph_spec is None
    # Compacted NeoX width: 8 active pairs -> 16 columns instead of 64.
    assert sub.cos_sin_cache.shape == (128, 16)


def test_mrope_section_none_matches_table_on_1d_and_rejects_2d():
    """Section-less M-RoPE / Ernie work on 1-D, ignore offsets, and reject 2-D."""
    from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
        Ernie4_5_VLRotaryEmbedding,
    )

    from vllm_torchtpu import rope_in_graph

    g = torch.Generator().manual_seed(17)
    pos_1d = torch.tensor([0, 3, 19, 63], dtype=torch.int64)
    pos_2d = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int64)
    offs = torch.tensor([10, 10, 10, 10], dtype=torch.int64)
    q = torch.randn(4, 4 * 64, generator=g, dtype=torch.float32)
    k = torch.randn(4, 2 * 64, generator=g, dtype=torch.float32)

    cfg = get_current_vllm_config()
    for cls in (MRotaryEmbedding, Ernie4_5_VLRotaryEmbedding):
        cfg.additional_config[rope_in_graph_flag.KEY] = False
        ref = cls(64, 64, 128, 10000.0, True, torch.float32, mrope_section=None)

        cfg.additional_config[rope_in_graph_flag.KEY] = True
        uut = cls(64, 64, 128, 10000.0, True, torch.float32, mrope_section=None)
        assert sum(b.numel() for b in uut.buffers()) == 0
        assert uut._in_graph_spec.dim_modalities is None

        ref_q, ref_k = ref(pos_1d, q.clone(), k.clone())
        out_q, out_k = uut(pos_1d, q.clone(), k.clone())
        torch.testing.assert_close(out_q, ref_q, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_k, ref_k, rtol=1e-6, atol=1e-6)

        if cls is MRotaryEmbedding:
            # Upstream MRotaryEmbedding ignores `offsets`; in-graph matches it.
            ref_oq, ref_ok = ref(pos_1d, q.clone(), k.clone(), offsets=offs)
            out_oq, out_ok = uut(pos_1d, q.clone(), k.clone(), offsets=offs)
            torch.testing.assert_close(out_oq, ref_oq, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(out_ok, ref_ok, rtol=1e-6, atol=1e-6)

        with pytest.raises(ValueError, match="mrope_section"):
            uut(pos_2d, q.clone(), k.clone())

    # Non-empty invalid mrope_section is still rejected by spec_from_module.
    bad_ernie = Ernie4_5_VLRotaryEmbedding(
        64, 64, 128, 10000.0, False, torch.float32, mrope_section=[10, 10, 12]
    )
    bad_ernie.mrope_section = [10, 12, 10]  # h != w
    bad_ernie.__dict__.pop("_in_graph_spec", None)
    assert rope_in_graph.spec_from_module(bad_ernie) is None


def test_mrope_2d_positions_wrong_leading_dim_raises():
    """2-D `[2, N]` positions raise ValueError instead of flattening to 2N tokens."""
    from vllm.model_executor.layers.rotary_embedding import (
        BailingMRotaryEmbedding,
        MRotaryEmbeddingInterleaved,
    )

    get_current_vllm_config().additional_config[rope_in_graph_flag.KEY] = True
    pos_bad = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int64)
    q = torch.randn(4, 4 * 64, dtype=torch.float32)
    k = torch.randn(4, 2 * 64, dtype=torch.float32)

    mrope = MRotaryEmbedding(
        64, 64, 128, 10000.0, True, torch.float32, mrope_section=[8, 12, 12]
    )
    bailing = BailingMRotaryEmbedding(
        64, 64, 128, 10000.0, True, torch.float32, mrope_section=[12, 10, 10]
    )
    interleaved_2d = MRotaryEmbeddingInterleaved(
        64, 64, 128, 10000.0, True, torch.float32, mrope_section=[16, 16]
    )

    for mod in (mrope, bailing, interleaved_2d):
        with pytest.raises(ValueError, match="leading dimension 3"):
            mod(pos_bad, q.clone(), k.clone())


def test_mrope_plugin_alias_subclass_produces_spec(monkeypatch: pytest.MonkeyPatch):
    """Plugin aliases like `vllm_omni`'s `OmniMRotaryEmbedding` are recognized."""
    from vllm.model_executor.layers import rotary_embedding as vllm_rope

    from vllm_torchtpu import rope_in_graph, rope_patch

    class FakeOmniMRotaryEmbedding(MRotaryEmbedding):
        pass

    aliases = rope_in_graph.mrope_aliases(FakeOmniMRotaryEmbedding)
    assert aliases == (FakeOmniMRotaryEmbedding, MRotaryEmbedding)

    monkeypatch.setattr(vllm_rope, "MRotaryEmbedding", FakeOmniMRotaryEmbedding)
    # Bypass CustomOp.__new__ so the instance's exact type is the plugin alias.
    mod = object.__new__(FakeOmniMRotaryEmbedding)
    MRotaryEmbedding.__init__(
        mod, 64, 64, 128, 10000.0, True, torch.float32, mrope_section=[8, 12, 12]
    )
    assert type(mod) is FakeOmniMRotaryEmbedding
    assert rope_patch.is_in_graph_rope_eligible(mod)
    spec = rope_in_graph.spec_from_module(mod)
    assert spec is not None
    assert spec.dim_modalities is not None
    assert len(spec.dim_modalities) == 32


def test_in_graph_mrope_requires_key_except_interleaved():
    """In-graph M-RoPE raises for `key=None`, except `MRotaryEmbeddingInterleaved`."""
    from vllm.model_executor.layers.rotary_embedding import (
        BailingMRotaryEmbedding,
        MRotaryEmbeddingInterleaved,
    )
    from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
        Ernie4_5_VLRotaryEmbedding,
    )

    get_current_vllm_config().additional_config[rope_in_graph_flag.KEY] = True
    pos = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int64)
    q = torch.randn(4, 4 * 64, dtype=torch.float32)

    for cls, section in (
        (MRotaryEmbedding, [8, 12, 12]),
        (Ernie4_5_VLRotaryEmbedding, [10, 10, 12]),
        (BailingMRotaryEmbedding, [12, 10, 10]),
    ):
        mod = cls(64, 64, 128, 10000.0, True, torch.float32, mrope_section=section)
        with pytest.raises(ValueError, match="requires key states"):
            mod(pos, q.clone(), None)

    interleaved = MRotaryEmbeddingInterleaved(
        64, 64, 128, 10000.0, True, torch.float32, mrope_section=[11, 11, 10]
    )
    out_q, out_k = interleaved(pos, q.clone(), None)
    assert out_q.shape == q.shape
    assert out_k is None


def test_tpu_mrope_table_forward_is_its_own_base_forward():
    """With the flag off, a plugin alias's `forward_native` override still runs."""
    calls: list[str] = []

    class _AliasMRotaryEmbedding(MRotaryEmbedding):
        def forward_native(self, positions, query, key=None, offsets=None):
            calls.append("alias")
            return super().forward_native(positions, query, key, offsets)

    # `_make_tpu_mrope_class` registers the class in `op_registry_oot`.
    try:
        tpu_cls = tpu_rope._make_tpu_mrope_class(_AliasMRotaryEmbedding)
        mod = tpu_cls(
            64, 64, 128, 10000.0, True, torch.float32, mrope_section=[8, 12, 12]
        )
        assert mod._in_graph_spec is None
        pos = torch.tensor(
            [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int64
        )
        mod.forward_native(pos, torch.randn(4, 4 * 64), torch.randn(4, 2 * 64))
        assert calls == ["alias"]
    finally:
        op_registry_oot.pop(_AliasMRotaryEmbedding.__name__, None)
