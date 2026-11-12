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


def test_vllm_rotary_subclass_exhaustiveness_guard():
    """Every upstream vLLM rotary class must belong to Tier 1, Tier 2, or §3.5."""
    import importlib
    import pkgutil

    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
    from vllm.model_executor.layers.rotary_embedding.dual_chunk_rope import (
        DualChunkRotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope import (
        Phi3LongRoPEScaledRotaryEmbedding,
    )

    from vllm_torchtpu import rope_patch
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

    tier1_classes = (
        set(tpu_rope.TIER1_STATIC_ROPE_CLASSES)
        | set(tpu_rope.TIER1_MROPE_CLASSES)
        | {upstream_rope.DeepseekScalingRotaryEmbedding}
    )
    out_of_scope_rotary_base = {
        upstream_rope.DeepseekV4ScalingRotaryEmbedding,
        upstream_rope.FourierRotaryEmbedding,
        upstream_rope.Llama4VisionRotaryEmbedding,
    }
    assert tier1_classes.isdisjoint(out_of_scope_rotary_base)
    expected_all = tier1_classes | out_of_scope_rotary_base
    assert upstream_rotary_base_subclasses == expected_all, (
        "Unclassified RotaryEmbeddingBase subclass(es): "
        f"{upstream_rotary_base_subclasses ^ expected_all}"
    )

    # Verify OOT registry wiring for all 13 Tier-1 RotaryEmbeddingBase classes.
    for cls in tier1_classes:
        assert cls.__name__ in op_registry_oot, f"Missing OOT registration for {cls}"

    # Verify standalone non-RotaryEmbeddingBase classes (Tier 2 vs §3.5).
    assert not issubclass(Phi3LongRoPEScaledRotaryEmbedding, RotaryEmbeddingBase)
    assert not issubclass(DualChunkRotaryEmbedding, RotaryEmbeddingBase)
    assert (
        rope_patch.spec_from_module(
            DualChunkRotaryEmbedding.__new__(DualChunkRotaryEmbedding)
        )
        is None
    )


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
def test_all_in_scope_rotary_classes_match_upstream_cpu_table_and_forward(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
):
    """All 14 in-scope rotary classes match upstream CPU FP32 tables & forwards."""
    import inspect

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

    from vllm_torchtpu import envs, rope_patch
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
            yarn_sig = inspect.signature(YaRNScalingRotaryEmbedding.__init__).parameters
            yarn_kw = (
                {"attention_factor": 1.25}
                if "attention_factor" in yarn_sig
                else {"attn_factor": 1.25}
            )
            return (
                YaRNScalingRotaryEmbedding(
                    64, 64, 128, 10000.0, True, 4.0, torch.float32, **yarn_kw
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

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.model_config = type("MC", (), {"max_model_len": 512})()  # type: ignore[assignment]
    with set_current_vllm_config(cfg):
        monkeypatch.setattr(envs, "TPU_ROPE_IN_GRAPH", False)
        ref, positions, offsets = _construct()
        assert sum(b.numel() for b in ref.buffers()) > 0

        monkeypatch.setattr(envs, "TPU_ROPE_IN_GRAPH", True)
        uut, _, _ = _construct()

    cache_attr = (
        "long_short_cos_sin_cache" if case == "phi3_long_rope" else "cos_sin_cache"
    )
    ref_table = getattr(ref, cache_attr)
    assert isinstance(ref_table, torch.Tensor)

    if case == "phi3_long_rope":
        # Tier 2 converts plain-nn.Module Phi3LongRoPE post-load.
        assert rope_patch.patch_in_graph_rotary_embedding(uut) == 1
    else:
        # Tier 1 allocates 0 buffer elements at __init__ BEFORE Tier 2 runs.
        assert sum(b.numel() for b in uut.buffers()) == 0
        assert rope_patch.patch_in_graph_rotary_embedding(uut) == 0

    assert sum(b.numel() for b in uut.buffers()) == 0
    uut_cache = getattr(uut, cache_attr)
    assert isinstance(uut_cache, rope_patch.InGraphCosSinCache)
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


def test_tier1_meta_to_empty_and_disable_materialization(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tier-1 `InGraphCosSinCache` tracks `to_empty()` and `TPU_ROPE_IN_GRAPH=0`."""
    from vllm.model_executor.layers.rotary_embedding import (
        DeepseekScalingRotaryEmbedding,
    )

    from vllm_torchtpu import envs, rope_patch
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    monkeypatch.setattr(envs, "TPU_ROPE_IN_GRAPH", True)
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
        assert isinstance(mod.cos_sin_cache, rope_patch.InGraphCosSinCache)
        assert mod.cos_sin_cache.device.type == "meta"
        assert sum(b.numel() for b in mod.buffers()) == 0

    model.to_empty(device="cpu")
    for mod in (rope, mrope, ds_rope):
        assert mod.cos_sin_cache.device == torch.device("cpu")

    cos, sin = rope.get_cos_sin(16)
    assert cos.device == torch.device("cpu") and cos.shape == (16, 32)
    assert sin.device == torch.device("cpu") and sin.shape == (16, 32)

    # Toggling TPU_ROPE_IN_GRAPH=0 post-init materializes caches back to buffers.
    assert rope_patch.materialize_model_cos_sin_caches(model) == 3
    for mod in (rope, mrope, ds_rope):
        assert isinstance(mod.cos_sin_cache, torch.Tensor)
        assert "cos_sin_cache" in mod._buffers


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

    from vllm_torchtpu import rope_patch
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
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
        rope_patch.patch_in_graph_rotary_embedding(r_phi)

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
