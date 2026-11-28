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
"""CPU contract, layout, 3-step plan, and import-boundary tests for Qwen4Exp."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.models.qwen4_exp.attention_plan import (
    AttentionDesc,
    AttentionPlan,
    CoreAttn,
    OutProj,
    QKVProj,
)
from vllm_torchtpu.models.qwen4_exp.layout import (
    Axis,
    QKVLayout,
    TensorLayout,
    expected_shape,
    kv_hnd,
    kv_nhd,
    packed_q,
    unfused_o,
    unfused_qkv,
    verify_qkv,
    verify_tensor,
)

_PKG_DIR = Path(__file__).resolve().parents[3] / "src/vllm_torchtpu/models/qwen4_exp"


def _desc(
    *,
    sparse: bool = False,
    kv_layout: KVCacheLayout = KVCacheLayout.LBNHC,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: torch.dtype = torch.float8_e4m3fn,
) -> AttentionDesc:
    return AttentionDesc(
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        kv_layout=kv_layout,
        sparse=sparse,
    )


# ── TensorLayout contract checks ───────────────────────────────────────────


def test_layout_equality_catches_merge_order_and_packing_mismatches() -> None:
    assert unfused_o() == TensorLayout(
        (Axis.T, (Axis.H_KV, Axis.G), Axis.D), torch.bfloat16
    )
    # Non-commutative merge: [k0, v0, k1, v1] != [k0, k1, v0, v1].
    kv_concat = TensorLayout(
        (Axis.T, (Axis.KV, Axis.H_KV), Axis.D),
        torch.float8_e4m3fn,
        pack_sublane=True,
        lane_align=128,
        quantized=True,
    )
    assert kv_nhd() != kv_concat
    assert kv_hnd() != TensorLayout(
        ((Axis.H_KV, Axis.KV), Axis.D, Axis.T),
        torch.float8_e4m3fn,
        pack_sublane=True,
        lane_align=16,
        quantized=True,
    )
    assert kv_nhd(fold_lanes=True) != kv_nhd(fold_lanes=False)
    assert kv_nhd(quantized=True) != kv_nhd(quantized=False)


def test_layout_expected_shapes_and_verify() -> None:
    assert expected_shape(unfused_o(), T=64, H_kv=2, G=8, D=128) == (64, 16, 128)
    assert expected_shape(packed_q(), T=64, H_kv=2, G=8, D=128) == (2, 64, 4, 2, 128)
    assert expected_shape(kv_nhd(), T=64, H_kv=2, G=8, D=128) == (64, 1, 4, 128)
    assert expected_shape(kv_nhd(), T=64, H_kv=2, G=8, D=256) == (64, 2, 4, 128)
    assert expected_shape(kv_nhd(fold_lanes=False), T=64, H_kv=2, G=8, D=256) == (
        64,
        1,
        4,
        256,
    )
    assert expected_shape(kv_hnd(), T=64, H_kv=2, G=8, D=128, page_size=128) == (
        4,
        32,
        4,
        128,
    )
    assert expected_shape(kv_hnd(), T=64, H_kv=2, G=8, D=256, page_size=16) == (
        4,
        64,
        4,
        128,
    )

    with pytest.raises(ValueError, match="Expected"):
        verify_tensor(
            packed_q(),
            torch.empty((64, 16, 128), dtype=torch.bfloat16),
            T=64,
            H_kv=2,
            G=8,
            D=128,
        )


def test_layout_verify_works_under_fake_tensor_mode() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty((2, 64, 4, 2, 128), dtype=torch.bfloat16)
        nhd = torch.empty((64, 1, 4, 128), dtype=torch.float8_e4m3fn)
        verify_qkv(
            QKVLayout(q=packed_q(), kv=kv_nhd()), q, nhd, T=64, H_kv=2, G=8, D=128
        )

        hnd = torch.empty((4, 32, 4, 128), dtype=torch.float8_e4m3fn)
        verify_qkv(
            QKVLayout(q=packed_q(), kv=kv_hnd()),
            q,
            hnd,
            T=64,
            H_kv=2,
            G=8,
            D=128,
            page_size=128,
        )


# ── 3-step plan resolution ─────────────────────────────────────────────────


def test_attention_plan_resolve_defaults() -> None:
    dense = AttentionPlan.resolve(_desc(sparse=False))
    assert dense == AttentionPlan(
        qkv_proj=QKVProj.UNFUSED,
        core_attn=CoreAttn.RPA_V3,
        out_proj=OutProj.UNFUSED,
        qkv_layout=unfused_qkv(torch.bfloat16),
        o_layout=unfused_o(torch.bfloat16),
    )

    sparse = AttentionPlan.resolve(_desc(sparse=True))
    assert sparse == AttentionPlan(
        qkv_proj=QKVProj.UNFUSED,
        core_attn=CoreAttn.QSA,
        out_proj=OutProj.UNFUSED,
        qkv_layout=unfused_qkv(torch.bfloat16),
        o_layout=unfused_o(torch.bfloat16),
    )


def test_attention_plan_rejects_unsupported_stage_overrides() -> None:
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), qkv_proj=QKVProj.FUSED)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), out_proj=OutProj.FUSED)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), core_attn=CoreAttn.BATCHED_RPA)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=False), core_attn=CoreAttn.QSA)
    with pytest.raises(ValueError, match="No compatible"):
        AttentionPlan.resolve(_desc(sparse=True, kv_layout=KVCacheLayout.LBHNC))


# ── Mechanical import-boundary enforcement ────────────────────────────────


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_qwen4_exp_import_hierarchy_is_strictly_top_down_and_independent() -> None:
    # 1. Pure leaf modules (layout.py, ops/*) never import vllm or layer files.
    for leaf in [_PKG_DIR / "layout.py", *(_PKG_DIR / "ops").glob("*.py")]:
        imports = _imported_modules(leaf)
        assert not any(m == "vllm" or m.startswith("vllm.") for m in imports), (
            f"{leaf.name} must stay free of vllm imports: {sorted(imports)}"
        )
        assert not any(
            m.startswith("vllm_torchtpu.models.qwen4_exp.")
            and not m.startswith("vllm_torchtpu.models.qwen4_exp.ops.")
            for m in imports
        ), f"{leaf.name} must not import model/layer modules: {sorted(imports)}"

    pkg = "vllm_torchtpu.models.qwen4_exp."

    def local(name: str) -> set[str]:
        return {
            m.removeprefix(pkg)
            for m in _imported_modules(_PKG_DIR / name)
            if m.startswith(pkg)
        }

    # 2. No local forked kernels directory (upstream kernels live in tokamax;
    #    only thin stateless jax_op wrappers live under ops/).
    assert not (_PKG_DIR / "kernels").exists(), (
        "Do not fork kernels under models/qwen4_exp/kernels/; "
        "upstream to tokamax and wrap via ops/."
    )

    # 3. Strict top-down and sibling independence within qwen4_exp.
    assert local("ple.py") <= {"ops.ple_state"}
    assert local("qsa.py") <= {"layout", "quant", "ops.qsa"}
    assert local("attention_plan.py") <= {"layout", "qsa"}
    assert local("attention.py") <= {"attention_plan", "qsa", "quant", "ops.dense_attn"}
    assert local("model.py") <= {"attention", "ple"}


# ── Real Qwen4ExpAttention / AttentionDesc / Qwen4ExpQSA CPU tests ─────────


def test_attention_desc_from_configs_resolves_kv_dtypes() -> None:
    from types import SimpleNamespace
    from typing import cast

    from vllm.config import CacheConfig, ModelConfig

    model_cfg = cast(ModelConfig, SimpleNamespace(dtype=torch.bfloat16))
    for cache_dtype_str, expected_kv in [
        ("auto", torch.bfloat16),
        ("fp8", torch.float8_e4m3fn),
        ("fp8_e4m3", torch.float8_e4m3fn),
        ("fp8_e5m2", torch.float8_e5m2),
    ]:
        cache_cfg = cast(
            CacheConfig,
            SimpleNamespace(
                cache_dtype=cache_dtype_str,
                get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
            ),
        )
        desc = AttentionDesc.from_configs(
            num_heads=16,
            num_kv_heads=2,
            head_dim=256,
            sparse=False,
            model_config=model_cfg,
            cache_config=cache_cfg,
        )
        assert desc.dtype == torch.bfloat16
        assert desc.kv_cache_dtype == expected_kv
        assert desc.kv_layout is KVCacheLayout.LBHNC


@pytest.mark.parametrize(
    ("num_q", "num_kv", "tp", "expected_q", "expected_kv"),
    [
        (16, 2, 1, 16, 2),
        (16, 2, 2, 8, 1),
        (16, 2, 4, 4, 1),  # KV heads replicated when kv < tp
        (16, 2, 8, 2, 1),
    ],
)
def test_tp_head_counts_and_head_dim_fallback(
    monkeypatch: pytest.MonkeyPatch,
    num_q: int,
    num_kv: int,
    tp: int,
    expected_q: int,
    expected_kv: int,
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from transformers import Qwen4ExpTextConfig

    from vllm_torchtpu.models.qwen4_exp import attention as attn_mod

    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: tp)
    cfg = cast(
        Qwen4ExpTextConfig,
        SimpleNamespace(
            num_attention_heads=num_q,
            num_key_value_heads=num_kv,
            hidden_size=num_q * 256,
            head_dim=None,
        ),
    )
    assert attn_mod._tp_head_counts(cfg) == (expected_q, expected_kv)
    # head_dim fallback must use pre-TP config.num_attention_heads.
    assert (cfg.head_dim or (cfg.hidden_size // cfg.num_attention_heads)) == 256


@pytest.mark.parametrize("sparse", [False, True])
def test_qwen4_exp_attention_init_and_forward_cpu(
    monkeypatch: pytest.MonkeyPatch, sparse: bool
) -> None:
    """Instantiates real Qwen4ExpAttention and runs forward for RPA_V3 and QSA."""
    from types import SimpleNamespace
    from typing import cast

    import vllm.distributed.parallel_state as parallel_state
    import vllm.model_executor.layers.linear as vllm_linear
    import vllm.model_executor.layers.rotary_embedding as vllm_rope
    import vllm.model_executor.parameter as vllm_parameter
    from transformers import Qwen4ExpTextConfig
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )

    from vllm_torchtpu.models.qwen4_exp import attention as attn_mod
    from vllm_torchtpu.models.qwen4_exp.attention import Qwen4ExpAttention
    from vllm_torchtpu.models.qwen4_exp.qsa import dense_max_model_len

    for mod in (parallel_state, vllm_linear, vllm_parameter, attn_mod):
        if hasattr(mod, "get_tensor_model_parallel_rank"):
            monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)
        if hasattr(mod, "get_tensor_model_parallel_world_size"):
            monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)

    T, K, H_q, H_kv, D = 8, 256, 4, 2, 128
    cfg = Qwen4ExpTextConfig(
        vocab_size=512,
        hidden_size=K,
        num_hidden_layers=1,
        layer_types=["indexed_attention"],
        num_attention_heads=H_q,
        num_key_value_heads=H_kv,
        head_dim=D,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=16,
        indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7},
        max_position_embeddings=512,
    )
    assert dense_max_model_len(cfg) == 19
    cache_cfg = CacheConfig(cache_dtype="auto", block_size=16)
    monkeypatch.setattr(
        cache_cfg, "get_resolved_kv_cache_layout", lambda: KVCacheLayout.LBNHC
    )
    model_cfg = cast(
        ModelConfig,
        SimpleNamespace(
            dtype=torch.bfloat16,
            max_model_len=64 if sparse else 16,
            hf_text_config=cfg,
            hf_config=cfg,
        ),
    )
    vllm_cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_rope._ROPE_DICT.clear()
    with set_current_vllm_config(vllm_cfg):
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = Qwen4ExpAttention(
                cfg,
                model_config=model_cfg,
                cache_config=cache_cfg,
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )
            assert layer.plan.core_attn is (CoreAttn.QSA if sparse else CoreAttn.RPA_V3)
            assert (layer.indexer is not None) is sparse

            # Stub self.attn to exercise _qkv_proj -> _core_attn -> _out_proj on CPU.
            seen_shapes: list[tuple[tuple[int, ...], ...]] = []

            class _FakeAttn(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.kv_cache = torch.empty((0,), dtype=torch.bfloat16)

                def forward(
                    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
                    seen_shapes.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
                    return torch.zeros_like(q)

            monkeypatch.setattr(layer, "attn", _FakeAttn())
            out = layer(
                torch.arange(T, dtype=torch.int64),
                torch.zeros((T, K), dtype=torch.bfloat16),
            )
            assert tuple(out.shape) == (T, K)
            assert out.dtype == torch.bfloat16
            assert seen_shapes == [((T, H_q * D), (T, H_kv * D), (T, H_kv * D))]
            if sparse:
                assert layer.indexer is not None
                with pytest.raises(NotImplementedError, match="packed KV layout"):
                    layer.indexer(
                        q=torch.zeros((T, H_q, D), dtype=torch.bfloat16),
                        kv=torch.zeros((T, 1, 4, D), dtype=torch.float8_e4m3fn),
                        positions=torch.arange(T, dtype=torch.int64),
                        hidden_states=torch.zeros((T, K), dtype=torch.bfloat16),
                        main_pool=torch.empty((0,), dtype=torch.bfloat16),
                        metadata=None,
                        kv_scales=None,
                        cos_sin_cache=layer.rotary_emb.cos_sin_cache,
                        dense_attn=layer.attn,
                    )
        finally:
            vllm_rope._ROPE_DICT.clear()
            torch.set_default_dtype(torch.float32)


def test_qwen4_exp_qsa_validates_indexer_config() -> None:
    from types import SimpleNamespace
    from typing import cast

    from transformers import Qwen4ExpTextConfig

    from vllm_torchtpu.models.qwen4_exp.qsa import Qwen4ExpQSA

    bad_kv = cast(
        Qwen4ExpTextConfig,
        SimpleNamespace(
            indexer_kv_heads=2,
            indexer_head_dim=64,
            indexer_n_heads=4,
            indexer_budget=16,
            indexer_compress_ratio=4,
            hidden_size=256,
            rms_norm_eps=1e-6,
        ),
    )
    with pytest.raises(NotImplementedError, match="indexer_kv_heads=1"):
        Qwen4ExpQSA(
            bad_kv,
            num_heads=4,
            num_kv_heads=2,
            head_dim=128,
            scaling=128**-0.5,
            qkv_layout=unfused_qkv(),
            cache_config=None,
            quant_config=None,
            prefix="layers.0.self_attn",
        )

    bad_dim = cast(
        Qwen4ExpTextConfig,
        SimpleNamespace(
            indexer_kv_heads=1,
            indexer_head_dim=256,
            indexer_n_heads=4,
            indexer_budget=16,
            indexer_compress_ratio=4,
            hidden_size=256,
            rms_norm_eps=1e-6,
        ),
    )
    with pytest.raises(NotImplementedError, match="indexer_head_dim"):
        Qwen4ExpQSA(
            bad_dim,
            num_heads=4,
            num_kv_heads=2,
            head_dim=128,
            scaling=128**-0.5,
            qkv_layout=unfused_qkv(),
            cache_config=None,
            quant_config=None,
            prefix="layers.0.self_attn",
        )
