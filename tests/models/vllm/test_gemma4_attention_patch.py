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
"""Gemma4 attention from fused qkv rows: the patch and what it hands the backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.gemma4 import Gemma4Attention

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm import gemma4_attention_patch as patch

pytestmark = pytest.mark.usefixtures("default_on_knobs_off")


class _Rope:
    is_neox_style = True

    def __init__(self, rotary_dim: int) -> None:
        self.cos_sin_cache = torch.randn(64, rotary_dim).to(torch.bfloat16)

    def _match_cos_sin_cache_dtype(self, x: torch.Tensor) -> torch.Tensor:
        return self.cos_sin_cache.to(x.dtype)


class _Impl:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def forward_from_qkv(self, layer, qkv, cos, sin, wq, wk, eps, kv_cache, md):
        self.calls.append((layer, qkv, cos, sin, wq, wk, eps, kv_cache, md))
        return qkv.new_zeros(qkv.shape[0], 4)


def _attn(rotary_dim: int = 8):
    impl = _Impl()
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn", impl=impl)
    norm = lambda: SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-6)  # noqa: E731
    return SimpleNamespace(
        attn=layer, rotary_emb=_Rope(rotary_dim), q_norm=norm(), k_norm=norm()
    ), impl


def test_fused_path_hands_the_backend_rotary_rows(monkeypatch):
    attn, impl = _attn()
    context = ("metadata", attn.attn, "kv_cache", None)
    monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_ATTN_PROLOGUE", True)
    monkeypatch.setattr(patch, "get_attention_context", lambda name: context)
    positions = torch.tensor([5, 0, 63])
    qkv = torch.randn(3, 32).to(torch.bfloat16)
    out = patch.attention_from_qkv(attn, positions, qkv)
    assert out.shape == (3, 4)
    ((layer, got_qkv, cos, sin, wq, wk, eps, kv_cache, md),) = impl.calls
    rows = attn.rotary_emb.cos_sin_cache[positions]
    assert (
        layer is attn.attn
        and got_qkv is qkv
        and (kv_cache, md) == ("kv_cache", "metadata")
    )
    torch.testing.assert_close(cos, rows[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(sin, rows[:, 4:], rtol=0, atol=0)
    assert wq is attn.q_norm.weight and wk is attn.k_norm.weight and eps == 1e-6


def test_flag_off_keeps_upstream_ops(monkeypatch):
    calls = []
    attn = SimpleNamespace(
        q_size=8,
        kv_size=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        q_norm=lambda x: calls.append("q_norm") or x,
        k_norm=lambda x: calls.append("k_norm") or x,
        v_norm=lambda x: calls.append("v_norm") or x,
        rotary_emb=lambda pos, q, k: (calls.append("rotary") or (q, k)),
        attn=lambda q, k, v: calls.append("attn") or q,
    )
    monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_ATTN_PROLOGUE", False)
    patch.attention_from_qkv(attn, torch.arange(3), torch.randn(3, 16))
    assert calls == ["q_norm", "k_norm", "rotary", "v_norm", "attn"]


def test_patch_needs_the_flag_and_a_gemma4_config(monkeypatch):
    original, original_init = Gemma4Attention.forward, Gemma4Attention.__init__
    gemma4 = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="gemma4_text"))
    other = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="llama"))
    try:
        monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_ATTN_PROLOGUE", False)
        assert patch.maybe_patch_gemma4_attention(gemma4) is None
        assert Gemma4Attention.forward is original
        monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_ATTN_PROLOGUE", True)
        assert patch.maybe_patch_gemma4_attention(None) is False
        assert patch.maybe_patch_gemma4_attention(other) is False
        assert Gemma4Attention.forward is original
        assert patch.maybe_patch_gemma4_attention(gemma4) is None
        assert Gemma4Attention.forward is patch._patched_attention_forward
        assert Gemma4Attention.__init__ is patch._patched_attention_init
    finally:
        Gemma4Attention.forward = original
        Gemma4Attention.__init__ = original_init


def test_init_refuses_a_backend_without_the_fused_entry(monkeypatch):
    def fake_init(self, *args, **kwargs):
        self.is_kv_shared_layer = False
        self.attn = SimpleNamespace(layer_name="layers.0.attn", impl=object())

    monkeypatch.setattr(patch, "_orig_attention_init", fake_init)
    with pytest.raises(NotImplementedError, match="CUSTOM"):
        patch._patched_attention_init(SimpleNamespace())


def test_non_neox_rotary_is_refused(monkeypatch):
    attn, _ = _attn()
    attn.rotary_emb.is_neox_style = False
    monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_ATTN_PROLOGUE", True)
    with pytest.raises(NotImplementedError, match="neox"):
        patch.attention_from_qkv(attn, torch.arange(2), torch.randn(2, 32))


def _fake_init(impl, shared=False, neox=True):
    def init(self, *args, **kwargs):
        self.is_kv_shared_layer = shared
        self.attn = SimpleNamespace(layer_name="layers.0.attn", impl=impl)
        self.rotary_emb = SimpleNamespace(is_neox_style=neox)

    return init


def test_init_names_the_flag_for_unsupported_configs(monkeypatch):
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl,
    )

    impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
    monkeypatch.setattr(impl, "qkv_prologue_unsupported", lambda: ["attention sinks"])
    monkeypatch.setattr(patch, "_orig_attention_init", _fake_init(impl))
    with pytest.raises(NotImplementedError, match="TPU_GEMMA4_FUSED_ATTN_PROLOGUE"):
        patch._patched_attention_init(SimpleNamespace())


def test_init_leaves_kv_shared_layers_alone(monkeypatch):
    monkeypatch.setattr(patch, "_orig_attention_init", _fake_init(object(), True))
    patch._patched_attention_init(SimpleNamespace())  # no impl checks


def test_init_records_the_epsilon_and_the_backend_builds_before_tracing(monkeypatch):
    """The op is built in initialize_kernel (before torch.compile), from the
    epsilon the init patch records; forward_from_qkv refuses an unbuilt op."""
    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackendImpl,
        PallasBatchedRPAAttentionBackendImpl,
    )

    impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
    impl.qkv_prologue_eps, impl._qkv_kernel = None, None
    monkeypatch.setattr(impl, "qkv_prologue_unsupported", lambda: [])

    def init(self, *args, **kwargs):
        self.is_kv_shared_layer = False
        self.attn = SimpleNamespace(layer_name="layers.0.attn", impl=impl)
        self.rotary_emb = SimpleNamespace(is_neox_style=True)
        self.q_norm = SimpleNamespace(variance_epsilon=1e-6)

    monkeypatch.setattr(patch, "_orig_attention_init", init)
    patch._patched_attention_init(SimpleNamespace())
    assert impl.qkv_prologue_eps == 1e-6

    built = []
    monkeypatch.setattr(
        PallasAttentionBackendImpl, "initialize_kernel", lambda self, layer: None
    )
    monkeypatch.setattr(
        impl, "_build_qkv_kernel", lambda ks, vs, eps: built.append(eps) or "op"
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    impl.initialize_kernel(layer)
    assert built == [1e-6] and impl._qkv_kernel == "op"


def test_forward_from_qkv_refuses_an_op_built_for_another_epsilon(monkeypatch):
    from vllm_torchtpu.layers.adapter import attention as A

    impl = object.__new__(A.PallasBatchedRPAAttentionBackendImpl)
    impl.qkv_prologue_eps, impl._qkv_kernel = None, None
    impl.num_heads, impl.head_size = 2, 128
    monkeypatch.setattr(impl, "qkv_prologue_unsupported", lambda: [])
    monkeypatch.setattr(
        A,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(kv_cache_bundle=None),
    )
    qkv, kv_cache = torch.zeros(4, 512), torch.zeros(8)
    with pytest.raises(RuntimeError, match="initialize_kernel"):
        impl.forward_from_qkv(None, qkv, None, None, None, None, 1e-6, kv_cache, None)
