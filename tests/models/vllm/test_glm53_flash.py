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
"""Unit tests for the TPU GLM-5.3-Flash model (`models/vllm/glm5_next/`)."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import types
from unittest import mock

import pytest
import torch

from vllm_torchtpu import envs, gdn_pool_layout
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.glm5_next import config as glm5_next_config
from vllm_torchtpu.models.vllm.glm5_next import model as glm5_next_model


def test_register_models_is_lazy_and_side_effect_free():
    """`register_models()` does not import heavy GLM-5.3 modules or patch vLLM."""
    code = """
import sys
from vllm.v1.core.sched.scheduler import Scheduler
orig_schedule = Scheduler.schedule
from vllm_torchtpu.models.vllm import register_models
register_models()
assert "vllm_torchtpu.models.vllm.glm5_next.model" not in sys.modules
assert "vllm_torchtpu.layers.adapter.custom_ops.glm53_kda_tpu" not in sys.modules
assert "vllm.third_party.flash_linear_attention.ops.kda" not in sys.modules
assert Scheduler.schedule is orig_schedule
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_model_registry_and_pooled_gdn_membership():
    register_models()
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    supported = ModelRegistry.get_supported_archs()
    for arch in (
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
        "DeepseekV4ForCausalLM",
    ):
        assert arch in supported

    assert (
        ModelRegistry.models["Glm5NextForCausalLM"].load_model_cls()
        is glm5_next_model.Glm5NextForCausalLM
    )
    assert (
        ModelRegistry.models["Glm5NextForConditionalGeneration"].load_model_cls()
        is glm5_next_model.Glm5NextForConditionalGeneration
    )
    assert (
        MODELS_CONFIG_MAP["Glm5NextForCausalLM"] is glm5_next_config.TPUGlm5NextConfig
    )
    assert (
        MODELS_CONFIG_MAP["Glm5NextForConditionalGeneration"]
        is glm5_next_config.TPUGlm5NextConfig
    )
    assert "Glm5NextForCausalLM" in gdn_pool_layout.POOLED_GDN_ARCHITECTURES
    assert (
        "Glm5NextForConditionalGeneration" in gdn_pool_layout.POOLED_GDN_ARCHITECTURES
    )


@dataclasses.dataclass
class _FakeArchConfig:
    is_deepseek_mla: bool = True
    head_size: int = 512
    total_num_kv_heads: int = 1


def _make_fake_vllm_config(
    *,
    qk_rope_head_dim: int = 0,
    mhc: bool = True,
    tp_size: int = 8,
    max_model_len: int = 4096,
):
    text_cfg = types.SimpleNamespace(
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=qk_rope_head_dim,
        num_key_value_heads=64,
        num_attention_heads=64,
        linear_num_heads=64,
        mhc=mhc,
        index_topk=2048,
        index_kpool=64,
        index_kpool_always_select_tail=True,
    )
    model_cfg = types.SimpleNamespace(
        hf_text_config=text_cfg,
        model_arch_config=_FakeArchConfig(),
        max_model_len=max_model_len,
    )
    par_cfg = types.SimpleNamespace(tensor_parallel_size=tp_size)
    return types.SimpleNamespace(model_config=model_cfg, parallel_config=par_cfg)


def test_tpu_glm5_next_config_mla_modes_and_validation(monkeypatch):
    from vllm.model_executor.models.config import Glm5NextModelConfig

    monkeypatch.setattr(
        Glm5NextModelConfig,
        "verify_and_update_model_config",
        staticmethod(lambda mc: None),
    )
    monkeypatch.setattr(
        Glm5NextModelConfig,
        "verify_and_update_config",
        staticmethod(lambda vc: None),
    )

    vcfg = _make_fake_vllm_config()
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_MLA_IMPL", "rpa", raising=False)
    glm5_next_config.TPUGlm5NextConfig.verify_and_update_model_config(vcfg.model_config)
    assert vcfg.model_config.model_arch_config.is_deepseek_mla is False
    assert vcfg.model_config.model_arch_config.head_size == 256
    assert vcfg.model_config.model_arch_config.total_num_kv_heads == 64

    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_MLA_IMPL", "absorbed", raising=False)
    glm5_next_config.TPUGlm5NextConfig.verify_and_update_model_config(vcfg.model_config)
    assert vcfg.model_config.model_arch_config.is_deepseek_mla is True
    assert (
        vcfg.model_config.model_arch_config.head_size
        == 512 + glm5_next_config.GLM53_MLA_ROPE_PAD
    )

    glm5_next_config.TPUGlm5NextConfig.verify_and_update_config(vcfg)

    with pytest.raises(NotImplementedError, match="qk_rope_head_dim=0"):
        glm5_next_config.TPUGlm5NextConfig.verify_and_update_config(
            _make_fake_vllm_config(qk_rope_head_dim=64)
        )
    with pytest.raises(NotImplementedError, match="mhc=True"):
        glm5_next_config.TPUGlm5NextConfig.verify_and_update_config(
            _make_fake_vllm_config(mhc=False)
        )
    with pytest.raises(ValueError, match="not divisible"):
        glm5_next_config.TPUGlm5NextConfig.verify_and_update_config(
            _make_fake_vllm_config(tp_size=6)
        )


def test_checkpoint_name_normalization_and_layer_helpers():
    norm = glm5_next_model.normalize_glm53_checkpoint_name
    assert norm("model.visual.patch_embed.weight", 45) is None
    assert norm("model.language_model.layers.45.enorm.weight", 45) is None
    assert (
        norm("model.language_model.layers.0.self_attn.indexer.wq_b.weight", 45) is None
    )
    assert (
        norm("model.language_model.layers.0.hc_attn_fn", 45)
        == "model.layers.0.attn_hc.fn"
    )
    assert (
        norm("model.language_model.layers.3.input_layernorm.weight", 45)
        == "model.layers.3.attn_hc.sublayer_norm_weight"
    )

    cfg = types.SimpleNamespace(
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        linear_num_heads=64,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    assert glm5_next_model.is_kda_layer(cfg, 0) is True
    assert glm5_next_model.is_kda_layer(cfg, 1) is False
    la_cfg = glm5_next_model.kda_linear_attn_config(cfg)
    assert la_cfg == {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -9.0,
        "safe_gate": True,
    }


def test_run_rpa_attention_avoids_symbolic_torch_empty() -> None:
    """Compiled RPA attention uses `zeros_like(v_flat)` instead of `torch.empty`."""
    attn = glm5_next_model.Attention.__new__(glm5_next_model.Attention)
    torch.nn.Module.__init__(attn)
    attn.num_heads = 2
    attn.num_kv_heads = 2
    attn.head_size = 8
    attn.head_size_v = 4
    attn.query_quant = None
    attn.use_direct_call = True
    attn.attn_backend = types.SimpleNamespace(forward_includes_kv_cache_update=False)
    attn.kv_sharing_target_layer_name = None
    attn.layer_name = "model.layers.3.self_attn.attn"

    q = torch.randn(6, 16)
    k_flat = torch.randn(6, 16)
    v_flat = torch.randn(6, 8)
    calls: list[str] = []

    def _fake_kv_update(k: torch.Tensor, v: torch.Tensor, layer_name: str):
        assert k.shape == (6, 2, 8)
        assert v.shape == (6, 2, 4)
        assert layer_name == attn.layer_name
        calls.append("kv_update")
        return torch.zeros(1)

    def _fake_attn(
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        out_3d: torch.Tensor,
        layer_name: str,
        kv_cache_dummy_dep=None,
    ) -> None:
        del layer_name
        assert q_3d.shape == (6, 2, 8)
        assert k_3d.shape == (6, 2, 8)
        assert v_3d.shape == (6, 2, 4)
        assert out_3d.shape == (6, 2, 4)
        assert bool(torch.all(out_3d == 0))
        assert kv_cache_dummy_dep is not None
        out_3d[:4].copy_(v_3d[:4] * 2.0)
        calls.append("attn")

    with (
        mock.patch.object(torch.compiler, "is_compiling", return_value=True),
        mock.patch.object(
            torch,
            "empty",
            side_effect=AssertionError("torch.empty should not be called"),
        ),
        mock.patch.object(
            torch,
            "empty_like",
            side_effect=AssertionError("torch.empty_like should not be called"),
        ),
        mock.patch.object(
            glm5_next_model, "unified_kv_cache_update", side_effect=_fake_kv_update
        ),
        mock.patch.object(
            glm5_next_model, "unified_attention_with_output", side_effect=_fake_attn
        ),
    ):
        got = glm5_next_model._run_rpa_attention(attn, q, k_flat, v_flat)

    assert calls == ["kv_update", "attn"]
    torch.testing.assert_close(got[:4], v_flat[:4] * 2.0)
    torch.testing.assert_close(got[4:], torch.zeros_like(v_flat[4:]))


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * weight.float()).to(x.dtype)


class _Norm(torch.nn.Module):
    """An RMSNorm stand-in: callable, with `weight` / `variance_epsilon`."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(size) + 0.5)
        self.variance_epsilon = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(x, self.weight, self.variance_epsilon)


def test_mla_rpa_forward_matches_reference():
    """The RPA MLA path: low-rank q / kv projections, their RMSNorms, the
    `kv_b_proj` expansion into per-head K / V, attention and `o_proj`."""
    torch.manual_seed(0)
    hidden, q_lora, kv_lora, heads, hd = 16, 12, 8, 2, 4

    def _lin(in_f: int, out_f: int):
        w = torch.randn(out_f, in_f) * 0.1
        return lambda inp: (torch.nn.functional.linear(inp, w), None)

    mla = glm5_next_model.Glm5NextMLAAttention.__new__(
        glm5_next_model.Glm5NextMLAAttention
    )
    torch.nn.Module.__init__(mla)
    mla.hidden_size = hidden
    mla.num_local_heads = heads
    mla.q_lora_rank = q_lora
    mla.kv_lora_rank = kv_lora
    mla.qk_nope_head_dim = hd
    mla.qk_rope_head_dim = 0
    mla.qk_head_dim = hd
    mla.v_head_dim = hd
    mla.q_a_proj = _lin(hidden, q_lora)
    mla.q_a_layernorm = _Norm(q_lora)
    mla.q_b_proj = _lin(q_lora, heads * hd)
    mla.kv_a_proj_with_mqa = _lin(hidden, kv_lora)
    mla.kv_a_layernorm = _Norm(kv_lora)
    mla.kv_b_proj = _lin(kv_lora, heads * 2 * hd)
    mla.o_proj = _lin(heads * hd, hidden)
    mla.mla_attn = None
    mla.attn = lambda q, k, v: q * 2.0 + k - v

    inp = torch.randn(8, hidden)
    got = mla(inp, torch.arange(8))

    q = mla.q_b_proj(mla.q_a_layernorm(mla.q_a_proj(inp)[0]))[0]
    kv = mla.kv_b_proj(mla.kv_a_layernorm(mla.kv_a_proj_with_mqa(inp)[0]))[0]
    k, v = kv.view(8, heads, 2 * hd).split([hd, hd], dim=-1)
    want = mla.o_proj(q * 2.0 + k.reshape(8, -1) - v.reshape(8, -1))[0]
    torch.testing.assert_close(got, want)


def test_enable_tpu_swiglu_clamp_resolves_the_clamped_kernel():
    method = types.SimpleNamespace(_resolve_tpu_activation=lambda layer: layer.act)
    experts = types.SimpleNamespace(
        routed_experts=types.SimpleNamespace(quant_method=method)
    )
    glm5_next_model._enable_tpu_swiglu_clamp(experts, 10.0)
    resolve = method._resolve_tpu_activation
    for act in ("silu", "SiLU", "silu_and_mul", "swiglu"):
        assert resolve(types.SimpleNamespace(act=act)) == "silu_and_mul_with_clamp"
    assert resolve(types.SimpleNamespace(act="gelu")) == "gelu"

    # A non-TPU MoE method is left alone.
    other = types.SimpleNamespace(quant_method=object())
    glm5_next_model._enable_tpu_swiglu_clamp(other, 10.0)

    # The TPU kernel's clamp is fixed at 10.
    tpu_method = types.SimpleNamespace(_resolve_tpu_activation=lambda layer: "silu")
    with pytest.raises(NotImplementedError, match="swiglu_limit=7"):
        glm5_next_model._enable_tpu_swiglu_clamp(
            types.SimpleNamespace(quant_method=tpu_method), 7.0
        )


def _hc_site(hidden: int = 8):
    hc = glm5_next_model.Glm5NextHyperConnection(
        hidden_size=hidden,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    with torch.no_grad():
        hc.sublayer_norm_weight.copy_(torch.rand(hidden) + 0.5)
    return hc


def test_hyper_connection_keeps_the_checkpoint_parameters():
    """Same names, shapes and dtypes as the checkpoint's mHC tensors (and
    the renames in `_CKPT_RENAMES`), so weight loading is unchanged."""
    hc = _hc_site(hidden=8)
    assert {n: tuple(p.shape) for n, p in hc.named_parameters()} == {
        "fn": (24, 32),
        "base": (24,),
        "scale": (3,),
        "sublayer_norm_weight": (8,),
    }
    assert all(p.dtype == torch.float32 for p in hc.parameters())
    targets = {dst.rsplit(".", 1)[-1] for _, dst in glm5_next_model._CKPT_RENAMES}
    assert targets == {n for n, _ in hc.named_parameters()}


def test_hyper_connection_runs_dsv4_ops_with_glm_constants(monkeypatch):
    """The DeepSeek-V4 ops are built once per site with GLM's constants (one
    eps for the pre gates and Sinkhorn, post multiplier 2), take the raw
    fp32 `fn` / `scale` / `base`, and their collapsed stream goes through
    the weighted sublayer RMSNorm."""
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
        deepseek_v4_mhc_op,
    )

    hc = _hc_site()
    built, calls = [], []

    def _pre(residual, fn, scale, base):
        calls.append(("pre", fn, scale, base))
        return "post0", "comb0", residual.sum(dim=-2)

    def _fused(x, residual, post, comb, fn, scale, base):
        calls.append(("fused", fn, scale, base))
        return residual + 1.0, f"{post}+", f"{comb}+", x * 3.0

    def _get_mhc_ops(*args):
        built.append(args)
        return deepseek_v4_mhc_op.MHCOps(pre=_pre, fused=_fused, post=None)

    monkeypatch.setattr(deepseek_v4_mhc_op, "get_mhc_ops", _get_mhc_ops)
    residual = torch.randn(5, 4, 8, dtype=torch.bfloat16)

    post, comb, normed = hc.pre(residual)
    assert (post, comb) == ("post0", "comb0")
    want = _rms(residual.sum(dim=-2), hc.sublayer_norm_weight, 1e-5)
    torch.testing.assert_close(normed, want)
    assert normed.dtype == torch.bfloat16

    x = torch.randn(5, 8, dtype=torch.bfloat16)
    new_res, post, comb, normed = hc.post_pre(x, residual, post, comb)
    torch.testing.assert_close(new_res, residual + 1.0)
    assert (post, comb) == ("post0+", "comb0+")
    torch.testing.assert_close(normed, _rms(x * 3.0, hc.sublayer_norm_weight, 1e-5))

    assert built == [(1e-5, 1e-6, 1e-6, 2.0, 20)]
    for _, fn, scale, base in calls:
        assert fn is hc.fn and scale is hc.scale and base is hc.base


def test_decoder_layer_defers_each_post_into_the_next_pre():
    """The FFN post is returned, not applied, and the next layer's attention
    pre settles it in one fused op."""
    log = []

    class _Site:
        def __init__(self, name: str) -> None:
            self.name = name

        def pre(self, residual):
            log.append((self.name, "pre"))
            return "p", "c", residual.sum(dim=-2)

        def post_pre(self, out, residual, post, comb):
            log.append((self.name, "post_pre", post))
            return residual + out.unsqueeze(-2), post + "'", comb + "'", out * 2.0

    layer = glm5_next_model.Glm5NextDecoderLayer.__new__(
        glm5_next_model.Glm5NextDecoderLayer
    )
    torch.nn.Module.__init__(layer)
    layer.attn_hc, layer.ffn_hc = _Site("attn"), _Site("ffn")
    layer._run_attention = lambda x, positions: x + 1.0
    layer.mlp = lambda x: x * 3.0

    streams = torch.ones(2, 4, 3)
    residual, deferred = layer(positions=None, hidden_streams=streams)
    assert log == [("attn", "pre"), ("ffn", "post_pre", "p")]
    attn_out = streams.sum(dim=-2) + 1.0
    torch.testing.assert_close(residual, streams + attn_out.unsqueeze(-2))
    torch.testing.assert_close(deferred.sublayer_out, attn_out * 2.0 * 3.0)
    assert deferred.residual is residual and deferred.post == "p'"

    log.clear()
    layer(positions=None, hidden_streams=residual, deferred=deferred)
    assert log == [("attn", "post_pre", "p'"), ("ffn", "post_pre", "p''")]
