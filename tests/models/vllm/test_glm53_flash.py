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
from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_collectives_tpu,
    glm53_mhc_tpu,
)
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


def test_sp_prefill_enabled_and_min_tokens(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_SP_PREFILL", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_SP_MIN_TOKENS", 256, raising=False)
    assert glm5_next_model.glm53_sp_prefill_enabled(256, 8, mhc_pallas=True)
    assert glm5_next_model.glm53_sp_prefill_enabled(2048, 8, mhc_pallas=True)
    assert not glm5_next_model.glm53_sp_prefill_enabled(64, 8, mhc_pallas=True)
    assert not glm5_next_model.glm53_sp_prefill_enabled(300, 8, mhc_pallas=True)
    assert not glm5_next_model.glm53_sp_prefill_enabled(2048, 1, mhc_pallas=True)
    assert not glm5_next_model.glm53_sp_prefill_enabled(2048, 8, mhc_pallas=False)
    assert not glm5_next_model.glm53_sp_prefill_enabled(
        2048, 8, mhc_pallas=True, pp_size=2
    )

    min_tokens = glm5_next_model.glm53_sp_min_tokens(8, mhc_pallas=True)
    assert min_tokens == 256
    for n in (16, 64, 256, 300, 512, 2048):
        assert glm5_next_model._sp_prefill_for(
            n, min_tokens, 8
        ) == glm5_next_model.glm53_sp_prefill_enabled(n, 8, mhc_pallas=True)

    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_SP_PREFILL", False, raising=False)
    assert not glm5_next_model.glm53_sp_prefill_enabled(2048, 8, mhc_pallas=True)


def test_local_token_shard_and_sequence_parallel_mlp_mla_equivalence():
    x = torch.arange(8 * 3).reshape(8, 3)
    for rank in range(4):
        fake_tp = types.SimpleNamespace(world_size=4, rank_in_group=rank)
        with mock.patch.object(glm5_next_model, "get_tp_group", lambda tp=fake_tp: tp):
            got = glm5_next_model._local_token_shard(x)
        torch.testing.assert_close(got, x[rank * 2 : (rank + 1) * 2])

    torch.manual_seed(0)
    hidden, inter, q_lora, kv_lora, heads, hd = 16, 24, 12, 8, 2, 4

    class _Lin:
        def __init__(self, in_f: int, out_f: int) -> None:
            self.w = torch.randn(out_f, in_f) * 0.1
            self.quant_method = types.SimpleNamespace(apply=self._apply)

        def _apply(self, _l, inp: torch.Tensor, bias=None) -> torch.Tensor:
            return torch.nn.functional.linear(inp, self.w, bias)

        def __call__(self, inp: torch.Tensor):
            return self.quant_method.apply(self, inp), None

    mlp = glm5_next_model.Glm5NextMLP.__new__(glm5_next_model.Glm5NextMLP)
    torch.nn.Module.__init__(mlp)
    mlp.swiglu_limit = 10.0
    mlp.gate_up_proj = _Lin(hidden, 2 * inter)
    mlp.down_proj = _Lin(inter, hidden)

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
    mla.q_a_proj = _Lin(hidden, q_lora)
    mla.q_a_layernorm = types.SimpleNamespace(
        weight=torch.ones(q_lora), variance_epsilon=1e-5
    )
    mla.q_b_proj = _Lin(q_lora, heads * hd)
    mla.kv_a_proj_with_mqa = _Lin(hidden, kv_lora)
    mla.kv_a_layernorm = types.SimpleNamespace(
        weight=torch.ones(kv_lora), variance_epsilon=1e-5
    )
    mla.kv_b_proj = _Lin(kv_lora, heads * 2 * hd)
    mla.o_proj = _Lin(heads * hd, hidden)
    mla.mla_attn = None
    mla.attn = lambda q, k, v: q + k + v

    inp = torch.randn(8, hidden)
    pos = torch.arange(8)
    want_mlp = mlp.forward(inp, sequence_parallel=False)
    want_mla = mla.forward(inp, pos, sequence_parallel=False)

    inp_shards = [inp[:4], inp[4:]]
    qkv_full = torch.cat(
        [
            glm53_mhc_tpu.rms_norm_weighted(
                mla.q_a_proj(inp)[0],
                mla.q_a_layernorm.weight,
                mla.q_a_layernorm.variance_epsilon,
            ),
            glm53_mhc_tpu.rms_norm_weighted(
                mla.kv_a_proj_with_mqa(inp)[0],
                mla.kv_a_layernorm.weight,
                mla.kv_a_layernorm.variance_epsilon,
            ),
        ],
        dim=-1,
    )
    for rank in (0, 1):
        fake_tp = types.SimpleNamespace(
            rank_in_group=rank,
            world_size=2,
            all_gather=lambda t, dim=0: (inp if t.shape[-1] == hidden else qkv_full),
            reduce_scatter=lambda t, dim=0, r=rank: t[r * 4 : (r + 1) * 4],
        )
        with (
            mock.patch.object(glm5_next_model, "get_tp_group", lambda tp=fake_tp: tp),
            mock.patch.object(
                glm53_collectives_tpu, "get_tp_group", lambda tp=fake_tp: tp
            ),
        ):
            got_mlp = mlp.forward(inp_shards[rank], sequence_parallel=True)
            got_mla = mla.forward(inp_shards[rank], pos, sequence_parallel=True)
        torch.testing.assert_close(got_mlp, want_mlp[rank * 4 : (rank + 1) * 4])
        torch.testing.assert_close(got_mla, want_mla[rank * 4 : (rank + 1) * 4])


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
