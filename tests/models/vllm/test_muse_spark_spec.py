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
"""CPU and TPU unit tests for Muse Spark DSpark speculative decoding (PR-C)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import vllm.models.muse_spark.common.dspark as up_dspark
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.transformers_utils.configs.muse_spark import MuseSparkDSparkConfig
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.models.vllm.test_muse_spark_text import (
    _use_cpu_platform,
    make_tiny_omni_config_dict,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.muse_spark.config import TPUMuseSparkConfig
from vllm_torchtpu.models.vllm.muse_spark.dspark import (
    TPUMuseSparkDSparkDecoderLayer,
    TPUMuseSparkDSparkForCausalLM,
    TPUMuseSparkDSparkModel,
    _swap_upstream_dspark_classes,
)
from vllm_torchtpu.models.vllm.muse_spark.model import TPUMuseSparkForCausalLM


def _make_tiny_dspark_dict(
    *,
    num_hidden_layers: int = 2,
    hidden_size: int = 64,
    target_hidden_size: int = 64,
    vocab_size: int = 256,
    causal: bool = True,
) -> dict[str, Any]:
    return {
        "model_type": "muse_spark_dspark",
        "architectures": ["MuseSparkDSparkModel"],
        "vocab_size": vocab_size,
        "draft_vocab_size": vocab_size,
        "mask_token_id": vocab_size - 1,
        "hidden_size": hidden_size,
        "target_hidden_size": target_hidden_size,
        "ffn_dim_multiplier": 2.0,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 131_072,
        "rms_norm_eps": 1e-5,
        "norm_zero_centered_gamma": True,
        "use_attn_o_gate": True,
        "use_attn_o_norm": True,
        "use_residual_gate": True,
        "residual_dtype": "fp32",
        "moe": {"is_enabled": False},
        "metap": {
            "use_metap": True,
            "router_has_one_inf_dim": True,
            "metap_mode": "sp",
            "base_width": 16,
        },
        "output_soft_cap_temp": 20.0,
        "global_attn_cfg": [2048],
        "layer_types": ["sliding_attention"] * num_hidden_layers,
        "target_layer_ids": [0, 1],
        "num_target_layers": 2,
        "markov_rank": 8,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "draft_position_mode": "rope",
        "dflash_config": {"causal": causal, "target_layer_ids": [0, 1]},
    }


def _make_dspark_vllm_config(
    tmp_path: Path,
    *,
    draft_dict: dict[str, Any] | None = None,
    target_quant_dict: dict[str, Any] | None = None,
    draft_tp: int | None = None,
) -> VllmConfig:
    target_cfg = make_tiny_omni_config_dict(num_hidden_layers=2)
    if target_quant_dict is not None:
        target_cfg["quantization_config"] = target_quant_dict
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "config.json").write_text(json.dumps(target_cfg))
    (draft_dir / "config.json").write_text(
        json.dumps(draft_dict if draft_dict is not None else _make_tiny_dspark_dict())
    )

    target_mc = ModelConfig(
        model=str(target_dir),
        tokenizer_mode="skip",
        dtype=torch.bfloat16,
        max_model_len=4096,
    )
    pc = ParallelConfig(tensor_parallel_size=1)
    spec_cfg = SpeculativeConfig(
        model=str(draft_dir),
        method="dspark",
        num_speculative_tokens=4,
        draft_tensor_parallel_size=draft_tp,
        target_model_config=target_mc,
        target_parallel_config=pc,
    )
    vc = VllmConfig(
        model_config=target_mc,
        cache_config=CacheConfig(block_size=128, cache_dtype="auto"),
        parallel_config=pc,
        device_config=DeviceConfig(device="cpu"),
        quant_config=VllmUnquantizedConfig(),
    )
    vc.speculative_config = spec_cfg
    VllmQuantConfig.set_configs(vc)
    TPUMuseSparkConfig.verify_and_update_config(vc)
    return vc


def _build_synthetic_dspark_weights(
    cfg: MuseSparkDSparkConfig,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(321)
    h = int(cfg.hidden_size)
    inter = int(cfg.intermediate_size_mlp)
    nq = int(cfg.num_attention_heads * cfg.head_dim)
    nkv = int(cfg.num_key_value_heads * cfg.head_dim)
    vocab = int(cfg.vocab_size)
    draft_vocab = int(cfg.draft_vocab_size)
    r = int(cfg.markov_rank)
    num_targets = len(cfg.target_layer_ids)

    w: dict[str, torch.Tensor] = {
        "fc.weight": torch.randn(
            h, int(cfg.target_hidden_size) * num_targets, dtype=torch.bfloat16
        )
        * 0.02,
        "hidden_norm.weight": torch.zeros(h, dtype=torch.float32),
        "norm.weight": torch.zeros(h, dtype=torch.float32),
        "lm_head.weight": torch.randn(draft_vocab, h, dtype=torch.bfloat16) * 0.02,
        "d2t": torch.zeros(draft_vocab, dtype=torch.long),
        "t2d": torch.zeros(vocab, dtype=torch.bool),
        "markov_head.markov_w1.weight": torch.randn(vocab, r, dtype=torch.bfloat16)
        * 0.02,
        "markov_head.markov_w2.weight": torch.randn(
            draft_vocab, r, dtype=torch.bfloat16
        )
        * 0.02,
    }
    if cfg.enable_confidence_head:
        cw = h + (r if cfg.confidence_head_with_markov else 0)
        w["confidence_head.proj.weight"] = (
            torch.randn(1, cw, dtype=torch.float32) * 0.02
        )
        w["confidence_head.proj.bias"] = torch.zeros(1, dtype=torch.float32)

    for li in range(int(cfg.num_hidden_layers)):
        p = f"layers.{li}"
        w[f"{p}.self_attn.pre_qkv_proj_norm.weight"] = torch.zeros(
            h, dtype=torch.float32
        )
        w[f"{p}.self_attn.q_proj.weight"] = (
            torch.randn(nq, h, dtype=torch.bfloat16) * 0.05
        )
        w[f"{p}.self_attn.k_proj.weight"] = (
            torch.randn(nkv, h, dtype=torch.bfloat16) * 0.05
        )
        w[f"{p}.self_attn.v_proj.weight"] = (
            torch.randn(nkv, h, dtype=torch.bfloat16) * 0.05
        )
        w[f"{p}.self_attn.o_gate_proj.weight"] = (
            torch.randn(nq, h, dtype=torch.bfloat16) * 0.05
        )
        w[f"{p}.self_attn.o_proj.weight"] = (
            torch.randn(h, nq, dtype=torch.bfloat16) * 0.05
        )
        w[f"{p}.post_attn_norm.weight"] = torch.zeros(h, dtype=torch.float32)
        w[f"{p}.add_attn.gate"] = torch.zeros(h, dtype=torch.float32)
        w[f"{p}.pre_ffn_norm.weight"] = torch.zeros(h, dtype=torch.float32)
        w[f"{p}.feed_forward.gate_proj.weight"] = (
            torch.randn(inter, h, dtype=torch.bfloat16) * 0.02
        )
        w[f"{p}.feed_forward.up_proj.weight"] = (
            torch.randn(inter, h, dtype=torch.bfloat16) * 0.02
        )
        w[f"{p}.feed_forward.down_proj.weight"] = (
            torch.randn(h, inter, dtype=torch.bfloat16) * 0.02
        )
        w[f"{p}.post_ffn_norm.weight"] = torch.zeros(h, dtype=torch.float32)
        w[f"{p}.add_ffn.gate"] = torch.zeros(h, dtype=torch.float32)
    return w


def test_registry_and_no_det_imports() -> None:
    """MuseSparkDSparkModel registers lazily with zero batch_invariant imports."""
    register_models()
    resolved = ModelRegistry.models["MuseSparkDSparkModel"].load_model_cls()
    assert resolved is TPUMuseSparkDSparkForCausalLM

    code = (
        "import sys, vllm_torchtpu.models.vllm.muse_spark.dspark;"
        "bad=[k for k in sys.modules if k.startswith('vllm_torchtpu') and "
        "('batch_invariant' in k or 'det_collectives' in k)];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_swap_restores_upstream_classes_on_exception() -> None:
    orig_layer = up_dspark.MuseSparkDSparkDecoderLayer
    orig_model = up_dspark.MuseSparkDSparkModel
    with pytest.raises(RuntimeError, match="boom"), _swap_upstream_dspark_classes():
        assert up_dspark.MuseSparkDSparkDecoderLayer is TPUMuseSparkDSparkDecoderLayer
        raise RuntimeError("boom")
    assert up_dspark.MuseSparkDSparkDecoderLayer is orig_layer
    assert up_dspark.MuseSparkDSparkModel is orig_model


def test_tpu_draft_layer_proposer_contract_and_full_kv_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Draft builds TPU layers, offsets KV names, and keeps FullAttentionSpec."""
    _use_cpu_platform(monkeypatch, request)
    vc = _make_dspark_vllm_config(tmp_path)
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(vc):
        draft = TPUMuseSparkDSparkForCausalLM(vllm_config=vc)

    assert isinstance(draft.model, TPUMuseSparkDSparkModel)
    assert all(
        isinstance(layer, TPUMuseSparkDSparkDecoderLayer)
        for layer in draft.model.layers
    )
    assert draft.has_own_lm_head is True
    assert draft.has_own_embed_tokens is False
    assert not getattr(type(draft), "owns_context_kv", False)
    for attr in (
        "compute_draft_logits",
        "markov_embed",
        "markov_bias",
        "map_draft_to_target",
        "has_own_lm_head",
        "has_own_embed_tokens",
    ):
        assert hasattr(draft, attr), attr

    assert draft.get_draft_attn_causal() == [True, True]
    assert draft.get_draft_kv_cache_layer_names() == [
        "model.layers.2.self_attn.attn",
        "model.layers.3.self_attn.attn",
    ]
    for layer in draft.model.layers:
        assert layer.self_attn.rotary_emb is not None
        assert layer.self_attn.rotary_emb.is_neox_style is True
        spec = layer.self_attn.attn.get_kv_cache_spec(vc)
        assert isinstance(spec, FullAttentionSpec)
        assert spec.block_size == 128


def test_noncausal_draft_and_tp_mismatch_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    _use_cpu_platform(monkeypatch, request)
    vc_noncausal = _make_dspark_vllm_config(
        tmp_path / "nc", draft_dict=_make_tiny_dspark_dict(causal=False)
    )
    with (
        set_current_vllm_config(vc_noncausal),
        pytest.raises(NotImplementedError, match="causal drafts only"),
    ):
        TPUMuseSparkDSparkForCausalLM(vllm_config=vc_noncausal)

    vc_tp = _make_dspark_vllm_config(tmp_path / "tp")
    assert vc_tp.speculative_config is not None
    vc_tp.speculative_config.draft_tensor_parallel_size = 2
    with pytest.raises(NotImplementedError, match="draft_tensor_parallel_size"):
        TPUMuseSparkConfig.verify_and_update_config(vc_tp)


def test_unlisted_arch_draft_with_nvfp4_target_repaired(tmp_path: Path) -> None:
    """D9/F037: non-Muse draft with NVFP4 target repairs to MuseSparkDSparkConfig."""
    register_models()
    raw_draft = _make_tiny_dspark_dict()
    raw_draft["model_type"] = "qwen3"
    raw_draft["architectures"] = ["DSparkDraftModel"]
    del raw_draft["global_attn_cfg"]
    raw_draft["sliding_window"] = 2048

    vc = _make_dspark_vllm_config(tmp_path, draft_dict=raw_draft)
    assert vc.speculative_config is not None
    # Simulate SpeculativeConfig.__post_init__ leaking target NVFP4 quantization.
    vc.model_config.quantization = "modelopt_fp4"  # type: ignore[assignment]
    vc.speculative_config.draft_model_config.quantization = "modelopt_fp4"  # type: ignore[assignment]
    TPUMuseSparkConfig.verify_and_update_config(vc)

    draft_mc = vc.speculative_config.draft_model_config
    assert isinstance(draft_mc.hf_config, MuseSparkDSparkConfig)
    assert draft_mc.architectures == ["MuseSparkDSparkModel"]
    assert draft_mc.hf_config.global_attn_cfg == [2048]
    assert draft_mc.quantization is None
    assert get_draft_quant_config(vc) is None

    # A genuinely quantized unlisted-arch draft preserves its quantization_config.
    quant_draft = dict(raw_draft)
    quant_draft["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
    }
    vc_qdraft = _make_dspark_vllm_config(tmp_path / "qdraft", draft_dict=quant_draft)
    TPUMuseSparkConfig.verify_and_update_config(vc_qdraft)
    assert vc_qdraft.speculative_config is not None
    qdraft_hf = vc_qdraft.speculative_config.draft_model_config.hf_config
    assert isinstance(qdraft_hf, MuseSparkDSparkConfig)
    assert qdraft_hf.quantization_config == quant_draft["quantization_config"]

    # A user-requested draft quantization (speculative_config.quantization)
    # on an unquantized draft checkpoint is kept, not cleared as a leak.
    vc_user_q = _make_dspark_vllm_config(tmp_path / "user_q", draft_dict=raw_draft)
    assert vc_user_q.speculative_config is not None
    vc_user_q.model_config.quantization = "modelopt_fp4"  # type: ignore[assignment]
    vc_user_q.speculative_config.quantization = "fp8"  # type: ignore[assignment]
    vc_user_q.speculative_config.draft_model_config.quantization = "modelopt_fp4"  # type: ignore[assignment]
    TPUMuseSparkConfig.verify_and_update_config(vc_user_q)
    assert vc_user_q.speculative_config.draft_model_config.quantization == "fp8"

    # Non-dspark spec method is rejected.
    vc_bad = _make_dspark_vllm_config(tmp_path / "bad_spec")
    assert vc_bad.speculative_config is not None
    vc_bad.speculative_config.method = "ngram"
    with pytest.raises(NotImplementedError, match="dspark"):
        TPUMuseSparkConfig.verify_and_update_config(vc_bad)


def test_target_aux_hidden_states_cast_to_bf16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """D8/D19: TPUMuseSparkModel.forward returns aux hidden states in bf16."""
    _use_cpu_platform(monkeypatch, request)
    vc = _make_dspark_vllm_config(tmp_path)
    text_vc = VllmConfig(
        model_config=ModelConfig(
            model=vc.model_config.model,
            tokenizer_mode="skip",
            dtype=torch.bfloat16,
            max_model_len=4096,
        ),
        cache_config=vc.cache_config,
        parallel_config=vc.parallel_config,
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        quant_config=VllmUnquantizedConfig(),
    )
    VllmQuantConfig.set_configs(text_vc)
    text_vc.model_config.hf_config = vc.model_config.hf_config.text_config
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(text_vc):
        target = TPUMuseSparkForCausalLM(vllm_config=text_vc)

    # Stub layer forward to avoid needing an active TPU attention backend context.
    for layer in target.model.layers:
        monkeypatch.setattr(
            layer,
            "forward",
            lambda _pos, x: x.float() if isinstance(x, torch.Tensor) else x.stream,
        )
    target.set_aux_hidden_state_layers((1, 2))
    out, aux = target.model(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        intermediate_tensors=None,
        inputs_embeds=None,
    )
    assert out.dtype == torch.bfloat16
    assert len(aux) == 2
    assert all(t.dtype == torch.bfloat16 and t.shape == (3, 64) for t in aux)


def test_neox_load_context_kv_hook_and_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """F033/F035/F036/NC-2/NC-3: NEOX permute, reload guard, fp32 K RoPE, logits."""
    _use_cpu_platform(monkeypatch, request)
    vc = _make_dspark_vllm_config(tmp_path)
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(vc):
        draft = TPUMuseSparkDSparkForCausalLM(vllm_config=vc)

    weights = _build_synthetic_dspark_weights(draft.config)
    raw_k0 = weights["layers.0.self_attn.k_proj.weight"].clone()
    loaded = draft.load_weights(weights.items())
    assert "lm_head.weight" in loaded

    # Verify k_proj rows were permuted from interleaved pairs to NEOX halves.
    hd = int(draft.config.head_dim)
    expected_k0 = (
        raw_k0.view(-1, hd // 2, 2, raw_k0.shape[-1]).transpose(1, 2).reshape_as(raw_k0)
    )
    q_size = draft.model.layers[0].self_attn.q_size
    kv_size = draft.model.layers[0].self_attn.kv_size
    loaded_k0 = draft.model.layers[0].self_attn.qkv_proj.weight[
        q_size : q_size + kv_size
    ]
    assert torch.equal(loaded_k0, expected_k0)

    # Verify context KV hook rotates K only in fp32 and matches independent reference
    # even at large positions (> 65536, testing fp32 angle precision).
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for attn_layer in draft.model._attn_layers:
        attn_layer.kv_cache = torch.empty(0)
        attn_layer.impl = SimpleNamespace(
            forward=lambda *, layer, query, key, value, kv_cache, attn_metadata: (
                captured.append((query.clone(), key.clone(), value.clone()))
            )
        )

    rope_calls: list[tuple[torch.dtype, Any]] = []
    orig_rope = draft.model._rotary_emb
    assert orig_rope is not None

    class _SpyRope(torch.nn.Module):
        def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
        ) -> Any:
            rope_calls.append((query.dtype, key))
            return orig_rope(positions, query, key)

    monkeypatch.setattr(draft.model, "_rotary_emb", _SpyRope())

    positions = torch.tensor([0, 70_000, 100_000], dtype=torch.long)
    target_hidden = torch.randn(3, 128, dtype=torch.bfloat16)
    ctx_out = draft.tpu_precompute_and_store_context_kv(
        target_hidden, positions, (None, None)
    )
    assert ctx_out.shape == (3, 64)
    assert len(rope_calls) == 1
    assert rope_calls[0][0] == torch.float32
    assert rope_calls[0][1] is None
    assert len(captured) == 2
    assert all(torch.count_nonzero(q) == 0 for q, _, _ in captured)

    # Check K0 against independent fp32 NEOX RoPE vs bf16-angle RoPE at pos >= 65536.
    ctx_ref = draft.combine_hidden_states(target_hidden)
    kv_proj = draft.model._project_muse_context_kv(ctx_ref).view(
        2, 3, 2, draft.model._num_kv_heads, hd
    )
    k0_raw = kv_proj[0, :, 0].float()
    k0_norm = k0_raw * torch.rsqrt(
        k0_raw.pow(2).mean(-1, keepdim=True) + draft.model._rms_norm_eps
    )
    with set_current_vllm_config(vc):
        ref_rope_fp32 = RotaryEmbedding(
            head_size=hd,
            rotary_dim=hd,
            max_position_embeddings=131_072,
            base=500_000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )
    k0_ref, _ = ref_rope_fp32(positions, k0_norm, None)
    assert torch.equal(captured[0][1], k0_ref.to(torch.bfloat16))

    # Verify embed_input_ids applies weightless fp32 RMSNorm (NC-2) and
    # compute_draft_logits / markov_bias apply output_multiplier & softcap (NC-3).
    # Draft shares target embed_tokens at runtime (has_own_embed_tokens=False), so
    # populate finite embedding weights here before invoking embed_input_ids.
    torch.manual_seed(99)
    draft.model.embed_tokens.weight.data.normal_(0.0, 0.02)
    tok_ids = torch.tensor([1, 5, 9], dtype=torch.long)
    raw_emb = draft.model.embed_tokens(tok_ids).float()
    expected_emb = raw_emb * torch.rsqrt(
        raw_emb.pow(2).mean(-1, keepdim=True) + draft.config.rms_norm_eps
    )
    torch.testing.assert_close(draft.model.embed_input_ids(tok_ids), expected_emb)

    hidden = torch.randn(3, 64, dtype=torch.bfloat16)
    raw_l = draft.logits_processor(draft.lm_head, hidden).float()
    expected_l = 20.0 * torch.tanh(raw_l * draft.output_multiplier / 20.0)
    torch.testing.assert_close(draft.compute_draft_logits(hidden), expected_l)

    m_emb = draft.markov_embed(tok_ids)
    raw_b = draft.model.markov_head.bias(m_emb, draft.logits_processor).float()
    torch.testing.assert_close(
        draft.markov_bias(m_emb), raw_b * draft.output_multiplier
    )

    # Simulate TPU PWAL weight flip and verify reload_weights is rejected cleanly (H3).
    setattr(draft.model.layers[0].self_attn.qkv_proj, WEIGHT_FLIPPED_ATTR, True)
    with pytest.raises(NotImplementedError, match="reload_weights is not supported"):
        draft.load_weights(weights.items())


def test_dummy_load_after_tpu_weight_flip_builds_fused_kv_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Dummy load (skipping load_weights) builds _stacked_kv_weight post-flip."""
    _use_cpu_platform(monkeypatch, request)
    vc = _make_dspark_vllm_config(tmp_path)
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(vc):
        draft = TPUMuseSparkDSparkForCausalLM(vllm_config=vc)

    for layer in draft.model.layers:
        qkv = layer.self_attn.qkv_proj
        qkv.weight = torch.nn.Parameter(
            qkv.weight.detach().t().contiguous(), requires_grad=False
        )
        setattr(qkv, WEIGHT_FLIPPED_ATTR, True)

    draft.process_weights_after_loading()
    attn0 = draft.model.layers[0].self_attn
    assert tuple(draft.model._stacked_kv_weight.shape) == (
        len(draft.model.layers),
        2 * attn0.kv_size,
        int(draft.config.hidden_size),
    )
