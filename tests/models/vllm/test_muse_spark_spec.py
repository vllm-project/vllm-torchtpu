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
"""CPU unit tests for Muse Spark speculative decoding draft models."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm.config import VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.muse_spark import dspark as dspark_module
from vllm_torchtpu.models.vllm.muse_spark.configuration import (
    MuseSparkConfig,
    MuseSparkDSparkConfig,
    register_muse_spark_configs,
)
from vllm_torchtpu.models.vllm.muse_spark.dspark import (
    MuseSparkDSparkAttention,
    MuseSparkDSparkModel,
    _remap_dspark_name,
)
from vllm_torchtpu.models.vllm.muse_spark.text_model import (
    MuseSparkForCausalLM,
    TPUMuseSparkConfig,
)


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return
    monkeypatch.setattr(tpu, "is_available", lambda: False)
    monkeypatch.setattr(tpu, "current_device", lambda: 0)
    monkeypatch.setattr(tpu, "device_count", lambda: 0)


@pytest.fixture
def _single_rank_dist(monkeypatch):
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)
    if not torch.distributed.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)


def test_spec_architectures_are_registered() -> None:
    register_models()
    register_muse_spark_configs()

    assert CONFIG_MAPPING["muse_spark_dspark"] is MuseSparkDSparkConfig
    assert (
        ModelRegistry.models["MuseSparkDSparkModel"].load_model_cls()
        is MuseSparkDSparkModel
    )
    # No TPU DFlash2 path exists yet, so no Muse DFlash2 architecture is
    # registered.
    assert "MuseSparkDFlash2Model" not in ModelRegistry.models
    assert MODELS_CONFIG_MAP["MuseSparkDSparkModel"] is TPUMuseSparkConfig


def test_muse_spark_dspark_config_defaults() -> None:
    cfg = MuseSparkDSparkConfig()
    assert cfg.model_type == "muse_spark_dspark"
    assert cfg.hidden_size == 8192
    assert cfg.intermediate_size == 16384
    assert cfg.num_hidden_layers == 5
    assert cfg.num_attention_heads == 128
    assert cfg.num_key_value_heads == 16
    assert cfg.head_dim == 64
    assert cfg.sliding_window == 2048
    assert cfg.target_layer_ids == [13, 29, 45, 57, 61]
    assert cfg.block_size == 7
    assert cfg.markov_rank == 512
    assert cfg.sample_from_anchor is True
    assert cfg.dflash_config["mask_token_id"] == 200018
    assert cfg.dflash_config["target_layer_ids"] == [13, 29, 45, 57, 61]
    assert cfg.dflash_config["causal"] is True


def test_remap_dspark_weights() -> None:
    assert _remap_dspark_name("fc.weight") == "model.fc.weight"
    assert _remap_dspark_name("hidden_norm.weight") == "model.hidden_norm.weight"
    assert _remap_dspark_name("norm.weight") == "model.norm.weight"
    assert _remap_dspark_name("lm_head.weight") == "lm_head.weight"
    assert _remap_dspark_name("d2t") == "model.d2t"
    assert (
        _remap_dspark_name("markov_head.markov_w1.weight")
        == "model.markov_head.markov_w1.weight"
    )
    assert _remap_dspark_name("confidence_head.proj.weight") is None
    assert _remap_dspark_name("model.confidence_head.proj.weight") is None

    # Layer weights
    assert (
        _remap_dspark_name("layers.0.self_attn.pre_qkv_proj_norm.weight")
        == "model.layers.0.input_layernorm.weight"
    )
    assert (
        _remap_dspark_name("layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        _remap_dspark_name("layers.0.add_attn.gate")
        == "model.layers.0.post_attention_residual_gate.gate"
    )
    assert (
        _remap_dspark_name("layers.0.pre_ffn_norm.weight")
        == "model.layers.0.pre_feedforward_layernorm.weight"
    )
    assert (
        _remap_dspark_name("layers.0.feed_forward.gate_proj.weight")
        == "model.layers.0.feed_forward.gate_proj.weight"
    )
    assert (
        _remap_dspark_name("layers.0.post_ffn_norm.weight")
        == "model.layers.0.post_feedforward_layernorm.weight"
    )
    assert (
        _remap_dspark_name("layers.0.add_ffn.gate")
        == "model.layers.0.post_feedforward_residual_gate.gate"
    )


def test_set_aux_hidden_state_layers_normalization() -> None:
    mock_model = mock.MagicMock()
    mock_model.layers = [mock.MagicMock() for _ in range(62)]
    causal_lm = mock.MagicMock(spec=MuseSparkForCausalLM)
    causal_lm.model = mock_model
    MuseSparkForCausalLM.set_aux_hidden_state_layers(causal_lm, (14, 30, 46, 58, 62))
    assert mock_model.aux_hidden_state_layers == (14, 30, 46, 58, 62)

    # Ids are stored as given (callers pass 1-indexed ids), with no guessing.
    MuseSparkForCausalLM.set_aux_hidden_state_layers(causal_lm, [13, 29, 45, 57, 61])
    assert mock_model.aux_hidden_state_layers == (13, 29, 45, 57, 61)

    MuseSparkForCausalLM.set_aux_hidden_state_layers(causal_lm, [3, 19, 31, 47, 59])
    assert mock_model.aux_hidden_state_layers == (3, 19, 31, 47, 59)


_T = 6
_HIDDEN = 64
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16
_VOCAB = 256
_TARGET_LAYERS = 3


def _tiny_dspark_config(**overrides) -> MuseSparkDSparkConfig:
    kwargs = dict(
        vocab_size=_VOCAB,
        draft_vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        target_hidden_size=_HIDDEN,
        num_target_layers=2,
        # Deliberately different from the target layer count: the draft must
        # take its layer offset from the target model config.
        target_num_hidden_layers=62,
        target_layer_ids=[0, 1],
        markov_rank=16,
        block_size=4,
    )
    kwargs.update(overrides)
    return MuseSparkDSparkConfig(**kwargs)


def _build_dspark(cfg: MuseSparkDSparkConfig | None = None):
    from vllm.config import set_current_vllm_config

    cfg = cfg or _tiny_dspark_config()
    vllm_config = VllmConfig()
    vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=cfg, quantization=None),
        num_speculative_tokens=4,
        enable_adaptive_verification=False,
    )
    vllm_config.model_config = SimpleNamespace(
        max_model_len=128,
        get_vocab_size=lambda: _VOCAB,
        get_num_layers=lambda parallel_config: _TARGET_LAYERS,
        dtype=torch.float32,
        is_mm_prefix_lm=False,
        head_dtype=None,
        rswa_window=None,
    )
    with set_current_vllm_config(vllm_config):
        model = MuseSparkDSparkModel(vllm_config=vllm_config)
    for p in model.parameters():
        with torch.no_grad():
            p.normal_(0.0, 0.2)
    model.process_weights_after_loading()
    return model


def _neox_order(head_dim: int) -> torch.Tensor:
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def _interleaved_rope(
    x: torch.Tensor, positions: torch.Tensor, theta: float
) -> torch.Tensor:
    """Reference GPT-J style RoPE on (2i, 2i+1) pairs; x is [T, H, D]."""
    d = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = positions.double()[:, None] * inv_freq[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x = x.double()
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out


def _causal_gqa_attention(q, k, v, num_heads, num_kv_heads, head_dim):
    t = q.shape[0]
    q = q.float().view(t, num_heads, head_dim).transpose(0, 1)
    k = k.float().view(t, num_kv_heads, head_dim).transpose(0, 1)
    v = v.float().view(t, num_kv_heads, head_dim).transpose(0, 1)
    rep = num_heads // num_kv_heads
    k = k.repeat_interleave(rep, dim=0)
    v = v.repeat_interleave(rep, dim=0)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(0, 1).reshape(t, num_heads * head_dim)


def test_muse_spark_dspark_model_hooks(_single_rank_dist) -> None:
    model = _build_dspark()
    assert model.owns_context_kv is True
    assert model.has_own_embed_tokens is False
    assert model.has_own_lm_head is True
    assert model.model.confidence_head is None
    assert model.get_draft_attn_causal() == [True, True]

    model.model.embed_tokens = torch.nn.Embedding(_VOCAB, _HIDDEN)
    embeds = model.embed_input_ids(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert embeds.shape == (4, _HIDDEN)

    combined = model.combine_hidden_states(
        [torch.randn(4, _HIDDEN), torch.randn(4, _HIDDEN)]
    )
    assert combined.shape == (4, _HIDDEN)

    mapped = model.map_draft_to_target(torch.tensor([10, 20], dtype=torch.int32))
    assert torch.equal(mapped, torch.tensor([10, 20], dtype=torch.int32))


def test_draft_layer_names_start_after_target_layers(_single_rank_dist) -> None:
    model = _build_dspark()
    # Offset comes from model_config.get_num_layers (3), not the draft
    # config's target_num_hidden_layers (62).
    assert model.get_draft_kv_cache_layer_names() == [
        f"model.layers.{_TARGET_LAYERS}.self_attn.attn",
        f"model.layers.{_TARGET_LAYERS + 1}.self_attn.attn",
    ]


def test_production_target_layer_count_matches_draft_default() -> None:
    from vllm.distributed.utils import get_pp_indices
    from vllm.transformers_utils.config import get_hf_text_config
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    target = MuseSparkConfig()
    total = ModelArchConfigConvertorBase(
        target, get_hf_text_config(target)
    ).get_num_hidden_layers()
    # PP=1 in production, so get_num_layers returns the full count, which is
    # the value the draft used before (target_num_hidden_layers=62).
    assert get_pp_indices(total, 0, 1) == (0, 62)
    assert MuseSparkDSparkConfig().target_num_hidden_layers == 62


@pytest.mark.parametrize(
    ("heads", "kv_heads", "match"),
    [(12, 8, "num_attention_heads"), (16, 4, "num_key_value_heads")],
)
def test_draft_attention_rejects_indivisible_heads(
    monkeypatch, heads, kv_heads, match
) -> None:
    monkeypatch.setattr(
        dspark_module, "get_tensor_model_parallel_world_size", lambda: 8
    )
    cfg = _tiny_dspark_config(num_attention_heads=heads, num_key_value_heads=kv_heads)
    with pytest.raises(ValueError, match=match):
        MuseSparkDSparkAttention(cfg)


def test_forward_runs_and_matches_context_kv_insert(_single_rank_dist) -> None:
    model = _build_dspark().to(torch.bfloat16)
    # Production builds RoPE with an fp32 cos/sin cache; keep it fp32 (vLLM
    # also shares the rotary module across models through get_rope's cache).
    for layer in model.model.layers:
        layer.self_attn.rotary_emb.to(torch.float32)
    model.model.embed_tokens = torch.nn.Embedding(_VOCAB, _HIDDEN).to(torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 7, 8, 300])

    forward_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    insert_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    metadata = tuple(object() for _ in model.model.layers)

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def fake_attn(q, k, v, _attn=attn):
            forward_kv.append((k.clone(), v.clone()))
            return _causal_gqa_attention(
                q, k, v, _attn.num_heads, _attn.num_kv_heads, _attn.head_dim
            ).to(v.dtype)

        def fake_impl_forward(
            *, layer, query, key, value, kv_cache, attn_metadata, _i=i
        ):
            assert attn_metadata is metadata[_i]
            insert_kv.append((key.clone(), value.clone()))

        attn.attn.forward = fake_attn
        attn.attn.impl = SimpleNamespace(forward=fake_impl_forward)
        attn.attn.kv_cache = [None]

    input_ids = torch.tensor([3, 1, 4, 1, 5, 9])
    out = model(input_ids, positions)
    assert out.shape == (_T, _HIDDEN)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert len(forward_kv) == len(model.model.layers)

    # The context-KV insert must write the K/V that the layer's own forward
    # computes for the same input.
    target_hidden = torch.randn(_T, 2 * _HIDDEN, dtype=torch.bfloat16)
    returned = model.precompute_and_store_context_kv(target_hidden, positions, metadata)
    assert returned is target_hidden
    context = model.combine_hidden_states(target_hidden)
    forward_kv.clear()
    for layer in model.model.layers:
        layer(positions, context)
    assert len(insert_kv) == len(forward_kv) == len(model.model.layers)
    for (k_ins, v_ins), (k_fwd, v_fwd) in zip(insert_kv, forward_kv):
        torch.testing.assert_close(k_ins, k_fwd, rtol=0, atol=0)
        torch.testing.assert_close(v_ins, v_fwd, rtol=0, atol=0)


def test_rope_permutation_matches_interleaved_rope(
    _single_rank_dist, monkeypatch
) -> None:
    torch.manual_seed(0)
    model = _build_dspark()
    attn = model.model.layers[0].self_attn
    rope = attn.rotary_emb
    assert rope.is_neox_style
    # get_rope shares one module per config across tests, and other tests may
    # have cast its cos/sin cache to bf16; use a fresh fp32 cache here.
    monkeypatch.setattr(
        rope, "cos_sin_cache", rope._compute_cos_sin_cache().to(torch.float32)
    )
    theta = float(model.config.rope_theta)
    w_q = torch.randn_like(attn.q_proj.weight)
    w_k = torch.randn_like(attn.k_proj.weight)
    with torch.no_grad():
        attn.q_proj.weight.copy_(w_q)
        attn.k_proj.weight.copy_(w_k)
    attn._muse_rope_permuted = False
    x = torch.randn(_T, _HIDDEN)
    # Kept small: vLLM builds the cos/sin cache in fp32, whose angle error
    # grows with position, while the reference uses fp64.
    positions = torch.tensor([0, 1, 5, 17, 100, 300])

    # Projections in fp64: other tests may lower the global fp32 matmul
    # precision, which would swamp the comparison.
    def proj(w: torch.Tensor) -> torch.Tensor:
        return x.double() @ w.double().T

    ref_q = _interleaved_rope(proj(w_q).view(_T, -1, _HEAD_DIM), positions, theta)
    ref_k = _interleaved_rope(proj(w_k).view(_T, -1, _HEAD_DIM), positions, theta)

    model.process_weights_after_loading()
    # Idempotent: a second call must not permute again.
    model.process_weights_after_loading()
    q_neox, k_neox = rope(
        positions,
        proj(attn.q_proj.weight).float(),
        proj(attn.k_proj.weight).float(),
    )
    order = _neox_order(_HEAD_DIM)
    torch.testing.assert_close(
        q_neox.view(_T, -1, _HEAD_DIM).double(),
        ref_q[..., order],
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        k_neox.view(_T, -1, _HEAD_DIM).double(),
        ref_k[..., order],
        rtol=1e-4,
        atol=1e-4,
    )
    # Same permutation on q and k: attention scores are unchanged.
    scores_neox = torch.einsum(
        "thd,shd->hts",
        q_neox.view(_T, -1, _HEAD_DIM)[:, :_KV_HEADS].double(),
        k_neox.view(_T, -1, _HEAD_DIM).double(),
    )
    scores_ref = torch.einsum("thd,shd->hts", ref_q[:, :_KV_HEADS], ref_k)
    torch.testing.assert_close(scores_neox, scores_ref, rtol=1e-4, atol=1e-3)


def test_compute_draft_logits_scales_then_softcaps(_single_rank_dist) -> None:
    model = _build_dspark()
    mult = model.output_multiplier
    cap = model.final_logit_softcapping
    assert cap is not None
    hidden = torch.randn(_T, _HIDDEN) * 30.0
    raw = model.model.norm(hidden).float() @ model.lm_head.weight.float().T
    expected = cap * torch.tanh(mult * raw[:, :_VOCAB] / cap)
    torch.testing.assert_close(
        model.compute_draft_logits(hidden), expected, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        model.compute_logits(hidden), expected, rtol=1e-5, atol=1e-5
    )

    emb = model.markov_embed(torch.tensor([5, 10]))
    assert emb.shape == (2, 16)
    expected_bias = mult * (emb.float() @ model.model.markov_head.markov_w2.weight.T)
    torch.testing.assert_close(
        model.markov_bias(emb), expected_bias[:, :_VOCAB], rtol=1e-5, atol=1e-5
    )


def _synthetic_dspark_checkpoint(cfg) -> dict[str, torch.Tensor]:
    h, i = cfg.hidden_size, cfg.intermediate_size
    q = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.num_key_value_heads * cfg.head_dim
    ckpt = {
        "fc.weight": torch.randn(h, cfg.target_hidden_size * 2),
        "hidden_norm.weight": torch.randn(h),
        "norm.weight": torch.randn(h),
        "lm_head.weight": torch.randn(cfg.draft_vocab_size, h),
        "d2t": torch.randint(0, 5, (cfg.draft_vocab_size,), dtype=torch.int64),
        "markov_head.markov_w1.weight": torch.randn(cfg.vocab_size, cfg.markov_rank),
        "markov_head.markov_w2.weight": torch.randn(
            cfg.draft_vocab_size, cfg.markov_rank
        ),
        # Skipped: no TPU confidence head; embeddings come from the target.
        "confidence_head.proj.weight": torch.randn(4, 4),
        "embed_tokens.weight": torch.randn(cfg.vocab_size, h),
    }
    for n in range(cfg.num_hidden_layers):
        p = f"layers.{n}."
        ckpt.update(
            {
                p + "self_attn.pre_qkv_proj_norm.weight": torch.randn(h),
                p + "self_attn.q_proj.weight": torch.randn(q, h),
                p + "self_attn.k_proj.weight": torch.randn(kv, h),
                p + "self_attn.v_proj.weight": torch.randn(kv, h),
                p + "self_attn.o_gate_proj.weight": torch.randn(q, h),
                p + "self_attn.o_proj.weight": torch.randn(h, q),
                p + "add_attn.gate": torch.randn(h),
                p + "pre_ffn_norm.weight": torch.randn(h),
                p + "feed_forward.gate_proj.weight": torch.randn(i, h),
                p + "feed_forward.up_proj.weight": torch.randn(i, h),
                p + "feed_forward.down_proj.weight": torch.randn(h, i),
                p + "post_ffn_norm.weight": torch.randn(h),
                p + "add_ffn.gate": torch.randn(h),
            }
        )
    return ckpt


def test_load_weights_synthetic_checkpoint(_single_rank_dist) -> None:
    model = _build_dspark()
    for layer in model.model.layers:
        layer.self_attn._muse_rope_permuted = False
    cfg = model.config
    ckpt = _synthetic_dspark_checkpoint(cfg)
    loaded = model.load_weights(iter(ckpt.items()))

    expected_names = set(dict(model.named_parameters())) | {"model.d2t"}
    assert loaded == expected_names

    state = model.state_dict()
    torch.testing.assert_close(state["model.fc.weight"], ckpt["fc.weight"])
    torch.testing.assert_close(state["lm_head.weight"], ckpt["lm_head.weight"])
    assert torch.equal(state["model.d2t"], ckpt["d2t"])
    torch.testing.assert_close(
        state["model.layers.1.post_feedforward_residual_gate.gate"],
        ckpt["layers.1.add_ffn.gate"],
    )
    torch.testing.assert_close(
        state["model.layers.0.input_layernorm.weight"],
        ckpt["layers.0.self_attn.pre_qkv_proj_norm.weight"],
    )
    # q/k rows are permuted per head from interleaved to NEOX order once.
    order = _neox_order(cfg.head_dim)
    for n, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert attn._muse_rope_permuted
        assert layer.post_attention_residual_gate.alpha is not None
        for proj in ("q_proj", "k_proj"):
            src = ckpt[f"layers.{n}.self_attn.{proj}.weight"]
            want = src.view(-1, cfg.head_dim, src.shape[-1])[:, order].reshape_as(src)
            torch.testing.assert_close(getattr(attn, proj).weight.data, want)
        torch.testing.assert_close(
            attn.v_proj.weight.data, ckpt[f"layers.{n}.self_attn.v_proj.weight"]
        )

    draft_ids = torch.tensor([0, 7, 200])
    assert torch.equal(
        model.map_draft_to_target(draft_ids), draft_ids + ckpt["d2t"][draft_ids]
    )


@pytest.mark.parametrize(
    "flag", ["use_qk_norm", "use_attn_o_gate", "use_attn_o_norm", "use_residual_gate"]
)
def test_dspark_config_rejects_disabled_blocks(flag) -> None:
    draft_cfg = SimpleNamespace(
        hf_config=MuseSparkDSparkConfig(**{flag: False}),
        quantization=None,
        quantization_config=None,
    )
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=MuseSparkConfig()),
        cache_config=SimpleNamespace(block_size=128, user_specified_block_size=True),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=draft_cfg, update_arch_=lambda: None
        ),
    )
    with pytest.raises(ValueError, match=flag):
        TPUMuseSparkConfig.verify_and_update_config(vllm_cfg)


def test_dspark_config_accepts_default_muse_draft() -> None:
    hf = MuseSparkDSparkConfig()
    draft_cfg = SimpleNamespace(
        hf_config=hf, quantization=None, quantization_config=None
    )
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=MuseSparkConfig()),
        cache_config=SimpleNamespace(block_size=128, user_specified_block_size=True),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=draft_cfg, update_arch_=lambda: None
        ),
    )
    TPUMuseSparkConfig.verify_and_update_config(vllm_cfg)
    assert hf.architectures == ["MuseSparkDSparkModel"]


def test_dspark_config_resolution_uses_target_type_not_directory_name() -> None:
    from vllm_torchtpu.models.vllm.muse_spark.configuration import MuseSparkConfig

    updated = []
    draft_hf = SimpleNamespace(model_type="qwen3_dspark", architectures=["Qwen3DSpark"])
    draft_cfg = SimpleNamespace(
        model="/checkpoints/custom-finetune-draft-v2",
        hf_config=draft_hf,
        quantization="nvfp4",
        quantization_config={"quant_method": "modelopt"},
    )
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=MuseSparkConfig()),
        cache_config=SimpleNamespace(block_size=128, user_specified_block_size=True),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=draft_cfg,
            update_arch_=lambda: updated.append(True),
        ),
    )
    TPUMuseSparkConfig.verify_and_update_config(vllm_cfg)
    assert draft_hf.model_type == "muse_spark_dspark"
    assert draft_hf.architectures == ["MuseSparkDSparkModel"]
    assert draft_cfg.quantization is None
    assert draft_cfg.quantization_config is None
    assert updated == [True]
