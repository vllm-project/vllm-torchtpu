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
"""CPU contract, math, quant, weight-loading, and config tests for Qwen4Exp MTP."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen4ExpTextConfig
from vllm.config import CompilationConfig, CompilationMode, set_current_vllm_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.models.qwen4_exp import model as qwen4_model
from vllm_torchtpu.models.qwen4_exp import mtp as qwen4_mtp
from vllm_torchtpu.models.qwen4_exp.config import TPUQwen4ExpConfig
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


def _tiny_text_config(**overrides) -> Qwen4ExpTextConfig:
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=97,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
        ple_conv_kernel_size=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=16,
        indexer_compress_ratio=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
        },
        max_position_embeddings=2048,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        mtp={
            "hybrid": True,
            "layer_types": ["full_attention"],
            "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 1,
            "rope_theta": 10000000,
        },
        tie_word_embeddings=False,
        eos_token_id=2,
        bos_token_id=1,
        seed=1234,
    )
    defaults.update(overrides)
    return Qwen4ExpTextConfig(**defaults)


def _mock_vllm_config(
    text_config: Qwen4ExpTextConfig | None = None,
    *,
    tp: int = 1,
    max_model_len: int = 1024,
    block_size: int = 64,
    cache_dtype: str = "auto",
    mamba_cache_mode: str = "align",
    enable_prefix_caching: bool = True,
    quantization: str | None = None,
    quant_config=None,
    speculative_config=None,
    kv_transfer_config=None,
):
    tc = text_config or _tiny_text_config()
    model_config = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        hf_text_config=tc,
        hf_config=tc,
        dtype=torch.bfloat16,
        head_dtype=None,
        max_model_len=max_model_len,
        quantization=quantization,
        get_num_kv_heads=lambda pc: max(
            1, tc.num_key_value_heads // pc.tensor_parallel_size
        ),
        get_head_size=lambda: tc.head_dim,
        get_hidden_size=lambda: tc.hidden_size,
    )
    cache_config = SimpleNamespace(
        block_size=block_size,
        cache_dtype=cache_dtype,
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="bfloat16",
        mamba_cache_mode=mamba_cache_mode,
        enable_prefix_caching=enable_prefix_caching,
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_sequence_parallel_moe=False,
    )
    compilation_config = CompilationConfig(
        mode=CompilationMode.NONE, custom_ops=["none"]
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        quant_config=quant_config,
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
        compilation_config=compilation_config,
        observability_config=SimpleNamespace(enable_layerwise_nvtx_tracing=False),
    )


def _mock_spec_config(
    draft_text_config: Qwen4ExpTextConfig,
    *,
    method: str = "mtp",
    num_spec: int = 3,
    draft_tp: int | None = None,
    index_share: bool = False,
    quantization: str | None = None,
):
    draft_hf = SimpleNamespace(
        hc_mult=draft_text_config.hc_count,
        index_share_for_mtp_iteration=index_share,
    )
    draft_model_config = SimpleNamespace(
        hf_config=draft_hf,
        hf_text_config=draft_text_config,
        quantization=quantization,
    )
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_spec,
        draft_tensor_parallel_size=draft_tp,
        draft_model_config=draft_model_config,
    )


# ── 1. Model registration ──────────────────────────────────────────────────


def test_qwen4_exp_mtp_is_registered_without_overriding_config_map() -> None:
    """Qwen4ExpMTP registers in ModelRegistry and leaves MODELS_CONFIG_MAP on
    upstream Qwen4ExpMTPConfig rather than binding TPUQwen4ExpConfig."""
    arch = "Qwen4ExpMTP"
    previous_model = ModelRegistry.models.get(arch)
    previous_config = MODELS_CONFIG_MAP.get(arch)
    try:
        register_layers()
        assert ModelRegistry.models[arch].load_model_cls() is qwen4_mtp.Qwen4ExpMTP
        assert MODELS_CONFIG_MAP.get(arch) is not TPUQwen4ExpConfig
    finally:
        if previous_model is None:
            ModelRegistry.models.pop(arch, None)
        else:
            ModelRegistry.models[arch] = previous_model
        if previous_config is None:
            MODELS_CONFIG_MAP.pop(arch, None)
        else:
            MODELS_CONFIG_MAP[arch] = previous_config


# ── 2. Quantization safety, unused-weight skipping, and MTP forward math ──


def test_shared_quant_config_preserves_all_mtp_ignored_layers() -> None:
    """Configuring target then draft against one shared Fp8Config must keep
    every mtp.* entry in ignored_layers intact."""
    mtp_ignored = [
        "mtp.fc_embedding",
        "mtp.fc_hidden",
        "mtp.hyper_connection_mixer.input_mix_weight_down",
        "mtp.hyper_connection_mixer.input_mix_weight_up",
        "mtp.layers.0.attn_hyper_connection.block_inject_weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_up",
        "mtp.layers.0.mlp.gate",
        "mtp.layers.0.mlp.shared_expert.down_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj",
        "mtp.layers.0.mlp.shared_expert.up_proj",
        "mtp.layers.0.mlp.shared_expert_gate",
        "mtp.layers.0.mlp_hyper_connection.block_inject_weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up",
        "mtp.layers.0.self_attn.indexer.index_qk_proj",
        "mtp.layers.0.self_attn.k_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.v_proj",
    ]
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=list(mtp_ignored),
        weight_block_size=[128, 128],
    )
    configure_quant_config(quant_config, qwen4_model.Qwen4ExpForCausalLM)
    configure_quant_config(quant_config, qwen4_mtp.Qwen4ExpMTP)

    assert not hasattr(qwen4_mtp.Qwen4ExpMTP, "hf_to_vllm_mapper")
    for entry in mtp_ignored:
        assert entry in quant_config.ignored_layers, (
            f"Shared ignored_layers lost {entry!r} after configuring Qwen4ExpMTP"
        )
    # Fused and unfused BF16 draft linears must all resolve as skipped; routed
    # MoE experts must remain quantized.
    for bf16_prefix in (
        "mtp.fc_embedding",
        "mtp.fc_hidden",
        "mtp.layers.0.self_attn.qkv_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.self_attn.indexer.index_qk_proj",
        "mtp.layers.0.mlp.shared_expert.gate_up_proj",
        "mtp.layers.0.mlp.shared_expert.down_proj",
        "mtp.layers.0.mlp.gate",
    ):
        assert is_layer_skipped(
            prefix=bf16_prefix,
            ignored_layers=quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
        ), f"Expected BF16 draft linear {bf16_prefix!r} to be skipped by Fp8Config"
    assert not is_layer_skipped(
        prefix="mtp.layers.0.mlp.experts",
        ignored_layers=quant_config.ignored_layers,
        fused_mapping=quant_config.packed_modules_mapping,
    )


def _test_param(
    shape: tuple[int, ...], *, init_scale: float = 0.0
) -> torch.nn.Parameter:
    data = torch.randn(shape) * init_scale if init_scale else torch.zeros(shape)
    param = torch.nn.Parameter(data)

    def _loader(target, loaded_weight, shard_id=None):
        resolved_shard = (
            shard_id
            if shard_id is not None
            else getattr(loaded_weight, "shard_id", None)
        )
        with torch.no_grad():
            if resolved_shard is None:
                target.copy_(loaded_weight)
            else:
                slot = {"q": 0, "k": 1, "v": 2}[resolved_shard]
                chunk = target.shape[0] // 3
                target[slot * chunk : (slot + 1) * chunk].copy_(loaded_weight)

    param.weight_loader = _loader  # type: ignore[attr-defined]
    return param


class _FakeEmbed(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = ""):
        super().__init__()
        self.weight = _test_param((num_embeddings, embedding_dim), init_scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(x, self.weight)


class _FakeLinear(torch.nn.Module):
    def __init__(self, in_f: int, out_f: int, **kwargs):
        super().__init__()
        self.weight = _test_param((out_f, in_f), init_scale=0.1)
        self.return_bias = kwargs.get("return_bias", True)

    def forward(self, x: torch.Tensor):
        out = torch.matmul(x, self.weight.T)
        return (out, None) if self.return_bias else out


def test_mtp_metadata_unused_weights_and_forward_math(monkeypatch) -> None:
    """Checks draft metadata, is_unused_checkpoint_weight, target emission,
    and Qwen4ExpMTP forward math against an independent reference."""

    class _FakeSublayer(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.indexer = None

        def forward(self, *args, **kwargs):
            h = kwargs.get("hidden_states")
            if h is None:
                h = args[-1]
            return h * 0.5

    for name in (
        "QwenGatedDeltaNetAttention",
        "Qwen4ExpAttention",
        "Qwen3NextSparseMoeBlock",
    ):
        monkeypatch.setattr(qwen4_model, name, _FakeSublayer)
    monkeypatch.setattr(qwen4_model, "VocabParallelEmbedding", _FakeEmbed)
    monkeypatch.setattr(qwen4_model, "ParallelLMHead", _FakeEmbed)
    monkeypatch.setattr(qwen4_mtp, "VocabParallelEmbedding", _FakeEmbed)
    monkeypatch.setattr(qwen4_mtp, "ParallelLMHead", _FakeEmbed)
    monkeypatch.setattr(qwen4_mtp, "ReplicatedLinear", _FakeLinear)

    torch.manual_seed(0)
    tc = _tiny_text_config(ple_layer_ids=[])
    spec = _mock_spec_config(tc, num_spec=3)
    vc_k0 = _mock_vllm_config(tc)
    vc = _mock_vllm_config(tc, speculative_config=spec)

    # Target emits (sample_hidden, [multi_hidden]) for MTP, bare tensor at K=0.
    with set_current_vllm_config(vc_k0):  # type: ignore[arg-type]
        target_k0 = qwen4_model.Qwen4ExpForCausalLM(vllm_config=vc_k0)
    with set_current_vllm_config(vc):  # type: ignore[arg-type]
        target_mtp = qwen4_model.Qwen4ExpForCausalLM(vllm_config=vc)
        draft = qwen4_mtp.Qwen4ExpMTP(vllm_config=vc)

    ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    pos = torch.tensor([0, 1, 2], dtype=torch.int32)
    out_k0 = target_k0(ids, pos)
    assert isinstance(out_k0, torch.Tensor) and out_k0.shape == (3, tc.hidden_size)
    out_mtp = target_mtp(ids, pos)
    assert isinstance(out_mtp, tuple) and len(out_mtp) == 2
    sample_h, aux_list = out_mtp
    assert sample_h.shape == (3, tc.hidden_size)
    assert len(aux_list) == 1 and aux_list[0].shape == (3, tc.hc_count * tc.hidden_size)
    assert target_mtp.mtp_emits_aux_hidden_state is True

    # Draft compile metadata and unused-weight filtering.
    vc_compile = _mock_vllm_config(tc, speculative_config=spec)
    vc_compile.compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, custom_ops=["none"]
    )
    with set_current_vllm_config(vc_compile):  # type: ignore[arg-type]
        compiled_draft = qwen4_mtp.Qwen4ExpMTP(vllm_config=vc_compile)
    assert draft.config is tc
    assert draft.has_own_embed_tokens is False
    assert draft.has_own_lm_head is False
    assert getattr(compiled_draft, "_dynamic_arg_dims", None) == {
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "inputs_embeds": 0,
    }
    assert draft.is_unused_checkpoint_weight(
        "model.language_model.layers.0.mlp.gate.weight"
    )
    assert draft.is_unused_checkpoint_weight("mtp.shared_head.head.weight")
    assert not draft.is_unused_checkpoint_weight("mtp.layers.0.mlp.gate.weight")
    assert not draft.is_unused_checkpoint_weight("mtp.fc_embedding.weight")
    assert not draft.is_unused_checkpoint_weight(
        "model.language_model.embed_tokens.weight"
    )

    # Set non-zero norm weights so an incorrect (w vs 1 + w) RMSNorm breaks.
    with torch.no_grad():
        draft.model.pre_fc_norm_embedding.weight.copy_(
            torch.randn_like(draft.model.pre_fc_norm_embedding.weight) * 0.3
        )
        draft.model.pre_fc_norm_hidden.weight.copy_(
            torch.randn_like(draft.model.pre_fc_norm_hidden.weight) * 0.3
        )

    def _gemma_rms_norm_ref(
        x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * (weight.float() + 1.0)).to(x.dtype)

    h_in = torch.randn(3, tc.hc_count * tc.hidden_size)
    got_sample, got_multi = draft.model(ids, pos, h_in, spec_step_idx=0)
    assert got_sample.shape == (3, tc.hidden_size)
    assert got_multi.shape == (3, tc.hc_count * tc.hidden_size)

    emb_raw = torch.nn.functional.embedding(ids, draft.model.embed_tokens.weight)
    emb_norm = _gemma_rms_norm_ref(
        emb_raw, draft.model.pre_fc_norm_embedding.weight, tc.rms_norm_eps
    )
    emb_ref = torch.nn.functional.linear(emb_norm, draft.model.fc_embedding.weight)

    h_norm = _gemma_rms_norm_ref(
        h_in, draft.model.pre_fc_norm_hidden.weight, tc.rms_norm_eps
    ).view(3, tc.hc_count, tc.hidden_size)
    h_proj = torch.nn.functional.linear(h_norm, draft.model.fc_hidden.weight)
    h_ref = (
        (h_proj.float() + emb_ref.float().unsqueeze(-2)).to(h_proj.dtype).flatten(-2)
    )
    multi_ref = draft.model.layers[0](pos, h_ref, input_ids=None)
    sample_ref, _ = draft.model.hyper_connection_mixer.mix(multi_ref)
    torch.testing.assert_close(got_multi, multi_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_sample, sample_ref, rtol=1e-5, atol=1e-5)

    with pytest.raises(NotImplementedError, match="PP"):
        draft.forward(ids, pos, h_in, intermediate_tensors=object())


def test_qwen4_exp_mtp_load_weights_remaps_fuses_and_skips(monkeypatch) -> None:
    """Exercises Qwen4ExpMTP.load_weights end-to-end, including skipping
    mtp.shared_head.head.* and failing fast when mtp.layers.0.* is absent."""

    class _FakeAttn(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.indexer = None
            self.qkv_proj = torch.nn.Module()
            self.qkv_proj.weight = _test_param((12, 4))

    class _FakeMlp(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.gate = torch.nn.Module()
            self.gate.weight = _test_param((4, 4))

    monkeypatch.setattr(qwen4_model, "Qwen4ExpAttention", _FakeAttn)
    monkeypatch.setattr(qwen4_model, "Qwen3NextSparseMoeBlock", _FakeMlp)
    monkeypatch.setattr(qwen4_mtp, "ReplicatedLinear", _FakeLinear)
    monkeypatch.setattr(qwen4_mtp, "VocabParallelEmbedding", _FakeEmbed)
    monkeypatch.setattr(qwen4_mtp, "ParallelLMHead", _FakeEmbed)

    tc = _tiny_text_config(hidden_size=4, hc_count=2, hc_lowrank=2, vocab_size=8)
    spec = _mock_spec_config(tc, num_spec=3)
    vc = _mock_vllm_config(tc, speculative_config=spec)
    with set_current_vllm_config(vc):  # type: ignore[arg-type]
        draft = qwen4_mtp.Qwen4ExpMTP(vllm_config=vc)

    orig_lm_head = draft.lm_head.weight.detach().clone()
    fc_emb_w = torch.full((4, 4), 1.5)
    fc_hid_w = torch.full((4, 4), 2.5)
    q_w = torch.full((4, 4), 3.0)
    k_w = torch.full((4, 4), 4.0)
    v_w = torch.full((4, 4), 5.0)
    gate_w = torch.full((4, 4), 6.0)
    norm_emb_w = torch.full((4,), 0.25)
    norm_hid_w = torch.full((8,), 0.75)

    synthetic_weights = [
        ("mtp.fc_embedding.weight", fc_emb_w),
        ("mtp.fc_hidden.weight", fc_hid_w),
        ("mtp.pre_fc_norm_embedding.weight", norm_emb_w),
        ("mtp.pre_fc_norm_hidden.weight", norm_hid_w),
        ("mtp.layers.0.self_attn.q_proj.weight", q_w),
        ("mtp.layers.0.self_attn.k_proj.weight", k_w),
        ("mtp.layers.0.self_attn.v_proj.weight", v_w),
        ("mtp.layers.0.mlp.gate.weight", gate_w),
        # Must be ignored so shared target lm_head is never overwritten:
        ("mtp.shared_head.head.weight", torch.full((8, 4), 9.0)),
        # Must be cleanly skipped without raising KeyError:
        ("mtp.hyper_connection_mixer.block_inject_weight", torch.zeros(2, 2)),
        ("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", torch.zeros(4, 4)),
        ("model.language_model.layers.0.mlp.gate.weight", torch.zeros(4, 4)),
    ]
    loaded = draft.load_weights(synthetic_weights)

    assert "model.fc_embedding.weight" in loaded
    assert "model.fc_hidden.weight" in loaded
    assert "model.layers.0.self_attn.qkv_proj.weight" in loaded
    assert "model.layers.0.mlp.gate.weight" in loaded
    assert "lm_head.weight" not in loaded
    torch.testing.assert_close(draft.lm_head.weight, orig_lm_head)
    torch.testing.assert_close(draft.model.fc_embedding.weight, fc_emb_w)
    torch.testing.assert_close(draft.model.fc_hidden.weight, fc_hid_w)
    torch.testing.assert_close(draft.model.pre_fc_norm_embedding.weight, norm_emb_w)
    torch.testing.assert_close(draft.model.pre_fc_norm_hidden.weight, norm_hid_w)
    torch.testing.assert_close(
        draft.model.layers[0].self_attn.qkv_proj.weight,
        torch.cat([q_w, k_w, v_w], dim=0),
    )
    torch.testing.assert_close(draft.model.layers[0].mlp.gate.weight, gate_w)

    # A checkpoint lacking mtp.layers.0.* must fail loudly rather than leaving
    # the draft layer on uninitialized weights.
    with pytest.raises(ValueError, match="mtp\\.layers\\.0"):
        draft.load_weights([("mtp.fc_embedding.weight", fc_emb_w)])


# ── 3. Config fail-fast gates and Eagle3Proposer HC seam ──────────────────


def test_config_accepts_valid_mtp_and_rejects_unsupported_combinations(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vllm_torchtpu.models.qwen4_exp.config.Qwen4ExpForCausalLMConfig.verify_and_update_config",
        staticmethod(lambda vc: None),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.qwen4_exp.config.unified_kv_layout_enabled_for_architecture",
        lambda arch: True,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.qwen4_exp.config.unified_block_major_enabled",
        lambda vc: False,
    )

    tc = _tiny_text_config()
    valid_vc = _mock_vllm_config(
        tc, tp=2, speculative_config=_mock_spec_config(tc, draft_tp=2)
    )
    TPUQwen4ExpConfig.verify_and_update_config(valid_vc)

    # draft_tp=None is also accepted (defaults to target TP in Eagle3Proposer).
    valid_none_tp = _mock_vllm_config(
        tc, tp=2, speculative_config=_mock_spec_config(tc, draft_tp=None)
    )
    TPUQwen4ExpConfig.verify_and_update_config(valid_none_tp)

    # MTP metadata is read from draft_text_config, so target without MTP layers
    # plus a valid 1-layer draft checkpoint succeeds.
    target_no_mtp = _tiny_text_config(mtp_num_hidden_layers=0, mtp=None)
    TPUQwen4ExpConfig.verify_and_update_config(
        _mock_vllm_config(target_no_mtp, speculative_config=_mock_spec_config(tc))
    )

    # Build a draft config with all MTP layer-count fields removed.
    missing_mtp_layers = _tiny_text_config(mtp=None)
    delattr(missing_mtp_layers, "mtp_num_hidden_layers")

    bad_cases = [
        (
            "method='mtp'",
            _mock_vllm_config(
                tc, tp=2, speculative_config=_mock_spec_config(tc, method="eagle3")
            ),
        ),
        (
            "draft_tensor_parallel_size",
            _mock_vllm_config(
                tc, tp=2, speculative_config=_mock_spec_config(tc, draft_tp=1)
            ),
        ),
        (
            "draft quantization",
            _mock_vllm_config(
                tc,
                quantization="fp8",
                speculative_config=_mock_spec_config(tc, quantization=None),
            ),
        ),
        (
            "only 1 MTP hidden layer",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(missing_mtp_layers),
            ),
        ),
        (
            "only 1 MTP hidden layer",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(mtp_num_hidden_layers=0, mtp=None)
                ),
            ),
        ),
        (
            "only 1 MTP hidden layer",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(mtp_num_hidden_layers=2)
                ),
            ),
        ),
        (
            "only 1 MTP hidden layer",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(
                        mtp={
                            "layer_types": ["full_attention"],
                            "num_hidden_layers": 2,
                        }
                    )
                ),
            ),
        ),
        (
            "only 1 MTP layer",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(
                        mtp={"layer_types": ["full_attention", "full_attention"]}
                    )
                ),
            ),
        ),
        (
            "mtp_use_dedicated_embeddings",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(mtp_use_dedicated_embeddings=True)
                ),
            ),
        ),
        (
            "index_share_for_mtp_iteration",
            _mock_vllm_config(
                tc, speculative_config=_mock_spec_config(tc, index_share=True)
            ),
        ),
        (
            "index_share_for_mtp_iteration",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(index_share_for_mtp_iteration=True)
                ),
            ),
        ),
        (
            "index_share_for_mtp_iteration",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(
                        mtp={
                            "layer_types": ["full_attention"],
                            "index_share_for_mtp_iteration": True,
                        }
                    )
                ),
            ),
        ),
        (
            "mamba_cache_mode='all'",
            _mock_vllm_config(tc, mamba_cache_mode="all", enable_prefix_caching=True),
        ),
        ("KV transfer", _mock_vllm_config(tc, kv_transfer_config=object())),
        (
            "rope_theta",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(
                        mtp={"layer_types": ["full_attention"], "rope_theta": 1e6}
                    )
                ),
            ),
        ),
        (
            "rope_theta",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(
                        mtp=SimpleNamespace(
                            layer_types=["full_attention"],
                            mtp_use_hidden_state_from_layer=None,
                            rope_theta=1e6,
                        )
                    )
                ),
            ),
        ),
        (
            "MTP layer types",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(mtp={"layer_types": ["linear_attention"]})
                ),
            ),
        ),
        (
            "mtp_use_hidden_state_from_layer",
            _mock_vllm_config(
                tc,
                speculative_config=_mock_spec_config(
                    _tiny_text_config(
                        mtp={
                            "layer_types": ["full_attention"],
                            "mtp_use_hidden_state_from_layer": 24,
                        }
                    )
                ),
            ),
        ),
    ]
    for pattern, vc in bad_cases:
        with pytest.raises(NotImplementedError, match=pattern):
            TPUQwen4ExpConfig.verify_and_update_config(vc)

    # Block-major rejection uses dense_max_model_len = budget + ratio - 1 (= 19
    # for budget=16, ratio=4), not raw indexer_budget (= 16).
    monkeypatch.setattr(
        "vllm_torchtpu.models.qwen4_exp.config.unified_block_major_enabled",
        lambda vc: True,
    )
    tc_no_ple = _tiny_text_config(
        ple_layer_ids=[], layer_types=["linear_attention"] * 4
    )
    TPUQwen4ExpConfig.verify_and_update_config(
        _mock_vllm_config(
            tc_no_ple, max_model_len=19, speculative_config=_mock_spec_config(tc)
        )
    )
    with pytest.raises(NotImplementedError, match="block-major"):
        TPUQwen4ExpConfig.verify_and_update_config(
            _mock_vllm_config(
                tc_no_ple, max_model_len=20, speculative_config=_mock_spec_config(tc)
            )
        )


def test_eagle3_proposer_feeds_the_draft_all_hc_count_streams() -> None:
    tc = _tiny_text_config(hidden_size=2560, hc_count=4)
    spec = _mock_spec_config(tc, num_spec=3)
    vc = _mock_vllm_config(tc, speculative_config=spec)

    proposer = Eagle3Proposer.__new__(Eagle3Proposer)
    proposer.vllm_config = vc
    proposer.speculative_config = spec
    proposer.draft_model = SimpleNamespace(config=tc)
    proposer._hc_mult = proposer._target_hc_mult()

    assert proposer._hc_mult == tc.hc_count
    assert proposer._draft_hidden_size() == 2560
    assert proposer._draft_input_hidden_size() == 2560 * tc.hc_count
