# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""CPU ML4 definition tests with torch references at the TPU kernel boundaries."""

from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CompilationMode, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUIndexerCache,
    VllmTPUSparseAttnIndexer,
)
from vllm_torchtpu.layers.core.quantization import e8m0_to_fp32, quantize_tensor
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.mistral_large4 import attention as attention_mod
from vllm_torchtpu.models.vllm.mistral_large4 import indexer as indexer_mod
from vllm_torchtpu.models.vllm.mistral_large4.attention import (
    MistralMLAAttention,
    SSMaxScaling,
)
from vllm_torchtpu.models.vllm.mistral_large4.checkpoint_config import get_layer_specs
from vllm_torchtpu.models.vllm.mistral_large4.config import TPUMistralLarge4Config
from vllm_torchtpu.models.vllm.mistral_large4.model import MistralLarge4ForCausalLM
from vllm_torchtpu.models.vllm.mistral_large4.rope import MistralRotaryEmbedding
from vllm_torchtpu.utils import tpu_bind_kv_cache


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    # torch_tpu registers a device module at import. CPU tests must not
    # initialize its runtime through manual_seed or Dynamo's device queries.
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 0)

    from vllm_torchtpu.layers.adapter.vision_attention import TpuMMEncoderAttention

    def cpu_vision_attention(self, q, k, v, cu_seqlens):
        boundaries = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        output = torch.zeros_like(q)
        for start, end in zip(boundaries, boundaries[1:]):
            output[:, start:end] = F.scaled_dot_product_attention(
                q[:, start:end].transpose(1, 2),
                k[:, start:end].transpose(1, 2),
                v[:, start:end].transpose(1, 2),
                scale=self.scale,
            ).transpose(1, 2)
        return output

    monkeypatch.setattr(TpuMMEncoderAttention, "_flash_attention", cpu_vision_attention)


def _text_config(**overrides) -> PretrainedConfig:
    fields = dict(
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        nope_layer_flags=[False, True],
        kv_source_layers=[0, 1],
        index_topk=None,
        n_routed_experts=0,
        ssmax_scaling=True,
        lerope=True,
        fp8_kv_cache=False,
        softmax_tanh_gating=5.0,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pixtral_packed_encoder_matches_upstream_and_isolates_images(
    native_model_context, dtype
):
    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
    from vllm.model_executor.models.pixtral import (
        PatchMerger,
        VisionEncoderArgs,
        VisionLanguageAdapter,
        VisionTransformer,
    )

    from vllm_torchtpu.models.vllm.mistral_large4.vision import (
        MistralLarge4ForConditionalGeneration,
    )
    from vllm_torchtpu.runner.mm_encoder_manager import MMEncoderManager

    args = VisionEncoderArgs(
        hidden_size=16,
        num_channels=3,
        image_size=112,
        patch_size=14,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        rope_theta=10000.0,
        image_token_id=10,
        adapter_bias=False,
        spatial_merge_size=2,
        add_pre_mm_projector_layer_norm=True,
        mm_projector_id="patch_merge",
    )
    model = MistralLarge4ForConditionalGeneration.__new__(
        MistralLarge4ForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
    model.vision_args = args
    model.vision_encoder = VisionTransformer(args)
    model.pre_mm_projector_norm = nn.RMSNorm(args.hidden_size, eps=1e-5)
    model.patch_merger = PatchMerger(args.hidden_size, args.spatial_merge_size)
    model.vision_language_adapter = VisionLanguageAdapter(args, dim=16)
    model.to(dtype=dtype)
    torch.manual_seed(73)
    for parameter in model.parameters():
        nn.init.normal_(parameter, std=0.1)
    _finalize_cpu_linears(model)
    images = [torch.randn(3, 56, 84), torch.randn(3, 28, 56)]
    expected_features = torch.cat(model.vision_encoder(images))
    expected_features = model.pre_mm_projector_norm(expected_features)
    expected_merged = model.patch_merger(expected_features, [(4, 6), (2, 4)])
    expected = model.vision_language_adapter(expected_merged)
    replay = model.prepare_encoder_cudagraph_replay_buffers(
        {"images": images}, 2, 0
    ).values
    manager = MMEncoderManager.__new__(MMEncoderManager)
    manager.budget_templates = {
        "default": {
            16: model.prepare_encoder_cudagraph_capture_inputs(
                16, 2, 0, torch.device("cpu"), dtype
            ).values
        }
    }
    manager.config = model.get_encoder_cudagraph_config()
    actual = model.encoder_cudagraph_forward(manager._pad_to_template(replay, 16))
    assert isinstance(model, SupportsEncoderCudaGraph)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-4, 2e-2)
    torch.testing.assert_close(actual[:8], expected, atol=atol, rtol=rtol)
    cached_outputs = {}
    model.postprocess_encoder_output(
        {"default": actual}, [0, 1], [6, 2], cached_outputs, clone=True
    )
    torch.testing.assert_close(cached_outputs[1], expected[6:], atol=atol, rtol=rtol)
    first = model.encoder_cudagraph_forward(
        model.prepare_encoder_cudagraph_replay_buffers(
            {"images": images[:1]}, 1, 0
        ).values
    )
    torch.testing.assert_close(actual[:6], first, atol=atol, rtol=rtol)
    assert [
        s.output_tokens
        for s in model.get_encoder_cudagraph_item_specs({"images": images})
    ] == [6, 2]


def _check_config(text_config=None):
    text_config = text_config or _text_config()
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            quantization=None,
            dtype=torch.float32,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
        ),
        speculative_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


def _sparse_config(**overrides):
    fields = dict(
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        index_topk=2,
        index_head_dim=128,
        index_n_heads=4,
        indexer_norm_eps=1e-6,
        indexer_ssmax_scaling=True,
        indexer_nope_layer_flags=[False, True],
        index_sharing=["full", "shared"],
    )
    fields.update(overrides)
    return _text_config(**fields)


@pytest.mark.parametrize(
    "overrides,dtype,cache,message",
    [
        ({"index_head_dim": 256}, torch.bfloat16, "fp8", "index_head_dim"),
        ({"kv_lora_rank": 256}, torch.bfloat16, "fp8", "kv_lora_rank"),
        ({"qk_rope_head_dim": 32}, torch.bfloat16, "fp8", "qk_rope_head_dim"),
        ({"index_topk": -1}, torch.bfloat16, "fp8", "positive"),
        ({"index_sharing": ["shared", "full"]}, torch.bfloat16, "fp8", "first"),
        ({}, torch.float32, "fp8", "dtype bfloat16"),
        ({}, torch.bfloat16, "auto", "fp8_e4m3"),
        ({}, torch.bfloat16, "fp8_ds_mla", "fp8_e4m3"),
    ],
)
def test_sparse_unsupported_geometry_or_cache_fails_early(
    overrides, dtype, cache, message
):
    config = _check_config(_sparse_config(**overrides))
    config.model_config.dtype = dtype
    config.cache_config.cache_dtype = cache
    with pytest.raises((NotImplementedError, ValueError), match=message):
        TPUMistralLarge4Config.verify_and_update_config(config)


def test_sparse_configuration_accepts_shared_indices_with_independent_kv():
    config = _check_config(_sparse_config())
    config.model_config.dtype = torch.bfloat16
    config.cache_config.cache_dtype = "fp8_e4m3"
    TPUMistralLarge4Config.verify_and_update_config(config)


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"q_lora_rank": 8}, "q_lora_rank"),
    ],
)
def test_unsupported_architecture_is_rejected_before_allocation(fields, message):
    config = _check_config(_text_config(**fields))
    with pytest.raises(NotImplementedError, match=message):
        MistralLarge4ForCausalLM(vllm_config=config)


@pytest.mark.parametrize(
    "target,field,value,message",
    [
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline_parallel"),
        ("parallel_config", "prefill_context_parallel_size", 2, "context parallel"),
        ("parallel_config", "decode_context_parallel_size", 2, "context parallel"),
        ("parallel_config", "enable_eplb", True, "load balancing"),
        ("model_config", "quantization", "gptq", "weight quantization"),
        ("model_config", "quantization", "modelopt_mixed", "weight quantization"),
        ("model_config", "quantization", "modelopt_fp4", "weight quantization"),
        ("cache_config", "cache_dtype", "fp8", "FP8 KV"),
    ],
)
def test_unsupported_execution_options_fail_early(target, field, value, message):
    config = _check_config()
    setattr(getattr(config, target), field, value)
    with pytest.raises(NotImplementedError, match=message):
        TPUMistralLarge4Config.verify_and_update_config(config)


def test_explicit_bf16_cache_overrides_checkpoint_fp8_preference():
    config = _check_config(_text_config(fp8_kv_cache=True))
    config.cache_config.cache_dtype = "bfloat16"
    TPUMistralLarge4Config.verify_and_update_config(config)
    assert config.cache_config.cache_dtype == "auto"
    # Config verification also runs in the model constructor.
    TPUMistralLarge4Config.verify_and_update_config(config)


def test_pixtral_text_child_accepts_copied_text_config():
    config = _check_config(_sparse_config())
    config.model_config.hf_text_config = deepcopy(config.model_config.hf_config)
    config.model_config.dtype = torch.bfloat16
    config.cache_config.cache_dtype = "fp8_e4m3"
    TPUMistralLarge4Config.verify_and_update_config(config)


@pytest.mark.parametrize("option", ["speculation", "multimodal"])
def test_unintegrated_wrappers_are_rejected(option):
    config = _check_config()
    if option == "speculation":
        config.speculative_config = object()
    else:
        config.model_config.hf_config = PretrainedConfig(
            text_config=config.model_config.hf_text_config
        )
    with pytest.raises(NotImplementedError, match="speculative|text-only"):
        TPUMistralLarge4Config.verify_and_update_config(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ssmax_scales_each_head_in_activation_dtype(dtype):
    scaling = SSMaxScaling(2).to(dtype)
    with torch.no_grad():
        scaling.s.copy_(torch.tensor([0.25, -0.125]))
        scaling.b.copy_(torch.tensor([1.0, 0.5]))
    q = torch.arange(24, dtype=dtype).reshape(3, 2, 4)
    log_positions = torch.log(torch.tensor([1.0, 4.0, 128.0]))
    expected = q.clone()
    for token in range(3):
        for head in range(2):
            scale = log_positions[token].to(dtype) * scaling.s[head]
            expected[token, head] *= scale + scaling.b[head]
    torch.testing.assert_close(scaling(q, log_positions), expected, atol=0, rtol=0)


class _ReferenceMLA(nn.Module):
    layers = {}

    def __init__(self, **kwargs):
        super().__init__()
        self.prefix = kwargs["prefix"]
        self.kv_sharing_target_layer_name = kwargs["kv_sharing_target_layer_name"]
        self.layers[self.prefix] = self
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.num_heads = kwargs["num_heads"]
        self.nope_dim = kwargs["qk_nope_head_dim"]
        self.v_dim = kwargs["v_head_dim"]
        self.scale = kwargs["scale"]
        self.use_sparse = kwargs["use_sparse"]
        self.topk_indices_buffer = kwargs["topk_indices_buffer"]

    def forward(self, q, kv_c_normed, k_pe, topk_indices=None):
        if self.kv_sharing_target_layer_name is not None:
            assert torch.count_nonzero(kv_c_normed) == 0
            assert torch.count_nonzero(k_pe) == 0
            owner = self.layers[self.kv_sharing_target_layer_name]
            kv_c_normed, k_pe = owner.cached_kv
        else:
            self.cached_kv = (kv_c_normed.clone(), k_pe.clone())
        q_nope, q_pe = q
        assert q_nope.shape[-1] == self.nope_dim
        kv = F.linear(kv_c_normed, self.kv_b_proj.weight).view(
            -1, self.num_heads, self.nope_dim + self.v_dim
        )
        k_nope, value = kv.split([self.nope_dim, self.v_dim], dim=-1)
        key = torch.cat((k_nope, k_pe.expand(-1, self.num_heads, -1)), dim=-1)
        query = torch.cat((q_nope, q_pe), dim=-1)
        logits = torch.einsum("thd,shd->hts", query, key) * self.scale
        mask = torch.ones(logits.shape[-2:], dtype=torch.bool).tril()
        self.last_topk = None if topk_indices is None else topk_indices.clone()
        if topk_indices is not None:
            selected = (topk_indices[:, :, None] == torch.arange(logits.shape[-1])).any(
                1
            )
            mask &= selected
        probs = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        return torch.einsum("hts,shd->thd", probs, value).flatten(1)


class _TorchIndexerOp(nn.Module):
    def __init__(
        self,
        cache,
        block_size,
        scale_fmt,
        topk,
        head_dim,
        max_model_len,
        max_total_seq_len,
        buffer,
    ):
        super().__init__()
        self.buffer = buffer
        self.topk = topk
        self.calls = []

    def forward(self, hidden_states, q_fp8, k, weights):
        self.calls.append(
            (q_fp8.detach().clone(), k.detach().clone(), weights.detach().clone())
        )
        k_fp8, k_scale = quantize_tensor(k, torch.float8_e4m3fn, use_ue8m0=True)
        logits = torch.einsum("thd,sd->ths", q_fp8.float(), k_fp8.float())
        logits = logits * e8m0_to_fp32(k_scale).T
        scores = (logits.relu() * weights[:, :, None]).sum(1)
        causal = torch.ones(scores.shape, dtype=torch.bool).tril()
        values, indices = scores.masked_fill(~causal, -torch.inf).topk(self.topk)
        indices = indices.masked_fill(~values.isfinite(), -1).to(torch.int32)
        self.buffer[: indices.shape[0]].copy_(indices)
        return indices


def _finalize_cpu_linears(module):
    for layer in module.modules():
        if isinstance(layer, LinearBase):
            # Keep the canonical checkpoint weight layout for the oracle;
            # packing for optimized CPU GEMMs would discard these weights.
            layer._cpu_skip_gemm_dispatch = True
            layer.quant_method.process_weights_after_loading(layer)
        elif isinstance(layer, VocabParallelEmbedding):
            layer.quant_method.process_weights_after_loading(layer)


@pytest.fixture
def native_model_context(monkeypatch):
    import torch.distributed as dist
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(_ReferenceMLA, "layers", {})
    monkeypatch.setattr(attention_mod, "VllmTPUMLAAttention", _ReferenceMLA)
    monkeypatch.setattr(indexer_mod, "VllmTPUSparseAttnIndexer", _TorchIndexerOp)
    config = VllmConfig()
    config.compilation_config.mode = CompilationMode.NONE
    config.attention_config.sparse_mla_prefill_workspace_factor = 1
    with set_current_vllm_config(config):
        if not dist.is_initialized():
            import portpicker

            port = portpicker.pick_unused_port()
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend="gloo",
            )
        ensure_model_parallel_initialized(1, 1)
        text_config = _text_config()
        config.model_config = SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            quantization=None,
            dtype=torch.float32,
            head_dtype=None,
            max_model_len=32,
        )
        yield config


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("ssmax", [False, True])
def test_attention_matches_explicit_mha(native_model_context, use_rope, ssmax):
    config = _text_config(nope_layer_flags=[not use_rope] * 2, ssmax_scaling=ssmax)
    spec = get_layer_specs(config)[0]
    layer = MistralMLAAttention(native_model_context, config, spec, "attn")
    rope = MistralRotaryEmbedding(config, config.qk_rope_head_dim)
    torch.manual_seed(7)
    for parameter in layer.parameters():
        nn.init.normal_(parameter, std=0.1)
    _finalize_cpu_linears(layer)
    hidden = torch.randn(3, config.hidden_size)
    positions = torch.arange(3)
    log_positions = torch.log(positions.float() + 1)
    actual = layer(positions, hidden, log_positions, rope)

    q = F.linear(hidden, layer.q_proj.weight).view(3, 2, 8)
    kv = F.linear(hidden, layer.kv_a_proj_with_mqa.weight)
    latent = kv[:, :8]
    latent = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + 1e-6)
    latent = latent * layer.kv_a_layernorm.weight
    keys_values = F.linear(latent, layer.kv_b_proj.weight).view(3, 2, -1)
    key, value = keys_values.split([layer.qk_nope_head_dim, 4], dim=-1)
    if use_rope:
        inv_freq = 10000.0 ** (-torch.arange(0, 4, 2).float() / 4)
        phase = positions[:, None] * inv_freq
        cos = phase.cos().repeat_interleave(2, -1)[:, None, :]
        sin = phase.sin().repeat_interleave(2, -1)[:, None, :]

        def rotate(x):
            pairs = x.reshape(*x.shape[:-1], -1, 2)
            return torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)

        q_rope = q[..., 4:]
        q = torch.cat((q[..., :4], q_rope * cos + rotate(q_rope) * sin), -1)
        k_rope = kv[:, None, 8:]
        k_rope = k_rope * cos + rotate(k_rope) * sin
        key = torch.cat((key, k_rope.expand(-1, 2, -1)), -1)
    if ssmax:
        q = q * (
            log_positions[:, None, None] * layer.ssmax.s[None, :, None]
            + layer.ssmax.b[None, :, None]
        )
    logits = torch.einsum("thd,shd->hts", q, key) * (8**-0.5)
    mask = torch.ones(3, 3, dtype=torch.bool).tril()
    probs = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    expected = F.linear(
        torch.einsum("hts,shd->thd", probs, value).flatten(1), layer.o_proj.weight
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("ssmax", [False, True])
@pytest.mark.parametrize("zero_query", [False, True])
def test_indexer_preprocessing_and_topk_match_reference(
    native_model_context, use_rope, ssmax, zero_query
):
    config = _sparse_config(indexer_ssmax_scaling=ssmax)
    buffer = torch.full((16, config.index_topk), -1, dtype=torch.int32)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        indexer = indexer_mod.MistralIndexer(
            native_model_context, config, buffer, "model.layers.0.self_attn.indexer"
        )
        rope = MistralRotaryEmbedding(config, config.qk_rope_head_dim)
    finally:
        torch.set_default_dtype(old_dtype)
    torch.manual_seed(23)
    for parameter in indexer.parameters():
        nn.init.normal_(parameter, std=0.1)
    if zero_query:
        nn.init.zeros_(indexer.wq_b.weight)
    _finalize_cpu_linears(indexer)
    hidden = torch.randn(6, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(hidden.shape[0])
    log_positions = (positions.float() + 1).log()
    actual = indexer(hidden, positions, log_positions, rope if use_rope else None)
    actual_q, actual_k, actual_weights = indexer.indexer_op.calls[-1]

    normalized = F.layer_norm(
        hidden,
        (config.hidden_size,),
        indexer.q_input_norm.weight,
        indexer.q_input_norm.bias,
        config.indexer_norm_eps,
    )
    q = F.linear(normalized, indexer.wq_b.weight).view(6, config.index_n_heads, 128)
    wk, weight_proj = indexer.wk_weights_proj.weight.split([128, config.index_n_heads])
    k = F.layer_norm(
        F.linear(hidden, wk).float(),
        (128,),
        indexer.k_norm.weight,
        indexer.k_norm.bias,
        config.indexer_norm_eps,
    ).to(hidden.dtype)
    weights = F.linear(hidden, weight_proj).float()
    if ssmax:
        scale = log_positions.to(q.dtype)[:, None, None] * indexer.ssmax.s[:, None]
        q = q * (scale + indexer.ssmax.b[:, None])
        assert not hasattr(indexer.ssmax.s, "weight_loader")
    if use_rope:
        freq = 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
        phase = positions[:, None] * freq
        cos = phase.cos().to(hidden.dtype)[:, None, :]
        sin = phase.sin().to(hidden.dtype)[:, None, :]

        def rotate(x):
            pairs = x[..., :64].float().reshape(*x.shape[:-1], 32, 2)
            even, odd = pairs[..., 0], pairs[..., 1]
            rotated = (
                torch.stack(
                    (
                        even * cos.float() - odd * sin.float(),
                        odd * cos.float() + even * sin.float(),
                    ),
                    dim=-1,
                )
                .flatten(-2)
                .to(x.dtype)
            )
            return torch.cat((rotated, x[..., 64:]), dim=-1)

        q = rotate(q)
        k = rotate(k[:, None, :]).squeeze(1)
    scales = torch.empty(6, config.index_n_heads, 1)
    for token in range(6):
        for head in range(config.index_n_heads):
            amax = float(q[token, head].detach().float().abs().max())
            raw = max(amax, 1e-10) / 448.0 if use_rope else max(amax / 448.0, 1e-10)
            scales[token, head] = 2.0 ** math.ceil(math.log2(raw))
    expected_q = (q.float() / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    expected_weights = weights * scales.squeeze(-1)
    if use_rope:
        expected_weights = expected_weights * 128**-0.5 * config.index_n_heads**-0.5
    else:
        expected_weights = expected_weights * (128**-0.5 * config.index_n_heads**-0.5)
    torch.testing.assert_close(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_k, k, atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)
    assert actual_weights.dtype == torch.float32

    k_fp8, scale_bytes = quantize_tensor(k, torch.float8_e4m3fn, use_ue8m0=True)
    key = k_fp8.float() * e8m0_to_fp32(scale_bytes)
    dot = torch.einsum("thd,sd->ths", expected_q.float(), key)
    scores = (dot.relu() * expected_weights[:, :, None]).sum(1)
    scores = scores.masked_fill(~torch.ones(6, 6, dtype=torch.bool).tril(), -torch.inf)
    values, topk = scores.topk(config.index_topk)
    topk = topk.masked_fill(~values.isfinite(), -1).to(torch.int32)
    torch.testing.assert_close(actual, topk)
    torch.testing.assert_close(buffer[:6], topk)
    assert (buffer[6:] == -1).all()


def test_indexer_uses_existing_tpu_operator_and_cache_contract(
    native_model_context, monkeypatch
):
    monkeypatch.setattr(
        indexer_mod, "VllmTPUSparseAttnIndexer", VllmTPUSparseAttnIndexer
    )
    # Construction of the JAX bridge needs a TPU mesh; the operator itself,
    # cache registration, backend selection and binding remain real here.
    monkeypatch.setattr(
        VllmTPUSparseAttnIndexer, "_build_streamidx_op", lambda self: None
    )
    config = _sparse_config()
    indexer = indexer_mod.MistralIndexer(
        native_model_context,
        config,
        torch.empty(16, 2, dtype=torch.int32),
        "model.layers.0.self_attn.indexer",
    )
    assert isinstance(indexer.indexer_op, VllmTPUSparseAttnIndexer)
    cache = indexer.k_cache
    assert isinstance(cache, (VllmTPUIndexerCache, DeepseekV32IndexerCache))
    assert cache.head_dim == 129
    spec = cache.get_kv_cache_spec(native_model_context)
    assert spec.head_size == 129 and spec.dtype == torch.uint8
    backend = cache.get_attn_backend()
    shape = backend.get_kv_cache_shape(1, 128, 1, spec.head_size, spec.dtype)
    allocated = torch.zeros(shape, dtype=spec.dtype)
    context = native_model_context.compilation_config.static_forward_context
    assert context[cache.prefix] is cache
    runner_caches = []
    tpu_bind_kv_cache({cache.prefix: allocated}, context, runner_caches)
    assert cache.kv_cache is allocated
    assert runner_caches[0] is allocated


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("share_kv", [False, True])
def test_model_forward_and_consolidated_checkpoint_roundtrip(
    native_model_context, sparse, share_kv
):
    text_config = (_sparse_config if sparse else _text_config)(
        kv_source_layers=[0, 0] if share_kv else [0, 1]
    )
    native_model_context.model_config.hf_config = text_config
    native_model_context.model_config.hf_text_config = text_config
    if sparse:
        native_model_context.model_config.hf_config = text_config
        native_model_context.model_config.hf_text_config = text_config
        native_model_context.model_config.dtype = torch.bfloat16
        native_model_context.cache_config.cache_dtype = "fp8_e4m3"
        native_model_context.scheduler_config = SimpleNamespace(
            max_num_batched_tokens=16
        )
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(native_model_context.model_config.dtype)
    try:
        source = MistralLarge4ForCausalLM(
            vllm_config=native_model_context, prefix="source"
        )
        target = MistralLarge4ForCausalLM(
            vllm_config=native_model_context, prefix="target"
        )
    finally:
        torch.set_default_dtype(old_dtype)
    torch.manual_seed(11)
    checkpoint = []
    attention_names = {
        "q_proj": "wq",
        "kv_a_proj_with_mqa": "wkv_a_with_mqa",
        "kv_a_layernorm": "kv_a_norm",
        "kv_b_proj": "wkv_b",
        "o_proj": "wo",
        "ssmax": "ssmax_scaling",
    }
    for name, parameter in source.named_parameters():
        nn.init.normal_(parameter, std=0.1)
        weight = parameter.detach().clone()
        name = name.removeprefix("model.")
        name = name.replace("embed_tokens.", "tok_embeddings.")
        name = name.replace("lm_head.", "output.")
        if name in ("tok_embeddings.weight", "output.weight"):
            weight = weight[: source.config.vocab_size]
        name = name.replace("input_layernorm.", "attention_norm.")
        name = name.replace("post_attention_layernorm.", "ffn_norm.")
        name = name.replace("self_attn.", "attention.")
        for param_name, ckpt_name in attention_names.items():
            name = name.replace(f".{param_name}.", f".{ckpt_name}.")
        name = name.replace("mlp.", "feed_forward.")
        if ".gate_up_proj." in name:
            gate, up = weight.chunk(2, dim=0)
            checkpoint.extend(
                [
                    (name.replace("gate_up_proj", "w1"), gate),
                    (name.replace("gate_up_proj", "w3"), up),
                ]
            )
        elif ".wk_weights_proj." in name:
            wk, weights_proj = weight.split(
                [source.config.index_head_dim, source.config.index_n_heads], dim=0
            )
            checkpoint.extend(
                [
                    (name.replace("wk_weights_proj", "wk"), wk),
                    (name.replace("wk_weights_proj", "weights_proj"), weights_proj),
                ]
            )
        else:
            checkpoint.append((name.replace("down_proj", "w2"), weight))
    alpha = torch.linspace(0.1, -0.2, source.config.qk_rope_head_dim // 2)
    source.model.rotary_emb.set_lerope_alpha(alpha)
    checkpoint.extend(
        [
            ("lerope.alpha", alpha),
            ("mtp_dspark.unused.weight", torch.empty(1)),
            ("layers.99.attention.wq.weight", torch.empty(1)),
        ]
    )
    if share_kv:
        assert not any("layers.1.attention.wkv_a" in name for name, _ in checkpoint)
    loaded = target.load_weights(iter(checkpoint))
    assert loaded == set(dict(target.named_parameters()))
    _finalize_cpu_linears(source)
    _finalize_cpu_linears(target)
    ids = torch.tensor([1, 3, 5, 2, 4, 6])
    positions = torch.arange(ids.shape[0])
    expected = source(ids, positions)
    torch.testing.assert_close(target(ids, positions), expected)
    torch.testing.assert_close(
        target(None, positions, inputs_embeds=target.embed_input_ids(ids)), expected
    )
    if sparse:
        first, shared = (layer.self_attn for layer in target.model.layers)
        assert shared.indexer is None
        assert first.attn.use_sparse and shared.attn.use_sparse
        assert first.attn.topk_indices_buffer is shared.attn.topk_indices_buffer
        torch.testing.assert_close(first.attn.last_topk, shared.attn.last_topk)
        assert first.attn.last_topk.shape == (ids.shape[0], target.config.index_topk)
        assert not torch.isnan(expected).any()
        assert not hasattr(first.indexer.ssmax.s, "weight_loader")
        target(ids[:3], positions[:3])
        assert shared.attn.last_topk.shape == (3, target.config.index_topk)
        torch.testing.assert_close(first.attn.last_topk, shared.attn.last_topk)
    logits = target.compute_logits(expected)
    raw_logits = F.linear(expected, target.lm_head.weight)[
        :, : target.config.vocab_size
    ]
    torch.testing.assert_close(logits, 5.0 * torch.tanh(raw_logits / 5.0))
    incomplete = [(name, value) for name, value in checkpoint if name != "norm.weight"]
    with pytest.raises(ValueError, match="Parameters not loaded.*model.norm.weight"):
        target.load_weights(iter(incomplete))


def test_registry_selects_tpu_model_and_config():
    arch = "MistralLarge4ForCausalLM"
    previous_model = ModelRegistry.models.get(arch)
    previous_config = MODELS_CONFIG_MAP.get(arch)
    try:
        register_models()
        assert ModelRegistry.models[arch].load_model_cls() is MistralLarge4ForCausalLM
        assert MODELS_CONFIG_MAP[arch] is TPUMistralLarge4Config
    finally:
        if previous_model is None:
            ModelRegistry.models.pop(arch, None)
        else:
            ModelRegistry.models[arch] = previous_model
        if previous_config is None:
            MODELS_CONFIG_MAP.pop(arch, None)
        else:
            MODELS_CONFIG_MAP[arch] = previous_config


@pytest.mark.parametrize("sources", [[1, 1], [-1, 1], [0, 2], [0, 0, 1]])
def test_kv_sharing_requires_an_earlier_owner(sources):
    config = _check_config(
        _text_config(
            num_hidden_layers=len(sources),
            kv_source_layers=sources,
            nope_layer_flags=[False] * len(sources),
        )
    )
    with pytest.raises(ValueError, match="earlier layer that owns"):
        TPUMistralLarge4Config.verify_and_update_config(config)


@pytest.mark.parametrize("source_rope", [False, True])
@pytest.mark.parametrize("reader_rope", [False, True])
def test_shared_attention_uses_source_latents_and_own_projection(
    native_model_context, source_rope, reader_rope
):
    config = _text_config(
        nope_layer_flags=[not source_rope, not reader_rope],
        kv_source_layers=[0, 0],
    )
    specs = get_layer_specs(config)
    owner = MistralMLAAttention(
        native_model_context, config, specs[0], "model.layers.0.self_attn"
    )
    reader = MistralMLAAttention(
        native_model_context,
        config,
        specs[1],
        "model.layers.1.self_attn",
        zero_kv_inputs=(torch.zeros(16, 8), torch.zeros(16, 1, 4)),
    )
    assert not hasattr(reader, "kv_a_proj_with_mqa")
    assert not hasattr(reader, "kv_a_layernorm")
    assert reader.attn.kv_sharing_target_layer_name == "model.layers.0.self_attn.attn"
    torch.manual_seed(37)
    for module in (owner, reader):
        for parameter in module.parameters():
            nn.init.normal_(parameter, std=0.2)
        _finalize_cpu_linears(module)
    rope = MistralRotaryEmbedding(config, config.qk_rope_head_dim)
    positions = torch.arange(3)
    log_positions = (positions.float() + 1).log()
    owner(positions, torch.randn(3, 16), log_positions, rope)
    latent, k_pe = (x.clone() for x in owner.attn.cached_kv)
    hidden = torch.randn(3, 16)
    actual = reader(positions, hidden, log_positions, rope)

    q = F.linear(hidden, reader.q_proj.weight).view(3, 2, 8)
    if reader_rope:
        q_pe, _ = rope(positions, q[..., 4:], torch.zeros(3, 1, 4))
        q = torch.cat((q[..., :4], q_pe), -1)
    else:
        q = F.pad(q, (0, 4))
    q *= (
        log_positions[:, None, None] * reader.ssmax.s[None, :, None]
        + reader.ssmax.b[None, :, None]
    )
    kv = F.linear(latent, reader.kv_b_proj.weight).view(3, 2, -1)
    key, value = kv.split([reader.qk_nope_head_dim, 4], -1)
    key = torch.cat((key, k_pe.expand(-1, 2, -1)), -1)
    logits = torch.einsum("thd,shd->hts", q, key) * (8**-0.5)
    causal = torch.ones(3, 3, dtype=torch.bool).tril()
    probs = logits.masked_fill(~causal, -torch.inf).softmax(-1)
    expected = F.linear(
        torch.einsum("hts,shd->thd", probs, value).flatten(1), reader.o_proj.weight
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(owner.attn.cached_kv[0], latent, atol=0, rtol=0)
    torch.testing.assert_close(owner.attn.cached_kv[1], k_pe, atol=0, rtol=0)


def _quantized_model(native_model_context, sparse=False):
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config

    text = (_sparse_config if sparse else _text_config)(
        hidden_size=128,
        intermediate_size=256,
        kv_lora_rank=512 if sparse else 128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        kv_source_layers=[0, 0],
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_intermediate_size=128,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        scoring_func="softmax",
        topk_method=None,
        n_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    native_model_context.model_config.hf_config = text
    native_model_context.model_config.hf_text_config = text
    native_model_context.model_config.dtype = torch.bfloat16
    native_model_context.model_config.quantization = "fp8"
    native_model_context.scheduler_config = SimpleNamespace(max_num_batched_tokens=16)
    if sparse:
        native_model_context.cache_config.cache_dtype = "fp8_e4m3"
    quant = VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
        ignored_layers=[
            f"layers.{i}.attention.{name}"
            for i in range(2)
            for name in (
                "wkv_a_with_mqa",
                "wkv_b",
                "indexer.wk",
                "indexer.weights_proj",
            )
        ],
    )
    quant.set_configs(native_model_context)
    native_model_context.quant_config = quant
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return MistralLarge4ForCausalLM(vllm_config=native_model_context, prefix="fp8")
    finally:
        torch.set_default_dtype(old_dtype)


def _mistral_checkpoint_name(name):
    name = name.removeprefix("model.")
    name = name.replace("embed_tokens.", "tok_embeddings.").replace(
        "lm_head.", "output."
    )
    name = name.replace("input_layernorm.", "attention_norm.").replace(
        "post_attention_layernorm.", "ffn_norm."
    )
    name = name.replace("self_attn.", "attention.")
    for native, checkpoint in {
        "q_proj": "wq",
        "kv_a_proj_with_mqa": "wkv_a_with_mqa",
        "kv_a_layernorm": "kv_a_norm",
        "kv_b_proj": "wkv_b",
        "o_proj": "wo",
        "ssmax": "ssmax_scaling",
    }.items():
        name = name.replace(f".{native}.", f".{checkpoint}.")
    name = name.replace(".mlp.shared_experts.", ".shared_experts.")
    name = name.replace(".mlp.experts.routed_experts.", ".experts.")
    name = name.replace(".mlp.gate.", ".gate.").replace(".mlp.", ".feed_forward.")
    return name


def _fp8_checkpoint(model, scale_spelling):
    checkpoint = []
    expected = {}
    torch.manual_seed(71)
    parameters = dict(model.named_parameters())
    for name, param in parameters.items():
        if name.endswith("weight_scale") and f"{name}_inv" in parameters:
            expected[name] = expected[f"{name}_inv"]
            continue
        if "scale" in name:
            value = (torch.rand(param.shape) * 0.015 + 0.01).to(param.dtype)
        else:
            value = (torch.randn(param.shape) * 0.05).to(param.dtype)
        expected[name] = value.clone()
        ckpt_name = _mistral_checkpoint_name(name)
        if ckpt_name in ("tok_embeddings.weight", "output.weight"):
            value = value[: model.config.vocab_size]
        if ".experts.w13_" in ckpt_name or ".experts.w2_" in ckpt_name:
            prefix, suffix = ckpt_name.split(".experts.")
            is_gate_up = suffix.startswith("w13_")
            suffix = suffix.removeprefix("w13_" if is_gate_up else "w2_")
            for expert in range(value.shape[0]):
                if is_gate_up:
                    shards = value[expert].chunk(2, dim=0)
                    for proj, shard in zip(("w1", "w3"), shards):
                        checkpoint.append(
                            (
                                f"{prefix}.experts.{expert}.{proj}.{suffix}",
                                shard.clone(),
                            )
                        )
                else:
                    checkpoint.append(
                        (
                            f"{prefix}.experts.{expert}.w2.{suffix}",
                            value[expert].clone(),
                        )
                    )
        elif ".wk_weights_proj." in ckpt_name:
            for proj, shard in zip(
                ("wk", "weights_proj"),
                value.split(
                    [model.config.index_head_dim, model.config.index_n_heads], dim=0
                ),
            ):
                checkpoint.append(
                    (ckpt_name.replace("wk_weights_proj", proj), shard.clone())
                )
        elif ".gate_up_proj." in ckpt_name:
            for proj, shard in zip(("w1", "w3"), value.chunk(2, dim=0)):
                checkpoint.append(
                    (ckpt_name.replace("gate_up_proj", proj), shard.clone())
                )
        else:
            checkpoint.append((ckpt_name.replace("down_proj", "w2"), value.clone()))
    checkpoint = [
        (name.replace("weight_scale_inv", scale_spelling), value)
        for name, value in checkpoint
    ]
    checkpoint.append(("lerope.alpha", torch.zeros(model.config.qk_rope_head_dim // 2)))
    return checkpoint, expected


def test_checkpoint_skips_recognized_nonlocal_experts(
    native_model_context, monkeypatch
):
    model = _quantized_model(native_model_context)
    checkpoint, _ = _fp8_checkpoint(model, "qscale_weight")
    experts = model.model.layers[1].mlp.experts.routed_experts
    for param in experts.parameters():
        original = param.weight_loader

        def load_local(*args, expert_id, _original=original, **kwargs):
            if expert_id == 1:
                return False
            return _original(*args, expert_id=expert_id, **kwargs)

        monkeypatch.setattr(param, "weight_loader", load_local)
    model.load_weights(iter(checkpoint))


@pytest.mark.parametrize("scale_spelling", ["weight_scale", "qscale_weight"])
@pytest.mark.parametrize("sparse", [False, True])
def test_fp8_checkpoint_loads_quantized_parameters(
    native_model_context, scale_spelling, sparse
):
    from vllm_torchtpu.layers.adapter.quantization.fp8 import (
        VllmFp8LinearMethodTPU,
        VllmFp8MoEMethodTPU,
    )

    model = _quantized_model(native_model_context, sparse)
    assert isinstance(
        model.model.layers[0].self_attn.q_proj.quant_method, VllmFp8LinearMethodTPU
    )
    assert (
        model.model.layers[0].self_attn.q_proj.weight_scale.data_ptr()
        == model.model.layers[0].self_attn.q_proj.weight_scale_inv.data_ptr()
    )
    experts = model.model.layers[1].mlp.experts.routed_experts
    assert isinstance(experts.quant_method, VllmFp8MoEMethodTPU)
    assert (
        model.model.layers[0].self_attn.kv_a_proj_with_mqa.weight.dtype
        == torch.bfloat16
    )
    assert model.model.layers[0].self_attn.kv_b_proj.weight.dtype == torch.bfloat16
    if sparse:
        indexer = model.model.layers[0].self_attn.indexer
        assert isinstance(indexer.wq_b.quant_method, VllmFp8LinearMethodTPU)
        assert indexer.wq_b.weight.dtype == torch.float8_e4m3fn
        assert indexer.wk_weights_proj.weight.dtype == torch.bfloat16
    checkpoint, expected = _fp8_checkpoint(model, scale_spelling)
    for param in model.parameters():
        param.data.zero_()
    loaded = model.load_weights(iter(checkpoint))
    assert loaded == set(dict(model.named_parameters()))
    for name, param in model.named_parameters():
        value = expected[name]
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            param = param[: model.config.vocab_size]
            value = value[: model.config.vocab_size]
        torch.testing.assert_close(
            param.float(), value.float(), atol=0, rtol=0, msg=name
        )
    incomplete = [
        (name, value)
        for name, value in checkpoint
        if ".experts." not in name or not name.endswith(scale_spelling)
    ]
    with pytest.raises(ValueError, match="Parameters not loaded.*weight_scale"):
        model.load_weights(iter(incomplete))


def test_fp8_weights_use_existing_tpu_preparation_and_numeric_paths(
    native_model_context, monkeypatch
):
    from vllm_torchtpu.layers.adapter.quantization import fp8

    model = _quantized_model(native_model_context)
    checkpoint, expected = _fp8_checkpoint(model, "weight_scale")
    model.load_weights(iter(checkpoint))
    linear = model.model.layers[0].self_attn.q_proj
    w = expected["model.layers.0.self_attn.q_proj.weight"].float()
    s = expected["model.layers.0.self_attn.q_proj.weight_scale_inv"]
    reference_weight = w * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    linear.quant_method.process_weights_after_loading(linear)
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.weight_scale.dtype == torch.float32

    def cpu_fp8_matmul(x, weight, scales, w8a16_max_rows=0):
        if 0 < x.shape[0] <= w8a16_max_rows:
            return ((x.float() @ weight.float()) * scales).to(x.dtype)
        quantized, input_scale = quantize_tensor(
            x.float(), torch.float8_e4m3fn, axis=-1
        )
        return ((quantized.float() @ weight.float()) * input_scale * scales).to(x.dtype)

    monkeypatch.setattr(fp8, "quantized_matmul", cpu_fp8_matmul)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    actual = linear(x)[0]
    reference = F.linear(x.float(), reference_weight)
    assert (actual.float() - reference).norm() / reference.norm() < 0.06
