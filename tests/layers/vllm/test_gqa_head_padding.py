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
"""Unit tests for GQA query-head padding and TPUTPPaddedQwen2Attention registration."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.gqa_head_padding import (
    GqaPaddingSpec,
    compute_padded_gqa_heads,
    pad_o_proj_weight,
    pad_qkv_weight,
    update_gqa_head_padding_config,
)


def test_compute_padded_gqa_heads_fin_r1_7b():
    # Fin-R1-7B / Qwen2.5-7B has 28 Q heads and 4 KV heads (7 Q heads per KV head).
    # TP=1, 2, 4 divide 28 evenly -> no padding required.
    assert compute_padded_gqa_heads(28, 4, 1) == (28, 0, 0)
    assert compute_padded_gqa_heads(28, 4, 2) == (28, 0, 0)
    assert compute_padded_gqa_heads(28, 4, 4) == (28, 0, 0)

    # TP=8 does not divide 28, but divides 4 KV heads with 2 ranks per KV head ->
    # pads 7 Q heads/KV group up to 8 Q heads/KV group (32 total Q heads).
    assert compute_padded_gqa_heads(28, 4, 8) == (32, 7, 8)


def test_update_gqa_head_padding_config_scope():
    tp8_no_dcp = SimpleNamespace(
        tensor_parallel_size=8,
        decode_context_parallel_size=1,
    )
    tp8_with_dcp = SimpleNamespace(
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
    )

    cfg = SimpleNamespace(
        model_arch_config=SimpleNamespace(
            total_num_attention_heads=28,
            head_size=128,
        ),
        get_total_num_kv_heads=lambda: 4,
    )
    # DCP > 1 is left untouched
    update_gqa_head_padding_config(cfg, tp8_with_dcp)
    assert cfg.model_arch_config.total_num_attention_heads == 28

    # DCP == 1 updates total_num_attention_heads to 32 idempotently
    update_gqa_head_padding_config(cfg, tp8_no_dcp)
    assert cfg.orig_total_num_attention_heads == 28
    assert cfg.model_arch_config.total_num_attention_heads == 32
    assert cfg.model_arch_config.head_size == 128

    update_gqa_head_padding_config(cfg, tp8_no_dcp)
    assert cfg.orig_total_num_attention_heads == 28
    assert cfg.model_arch_config.total_num_attention_heads == 32


def test_pad_helpers_reject_unsupported_shapes():
    head_dim = 16
    hidden_size = 64
    spec = GqaPaddingSpec(
        num_kv_heads=4, orig_q_per_kv=7, padded_q_per_kv=8, head_dim=head_dim
    )
    q_param = SimpleNamespace(output_dim=0)
    o_param = SimpleNamespace(input_dim=1)

    # o_proj params without input_dim are passed through unchanged.
    o_w = torch.randn(hidden_size, 28 * head_dim)
    assert pad_o_proj_weight(SimpleNamespace(), o_w, spec) is o_w

    # Widths other than head_dim (wrong head count, packed/quantized layouts)
    # raise instead of being silently mis-padded.
    bad_tensors = [
        (q_param, torch.randn(13 * head_dim, hidden_size), "q"),
        (q_param, torch.randn(28 * (head_dim // 8), hidden_size), "q"),
        (q_param, torch.randn(13 * head_dim, hidden_size), None),
    ]
    for param, tensor, shard_id in bad_tensors:
        with pytest.raises(ValueError, match="Unsupported"):
            pad_qkv_weight(param, tensor, shard_id, spec)
    with pytest.raises(ValueError, match="Unsupported"):
        pad_o_proj_weight(o_param, torch.randn(hidden_size, 13 * head_dim), spec)


def test_register_models_and_tpu_qwen2_attention(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    import vllm.model_executor.models.config as models_config_mod
    import vllm.model_executor.models.qwen2 as qwen2_mod

    from vllm_torchtpu.models.vllm import register_models
    from vllm_torchtpu.models.vllm.qwen2 import TPUTPPaddedQwen2Attention
    from vllm_torchtpu.models.vllm.qwen2_patch import maybe_patch_qwen2

    # register_models() and maybe_patch_qwen2() mutate module-level vLLM state;
    # swap in copies via monkeypatch so the originals are restored on teardown.
    monkeypatch.setattr(qwen2_mod, "Qwen2Attention", qwen2_mod.Qwen2Attention)
    monkeypatch.setattr(
        models_config_mod,
        "MODELS_CONFIG_MAP",
        dict(models_config_mod.MODELS_CONFIG_MAP),
    )

    register_models()
    assert "Qwen2ForCausalLM" in models_config_mod.MODELS_CONFIG_MAP

    # Patch is scoped to Qwen2ForCausalLM and retries for other/missing configs.
    upstream_attn = qwen2_mod.Qwen2Attention
    assert maybe_patch_qwen2(None) is False
    llama_cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
    assert maybe_patch_qwen2(llama_cfg) is False
    assert qwen2_mod.Qwen2Attention is upstream_attn
    qwen2_cfg = SimpleNamespace(architectures=["Qwen2ForCausalLM"])
    assert maybe_patch_qwen2(qwen2_cfg) is None
    assert qwen2_mod.Qwen2Attention is TPUTPPaddedQwen2Attention

    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(world_size=8, rank_in_group=1),
    )
    monkeypatch.setattr(qwen2_mod, "Attention", lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr(qwen2_mod, "get_rope", lambda *a, **kw: torch.nn.Identity())

    attn = qwen2_mod.Qwen2Attention(
        hidden_size=3584,
        num_heads=28,
        num_kv_heads=4,
        rope_parameters={},
    )
    assert attn.orig_total_num_heads == 28
    assert attn.total_num_heads == 32
    assert attn.num_heads == 4
    assert attn.qkv_proj.total_num_heads == 32
    assert attn.o_proj.input_size == 4096

    # Exercise real QKVParallelLinear and RowParallelLinear weight loaders.
    # Rank 1 (KV group 0, Q heads 4..7): Q heads 4..6 are 1, Q head 7 is 0.
    q_w = torch.ones(28 * 128, 3584, dtype=attn.qkv_proj.weight.dtype)
    attn.qkv_proj.weight.weight_loader(attn.qkv_proj.weight, q_w, "q")
    assert torch.all(attn.qkv_proj.weight.data[: 3 * 128] == 1)
    assert torch.all(attn.qkv_proj.weight.data[3 * 128 : 4 * 128] == 0)

    # Qwen2 Q bias is padded the same way.
    q_b = torch.ones(28 * 128, dtype=attn.qkv_proj.bias.dtype)
    attn.qkv_proj.bias.weight_loader(attn.qkv_proj.bias, q_b, "q")
    assert torch.all(attn.qkv_proj.bias.data[: 3 * 128] == 1)
    assert torch.all(attn.qkv_proj.bias.data[3 * 128 : 4 * 128] == 0)

    # Fused QKV via weight_loader_v2 (exercises re-entrant "q" shard loading).
    fused_qkv_w = torch.full(
        ((28 + 2 * 4) * 128, 3584),
        2.0,
        dtype=attn.qkv_proj.weight.dtype,
    )
    attn.qkv_proj.weight_loader_v2(attn.qkv_proj.weight, fused_qkv_w)
    assert torch.all(attn.qkv_proj.weight.data[: 3 * 128] == 2)
    assert torch.all(attn.qkv_proj.weight.data[3 * 128 : 4 * 128] == 0)
    assert torch.all(attn.qkv_proj.weight.data[4 * 128 : 6 * 128] == 2)

    o_w = torch.ones(3584, 28 * 128, dtype=attn.o_proj.weight.dtype)
    attn.o_proj.weight.weight_loader(attn.o_proj.weight, o_w)
    assert torch.all(attn.o_proj.weight.data[:, : 3 * 128] == 1)
    assert torch.all(attn.o_proj.weight.data[:, 3 * 128 : 4 * 128] == 0)

    # No padding needed (TP=4): arguments forwarded unchanged to upstream.
    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(world_size=4, rank_in_group=0),
    )
    attn_tp4 = qwen2_mod.Qwen2Attention(
        hidden_size=3584,
        num_heads=28,
        num_kv_heads=4,
        rope_parameters={},
    )
    assert attn_tp4.total_num_heads == 28
    assert attn_tp4.o_proj.input_size == 3584
