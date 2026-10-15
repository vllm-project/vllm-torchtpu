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
"""Tests for load-time FP8 quantization of VocabParallelEmbedding and ParallelLMHead."""

import torch
import torch.nn as nn
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
)


class _DummyEmbed(VocabParallelEmbedding):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 64):
        nn.Module.__init__(self)
        self.params_dtype = torch.bfloat16
        self.weight = nn.Parameter(
            torch.randn((vocab_size, hidden_size), dtype=torch.bfloat16),
            requires_grad=False,
        )


class _DummyLMHead(ParallelLMHead):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 64):
        nn.Module.__init__(self)
        self.params_dtype = torch.bfloat16
        self.weight = nn.Parameter(
            torch.randn((vocab_size, hidden_size), dtype=torch.bfloat16),
            requires_grad=False,
        )


def test_unquantized_config_dispatches_fp8_embedding_on_load(monkeypatch):
    cfg = VllmUnquantizedConfig()
    dummy = VocabParallelEmbedding.__new__(VocabParallelEmbedding)

    monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn,shared_experts")
    assert cfg.get_quant_method(dummy, "model.embed_tokens") is None

    monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "embed_tokens,lm_head")
    method = cfg.get_quant_method(dummy, "model.embed_tokens")
    assert isinstance(method, VllmFp8LinearMethodTPU)


def test_fp8_embedding_and_lm_head_numerics(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")
    monkeypatch.delenv("REQUANTIZE_BLOCK_SIZE", raising=False)
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")

    embed = _DummyEmbed(vocab_size=128, hidden_size=64)
    lm_head = _DummyLMHead(vocab_size=128, hidden_size=64)

    token_ids = torch.tensor([[1, 7, 42], [0, 19, 127]], dtype=torch.long)
    hidden = torch.randn((6, 64), dtype=torch.bfloat16)

    ref_embed = embed.weight[token_ids]
    ref_logits = hidden @ lm_head.weight.t()

    fp8_cfg = VllmFp8Config(
        is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
    )
    method = VllmFp8LinearMethodTPU(fp8_cfg)
    method.process_weights_after_loading(embed)
    method.process_weights_after_loading(lm_head)

    assert embed.weight.dtype == torch.float8_e4m3fn
    assert tuple(embed.weight.shape) == (128, 64)
    assert tuple(embed.weight_scale.shape) == (128,)

    assert lm_head.weight.dtype == torch.float8_e4m3fn
    assert tuple(lm_head.weight.shape) == (64, 128)
    assert tuple(lm_head.weight_scale.shape) == (128,)

    got_embed = method.embedding(embed, token_ids)
    assert got_embed.shape == (2, 3, 64)
    assert got_embed.dtype == torch.bfloat16
    torch.testing.assert_close(got_embed, ref_embed, rtol=0.05, atol=0.1)

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul",
        lambda x, w, s: (x.float() @ (w.float() * s.unsqueeze(0))).to(x.dtype),
    )
    got_logits = method.apply(lm_head, hidden)
    assert got_logits.shape == (6, 128)
    assert got_logits.dtype == torch.bfloat16
    cos_sim = torch.nn.functional.cosine_similarity(
        got_logits.float(), ref_logits.float(), dim=-1
    )
    assert (cos_sim > 0.999).all()
