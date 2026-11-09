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
"""100% full-module and diff coverage tests for DeepSeek-V4.1 v2 DSpark.

Target modules:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.dspark (full module lines 1-509)
- diffs in:
  - vllm_torchtpu.models.vllm.deepseek_v41.v2.model
  - vllm_torchtpu.models.vllm.deepseek_v41.v2.attention
  - vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels
  - vllm_torchtpu.models.vllm.__init__
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    dspark as v2_dspark,
)


class _FakeDraftAttn(nn.Module):
    def __init__(self, hidden_size: int, window_size: int = 512, prefix: str = ""):
        super().__init__()
        self.window_size = window_size
        self.non_causal_multi_token_decode = False
        self.dspark_block_size = 0
        self.swa_cache_layer = SimpleNamespace(
            prefix=f"{prefix}.swa_cache",
            kv_cache=torch.zeros((4, 16, 1, hidden_size)),
        )
        self.fused_wqa_wkv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(4))

    def _fused_wqa_wkv_gemm(self, x: torch.Tensor) -> torch.Tensor:
        return self.fused_wqa_wkv(x)

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return qr_kv, qr_kv, qr_kv

    def _rotate(self, kv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return kv


class _FakeDraftFFN(nn.Module):
    def __init__(self, hidden_size: int, use_native_mega_moe: bool = False):
        super().__init__()
        self.use_native_mega_moe = use_native_mega_moe
        self.gate = nn.Module()
        self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(4))
        self.shared_experts = nn.Module()
        self.shared_experts.gate_up_proj = nn.Parameter(
            torch.zeros(hidden_size, hidden_size)
        )
        self.shared_experts.down_proj = nn.Parameter(
            torch.zeros(hidden_size, hidden_size)
        )
        self.experts = nn.Module()
        self.experts.w13_weight = nn.Parameter(torch.zeros(2, hidden_size, hidden_size))
        self.experts.w2_weight = nn.Parameter(torch.zeros(2, hidden_size, hidden_size))
        self.experts.w13_weight_scale = nn.Parameter(
            torch.zeros(2, 4, dtype=torch.uint8), requires_grad=False
        )
        self.experts.w13_weight_scale_inv = nn.Parameter(
            torch.zeros(2, 4, dtype=torch.float32), requires_grad=False
        )
        self.finalized = False

    def finalize_mega_moe_weights(self) -> None:
        self.finalized = True


class _FakeDraftDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: Any,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        hidden_size = 16
        self.attn = _FakeDraftAttn(
            hidden_size, window_size=512, prefix=f"{prefix}.attn"
        )
        self.ffn = _FakeDraftFFN(hidden_size)
        self.attn_norm = v2_dspark.RMSNorm(hidden_size, eps=1e-6)
        self.ffn_norm = v2_dspark.RMSNorm(hidden_size, eps=1e-6)
        self.hc_attn_fn = nn.Parameter(
            torch.zeros((2, hidden_size), dtype=torch.float32)
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.zeros((2, hidden_size), dtype=torch.float32)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ):
        n, k, d = residual.shape if residual is not None else hidden_states.shape
        x = (
            hidden_states + 1.0
            if hidden_states.ndim == 2
            else hidden_states[:, 0] + 1.0
        )
        return (
            x,
            torch.zeros((n, k, d)),
            torch.ones((n, k, 1)),
            torch.eye(k).unsqueeze(0).expand(n, -1, -1),
            torch.full((n, k), 1.0 / k),
            None,
        )


def _make_dspark_vllm_config(
    *,
    num_speculative_tokens: int = 15,
    enable_confidence_head: bool = True,
    expert_dtype: str = "fp4",
    weight_block_size: list[int] | None = None,
) -> Any:
    hf_config = SimpleNamespace(
        hidden_size=16,
        hc_mult=2,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        num_hidden_layers=4,
        dspark_target_layer_ids=[1, 2, 3],
        n_mtp_layers=2,
        index_topk=4,
        dspark_block_size=num_speculative_tokens + 1,
        vocab_size=32,
        draft_vocab_size=32,
        dspark_markov_rank=8,
        enable_confidence_head=enable_confidence_head,
        dspark_n_routed_experts=2,
        n_routed_experts=2,
        num_attention_heads=4,
        expert_dtype=expert_dtype,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    quant_config = (
        SimpleNamespace(
            weight_block_size=weight_block_size,
            online_quantization_config=None,
            get_quant_method=lambda layer, prefix: UnquantizedLinearMethod(),
        )
        if weight_block_size is not None
        else None
    )
    return SimpleNamespace(
        additional_config={},
        engram_config=None,
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=num_speculative_tokens,
            draft_model_config=SimpleNamespace(hf_config=hf_config),
        ),
        quant_config=quant_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        model_config=SimpleNamespace(max_model_len=1024),
    )


def _patch_dspark_env(monkeypatch: pytest.MonkeyPatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark_sampling

    monkeypatch.setattr(dspark_sampling, "initialize_collapse", MagicMock)
    fake_cur_cfg = SimpleNamespace(
        additional_config={},
        engram_config=None,
        model_config=SimpleNamespace(
            max_logprobs=20, logits_processor_pattern=None, head_dtype=None
        ),
        kernel_config=SimpleNamespace(linear_backend="auto"),
        compilation_config=SimpleNamespace(
            custom_ops=["none"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
        ),
    )
    monkeypatch.setattr("vllm.config.vllm._current_vllm_config", fake_cur_cfg)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            is_first_rank=True,
            is_last_rank=True,
            all_reduce=lambda x: x,
            gather=lambda x, dst=0, dim=-1: x,
        ),
    )
    monkeypatch.setattr(
        "vllm.model_executor.custom_op.get_cached_compilation_config",
        lambda: fake_cur_cfg.compilation_config,
    )
    monkeypatch.setattr(v2_dspark, "initialize_cache_ops", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_delayed_mhc", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_mhc_post", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_paged_attention", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_indexer", lambda: None)
    monkeypatch.setattr(v2_dspark, "compact_rope_caches", lambda *a: None)
    monkeypatch.setattr(v2_dspark, "get_current_vllm_config", lambda: SimpleNamespace())
    monkeypatch.setattr(v2_dspark, "DeepseekV41DecoderLayer", _FakeDraftDecoderLayer)
    monkeypatch.setattr(
        v2_dspark,
        "pack_mix_weights",
        lambda layers: [
            setattr(
                layer,
                "hc_attn_fn",
                nn.Parameter(layer.hc_attn_fn.to(torch.bfloat16), requires_grad=False),
            )
            for layer in layers
        ],
    )
    monkeypatch.setattr(v2_dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(v2_dspark, "get_tensor_model_parallel_rank", lambda: 0)


def test_dspark_deepseek_v41_model_edge_cases(monkeypatch: pytest.MonkeyPatch):
    """Test full branches in DSparkDeepseekV41Model: confidence head disabled,
    fallback layer counts and block sizes.
    """
    _patch_dspark_env(monkeypatch)

    # 1. enable_confidence_head = False, num_nextn_predict_layers fallback,
    # block_size fallback, draft_vocab_size fallback
    vllm_cfg = _make_dspark_vllm_config(enable_confidence_head=False)
    hf_cfg = vllm_cfg.speculative_config.draft_model_config.hf_config
    hf_cfg.n_mtp_layers = None
    hf_cfg.num_nextn_predict_layers = 2
    hf_cfg.dspark_block_size = None
    hf_cfg.block_size = 12
    hf_cfg.draft_vocab_size = None

    draft_lm = v2_dspark.DSparkDeepseekV41ForCausalLM(vllm_config=vllm_cfg)
    assert draft_lm.model.confidence_head is None
    assert draft_lm.model.num_dspark_layers == 2
    assert draft_lm.model.layers[0].attn.dspark_block_size == 12

    # compute_confidence raises AssertionError when confidence_head is None
    with pytest.raises(AssertionError):
        draft_lm.compute_confidence(torch.randn(4, 16), torch.randn(4, 8))

    # _remap_dspark_name returns None for confidence_head when disabled
    assert draft_lm._remap_dspark_name("mtp.0.confidence_head.w1.weight") is None

    # 2. Both n_mtp_layers and num_nextn_predict_layers are None -> default 3 layers
    vllm_cfg2 = _make_dspark_vllm_config(num_speculative_tokens=5)
    hf_cfg2 = vllm_cfg2.speculative_config.draft_model_config.hf_config
    hf_cfg2.n_mtp_layers = None
    hf_cfg2.num_nextn_predict_layers = None
    hf_cfg2.dspark_block_size = None
    hf_cfg2.block_size = None

    draft_lm2 = v2_dspark.DSparkDeepseekV41ForCausalLM(vllm_config=vllm_cfg2)
    assert draft_lm2.model.num_dspark_layers == 3
    assert draft_lm2.model.layers[0].attn.dspark_block_size == 5

    # 3. finalize_mhc_broadcast_weights when layers is empty
    draft_lm2.model.layers = nn.ModuleList([])
    draft_lm2.model.finalize_mhc_broadcast_weights()


def test_dspark_load_weights_expert_mapping_and_stacked_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test full line+branch coverage in load_weights: expert mapping
    mismatch/failure, unmapped expert weights, and stacked param filters.
    """
    _patch_dspark_env(monkeypatch)
    vllm_cfg = _make_dspark_vllm_config(
        enable_confidence_head=False,
        expert_dtype="fp8",
        weight_block_size=None,  # pad_shared_expert = False
    )
    vllm_cfg.speculative_config.draft_model_config.hf_config.dspark_n_routed_experts = 0
    draft_lm = v2_dspark.DSparkDeepseekV41ForCausalLM(vllm_config=vllm_cfg)
    assert draft_lm.pad_shared_expert is False

    # expert_mapping has 2 entries:
    # 1. "experts.0.w1_mismatch.weight" -> triggers line 452 (weight_name not in name)
    # 2. "experts.0.w1_fail.weight" -> returns False (triggers line 463->450)
    monkeypatch.setattr(
        v2_dspark,
        "fused_moe_make_expert_params_mapping",
        lambda *a, **kw: [
            ("experts.w13_weight", "experts.0.w1_mismatch.weight", 0, "w1"),
            ("experts.w13_weight", "experts.0.w1_fail.weight", 0, "w1"),
        ],
    )
    layer0 = draft_lm.model.layers[0]
    layer0.ffn.experts.w13_weight.weight_loader = lambda *a, **kw: False
    layer0.attn_norm.weight.weight_loader = lambda *a, **kw: None
    draft_lm.model.norm.weight.weight_loader = lambda *a, **kw: None

    weights = [
        # Matches .experts., hits line 452 continue on first item, calls
        # weight_loader on second -> returns False -> loop finishes (line 450->466)
        ("mtp.0.ffn.experts.0.w1_fail.weight", torch.zeros(16, 16)),
        # Matches .experts., neither mapping matches -> loop finishes (450->466)
        ("mtp.0.ffn.experts.0.unmatched.weight", torch.zeros(16, 16)),
        # Non-layer parameter -> stacked_params_mapping checks "not is_layer_param"
        ("mtp.0.norm.weight", torch.ones(16)),
        # Layer param but doesn't match stacked_params_mapping -> continue
        ("mtp.0.attn_norm.weight", torch.ones(16)),
    ]
    loaded = draft_lm.load_weights(weights)
    assert "model.norm.weight" in loaded
    assert "model.layers.0.attn_norm.weight" in loaded
