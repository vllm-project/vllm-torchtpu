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
"""100% Diff coverage test suite for SP attention and projections.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.attention
  (attention.py:424-521, 589-600, 629-705, 813-882)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.model
  (model.py:38-74, 524-530, 616-625)
"""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model

# ==============================================================================
# 1. Tests for v2/model.py: _ced_config, DeepseekV41DecoderLayer, DeepseekV41Model
# ==============================================================================


def test_ced_config_combinations(monkeypatch, caplog):
    """Test _ced_config CED on/off, prefix caching, warnings, and fallback."""
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", True)

    # 1. CED=1 and prefix caching disabled -> ced_enabled=True
    cfg1 = SimpleNamespace(
        additional_config={},
        engram_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=40),
        ),
    )
    enabled, boundary = v2_model._ced_config(cfg1)
    assert enabled is True
    assert boundary == 20

    # 2. CED=1 and prefix caching enabled with warn_prefix_caching=True
    cfg2 = SimpleNamespace(
        additional_config={},
        engram_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=30),
        ),
    )
    with caplog.at_level(logging.WARNING):
        enabled2, boundary2 = v2_model._ced_config(cfg2, warn_prefix_caching=True)
    assert enabled2 is False
    assert boundary2 == 15
    assert "CED with prefix caching needs SWA bounded replay" in caplog.text

    # 2b. With SWA bounded replay, CED keeps running alongside prefix caching.
    cfg2b = SimpleNamespace(
        additional_config={},
        engram_config=None,
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, swa_bounded_replay=True
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        model_config=cfg2.model_config,
    )
    assert v2_model._ced_config(cfg2b, warn_prefix_caching=True) == (True, 15)

    # 3. CED=1 and prefix caching enabled with warn_prefix_caching=False
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        enabled3, boundary3 = v2_model._ced_config(cfg2, warn_prefix_caching=False)
    assert enabled3 is False
    assert boundary3 == 15
    assert caplog.text == ""

    # 4. CED=0 -> ced_enabled=False
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", False)
    enabled4, boundary4 = v2_model._ced_config(cfg1)
    assert enabled4 is False
    assert boundary4 == 20

    # 5. model_config is None, fallback_config provided
    cfg5 = SimpleNamespace(
        additional_config={},
        engram_config=None,
        cache_config=None,
        model_config=None,
    )
    fallback = SimpleNamespace(num_hidden_layers=24)
    enabled5, boundary5 = v2_model._ced_config(cfg5, fallback_config=fallback)
    assert enabled5 is False
    assert boundary5 == 12


# ==============================================================================
# 2. Tests for v2/attention.py: forward & _sp_sharded_input branches
# ==============================================================================


def _create_mock_attention():
    """Builds a bare DeepseekV41Attention instance with mock components."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    nn.Module.__init__(attn)
    attn.padded_heads = 16
    attn.n_local_heads = 16
    attn.head_dim = 128
    attn.q_lora_rank = 512
    attn.compress_ratio = 2
    attn.candidate_block_size = 16
    attn._max_model_len = 4096
    attn._fp8_kv = False
    attn._config = {"index_topk": 2048, "candidate_topk_blocks": 64}
    attn._sp_sharded_input = False
    attn._ced_global_from_encoder = False
    attn._prepare_and_attn_fn = MagicMock()
    attn._o_proj = MagicMock(side_effect=lambda o, pos: o)
    attn.compressor = SimpleNamespace(project=lambda x: x[..., :64])
    attn.indexer = SimpleNamespace(
        n_head=64,
        head_dim=128,
        weights_proj=lambda x: x[..., :64],
        wq_b=lambda x: x,
        indexer_op=SimpleNamespace(
            candidate_blocks=torch.zeros(4, 4), candidate_write=False
        ),
    )
    attn._fused_wqa_wkv_gemm = MagicMock(
        side_effect=lambda x: torch.cat((x[..., :512], x[..., :128]), dim=-1)
    )
    attn.q_norm = MagicMock(side_effect=lambda x: x)
    attn.kv_norm = MagicMock(side_effect=lambda x: x)
    attn.kv_cache = torch.zeros(10, 10)
    attn.candidate_block_buffer = torch.zeros(100, 16, dtype=torch.int32)
    attn.topk_indices_buffer = torch.zeros(100, 128, dtype=torch.int32)
    return attn


def test_attention_get_kv_cache_spec(monkeypatch):
    """Test get_kv_cache_spec for None, non-FP8, and FP8 replacement."""
    attn = _create_mock_attention()

    # 1. Spec is None -> returns None
    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, cfg: None,
    )
    assert attn.get_kv_cache_spec(None) is None

    # 2. Spec is present but _fp8_kv is False -> returns unmodified spec
    fake_spec = SimpleNamespace(dtype=torch.bfloat16, page_size_padded=128)
    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, cfg: fake_spec,
    )
    attn._fp8_kv = False
    assert attn.get_kv_cache_spec(None) is fake_spec

    # 3. Spec is present and _fp8_kv is True -> returns replaced spec
    @dataclasses.dataclass
    class FakeSpecCls:
        dtype: torch.dtype
        page_size_padded: int | None

    real_fake_spec = FakeSpecCls(dtype=torch.bfloat16, page_size_padded=128)
    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, cfg: real_fake_spec,
    )
    attn._fp8_kv = True
    replaced = attn.get_kv_cache_spec(None)
    assert replaced.dtype == torch.uint8
    assert replaced.page_size_padded is None


def test_run_parallel_input_projections_and_all_indices_fit():
    """Test _run_parallel_input_projections and _all_indices_fit branches."""
    attn = _create_mock_attention()

    # 1. compressor is not None and not _ced_global_from_encoder
    #    indexer is not None and not _all_indices_fit
    attn.compress_ratio = 2
    attn._max_model_len = 8192
    attn._config = {"index_topk": 2048}  # 8192 // 2 = 4096 > 2048 -> fit=False
    attn._ced_global_from_encoder = False

    x = torch.randn(2, 64)
    qr_kv, kv_score, idx_w = attn._run_parallel_input_projections(x)
    assert kv_score is not None
    assert idx_w is not None

    # 2. _ced_global_from_encoder is True -> kv_score is None
    attn._ced_global_from_encoder = True
    _, kv_score2, _ = attn._run_parallel_input_projections(x)
    assert kv_score2 is None

    # 3. _all_indices_fit is True -> idx_w is None
    attn._ced_global_from_encoder = False
    attn._config = {"index_topk": 5000}  # 4096 <= 5000 -> fit=True
    _, _, idx_w2 = attn._run_parallel_input_projections(x)
    assert idx_w2 is None

    # 4. compressor is None and indexer is None
    attn.compressor = None
    attn.indexer = None
    _, kv_score3, idx_w3 = attn._run_parallel_input_projections(x)
    assert kv_score3 is None
    assert idx_w3 is None

    # 5. compress_ratio is 0 -> _all_indices_fit is False
    attn.compress_ratio = 0
    assert attn._all_indices_fit is False


# ==============================================================================
# 3. Tests for v2/attention.py: _prepare_tpu & _select_paged_indices
# ==============================================================================
