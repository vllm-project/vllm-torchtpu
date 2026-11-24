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
"""TPU_MINIMAX_M3_NORM_WEIGHT_HOIST (`minimax_m3.norm_hoist`): the hoisted
Gemma RMSNorm forward and q/k norm table give the same bits as the per-call
``weight.float() + 1.0`` forms, and a weight change after hoisting is not
seen (positive control: the buffer is really what the forward reads)."""

import types

import pytest
import torch
from torch import nn


@pytest.fixture
def cpu_config():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _norm(hidden, seed):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    norm = GemmaRMSNorm(hidden, eps=1e-6)
    g = torch.Generator().manual_seed(seed)
    norm.weight.data = (torch.randn(hidden, generator=g) * 0.3).to(torch.bfloat16)
    return norm


@pytest.mark.parametrize("rows", [8, 96])
def test_gemma_norm_hoist_bitwise(cpu_config, rows):
    import vllm_torchtpu.layers.adapter.ir_ops  # noqa: F401  (TPU provider)
    from vllm_torchtpu.models.minimax_m3 import norm_hoist

    hidden = 256
    ref = _norm(hidden, 1)
    hot = _norm(hidden, 1)
    root = nn.Module()
    root.a = hot
    assert norm_hoist.hoist(root) == 1
    g = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, hidden, generator=g).to(torch.bfloat16) * 3
    r = torch.randn(rows, hidden, generator=g).to(torch.bfloat16)
    assert torch.equal(ref(x).view(torch.int16), hot(x).view(torch.int16))
    (o1, s1), (o2, s2) = ref(x, r), hot(x, r)
    assert torch.equal(o1.view(torch.int16), o2.view(torch.int16))
    assert torch.equal(s1.view(torch.int16), s2.view(torch.int16))
    assert torch.equal(norm_hoist.gemma_weight(hot), ref.weight.float() + 1.0)
    # Positive control: the hoisted forward reads the buffer, not the weight.
    hot.weight.data.zero_()
    assert torch.equal(hot(x).view(torch.int16), ref(x).view(torch.int16))
    assert not torch.equal(norm_hoist.gemma_weight(ref), torch.ones(hidden))


def test_qk_norm_table_matches_inline(cpu_config):
    from vllm_torchtpu.models.minimax_m3 import norm_hoist
    from vllm_torchtpu.models.minimax_m3.attention import norm_rope_heads

    heads, kv_heads, dim = 8, 1, 128
    norms = tuple(_norm(dim, s) for s in (2, 3, 4, 5))
    rope = types.SimpleNamespace(
        is_neox_style=True,
        head_size=dim,
        _match_cos_sin_cache_dtype=lambda qkv: torch.zeros(64, dim),
    )
    pos = torch.zeros(4, dtype=torch.long)
    op = lambda qkv, cos_sin, weight: weight  # noqa: E731
    inline = norm_rope_heads(None, pos, heads, kv_heads, dim, norms, rope, op)
    table = norm_hoist.qk_norm_table(norms, heads, kv_heads, dim)
    given = norm_rope_heads(None, pos, heads, kv_heads, dim, norms, rope, op, table)
    assert inline.dtype == torch.float32 and inline.shape == (heads + 2 * kv_heads + 1, dim)
    assert torch.equal(inline, table) and given is table
