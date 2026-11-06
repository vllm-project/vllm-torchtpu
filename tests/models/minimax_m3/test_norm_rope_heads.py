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
"""`norm_rope_heads` equals the per-projection norms and rotations bit for bit."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_torchtpu.models.minimax_m3.attention import norm_rope_heads

D = 128


def _reference(qkv, positions, nq, nkv, norms, rope):
    q, k, v, iq, ik = qkv.split([nq * D, nkv * D, nkv * D, nkv * D, D], dim=-1)

    def per_head(x, norm):
        return norm(x.reshape(-1, D)).reshape(x.shape)

    q, k, iq = per_head(q, norms[0]), per_head(k, norms[1]), per_head(iq, norms[2])
    ik = norms[3](ik)
    q, k = rope(positions, q, k)
    iq, ik = rope(positions, iq, ik)
    return q, k, v, iq, ik


@pytest.mark.parametrize("nq,nkv", [(16, 1), (8, 1)])  # TP4 / TP8 per rank
def test_matches_separate_norms_and_ropes(monkeypatch, nq, nkv):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        norms = tuple(GemmaRMSNorm(D, eps=1e-6) for _ in range(4))
        for n in norms:
            n.weight.data = torch.randn(D) * 0.1
        rope = get_rope(
            D,
            max_position=4096,
            rope_parameters={"rope_theta": 1e7, "partial_rotary_factor": 0.5},
        )
        tokens = 7
        qkv = torch.randn(tokens, (nq + 3 * nkv + 1) * D).to(torch.bfloat16)
        positions = torch.tensor([0, 1, 5, 127, 128, 3000, 4095])
        expected = _reference(qkv, positions, nq, nkv, norms, rope)
        got = norm_rope_heads(qkv, positions, nq, nkv, D, norms, rope)
    for name, e, g in zip(["q", "k", "v", "index_q", "index_k"], expected, got):
        assert g.shape == e.shape, name
        assert g.dtype == e.dtype, name
        assert torch.equal(g, e), name
