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
"""The fused q/k norm + rope kernel (additional_config
minimax_qk_norm_rope_kernel) against the XLA path `norm_rope_heads`, on CPU
(Pallas interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_torchtpu.models.minimax_m3.attention import neox_swap_matrix, norm_rope_heads
from vllm_torchtpu.models.minimax_m3.kernels import qk_norm_rope

D = 128


def _to_jax(t):
    return jnp.asarray(t.detach().float().numpy()).astype(
        jnp.bfloat16 if t.dtype == torch.bfloat16 else jnp.float32
    )


def _to_torch(a):
    return torch.from_numpy(np.asarray(a.astype(jnp.float32)))


@pytest.mark.parametrize("num_tokens", [5, 300])
@pytest.mark.parametrize("nq,nkv", [(8, 1), (16, 2)])
def test_kernel_matches_norm_rope_heads(num_tokens, nq, nkv, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    torch.manual_seed(num_tokens + nq)
    with set_current_vllm_config(VllmConfig()):
        norms = tuple(GemmaRMSNorm(D, eps=1e-6) for _ in range(4))
        rope = get_rope(
            D,
            max_position=1 << 20,
            rope_parameters={"rope_theta": 1e7, "partial_rotary_factor": 0.5},
            dtype=torch.bfloat16,
        )
    for n in norms:
        n.weight.data = (torch.randn(D) * 0.1).to(n.weight.dtype)
    qkv = torch.randn(num_tokens, (nq + 3 * nkv + 1) * D).to(torch.bfloat16) * 3
    positions = torch.randint(0, 200_000, (num_tokens,))
    swap = neox_swap_matrix(D, rope.rotary_dim, torch.bfloat16)
    want = norm_rope_heads(qkv, positions, nq, nkv, D, norms, rope, swap)

    weight = (
        torch.cat(
            [
                norms[0].weight.expand(nq, D),
                norms[1].weight.expand(nkv, D),
                norms[2].weight.expand(nkv, D),
                norms[3].weight.expand(1, D),
            ]
        ).float()
        + 1.0
    )
    cos_sin = rope._match_cos_sin_cache_dtype(qkv).index_select(0, positions)
    got = qk_norm_rope.qk_norm_rope(
        _to_jax(qkv),
        _to_jax(cos_sin),
        _to_jax(weight),
        num_heads=nq,
        num_kv_heads=nkv,
        head_dim=D,
        eps=1e-6,
        interpret=True,
    )
    for name, g, w in zip(("q", "k", "v", "index_q", "index_k"), got, want):
        g = _to_torch(g)
        assert g.shape == w.shape, name
        w = w.float()
        if name == "v":
            assert torch.equal(g, w)
            continue
        # One bf16 rounding of the norm and one of the rotation, possibly on
        # the other side of a tie: an ulp of the value, or an ulp of a normed
        # input where the rotation cancels (|x| ~ 1 after the norm).
        diff = (g - w).abs().detach()
        tol = w.abs().detach() * 2**-7 + 2**-7
        assert torch.all(diff <= tol), (name, diff.max())
        assert (diff > 0).float().mean() < 0.02, name
