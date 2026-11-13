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
"""One-pass q / k / index norm + RoPE kernel vs the XLA path it replaces.

The reference is `attention.norm_rope_heads`' unfused graph written with the
same torch ops (`ir_ops.rms_norm`'s FP32 normalize, `ApplyRotaryEmb`'s FP32
NeoX rotation of the first ``rot`` lanes). ``interpret=True`` runs the kernel
body on the host; it does not gate Mosaic lowering.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_torchtpu.layers.adapter.ir_ops import _normalize_fp32
from vllm_torchtpu.models.minimax_m3.kernels.norm_rope import norm_rope

HEAD_DIM = 128
ROT = 64
EPS = 1e-6


def _inputs(tokens, nq, nkv, seed):
    g = torch.Generator().manual_seed(seed)
    heads = nq + 3 * nkv + 1
    qkv = (torch.randn(tokens, heads * HEAD_DIM, generator=g) * 3).to(torch.bfloat16)
    norm_w = torch.randn(nq + 2 * nkv + 1, HEAD_DIM, generator=g) * 0.2 + 1.0
    positions = torch.randint(0, 300000, (tokens,), generator=g)
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, dtype=torch.float) / ROT))
    freqs = positions.float()[:, None] * inv[None, :]
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    return qkv, norm_w, cos_sin


def _reference(qkv, norm_w, cos_sin, nq, nkv):
    tokens = qkv.shape[0]
    h = qkv.view(tokens, -1, HEAD_DIM)
    x = torch.cat([h[:, : nq + nkv], h[:, nq + 2 * nkv :]], dim=1)
    y = _normalize_fp32(x.float(), norm_w, EPS, None).to(torch.bfloat16)
    cos, sin = cos_sin.chunk(2, dim=-1)
    rot = ApplyRotaryEmb.forward_static(
        y[..., :ROT], cos, sin, True, enable_fp32_compute=True
    )
    out = torch.cat([rot, y[..., ROT:]], dim=-1)
    variance = x.float().pow(2).mean(dim=-1)
    return out, torch.rsqrt(variance + EPS)


def _kernel(qkv, r, norm_w, cos_sin, nq, nkv):
    pad = torch.nn.functional.pad(cos_sin, (0, HEAD_DIM - ROT))
    f = lambda t: jnp.asarray(t.float().numpy()).astype(  # noqa: E731
        jnp.bfloat16 if t.dtype == torch.bfloat16 else jnp.float32
    )
    out = norm_rope(
        f(qkv), f(r), f(norm_w), f(pad), nq, nkv, HEAD_DIM, ROT, interpret=True
    )
    return torch.from_numpy(np.asarray(out.astype(jnp.float32))).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [8, 200, 512, 1100])
@pytest.mark.parametrize("nq,nkv", [(8, 1), (16, 2)])
def test_matches_unfused(tokens, nq, nkv):
    qkv, norm_w, cos_sin = _inputs(tokens, nq, nkv, seed=tokens + nq)
    ref, r = _reference(qkv, norm_w, cos_sin, nq, nkv)
    got = _kernel(qkv, r, norm_w, cos_sin, nq, nkv).view(tokens, -1, HEAD_DIM)
    assert got.shape == ref.shape
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_fault_is_caught():
    """Positive control: a kernel fed the wrong rotation (cos and sin swapped)
    must differ from the reference."""
    qkv, norm_w, cos_sin = _inputs(64, 8, 1, seed=3)
    ref, r = _reference(qkv, norm_w, cos_sin, 8, 1)
    swapped = torch.cat(cos_sin.chunk(2, dim=-1)[::-1], dim=-1)
    got = _kernel(qkv, r, norm_w, swapped, 8, 1).view(64, -1, HEAD_DIM)
    assert not torch.equal(got, ref)
    # The passed lanes are untouched by the rotation.
    assert torch.equal(got[..., ROT:], ref[..., ROT:])
