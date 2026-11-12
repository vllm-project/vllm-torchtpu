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
"""MiniMax M3 MXFP8 W8A16 decode path (TPU_MINIMAX_M3_MXFP8_W8A16_MAX_ROWS)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.minimax_m3.quant import dequantize_mxfp8, mxfp8_to_w8a16


def _mxfp8(n: int, k: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """OCP-style MXFP8 of a gaussian [n, k] weight (block 32 along k)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k // 32, 32, generator=g) * 0.02
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    e = torch.floor(torch.log2(amax)) - 8
    q = (w / torch.exp2(e)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(n, k), (e[..., 0] + 127).to(torch.uint8)


@pytest.mark.cpu_test
@pytest.mark.parametrize("n,k", [(1536, 6144), (6144, 1024), (6144, 384)])
def test_w8a16_copy_dequantizes_to_the_bf16_weight(n, k):
    w8, s = _mxfp8(n, k)
    w_kn, scale = mxfp8_to_w8a16(w8, s)
    assert w_kn.dtype == torch.float8_e4m3fn and w_kn.shape == (k, n)
    assert scale.dtype == torch.bfloat16 and scale.shape == (k // 32, n)
    deq = w_kn.to(torch.bfloat16) * scale.repeat_interleave(32, dim=0)
    assert torch.equal(deq.view(torch.int16), dequantize_mxfp8(w8, s).t().view(torch.int16))


def _on_tpu() -> bool:
    try:
        return jax.devices()[0].platform == "tpu"
    except RuntimeError:
        return False


@pytest.mark.skipif(not _on_tpu(), reason="Pallas TPU kernel")
@pytest.mark.parametrize("m", [8, 24, 64])
@pytest.mark.parametrize("n,k", [(1536, 6144), (768, 6144), (6144, 1024), (6144, 384)])
def test_kernel_matches_bf16_matmul(m, n, k):
    from vllm_torchtpu.models.minimax_m3.kernels.mxfp8_w8a16 import mxfp8_w8a16_matmul

    w8, s = _mxfp8(n, k, seed=m)
    w_kn, scale = mxfp8_to_w8a16(w8, s)
    to_jax = lambda t: jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    x = jnp.asarray(np.random.default_rng(m).standard_normal((m, k)), jnp.bfloat16)
    out = mxfp8_w8a16_matmul(x, to_jax(w_kn), jnp.asarray(scale.float().numpy(), jnp.bfloat16))
    w_bf16 = jnp.asarray(dequantize_mxfp8(w8, s).t().float().numpy(), jnp.bfloat16)
    ref = jnp.matmul(x, w_bf16, preferred_element_type=jnp.float32)
    # Same bf16 products; only the f32 accumulation order differs (K chunks).
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)), np.asarray(ref), rtol=1e-2, atol=1e-2
    )
