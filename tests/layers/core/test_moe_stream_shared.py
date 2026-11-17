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
"""The shared expert computed inside fused_moe_func (TPU_MOE_STREAM_SHARED_EXPERT)."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func, shared_expert_xla


def _shared_weights(key, hidden, fs):
    k = jax.random.split(key, 4)
    sw1 = (jax.random.normal(k[0], (hidden, 2 * fs)) * 8).astype(jnp.float8_e4m3fn)
    sw2 = (jax.random.normal(k[1], (fs, hidden)) * 8).astype(jnp.float8_e4m3fn)
    s1 = (2.0 ** jnp.round(jax.random.uniform(k[2], (2 * fs,), minval=-9, maxval=-6)))
    s2 = (2.0 ** jnp.round(jax.random.uniform(k[3], (hidden,), minval=-9, maxval=-6)))
    return sw1, s1.astype(jnp.bfloat16), sw2, s2.astype(jnp.bfloat16)


def _torch_shared_mlp(x, sw1, s1, sw2, s2):
    """MiniMaxM3MLP on MiniMaxM3Mxfp8LinearMethod's e4m3 weights."""
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    t = lambda a: torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(torch.bfloat16)
    xt = t(x)
    act = SiluAndMulWithClamp(swiglu_limit=7.0, alpha=1.702, beta=1.0)
    gu = torch.matmul(xt, t(sw1)) * t(s1)
    h = act.forward_native(gu)
    return (torch.matmul(h, t(sw2)) * t(s2)).float().numpy()


def test_shared_expert_xla_matches_torch_mlp():
    hidden, fs, n = 256, 128, 8
    sw1, s1, sw2, s2 = _shared_weights(jax.random.key(0), hidden, fs)
    x = jax.random.uniform(jax.random.key(1), (n, hidden), minval=-1, maxval=1).astype(
        jnp.bfloat16
    )
    ref = _torch_shared_mlp(x, sw1, s1, sw2, s2)
    got = np.asarray(
        shared_expert_xla(
            x, sw1, s1.reshape(1, -1).astype(jnp.float32), sw2,
            s2.reshape(1, -1).astype(jnp.float32),
        ).astype(jnp.float32)
    )
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() <= 1e-2 * scale
    assert np.abs(got - ref).mean() <= 1e-3 * scale


def test_fused_moe_func_adds_shared_in_bf16():
    num_tokens, hidden, inter, experts, topk, fs = 8, 256, 128, 4, 2, 128
    k = jax.random.split(jax.random.key(2), 4)
    x = jax.random.uniform(k[0], (num_tokens, hidden), minval=-1, maxval=1).astype(
        jnp.bfloat16
    )
    w1 = (jax.random.normal(k[1], (experts, hidden, 2 * inter)) / 16).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k[2], (experts, inter, hidden)) / 16).astype(jnp.bfloat16)
    topk_weights = jnp.full((num_tokens, topk), 0.5, jnp.float32)
    topk_ids = jnp.stack(
        [jnp.arange(num_tokens) % experts, (jnp.arange(num_tokens) + 1) % experts], 1
    ).astype(jnp.int32)
    sw1, s1, sw2, s2 = _shared_weights(k[3], hidden, fs)
    common = dict(
        hidden_states=x, w1=w1, w2=w2, w1_scale=None, w2_scale=None,
        w1_bias=None, w2_bias=None, topk_weights=topk_weights, topk_ids=topk_ids,
        topk=topk, activation="silu",
    )
    base = fused_moe_func(**common)
    got = fused_moe_func(**common, shared=(sw1, s1, sw2, s2))
    sh = shared_expert_xla(
        x, sw1, s1.reshape(1, -1).astype(jnp.float32), sw2,
        s2.reshape(1, -1).astype(jnp.float32),
    )
    assert got.dtype == base.dtype
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)),
        np.asarray((sh.astype(base.dtype) + base).astype(jnp.float32)),
    )
