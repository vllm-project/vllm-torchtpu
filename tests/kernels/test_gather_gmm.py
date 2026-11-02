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
"""Gather-fused MoE kernels, in the TPU interpreter against a JAX reference.

The reference repeats gmm_v2's per-row arithmetic (512-column FP8 activation
quantization, BF16 accumulation, the fused activation, the down projection)
with the routed rows materialized, so a mismatch is a gather, tiling or
packing bug in the kernels.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import gather_gmm as G
from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_gated_act

_VMEM = 64 * 1024 * 1024


def _quantize(block):
    amax = jnp.max(jnp.abs(block), axis=1, keepdims=True)
    scale = amax / float(jnp.finfo(jnp.float8_e4m3fn).max)
    inv = jnp.where(scale == 0, 0, 1 / scale)
    return (block * inv).astype(jnp.float8_e4m3fn), scale.astype(jnp.bfloat16)


def _expert_rows(xs, w1, w1s, w2, w2s, act):
    """gmm_v2 arithmetic for rows that all belong to one expert."""
    h = xs.shape[1]
    n1 = w1.shape[1]
    acc = jnp.zeros((xs.shape[0], n1), jnp.bfloat16)
    for k0 in range(0, h, G.QUANT_BLOCK):
        k1 = min(h, k0 + G.QUANT_BLOCK)
        q, sc = _quantize(xs[:, k0:k1])
        for n0 in range(0, n1, 256):
            prod = jnp.dot(
                q, w1[k0:k1, n0 : n0 + 256], preferred_element_type=jnp.float32
            )
            prod = (
                prod.astype(jnp.bfloat16)
                * sc
                * w1s[:, n0 : n0 + 256].astype(jnp.bfloat16)
            )
            acc = acc.at[:, n0 : n0 + 256].add(prod)
    inter = n1 // 2
    mid = apply_gated_act(acc[:, :inter], acc[:, inter:], act)
    q, sc = _quantize(mid)
    out = jnp.dot(q, w2, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    return out * sc * w2s.astype(jnp.bfloat16)


def _reference(x, token_ids, group_sizes, w1, w1s, w2, w2s, act):
    xs = x[token_ids]
    outs, start = [], 0
    for e, n in enumerate(np.asarray(group_sizes)):
        outs.append(
            _expert_rows(xs[start : start + n], w1[e], w1s[e], w2[e], w2s[e], act)
        )
        start += int(n)
    return jnp.concatenate(outs, axis=0)


def _routing(key, tokens, experts, topk, skew):
    logits = jax.random.normal(key, (tokens, experts)) + skew * jnp.arange(experts)
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
    flat = ids.reshape(-1)
    order = jnp.argsort(flat, stable=True)
    group_sizes = jnp.bincount(flat, length=experts).astype(jnp.int32)
    positions = jnp.argsort(order).reshape(tokens, topk).astype(jnp.int32)
    return (
        (order // topk).astype(jnp.int32),
        group_sizes,
        positions,
        weights.astype(jnp.bfloat16),
    )


def _weights(key, experts, hidden, inter):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w1 = (jax.random.normal(k1, (experts, hidden, 2 * inter)) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    w2 = (jax.random.normal(k2, (experts, inter, hidden)) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    w1s = jax.random.uniform(k3, (experts, 1, 2 * inter), minval=0.01, maxval=0.03)
    w2s = jax.random.uniform(k4, (experts, 1, hidden), minval=0.01, maxval=0.03)
    return w1, w1s, w2, w2s


def test_pack_rows_roundtrip():
    x = jax.random.normal(jax.random.PRNGKey(0), (6, 768)).astype(jnp.bfloat16)
    packed = G.pack_rows(x)
    assert packed.shape == (6 * 3, 128) and packed.dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(G.unpack_rows(packed, 6)), np.asarray(x))


@pytest.mark.parametrize(
    "tokens,experts,topk,tile_m,skew",
    [
        (64, 8, 2, 64, 0.0),  # several tiles per expert, partial last tiles
        (48, 8, 4, 64, 0.5),  # skewed routing: empty and multi-tile experts
        (32, 4, 1, 32, 3.0),  # one expert takes nearly every row
    ],
)
def test_gather_fused_moe_matches_reference(tokens, experts, topk, tile_m, skew):
    hidden, inter = 768, 128  # 1.5 quantization blocks, like hidden 2816
    kx, kr, kw = jax.random.split(jax.random.PRNGKey(1), 3)
    x = (jax.random.normal(kx, (tokens, hidden)) * 2.0).astype(jnp.bfloat16)
    token_ids, group_sizes, _, _ = _routing(kr, tokens, experts, topk, skew)
    w1, w1s, w2, w2s = _weights(kw, experts, hidden, inter)
    out = G.gather_fused_moe(
        G.pack_rows(x),
        token_ids,
        group_sizes,
        w1,
        w1s,
        w2,
        w2s,
        act="gelu_tanh",
        tile_m=tile_m,
        height_step=tile_m // 2,
        vmem_limit_bytes=_VMEM,
        interpret=True,
    )
    rows = tokens * topk
    got = G.unpack_rows(out, rows)
    want = _reference(x, token_ids, group_sizes, w1, w1s, w2, w2s, "gelu_tanh")
    assert np.isfinite(np.asarray(want, np.float32)).all()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_gather_combine_matches_reference():
    tokens, topk, hidden = 64, 4, 768
    kr, ky = jax.random.split(jax.random.PRNGKey(2))
    _, _, positions, weights = _routing(kr, tokens, 16, topk, 0.0)
    y = jax.random.normal(ky, (tokens * topk, hidden)).astype(jnp.bfloat16)
    out = G.gather_combine(
        G.pack_rows(y),
        positions,
        weights,
        hidden_size=hidden,
        tile_tokens=16,
        interpret=True,
    )
    gathered = (
        y[positions.reshape(-1)].reshape(tokens, topk, hidden).astype(jnp.float32)
    )
    acc = jnp.zeros((tokens, hidden), jnp.float32)
    for k in range(topk):
        prod = gathered[:, k] * weights[:, k : k + 1].astype(jnp.float32)
        acc = acc + prod.astype(jnp.bfloat16).astype(jnp.float32)
    want = acc.astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(want))


def test_routed_moe_end_to_end():
    """Routing permutation -> gather_fused_moe -> gather_combine vs the reference."""
    tokens, experts, topk, hidden, inter = 64, 8, 4, 768, 128
    kx, kr, kw = jax.random.split(jax.random.PRNGKey(3), 3)
    x = jax.random.normal(kx, (tokens, hidden)).astype(jnp.bfloat16)
    token_ids, group_sizes, positions, weights = _routing(
        kr, tokens, experts, topk, 0.3
    )
    w1, w1s, w2, w2s = _weights(kw, experts, hidden, inter)
    rows = G.gather_fused_moe(
        G.pack_rows(x),
        token_ids,
        group_sizes,
        w1,
        w1s,
        w2,
        w2s,
        act="gelu_tanh",
        tile_m=64,
        height_step=32,
        vmem_limit_bytes=_VMEM,
        interpret=True,
    )
    out = G.gather_combine(
        rows, positions, weights, hidden_size=hidden, tile_tokens=16, interpret=True
    )
    y = _reference(x, token_ids, group_sizes, w1, w1s, w2, w2s, "gelu_tanh")
    gathered = (
        y[positions.reshape(-1)].reshape(tokens, topk, hidden).astype(jnp.float32)
    )
    acc = jnp.zeros((tokens, hidden), jnp.float32)
    for k in range(topk):
        prod = gathered[:, k] * weights[:, k : k + 1].astype(jnp.float32)
        acc = acc + prod.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(acc.astype(jnp.bfloat16)))
