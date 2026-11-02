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


def _pack_rows(x):
    """BF16 [n, hidden] -> the packed uint32 [n * hidden // 256, 128] row form."""
    n, h = x.shape
    s = G.packed_sublanes(h)
    bits = jax.lax.bitcast_convert_type(x.reshape(n, s, 2, 128), jnp.uint16)
    bits = bits.astype(jnp.uint32)
    return (bits[:, :, 0, :] | (bits[:, :, 1, :] << 16)).reshape(n * s, 128)


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
    out = jnp.zeros((xs.shape[0], w2.shape[1]), jnp.bfloat16)
    for k0 in range(0, inter, G.QUANT_BLOCK):
        k1 = min(inter, k0 + G.QUANT_BLOCK)
        q, sc = _quantize(mid[:, k0:k1])
        prod = jnp.dot(q, w2[k0:k1], preferred_element_type=jnp.float32)
        out = out + prod.astype(jnp.bfloat16) * sc * w2s.astype(jnp.bfloat16)
    return out


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
    packed = _pack_rows(x)
    assert packed.shape == (6 * 3, 128) and packed.dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(G.unpack_rows(packed, 6)), np.asarray(x))


@pytest.mark.parametrize("tokens", [32, 1088])  # 1088 = 17 x 64: not a multiple of 256
def test_quantize_pack_rows_layout(tokens):
    hidden = 768
    x = (jax.random.normal(jax.random.PRNGKey(4), (tokens, hidden)) * 3).astype(
        jnp.bfloat16
    )
    x = x.at[5].set(0)  # an all-zero row has scale 0 and scale_inv 0
    packed = G.quantize_pack_rows(x, interpret=True)
    rows = np.asarray(packed).reshape(tokens, 4, 128)
    want_scaled, want_scales = [], []
    for k0 in range(0, hidden, G.QUANT_BLOCK):
        scaled, scale = jax.jit(G._scale_rows)(x[:, k0 : k0 + G.QUANT_BLOCK])
        want_scaled.append(scaled)
        want_scales.append(scale.astype(jnp.float32))
    got_scaled = G.unpack_rows(jnp.asarray(rows[:, :3].reshape(-1, 128)), tokens)
    np.testing.assert_array_equal(
        np.asarray(got_scaled), np.asarray(jnp.concatenate(want_scaled, axis=1))
    )
    got_scales = rows[:, 3, :2].view(np.float32)
    np.testing.assert_array_equal(
        got_scales, np.asarray(jnp.concatenate(want_scales, 1))
    )
    assert not rows[:, 3, 2:].any()


@pytest.mark.parametrize(
    "tokens,experts,topk,tile_m,skew,inter",
    [
        (64, 8, 2, 64, 0.0, 128),  # several tiles per expert, partial last tiles
        (48, 8, 4, 64, 0.5, 128),  # skewed routing: empty and multi-tile experts
        (32, 4, 1, 32, 3.0, 128),  # one expert takes nearly every row
        (64, 8, 2, 64, 0.0, 640),  # intermediate of 1.25 quantization blocks
    ],
)
def test_gather_fused_moe_matches_reference(tokens, experts, topk, tile_m, skew, inter):
    hidden = 768  # 1.5 quantization blocks, like hidden 2816
    kx, kr, kw = jax.random.split(jax.random.PRNGKey(1), 3)
    x = (jax.random.normal(kx, (tokens, hidden)) * 2.0).astype(jnp.bfloat16)
    token_ids, group_sizes, _, _ = _routing(kr, tokens, experts, topk, skew)
    w1, w1s, w2, w2s = _weights(kw, experts, hidden, inter)
    out = G.gather_fused_moe(
        G.quantize_pack_rows(x, tile_tokens=16, interpret=True),
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
        _pack_rows(y),
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
        acc = acc + prod
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
        G.quantize_pack_rows(x, tile_tokens=16, interpret=True),
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
        acc = acc + prod
    np.testing.assert_array_equal(np.asarray(out), np.asarray(acc.astype(jnp.bfloat16)))


def test_gather_combine_masked_routes():
    """Negative positions contribute zero, even when the buffer slot is stale NaN."""
    tokens, topk, hidden, tt = 48, 4, 768, 16  # three tiles: slots 0, 1, 0
    kr, ky, km = jax.random.split(jax.random.PRNGKey(6), 3)
    _, _, positions, weights = _routing(kr, tokens, 16, topk, 0.0)
    valid = jax.random.bernoulli(km, 0.5, (tokens, topk))
    # Route (t=0, k=0) of tile 0 is valid and reads a NaN row; the same slot
    # position in tile 2 (t=32, k=0) is skipped, so that slot is stale there.
    valid = valid.at[0, 0].set(True).at[2 * tt, 0].set(False)
    y = jax.random.normal(ky, (tokens * topk, hidden)).astype(jnp.bfloat16)
    y = y.at[positions[0, 0]].set(jnp.nan)
    out = G.gather_combine(
        _pack_rows(y),
        jnp.where(valid, positions, -1),
        weights,
        hidden_size=hidden,
        tile_tokens=tt,
        masked=True,
        interpret=True,
    )
    gathered = (
        y[positions.reshape(-1)].reshape(tokens, topk, hidden).astype(jnp.float32)
    )
    acc = jnp.zeros((tokens, hidden), jnp.float32)
    for k in range(topk):
        prod = gathered[:, k] * weights[:, k : k + 1].astype(jnp.float32)
        acc = acc + jnp.where(valid[:, k : k + 1], prod, 0.0)
    got = np.asarray(out, np.float32)
    assert np.isnan(got[0]).all() and np.isfinite(got[1:]).all()
    np.testing.assert_array_equal(
        got[1:], np.asarray(acc.astype(jnp.bfloat16), np.float32)[1:]
    )


@pytest.mark.parametrize("skip_padded", [False, True])
def test_masked_routes_end_to_end(skip_padded):
    """EP-style routing (-1 ids) and zero-weight padded tokens through the
    routing permutation, the expert kernel and the masked combine."""
    tokens, experts, topk, hidden, inter = 64, 8, 4, 768, 128
    kx, kr, kw = jax.random.split(jax.random.PRNGKey(7), 3)
    x = jax.random.normal(kx, (tokens, hidden)).astype(jnp.bfloat16)
    logits = jax.random.normal(kr, (tokens, 2 * experts))
    weights, global_ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
    weights = weights.astype(jnp.bfloat16)
    # This rank owns global experts [0, experts); the others are remote.
    ids = jnp.where(global_ids < experts, global_ids, -1).astype(jnp.int32)
    if skip_padded:
        weights = weights.at[-8:].set(0)  # padded tokens
    valid = (ids >= 0) & (weights != 0) if skip_padded else ids >= 0
    flat_valid = valid.reshape(-1)
    keys = jnp.where(flat_valid, ids.reshape(-1), experts)
    order = jnp.argsort(keys, stable=True)
    group_sizes = jnp.bincount(keys, length=experts + 1)[:experts].astype(jnp.int32)
    positions = jnp.argsort(order).reshape(tokens, topk).astype(jnp.int32)
    w1, w1s, w2, w2s = _weights(kw, experts, hidden, inter)
    rows = G.gather_fused_moe(
        G.quantize_pack_rows(x, interpret=True),
        (order // topk).astype(jnp.int32),
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
        rows,
        jnp.where(valid, positions, -1),
        weights,
        hidden_size=hidden,
        tile_tokens=16,
        masked=True,
        interpret=True,
    )
    n_valid = int(group_sizes.sum())
    y = _reference(
        x, order[:n_valid] // topk, group_sizes, w1, w1s, w2, w2s, "gelu_tanh"
    )
    y = jnp.concatenate([y, jnp.full((len(order) - n_valid, hidden), jnp.nan, y.dtype)])
    gathered = (
        y[positions.reshape(-1)].reshape(tokens, topk, hidden).astype(jnp.float32)
    )
    acc = jnp.zeros((tokens, hidden), jnp.float32)
    for k in range(topk):
        prod = gathered[:, k] * weights[:, k : k + 1].astype(jnp.float32)
        acc = acc + jnp.where(valid[:, k : k + 1], prod, 0.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(acc.astype(jnp.bfloat16)))
