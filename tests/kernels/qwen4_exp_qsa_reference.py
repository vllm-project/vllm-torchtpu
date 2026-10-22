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
"""Test-only JAX oracle for native QSA state and attention comparisons.

This module is never imported by the model or production kernels. It preserves
the dense selection/gather math to check the shared StreamIndex/RPA path.

JAX address inputs are bounded int32. Persistent RoPE tags cross the bridge as
an explicit pair of int32 bit planes; their public owner is always int64.
"""

import jax
import jax.numpy as jnp


def bf16(x):
    # Integer RNE makes otherwise removable BF16 boundaries observable to XLA.
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    bits = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def gemma_norm(x, weight, norm_eps):
    x = x.astype(jnp.float32)
    return bf16(
        x
        * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + norm_eps)
        * (1 + weight.astype(jnp.float32))
    )


def _fma(a, b, c):
    product = a.astype(jnp.float32) * b.astype(jnp.float32)
    c = c.astype(jnp.float32)
    high = jax.lax.optimization_barrier(product + c)
    z = jax.lax.optimization_barrier(high - product)
    residual = (product - (high - z)) + (c - z)
    bits = jax.lax.bitcast_convert_type(high, jnp.uint32)
    tie = (bits & jnp.uint32(0xFFFF)) == jnp.uint32(0x8000)
    adjusted = jnp.where(
        tie & (residual > 0),
        jnp.nextafter(high, jnp.float32(jnp.inf)),
        jnp.where(
            tie & (residual < 0), jnp.nextafter(high, jnp.float32(-jnp.inf)), high
        ),
    )
    return bf16(adjusted)


def _rope_parts(x, positions, cache):
    cs = cache[positions].astype(jnp.float32)
    half = cache.shape[-1]
    shape = (len(x),) + (1,) * (x.ndim - 2) + (half,)
    return (
        x[..., :half].astype(jnp.float32),
        x[..., half : 2 * half].astype(jnp.float32),
        cs[:, 0].reshape(shape),
        cs[:, 1].reshape(shape),
    )


def main_rope(x, positions, cache):
    """Main rotary uses promoted BF16 trig and FP32 two-product arithmetic."""
    a, b, c, s = _rope_parts(x, positions, cache)
    return jnp.concatenate(
        [bf16(a * c - b * s), bf16(b * c + a * s), x[..., 2 * cache.shape[-1] :]], -1
    )


def index_rope(x, positions, cache):
    """Historical compensated index-RoPE comparison reference."""
    a, b, c, s = _rope_parts(x, positions, cache)
    return jnp.concatenate(
        [
            _fma(a, c, -bf16(b * s)),
            _fma(a, s, bf16(b * c)),
            x[..., 2 * cache.shape[-1] :],
        ],
        -1,
    )


def produce(qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps):
    head_dim, index_dim = nq.shape[0], niq.shape[0]
    qgate = qgate.reshape(len(qgate), qgate.shape[1] // (2 * head_dim), 2 * head_dim)
    key = key.reshape(len(key), key.shape[1] // head_dim, head_dim)
    value = value.reshape(len(value), value.shape[1] // head_dim, head_dim)
    q = main_rope(gemma_norm(qgate[..., :head_dim], nq, norm_eps), positions, trig)
    key = main_rope(gemma_norm(key, nk, norm_eps), positions, trig)
    index_heads = index_qk.shape[1] // index_dim - 1
    iq = index_rope(
        gemma_norm(
            index_qk[:, :-index_dim].reshape(len(index_qk), index_heads, index_dim),
            niq,
            norm_eps,
        ),
        positions,
        trig,
    )
    return q, key, value, qgate[..., head_dim:], iq, index_qk[:, -index_dim:]


def _token_metadata(count, starts, context, slots):
    t = jnp.arange(count, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), len(slots) - 1)
    logical = context[req] + t - starts[req]
    valid = (t < starts[-1]) & (slots[req] >= 0)
    return req, logical, valid


def quantize_kv(value, scale):
    """CUDA reshape_and_cache: scalar FP32 division and saturated E4M3 RNE."""
    encoded = value.astype(jnp.float32) / jnp.float32(scale)
    return jnp.clip(encoded, -448, 448).astype(jnp.float8_e4m3fn)


def update_caches(
    key,
    value,
    raw,
    positions,
    main_k,
    main_v,
    compressed,
    ring,
    tags,
    norm_k,
    trig,
    starts,
    context,
    slots,
    main_table,
    compressed_table,
    compress_ratio,
    norm_eps,
):
    """Close groups using old ring + current chunk BEFORE updating the ring.

    Ring has compress_ratio logical lanes and backend-padded rows; tags [...,2] are the
    low/high int32 bit planes of public int64 RoPE positions. Only group-end
    tokens write compressed rows. Native compressed page span remains 128.
    """
    count = len(raw)
    if count == 0 or len(slots) == 0:
        return main_k, main_v, compressed, ring, tags
    req, logical, valid = _token_metadata(count, starts, context, slots)
    block_size = main_k.shape[1]
    page = main_table[req, logical // block_size]
    page = jnp.where(valid, page, main_k.shape[0])
    main_k = main_k.at[page, logical % block_size].set(key, mode="drop")
    main_v = main_v.at[page, logical % block_size].set(value, mode="drop")

    compressed, ring, tags = _update_indexer(
        raw,
        positions,
        compressed,
        ring,
        tags,
        norm_k,
        trig,
        starts,
        context,
        slots,
        compressed_table,
        compress_ratio,
        norm_eps,
    )
    return main_k, main_v, compressed, ring, tags


def _update_indexer(
    raw,
    positions,
    compressed,
    ring,
    tags,
    norm_k,
    trig,
    starts,
    context,
    slots,
    compressed_table,
    compress_ratio,
    norm_eps,
):
    count = len(raw)
    if count == 0 or len(slots) == 0:
        return compressed, ring, tags
    req, logical, valid = _token_metadata(count, starts, context, slots)
    safe_slots = jnp.maximum(slots, 0)
    ratio = compress_ratio
    groups_per_page = compressed.shape[1]
    group_start = logical // ratio * ratio
    members = group_start[:, None] + jnp.arange(ratio)[None]
    chunk_indices = starts[req, None] + members - context[req, None]
    current = raw[jnp.clip(chunk_indices, 0, count - 1)]
    previous = ring[safe_slots[req, None], members % ratio]
    previous = jnp.where((context[req] > 0)[:, None, None], previous, 0)
    joined = jnp.where(
        (chunk_indices >= starts[req, None])[..., None], current, previous
    )
    pooled = bf16(jnp.mean(joined.astype(jnp.float32), axis=1))
    first_index = starts[req] + group_start - context[req]
    old_position = tags[safe_slots[req], group_start % ratio, 0]
    first_position = jnp.where(
        first_index >= starts[req],
        positions[jnp.clip(first_index, 0, count - 1)],
        old_position,
    )
    completed = index_rope(
        gemma_norm(pooled, norm_k, norm_eps), jnp.maximum(first_position, 0), trig
    )
    complete = valid & (logical % ratio == ratio - 1)
    cp = jnp.where(
        complete,
        compressed_table[req, logical // (groups_per_page * ratio)],
        compressed.shape[0],
    )
    compressed = compressed.at[cp, (logical // ratio) % groups_per_page, 0].set(
        completed, mode="drop"
    )

    # Exactly one scatter per request/lane: no duplicate-index last-writer race.
    lengths = starts[1:] - starts[:-1]
    end = context + lengths
    lanes = jnp.arange(ring.shape[1])[None]
    last = end[:, None] - 1 - ((end[:, None] - 1 - lanes) % ratio)
    local = starts[:-1, None] + last - context[:, None]
    old_ring = ring[safe_slots]
    old_tags = tags[safe_slots]
    old_ring = jnp.where((context > 0)[:, None, None], old_ring, 0)
    old_tags = jnp.where((context > 0)[:, None, None], old_tags, -1)
    new_lane = (lanes < ratio) & (last >= context[:, None]) & (last < end[:, None])
    rr = jnp.where(new_lane[..., None], raw[jnp.clip(local, 0, count - 1)], old_ring)
    pp = positions[jnp.clip(local, 0, count - 1)]
    tag_pairs = jnp.stack([pp, jnp.zeros_like(pp)], -1)
    tt = jnp.where(new_lane[..., None], tag_pairs, old_tags)
    dest = jnp.where((lengths > 0) & (slots >= 0), slots, ring.shape[0])
    ring = ring.at[dest].set(rr, mode="drop")
    tags = tags.at[dest].set(tt, mode="drop")
    return compressed, ring, tags


def pack_fp8_kv(key, value):
    """Paired FP8 head bytes -> INT32 words, with no arithmetic conversion."""
    joined = jnp.concatenate((key, value), axis=-1)
    return jax.lax.bitcast_convert_type(
        joined.reshape(*joined.shape[:-1], joined.shape[-1] // 4, 4), jnp.int32
    )


def unpack_fp8_kv(words):
    joined = jax.lax.bitcast_convert_type(words, jnp.float8_e4m3fn).reshape(
        *words.shape[:-1], words.shape[-1] * 4
    )
    head_dim = joined.shape[-1] // 2
    return joined[..., :head_dim], joined[..., head_dim:]


def update_caches_fp8(
    key,
    value,
    raw,
    positions,
    packed_kv,
    compressed,
    ring,
    tags,
    norm_k,
    trig,
    starts,
    context,
    slots,
    main_table,
    compressed_table,
    compress_ratio,
    norm_eps,
    k_scale,
    v_scale,
):
    if len(raw) == 0 or len(slots) == 0:
        return packed_kv, compressed, ring, tags
    req, logical, valid = _token_metadata(len(raw), starts, context, slots)
    block_size = packed_kv.shape[1]
    page = jnp.where(valid, main_table[req, logical // block_size], packed_kv.shape[0])
    key = quantize_kv(key, k_scale)
    value = quantize_kv(value, v_scale)
    packed_kv = packed_kv.at[page, logical % block_size].set(
        pack_fp8_kv(key, value), mode="drop"
    )
    compressed, ring, tags = _update_indexer(
        raw,
        positions,
        compressed,
        ring,
        tags,
        norm_k,
        trig,
        starts,
        context,
        slots,
        compressed_table,
        compress_ratio,
        norm_eps,
    )
    return packed_kv, compressed, ring, tags


def select_groups(query, compressed, logical, compress_ratio, token_topk):
    """Rank visible compressed groups; lower group ID breaks exact ties."""
    groups = compressed.shape[1]
    block_topk = token_topk // compress_ratio
    padding = max(block_topk, groups) - groups
    compressed = jnp.pad(compressed, ((0, 0), (0, padding), (0, 0)))
    ids = jnp.arange(groups + padding, dtype=jnp.int32)
    scores = jnp.maximum(
        jnp.einsum(
            "qhd,qgd->qhg",
            query,
            compressed,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ),
        0,
    ).sum(1) / jnp.sqrt(jnp.float32(query.shape[-1]))
    visible = (logical + 1) // compress_ratio
    scores = jnp.where(
        (ids[None] < visible[:, None]) & (ids[None] < groups), scores, -jnp.inf
    )
    _, order = jax.lax.sort(
        (-scores, jnp.broadcast_to(ids, scores.shape)), dimension=1, num_keys=2
    )
    count = jnp.minimum(visible, block_topk)
    selected = jnp.where(
        jnp.arange(block_topk)[None] < count[:, None], order[:, :block_topk], -1
    )
    tokens = jnp.where(
        selected[:, :, None] >= 0,
        selected[:, :, None] * compress_ratio + jnp.arange(compress_ratio),
        -1,
    ).reshape(len(query), token_topk)
    result = jnp.pad(tokens, ((0, 0), (0, compress_ratio - 1)), constant_values=-1)
    for tail in range(compress_ratio - 1):
        token = visible * compress_ratio + tail
        result = result.at[jnp.arange(len(query)), count * compress_ratio + tail].set(
            jnp.where(token <= logical, token, -1)
        )
    return result


def qsa_attention(
    q,
    iq,
    gate,
    main_k,
    main_v,
    compressed,
    starts,
    context,
    slots,
    main_table,
    compressed_table,
    compress_ratio,
    token_topk,
    k_scale=None,
    v_scale=None,
    packed_kv=None,
):
    """Only selected-token softmax; at most eight queries gather KV together."""
    count = len(q)
    selected_width = token_topk + compress_ratio - 1
    head_dim = q.shape[-1]
    block_size = (main_k if packed_kv is None else packed_kv).shape[1]
    groups_per_page = compressed.shape[1]
    if count == 0 or len(slots) == 0:
        return jnp.zeros_like(q), jnp.full((count, selected_width), -1, jnp.int32)
    req, logical, valid = _token_metadata(count, starts, context, slots)
    padded = (-count) % 8

    def pad(x, value=0):
        return jnp.pad(
            x, ((0, padded),) + ((0, 0),) * (x.ndim - 1), constant_values=value
        )

    args = (
        pad(q).reshape(-1, 8, q.shape[1], head_dim),
        pad(iq).reshape(-1, 8, iq.shape[1], iq.shape[2]),
        pad(gate).reshape(-1, 8, q.shape[1], head_dim),
        pad(req).reshape(-1, 8),
        pad(logical).reshape(-1, 8),
        pad(valid, False).reshape(-1, 8),
    )
    group_ids = jnp.arange(compressed_table.shape[1] * groups_per_page, dtype=jnp.int32)
    # CUDA folds layer scales into QK and the output normalizer. Do not
    # multiply decoded FP8 values by a scale and round them back to BF16.
    softmax_scale = head_dim**-0.5 * (1.0 if packed_kv is None else k_scale)
    output_scale = 1.0 if packed_kv is None else v_scale
    heads = (main_k if packed_kv is None else packed_kv).shape[2]
    head_map = jnp.arange(q.shape[1]) // (q.shape[1] // heads)

    def attend(args):
        qq, indexq, gg, rr, pp, vv = args
        pages = compressed_table[rr[:, None], group_ids[None] // groups_per_page]
        ck = compressed[pages, group_ids[None] % groups_per_page, 0]
        selected = select_groups(indexq, ck, pp, compress_ratio, token_topk)
        selected = jnp.where(vv[:, None], selected, -1)
        safe = jnp.maximum(selected, 0)
        pages = main_table[rr[:, None], safe // block_size]
        if packed_kv is None:
            key = main_k[pages, safe % block_size]
            value = main_v[pages, safe % block_size]
        else:
            key, value = unpack_fp8_kv(packed_kv[pages, safe % block_size])
            key = key.astype(jnp.bfloat16)
            value = value.astype(jnp.float16)
        key, value = key[:, :, head_map, :], value[:, :, head_map, :]
        score = (
            jnp.einsum(
                "qhd,qshd->qhs",
                qq,
                key,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            * softmax_scale
        )
        score = jnp.where((selected >= 0)[:, None, :], score, -jnp.inf)
        # Invalid padded queries have zero probability, avoiding all--inf NaNs.
        maximum = jnp.max(score, axis=-1, keepdims=True)
        ex = jnp.exp(score - jnp.where(vv[:, None, None], maximum, 0))
        normalizer = ex.sum(-1, keepdims=True)
        if packed_kv is None:
            prob = ex / jnp.maximum(normalizer, jnp.float32(1e-30))
            multiplier = 1.0
        else:
            # Match CUDA's FP16 PV operands and scale after accumulation.
            # Normalizing probabilities before the FP16 cast rounds differently.
            prob = ex.astype(jnp.float16)
            multiplier = output_scale / jnp.maximum(normalizer, jnp.float32(1e-20))
        output = bf16(
            jnp.einsum(
                "qhs,qshd->qhd",
                prob,
                value,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            * multiplier
        )
        output = bf16(output * bf16(jax.nn.sigmoid(gg.astype(jnp.float32))))
        return jnp.where(vv[:, None, None], output, 0), selected

    output, selected = jax.lax.map(attend, args)
    return output.reshape(-1, q.shape[1], head_dim)[:count], selected.reshape(
        -1, selected_width
    )[:count]


def qsa_step(
    qgate: jax.Array,
    key: jax.Array,
    value: jax.Array,
    index_qk: jax.Array,
    nq: jax.Array,
    nk: jax.Array,
    niq: jax.Array,
    nik: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    main_k: jax.Array,
    main_v: jax.Array,
    compressed: jax.Array,
    ring: jax.Array,
    tags: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    slots: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    q, k, v, gate, iq, raw = produce(
        qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps
    )
    state = update_caches(
        k,
        v,
        raw,
        positions,
        main_k,
        main_v,
        compressed,
        ring,
        tags,
        nik,
        trig,
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        norm_eps,
    )
    output, selected = qsa_attention(
        q,
        iq,
        gate,
        *state[:3],
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
    )
    return (*state, output.reshape(len(qgate), qgate.shape[1] // 2), selected)


def qsa_step_fp8(
    qgate: jax.Array,
    key: jax.Array,
    value: jax.Array,
    index_qk: jax.Array,
    nq: jax.Array,
    nk: jax.Array,
    niq: jax.Array,
    nik: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    packed_kv: jax.Array,
    compressed: jax.Array,
    ring: jax.Array,
    tags: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    slots: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    k_scale: float,
    v_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    q, k, v, gate, iq, raw = produce(
        qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps
    )
    state = update_caches_fp8(
        k,
        v,
        raw,
        positions,
        packed_kv,
        compressed,
        ring,
        tags,
        nik,
        trig,
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        norm_eps,
        k_scale,
        v_scale,
    )
    output, selected = qsa_attention(
        q,
        iq,
        gate,
        None,
        None,
        state[1],
        starts,
        context,
        slots,
        main_table,
        compressed_table,
        compress_ratio,
        token_topk,
        k_scale=k_scale,
        v_scale=v_scale,
        packed_kv=state[0],
    )
    return (*state, output.reshape(len(qgate), qgate.shape[1] // 2), selected)
