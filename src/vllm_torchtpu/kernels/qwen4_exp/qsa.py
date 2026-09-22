# SPDX-License-Identifier: Apache-2.0
"""QSA BF16 producers, request-owned updates and bounded sparse attention.

JAX address inputs are bounded int32. Persistent RoPE tags cross the bridge as
an explicit pair of int32 bit planes; their public owner is always int64.
"""

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


def bf16(x):
    # Integer RNE makes otherwise removable BF16 boundaries observable to XLA.
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    bits = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def gemma_norm(x, weight):
    x = x.astype(jnp.float32)
    return bf16(
        x
        * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
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
    shape = (len(x),) + (1,) * (x.ndim - 2) + (32,)
    return (
        x[..., :32].astype(jnp.float32),
        x[..., 32:64].astype(jnp.float32),
        cs[:, 0].reshape(shape),
        cs[:, 1].reshape(shape),
    )


def main_rope(x, positions, cache):
    """Main rotary64 uses promoted BF16 trig and FP32 two-product arithmetic."""
    a, b, c, s = _rope_parts(x, positions, cache)
    return jnp.concatenate([bf16(a * c - b * s), bf16(b * c + a * s), x[..., 64:]], -1)


def index_rope(x, positions, cache):
    """Frozen Triton3.8 contract B(a*c-B(b*s)), B(a*s+B(b*c))."""
    a, b, c, s = _rope_parts(x, positions, cache)
    return jnp.concatenate(
        [_fma(a, c, -bf16(b * s)), _fma(a, s, bf16(b * c)), x[..., 64:]], -1
    )


def produce(qgate, key, value, index_qk, nq, nk, niq, positions, trig):
    qgate = qgate.reshape(len(qgate), qgate.shape[1] // 512, 512)
    key = key.reshape(len(key), key.shape[1] // 256, 256)
    value = value.reshape(len(value), value.shape[1] // 256, 256)
    q = main_rope(gemma_norm(qgate[..., :256], nq), positions, trig)
    key = main_rope(gemma_norm(key, nk), positions, trig)
    iq = index_rope(
        gemma_norm(index_qk[:, :512].reshape(-1, 4, 128), niq), positions, trig
    )
    return q, key, value, qgate[..., 256:], iq, index_qk[:, 512:]


def _token_metadata(count, starts, context, slots):
    t = jnp.arange(count, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), len(slots) - 1)
    logical = context[req] + t - starts[req]
    valid = (t < starts[-1]) & (slots[req] >= 0)
    return req, logical, valid


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
):
    """Close groups using old ring + current chunk BEFORE updating the ring.

    Ring has four logical lanes (eight physical rows); tags [...,2] are the
    low/high int32 bit planes of public int64 RoPE positions. Only group-end
    tokens write compressed rows. Native compressed page span remains 128.
    """
    count = len(raw)
    if count == 0 or len(slots) == 0:
        return main_k, main_v, compressed, ring, tags
    req, logical, valid = _token_metadata(count, starts, context, slots)
    safe_slots = jnp.maximum(slots, 0)
    page = main_table[req, logical // 128]
    page = jnp.where(valid, page, main_k.shape[0])
    main_k = main_k.at[page, logical % 128].set(key, mode="drop")
    main_v = main_v.at[page, logical % 128].set(value, mode="drop")

    group_start = logical // 4 * 4
    members = group_start[:, None] + jnp.arange(4)[None]
    chunk_indices = starts[req, None] + members - context[req, None]
    current = raw[jnp.clip(chunk_indices, 0, count - 1)]
    previous = ring[safe_slots[req, None], members % 4]
    previous = jnp.where((context[req] > 0)[:, None, None], previous, 0)
    joined = jnp.where(
        (chunk_indices >= starts[req, None])[..., None], current, previous
    )
    pooled = bf16(jnp.mean(joined.astype(jnp.float32), axis=1))
    first_index = starts[req] + group_start - context[req]
    old_position = tags[safe_slots[req], group_start % 4, 0]
    first_position = jnp.where(
        first_index >= starts[req],
        positions[jnp.clip(first_index, 0, count - 1)],
        old_position,
    )
    completed = index_rope(
        gemma_norm(pooled, norm_k), jnp.maximum(first_position, 0), trig
    )
    complete = valid & (logical % 4 == 3)
    cp = jnp.where(complete, compressed_table[req, logical // 128], compressed.shape[0])
    compressed = compressed.at[cp, (logical // 4) % 32, 0].set(completed, mode="drop")

    # Exactly one scatter per request/lane: no duplicate-index last-writer race.
    lengths = starts[1:] - starts[:-1]
    end = context + lengths
    lanes = jnp.arange(ring.shape[1])[None]
    last = end[:, None] - 1 - ((end[:, None] - 1 - lanes) % 4)
    local = starts[:-1, None] + last - context[:, None]
    old_ring = ring[safe_slots]
    old_tags = tags[safe_slots]
    old_ring = jnp.where((context > 0)[:, None, None], old_ring, 0)
    old_tags = jnp.where((context > 0)[:, None, None], old_tags, -1)
    new_lane = (lanes < 4) & (last >= context[:, None]) & (last < end[:, None])
    rr = jnp.where(new_lane[..., None], raw[jnp.clip(local, 0, count - 1)], old_ring)
    pp = positions[jnp.clip(local, 0, count - 1)]
    tag_pairs = jnp.stack([pp, jnp.zeros_like(pp)], -1)
    tt = jnp.where(new_lane[..., None], tag_pairs, old_tags)
    dest = jnp.where((lengths > 0) & (slots >= 0), slots, ring.shape[0])
    ring = ring.at[dest].set(rr, mode="drop")
    tags = tags.at[dest].set(tt, mode="drop")
    return main_k, main_v, compressed, ring, tags


def select_groups(query, compressed, logical):
    """Per-query compressed [Q,G,128], lower group ID breaks exact ties."""
    groups = compressed.shape[1]
    padding = max(512, groups) - groups
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
    ).sum(1) / jnp.sqrt(jnp.float32(128))
    visible = (logical + 1) // 4
    scores = jnp.where(
        (ids[None] < visible[:, None]) & (ids[None] < groups), scores, -jnp.inf
    )
    _, order = jax.lax.sort(
        (-scores, jnp.broadcast_to(ids, scores.shape)), dimension=1, num_keys=2
    )
    count = jnp.minimum(visible, 512)
    selected = jnp.where(jnp.arange(512)[None] < count[:, None], order[:, :512], -1)
    tokens = jnp.where(
        selected[:, :, None] >= 0, selected[:, :, None] * 4 + jnp.arange(4), -1
    ).reshape(-1, 2048)
    result = jnp.pad(tokens, ((0, 0), (0, 3)), constant_values=-1)
    for tail in range(3):
        token = visible * 4 + tail
        result = result.at[jnp.arange(len(query)), count * 4 + tail].set(
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
):
    """Only selected-token softmax; at most eight queries gather KV together."""
    count = len(q)
    if count == 0 or len(slots) == 0:
        return jnp.zeros_like(q), jnp.full((count, 2051), -1, jnp.int32)
    req, logical, valid = _token_metadata(count, starts, context, slots)
    padded = (-count) % 8

    def pad(x, value=0):
        return jnp.pad(
            x, ((0, padded),) + ((0, 0),) * (x.ndim - 1), constant_values=value
        )

    args = (
        pad(q).reshape(-1, 8, q.shape[1], 256),
        pad(iq).reshape(-1, 8, 4, 128),
        pad(gate).reshape(-1, 8, q.shape[1], 256),
        pad(req).reshape(-1, 8),
        pad(logical).reshape(-1, 8),
        pad(valid, False).reshape(-1, 8),
    )
    group_ids = jnp.arange(compressed_table.shape[1] * 32, dtype=jnp.int32)
    head_map = jnp.arange(q.shape[1]) // (q.shape[1] // main_k.shape[2])

    def attend(args):
        qq, indexq, gg, rr, pp, vv = args
        pages = compressed_table[rr[:, None], group_ids[None] // 32]
        ck = compressed[pages, group_ids[None] % 32, 0]
        selected = select_groups(indexq, ck, pp)
        selected = jnp.where(vv[:, None], selected, -1)
        safe = jnp.maximum(selected, 0)
        pages = main_table[rr[:, None], safe // 128]
        key = main_k[pages, safe % 128][:, :, head_map, :]
        value = main_v[pages, safe % 128][:, :, head_map, :]
        score = (
            jnp.einsum(
                "qhd,qshd->qhs",
                qq,
                key,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            / 16
        )
        score = jnp.where((selected >= 0)[:, None, :], score, -jnp.inf)
        # Invalid padded queries have zero probability, avoiding all--inf NaNs.
        maximum = jnp.max(score, axis=-1, keepdims=True)
        ex = jnp.exp(score - jnp.where(vv[:, None, None], maximum, 0))
        prob = ex / jnp.maximum(ex.sum(-1, keepdims=True), jnp.float32(1e-30))
        output = bf16(
            jnp.einsum(
                "qhs,qshd->qhd",
                prob,
                value,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        )
        output = bf16(output * bf16(jax.nn.sigmoid(gg.astype(jnp.float32))))
        return jnp.where(vv[:, None, None], output, 0), selected

    output, selected = jax.lax.map(attend, args)
    return output.reshape(-1, q.shape[1], 256)[:count], selected.reshape(-1, 2051)[
        :count
    ]


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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    q, k, v, gate, iq, raw = produce(
        qgate, key, value, index_qk, nq, nk, niq, positions, trig
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
    )
    output, selected = qsa_attention(
        q, iq, gate, *state[:3], starts, context, slots, main_table, compressed_table
    )
    return (*state, output.reshape(len(qgate), qgate.shape[1] // 2), selected)


@functools.cache
def build_qsa_op():
    op = pallas.jax_op(
        "qwen4_exp::qsa_sparse", qsa_step, donate_argnums=(10, 11, 12, 13, 14)
    )

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        main_k,
        main_v,
        compressed,
        ring,
        tags,
        *metadata,
    ):
        return (
            torch.empty_like(main_k),
            torch.empty_like(main_v),
            torch.empty_like(compressed),
            torch.empty_like(ring),
            torch.empty_like(tags),
            qgate.new_empty((qgate.shape[0], qgate.shape[1] // 2)),
            positions.new_empty((qgate.shape[0], 2051)),
        )

    def run(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        main_k,
        main_v,
        compressed,
        ring,
        tags,
        starts,
        context,
        slots,
        main_table,
        compressed_table,
    ):
        # Bit codec, never numerical BF16 reinterpretation of int64 tags.
        planes = torch.stack((tags.to(torch.int32), (tags >> 32).to(torch.int32)), -1)
        k, v, c, r, p, output, selected = op(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions.to(torch.int32),
            trig,
            main_k,
            main_v,
            compressed,
            ring,
            planes,
            starts.to(torch.int32),
            context.to(torch.int32),
            slots.to(torch.int32),
            main_table.to(torch.int32),
            compressed_table.to(torch.int32),
        )
        main_k.copy_(k)
        main_v.copy_(v)
        compressed.copy_(c)
        ring.copy_(r)
        tags.copy_(
            (p[..., 1].to(torch.int64) << 32) | (p[..., 0].to(torch.int64) & 0xFFFFFFFF)
        )
        return output, selected

    return run
