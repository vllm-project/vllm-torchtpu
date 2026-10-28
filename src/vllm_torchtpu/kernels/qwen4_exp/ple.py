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
"""Compact embedding lookup and request-owned dilated convolution."""

import math

import jax
import jax.numpy as jnp


def compact_rows(table, ids, *, head_dim: int):
    """Gather rows from 128-element storage segments without row padding.

    The segment width is a TPU vector layout choice. The number of rows per
    complete storage group follows from the configured head dimension. Form
    segment indices by groups to avoid overflowing int32 in ``id*head_dim``.
    """
    segment_width = table.shape[-1]
    alignment = math.gcd(head_dim, segment_width)
    rows_per_group = segment_width // alignment
    segments_per_group = head_dim // alignment
    ids = ids.astype(jnp.int32)
    in_group = ids % rows_per_group
    element_offset = in_group * head_dim
    segment = (
        ids // rows_per_group
    ) * segments_per_group + element_offset // segment_width
    segment_count = math.ceil((head_dim + segment_width - alignment) / segment_width)
    parts = table[segment[..., None] + jnp.arange(segment_count)].reshape(
        ids.shape + (segment_count * segment_width,)
    )
    offset = element_offset % segment_width
    rows = parts[..., :head_dim]
    for start in range(alignment, segment_width, alignment):
        rows = jnp.where(
            (offset == start)[..., None], parts[..., start : start + head_dim], rows
        )
    return rows


def _bf16(x):
    # Keep source BF16 rounding boundaries even when XLA can fuse neighbors.
    return jax.lax.optimization_barrier(x.astype(jnp.bfloat16))


def resident_lookup_raw(
    table, ids, *, vocab_start: int | jax.Array, local_rows: int, head_dim: int
):
    """Owner-masked integer bits [T,heads*head_dim]: INT8/FP8 or INT16/BF16."""
    token_count, heads = ids.shape
    embedding_dim = heads * head_dim
    carrier = jnp.int16 if table.dtype == jnp.int16 else jnp.int8
    if token_count == 0:
        return jnp.zeros((0, embedding_dim), carrier)
    ids = ids.astype(jnp.int32)
    owned = (ids >= vocab_start) & (ids < vocab_start + local_rows)
    local = jnp.where(owned, ids - vocab_start, 0)
    bits = compact_rows(table, local, head_dim=head_dim)
    bits = jnp.where(owned[..., None], bits, jnp.zeros((), table.dtype))
    return jax.lax.bitcast_convert_type(bits, carrier).reshape(
        token_count, embedding_dim
    )


def decode_embeddings(raw_embeddings, scale):
    if raw_embeddings.dtype == jnp.int16:
        return jax.lax.bitcast_convert_type(raw_embeddings, jnp.bfloat16)
    values = jax.lax.bitcast_convert_type(raw_embeddings, jnp.float8_e4m3fn)
    return _bf16(_bf16(values) * scale.astype(jnp.bfloat16))


def resident_lookup(
    table, ids, scale, *, vocab_start: int, local_rows: int, head_dim: int
):
    """Local numerical entry sharing exactly the production raw lookup."""
    return decode_embeddings(
        resident_lookup_raw(
            table,
            ids,
            vocab_start=vocab_start,
            local_rows=local_rows,
            head_dim=head_dim,
        ),
        scale,
    )


def short_conv(
    x,
    weight,
    history,
    query_start_loc,
    state_indices,
    has_initial_states,
    *,
    dilation: int,
    request_owned_history: bool = False,
):
    """Configured dilated convolution with BF16 history [slots,state_len,C].

    Starts and slot indices are int32. The step metadata builder owns
    monotone starts beginning at zero in [0,T] and unique in-range slots;
    -1 denotes padded requests. Empty requests leave their slots untouched.
    History stores normalized gated values, not convolution results. Output is
    already SiLU-activated and rounded to BF16.

    Request-owned history uses identity slots for active requests. Its caller
    masks idle writes, so updated rows can be returned without a slot scatter.
    """
    token_count, channels = x.shape
    requests = state_indices.shape[0]
    if token_count == 0 or requests == 0:
        return jnp.zeros_like(x), history
    starts = query_start_loc.astype(jnp.int32)
    slots = state_indices.astype(jnp.int32)
    t = jnp.arange(token_count, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), requests - 1)
    relative = t - starts[req]
    valid = (t < starts[-1]) & (slots[req] >= 0)
    safe_slots = jnp.clip(slots, 0, history.shape[0] - 1)
    state_len = history.shape[1]
    accum = jnp.zeros((token_count, channels), jnp.float32)
    for tap in range(weight.shape[-1]):
        distance = state_len - tap * dilation
        position = relative - distance
        # A one-tap convolution has no historical state and only distance 0.
        current = x[jnp.clip(t - distance, 0, token_count - 1)]
        if request_owned_history and distance == 0:
            # Packed positions are nonnegative, so distance 0 reads no history.
            value = current
        elif (
            request_owned_history
            and state_len
            and distance > 0
            and requests * distance < token_count
        ):
            # History supplies rows with relative < distance; this gather has
            # O(R*distance*C) elements. Inactive scatter destinations are out of
            # bounds and are dropped.
            offsets = jnp.arange(distance, dtype=jnp.int32)
            past = history[safe_slots[:, None], state_len - distance + offsets]
            past = jnp.where(has_initial_states[:, None, None], past, 0)
            destinations = starts[:-1, None] + offsets
            lengths = starts[1:] - starts[:-1]
            active = (offsets < lengths[:, None]) & (slots[:, None] >= 0)
            destinations = jnp.where(active, destinations, token_count)
            value = jnp.where((position >= 0)[:, None], current, 0)
            value = value.at[destinations.reshape(-1)].set(
                past.reshape(-1, channels), mode="drop"
            )
        else:
            past = (
                history[
                    safe_slots[req], jnp.clip(position + state_len, 0, state_len - 1)
                ]
                if state_len
                else jnp.zeros_like(x)
            )
            past = jnp.where(has_initial_states[req, None], past, 0)
            value = jnp.where((position >= 0)[:, None], current, past)
        accum = accum + value.astype(jnp.float32) * weight[:, tap].astype(jnp.float32)
    pre = _bf16(accum)
    output = _bf16(jax.nn.silu(pre.astype(jnp.float32)))
    output = jnp.where(valid[:, None], output, 0)

    if state_len == 0:
        return output, history

    lengths = starts[1:] - starts[:-1]
    positions = lengths[:, None] - state_len + jnp.arange(state_len)[None, :]
    past = history[
        safe_slots[:, None], jnp.clip(positions + state_len, 0, state_len - 1)
    ]
    past = jnp.where(has_initial_states[:, None, None], past, 0)
    current = x[jnp.clip(starts[:-1, None] + positions, 0, token_count - 1)]
    updated = jnp.where((positions >= 0)[..., None], current, past)
    if request_owned_history:
        return output, updated
    # Scatter padding out of bounds with drop, so it cannot alias an active
    # slot (including when starts has repeated trailing entries).
    destinations = jnp.where((lengths > 0) & (slots >= 0), slots, history.shape[0])
    history = history.at[destinations].set(updated, mode="drop")
    return output, history
