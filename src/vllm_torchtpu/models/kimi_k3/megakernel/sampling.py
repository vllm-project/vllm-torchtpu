# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/sampling.py
"""Token choice on the K3 decoder's vocabulary-sharded logits, inside its ops.

Every function runs inside ``shard_map`` over the 32-rank ``tp`` axis on this
rank's logits ``[rows, V / 32]`` and returns replicated results: all ranks
reach the same tokens through collectives and deterministic draws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .program import TP_SIZE, kernel_modules

if TYPE_CHECKING:
    from collections.abc import Callable

AXIS = "tp"
CANDIDATES = 128
_FP32_EXACT_INT_BOUND = 2**24
_MIN_SPLIT_ROWS = 16
_MIN_SPLIT_WIDTH = 2048
_HALF_DIVISOR = 2
_BITS_PER_WORD = 32
_MASKED_LOGIT_FLOOR = -1e30
_UNIFORM_COLS = 2
_PACKET_HEADER_FIELDS = 2


def _nucleus_accept() -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Return the DSpark nucleus acceptance function from kernel modules."""
    (dspark,) = kernel_modules("dspark")
    accept_fn: Callable[..., tuple[jax.Array, jax.Array]] = dspark.nucleus_accept
    return accept_fn


def greedy(local_logits: jax.Array) -> jax.Array:
    """Return ``[rows]`` int32 argmax over the full vocabulary."""
    rows, width = local_logits.shape
    rank = jax.lax.axis_index(AXIS)
    local_id = jnp.argmax(local_logits, axis=1).astype(jnp.int32)
    local_max = jnp.take_along_axis(local_logits, local_id[:, None], axis=1)[:, 0]
    global_id = local_id + rank * width
    if (
        rows > 0
        and local_logits.dtype in (jnp.float32, jnp.bfloat16)
        and width * TP_SIZE <= _FP32_EXACT_INT_BOUND
    ):
        packet = jnp.concatenate(
            (local_max.astype(jnp.float32), global_id.astype(jnp.float32))
        )
        gathered = jax.lax.all_gather(packet, AXIS, axis=0, tiled=True).reshape(
            -1, _PACKET_HEADER_FIELDS, rows
        )
        maxima, ids = gathered[:, 0], gathered[:, 1].astype(jnp.int32)
    else:
        maxima, ids = jax.lax.all_gather((local_max, global_id), AXIS, axis=0)
    winner = jnp.argmax(maxima, axis=0)
    return jnp.take_along_axis(ids, winner[None], axis=0)[0]


def _local_top_k(logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute exact top-128 via two narrower selections with stable tie order."""
    rows, width = logits.shape
    if rows < _MIN_SPLIT_ROWS or width < _MIN_SPLIT_WIDTH or width % _HALF_DIVISOR != 0:
        return jax.lax.top_k(logits, CANDIDATES)
    half_width = width // _HALF_DIVISOR
    values, indices = jax.lax.top_k(
        logits.reshape(rows * _HALF_DIVISOR, half_width),
        CANDIDATES,
    )
    indices = indices.reshape(rows, _HALF_DIVISOR, CANDIDATES) + (
        jnp.arange(_HALF_DIVISOR, dtype=jnp.int32)[None, :, None] * half_width
    )
    merged_width = _HALF_DIVISOR * CANDIDATES
    values, selected = jax.lax.top_k(
        values.reshape(rows, merged_width),
        CANDIDATES,
    )
    indices = jnp.take_along_axis(
        indices.reshape(rows, merged_width),
        selected,
        axis=1,
    )
    return values, indices


def _local_candidates(
    local_logits: jax.Array,
    temperature: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute local normalizer and top candidates with global vocabulary IDs."""
    _, width = local_logits.shape
    scaled = local_logits.astype(jnp.float32) / temperature[:, None]
    local_max = jnp.max(scaled, axis=1)
    local_lse = jnp.log(jnp.sum(jnp.exp(scaled - local_max[:, None]), axis=1))
    values, ids = _local_top_k(scaled)
    ids = ids.astype(jnp.int32) + jax.lax.axis_index(AXIS) * width
    return local_max, local_lse, values, ids


def _limit_candidates(values: jax.Array, top_k: jax.Array) -> jax.Array:
    """Mask candidates below each row's k-th largest logit when top_k > 0."""

    def limit(candidate_values: jax.Array) -> jax.Array:
        """Apply per-row top-k threshold preserving ties."""
        ordered = -jnp.sort(-candidate_values, axis=1)
        k_clipped = jnp.clip(top_k, 1, candidate_values.shape[1])
        threshold = jnp.take_along_axis(ordered, (k_clipped - 1)[:, None], axis=1)
        return jnp.where(
            (top_k > 0)[:, None] & (candidate_values < threshold),
            -jnp.inf,
            candidate_values,
        )

    return jax.lax.cond(jnp.any(top_k > 0), limit, lambda v: v, values)


def _candidates(
    local_logits: jax.Array,
    temperature: jax.Array,
    top_k: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather every rank's ``CANDIDATES`` largest tempered logits."""
    rows, width = local_logits.shape
    local_max, local_lse, values, ids = _local_candidates(local_logits, temperature)
    if width * TP_SIZE <= _FP32_EXACT_INT_BOUND:
        packet = jnp.concatenate(
            (
                local_max,
                local_lse,
                values.reshape(-1),
                ids.astype(jnp.float32).reshape(-1),
            )
        )
        packet = jax.lax.all_gather(packet, AXIS, axis=0)
        header_end = _PACKET_HEADER_FIELDS * rows
        values_end = header_end + rows * CANDIDATES
        maxima, lses = packet[:, :rows], packet[:, rows:header_end]
        values = packet[:, header_end:values_end].reshape(-1, rows, CANDIDATES)
        ids = packet[:, values_end:].astype(jnp.int32).reshape(-1, rows, CANDIDATES)
    else:
        maxima, lses, values, ids = jax.lax.all_gather(
            (local_max, local_lse, values, ids),
            AXIS,
            axis=0,
        )
    best = jnp.max(maxima, axis=0)
    lse = best + jnp.log(jnp.sum(jnp.exp(maxima - best[None] + lses), axis=0))
    ranks = maxima.shape[0]
    values = jnp.transpose(values, (1, 0, 2)).reshape(rows, ranks * CANDIDATES)
    ids = jnp.transpose(ids, (1, 0, 2)).reshape(rows, ranks * CANDIDATES)
    return _limit_candidates(values, top_k), ids, lse


def _request_candidates(
    local_logits: jax.Array,
    temperature: jax.Array,
    top_k: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exchange only the request assigned to this rank preserving source order."""
    requests, rows, width = local_logits.shape
    maximum, lse, values, ids = _local_candidates(
        local_logits.reshape(requests * rows, width),
        jnp.repeat(temperature, rows),
    )
    fields = _PACKET_HEADER_FIELDS + _HALF_DIVISOR * CANDIDATES
    packet = jnp.concatenate(
        (maximum[:, None], lse[:, None], values, ids.astype(jnp.float32)),
        axis=1,
    ).reshape(requests, rows, fields)
    packet = jnp.broadcast_to(
        packet[None],
        (TP_SIZE // requests, requests, rows, fields),
    ).reshape(TP_SIZE, rows, fields)
    packet = jax.lax.all_to_all(
        packet,
        AXIS,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )
    maxima, lses = packet[:, :, 0], packet[:, :, 1]
    best = jnp.max(maxima, axis=0)
    lse = best + jnp.log(jnp.sum(jnp.exp(maxima - best[None] + lses), axis=0))
    val_end = _PACKET_HEADER_FIELDS + CANDIDATES
    values = (
        packet[:, :, _PACKET_HEADER_FIELDS:val_end].transpose(1, 0, 2).reshape(rows, -1)
    )
    ids = packet[:, :, val_end:].astype(jnp.int32)
    ids = ids.transpose(1, 0, 2).reshape(rows, -1)
    request = jax.lax.axis_index(AXIS) % requests
    values = _limit_candidates(values, jnp.full((rows,), top_k[request], jnp.int32))
    return values, ids, lse


def mask_logits(
    local_logits: jax.Array,
    mask: jax.Array,
    rank: int | jax.Array,
) -> jax.Array:
    """Return ``rank``'s ``[rows, V / 32]`` logits with disallowed tokens masked."""
    rows, width = local_logits.shape
    local = jax.lax.dynamic_index_in_dim(
        mask.reshape(rows, TP_SIZE, width // _BITS_PER_WORD),
        rank,
        axis=1,
        keepdims=False,
    )
    bit_shifts = jnp.arange(_BITS_PER_WORD, dtype=jnp.int32)
    allowed = ((local[:, :, None] >> bit_shifts) & 1) != 0
    return jnp.where(allowed.reshape(rows, width), local_logits, _MASKED_LOGIT_FLOOR)


def apply_grammar_mask(local_logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Mask each row's disallowed tokens at this TP rank."""
    return mask_logits(local_logits, mask, jax.lax.axis_index(AXIS))


def uniforms(seed: jax.Array, position: jax.Array, rows: int) -> jax.Array:
    """Return ``[rows, 2]`` deterministic uniform draws for one step."""
    key = jax.random.fold_in(jax.random.PRNGKey(seed.astype(jnp.uint32)), position)
    return jax.random.uniform(key, (rows, _UNIFORM_COLS), jnp.float32)


def sample_rows(local_logits: jax.Array, *params: jax.Array) -> jax.Array:
    """Sample ``[rows]`` int32 tokens independently from per-row parameters."""
    temperature, top_p, top_k, seeds, positions = params
    accept = _nucleus_accept()
    values, ids, lse = _candidates(local_logits, temperature, top_k)
    draws = jax.vmap(lambda seed, position: uniforms(seed, position, 1))(
        seeds,
        positions,
    )

    def one(
        row_values: jax.Array,
        row_ids: jax.Array,
        row_lse: jax.Array,
        row_top_p: jax.Array,
        draw: jax.Array,
    ) -> jax.Array:
        """Draw a single token from one row's gathered candidates."""
        _, token = accept(
            row_values[None],
            row_ids[None],
            row_lse[None],
            jnp.full((1,), -1, jnp.int32),
            jnp.int32(0),
            row_top_p,
            draw,
        )
        return token

    return jax.vmap(one)(values, ids, lse, top_p, draws)


def accept_rows(
    local_logits: jax.Array,
    *params: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(count, bonus)`` for one sequence's verify rows."""
    drafts, forced_count, temperature, top_p, top_k, seed, position = params
    rows = local_logits.shape[0]
    accept = _nucleus_accept()
    values, ids, lse = _candidates(
        local_logits,
        jnp.full((rows,), temperature, jnp.float32),
        jnp.full((rows,), top_k, jnp.int32),
    )
    return accept(
        values,
        ids,
        lse,
        drafts,
        forced_count,
        top_p,
        uniforms(seed, position, rows),
    )


def accept_batched(
    local_logits: jax.Array,
    *params: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run independent identically-seeded acceptance with replicated decisions."""
    drafts, forced_count, temperature, top_p, top_k, seed, position = params
    requests, rows, width = local_logits.shape
    if (
        1 < requests <= TP_SIZE
        and TP_SIZE % requests == 0
        and requests * rows >= _MIN_SPLIT_ROWS
        and width * TP_SIZE <= _FP32_EXACT_INT_BOUND
    ):
        values, ids, lse = _request_candidates(local_logits, temperature, top_k)
        request = jax.lax.axis_index(AXIS) % requests
        count, bonus = _nucleus_accept()(
            values,
            ids,
            lse,
            drafts[request],
            forced_count[request],
            top_p[request],
            uniforms(seed[request], position[request], rows),
        )
        decisions = jax.lax.all_gather(jnp.stack((count, bonus)), AXIS)[:requests]
        return decisions[:, 0], decisions[:, 1]
    values, ids, lse = _candidates(
        local_logits.reshape(requests * rows, width),
        jnp.repeat(temperature, rows),
        jnp.repeat(top_k, rows),
    )
    draws = jax.vmap(lambda req_seed, pos: uniforms(req_seed, pos, rows))(
        seed,
        position,
    )
    return jax.vmap(_nucleus_accept())(
        values.reshape(requests, rows, -1),
        ids.reshape(requests, rows, -1),
        lse.reshape(requests, rows),
        drafts,
        forced_count,
        top_p,
        draws,
    )


def verify_acceptance(
    local_logits: jax.Array,
    verify_rows: tuple[jax.Array, jax.Array, jax.Array],
    parameters: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    synthetic_rates: jax.Array | None = None,
    single: bool = False,
) -> jax.Array:
    """Return replicated ``[requests, 2]`` accepted-count and bonus decisions."""
    tokens, positions, modes = verify_rows
    temperature, top_p, top_k, seed = parameters
    requests, rows = tokens.shape
    forced = jnp.where(modes == 0, 0, -1).astype(jnp.int32)
    if synthetic_rates is not None:
        (dspark,) = kernel_modules("dspark")
        rates = synthetic_rates[: rows - 1]
        drawn = jax.vmap(
            lambda req_seed, pos: dspark.synthetic_count(rates, req_seed, pos)
        )(seed, positions)
        forced = jnp.where(modes == 0, 0, drawn).astype(jnp.int32)

    def greedy_decision(_unused: None) -> jax.Array:
        """Compute greedy verification acceptance counts and bonus tokens."""
        targets = greedy(local_logits).reshape(requests, rows)
        match = tokens[:, 1:] == targets[:, :-1]
        first = jnp.min(jnp.where(match, rows - 1, jnp.arange(rows - 1)), axis=1)
        count = jnp.where(forced >= 0, forced, first)
        bonus = jnp.take_along_axis(targets, count[:, None], axis=1)[:, 0]
        return jnp.stack((count, bonus), axis=1)

    if synthetic_rates is not None:
        return greedy_decision(None)

    def sampled_decision(_unused: None) -> jax.Array:
        """Compute nucleus-sampled verification acceptance counts and bonus tokens."""
        drafts = jnp.concatenate(
            (tokens[:, 1:], jnp.full((requests, 1), -1, jnp.int32)),
            axis=1,
        )
        safe_temperature = jnp.where(temperature > 0, temperature, 1.0)
        if single:
            count, bonus = accept_rows(
                local_logits,
                drafts[0],
                forced[0],
                safe_temperature[0],
                top_p[0],
                top_k[0],
                seed[0],
                positions[0],
            )
            selected = jnp.stack((count, bonus))[None]
        else:
            count, bonus = accept_batched(
                local_logits.reshape(requests, rows, -1),
                drafts,
                forced,
                safe_temperature,
                top_p,
                top_k,
                seed,
                positions,
            )
            selected = jnp.stack((count, bonus), axis=1)
        return jax.lax.cond(
            jnp.all(temperature > 0),
            lambda _: selected,
            lambda _: jnp.where(
                (temperature > 0)[:, None],
                selected,
                greedy_decision(None),
            ),
            None,
        )

    return jax.lax.cond(
        jnp.any(temperature > 0),
        sampled_decision,
        greedy_decision,
        None,
    )
