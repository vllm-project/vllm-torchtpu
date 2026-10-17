# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token choice on the K3 decoder's vocabulary-sharded logits, inside its ops.

Every function runs inside ``shard_map`` over the 32-rank ``tp`` axis on this
rank's logits ``[rows, V / 32]`` and returns replicated results: all ranks
reach the same tokens through collectives and deterministic draws.

- :func:`greedy`: the argmax (the first shard holding the maximum on ties).
- :func:`sample_rows`: independent rows (a plain batch step): each row's
  token drawn from its temperature / top-k / top-p distribution.
- :func:`accept_rows`: one sequence's verify rows (DSpark): exact rejection
  sampling of the draft's (deterministic) proposals, then the bonus token
  (``kernels.dspark.nucleus_accept``).

Sampled rows need ``temperature > 0``; callers select :func:`greedy` for
greedy rows. Top-p and top-k act on the ``CANDIDATES`` largest logits of each
rank (exact unless the nucleus needs more than that many tokens of one rank).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.megakernel.kimi_k3.program import TP_SIZE, kernel_modules

AXIS = "tp"
CANDIDATES = 128


def _nucleus_accept():
    (dspark,) = kernel_modules("dspark")
    return dspark.nucleus_accept


def greedy(local_logits: jax.Array) -> jax.Array:
    """``[rows]`` int32 argmax over the full vocabulary."""
    rows, width = local_logits.shape
    rank = jax.lax.axis_index(AXIS)
    local_id = jnp.argmax(local_logits, axis=1).astype(jnp.int32)
    local_max = jnp.take_along_axis(local_logits, local_id[:, None], axis=1)[:, 0]
    global_id = local_id + rank * width
    if (
        rows > 0
        and local_logits.dtype in (jnp.float32, jnp.bfloat16)
        and width * TP_SIZE <= 2**24
    ):
        # These IDs are exact in FP32. A flat payload lowers to one TPU
        # all-reduce; the differently typed tuple requires two collectives.
        packet = jnp.concatenate(
            (local_max.astype(jnp.float32), global_id.astype(jnp.float32))
        )
        gathered = jax.lax.all_gather(packet, AXIS, axis=0, tiled=True).reshape(
            -1, 2, rows
        )
        maxima, ids = gathered[:, 0], gathered[:, 1].astype(jnp.int32)
    else:
        maxima, ids = jax.lax.all_gather(
            (local_max, global_id), AXIS, axis=0
        )  # [ranks, rows]
    winner = jnp.argmax(maxima, axis=0)
    return jnp.take_along_axis(ids, winner[None], axis=0)[0]


def _local_top_k(logits):
    """Exact top-128 via two narrower selections, with a single top_k's tie order.

    A discarded entry already has 128 earlier/better entries in its half, so
    cannot belong to the global top-128. Concatenating contiguous halves keeps
    lower vocabulary IDs first on ties in the final selection.
    """
    rows, width = logits.shape
    # Below 16 rows one top_k is cheaper than the split/merge's extra layout work.
    if rows < 16 or width < 2048 or width % 2:
        return jax.lax.top_k(logits, CANDIDATES)
    values, indices = jax.lax.top_k(logits.reshape(rows * 2, width // 2), CANDIDATES)
    indices = indices.reshape(rows, 2, CANDIDATES) + (
        jnp.arange(2, dtype=jnp.int32)[None, :, None] * (width // 2)
    )
    values, selected = jax.lax.top_k(values.reshape(rows, 2 * CANDIDATES), CANDIDATES)
    indices = jnp.take_along_axis(
        indices.reshape(rows, 2 * CANDIDATES), selected, axis=1
    )
    return values, indices


def _local_candidates(local_logits, temperature):
    """Local normalizer and stable candidates, with global vocabulary IDs."""
    _, width = local_logits.shape
    scaled = local_logits.astype(jnp.float32) / temperature[:, None]
    local_max = jnp.max(scaled, axis=1)
    local_lse = jnp.log(jnp.sum(jnp.exp(scaled - local_max[:, None]), axis=1))
    values, ids = _local_top_k(scaled)
    ids = ids.astype(jnp.int32) + jax.lax.axis_index(AXIS) * width
    return local_max, local_lse, values, ids


def _limit_candidates(values, top_k):
    # Keep ties at the k-th largest value, matching vLLM's sampler.
    def limit(values):
        ordered = -jnp.sort(-values, axis=1)
        k = jnp.clip(top_k, 1, values.shape[1])
        threshold = jnp.take_along_axis(ordered, (k - 1)[:, None], axis=1)
        return jnp.where((top_k > 0)[:, None] & (values < threshold), -jnp.inf, values)

    # Keep this outside vmap: disabled top-k must skip its sort.
    return jax.lax.cond(jnp.any(top_k > 0), limit, lambda v: v, values)


def _candidates(local_logits, temperature, top_k):
    """Every rank's ``CANDIDATES`` largest tempered logits, gathered:
    ``(values [rows, n], ids [rows, n], lse [rows])``, top-k applied."""
    rows, width = local_logits.shape
    local_max, local_lse, values, ids = _local_candidates(local_logits, temperature)
    # Vocabulary IDs are exact in FP32; one flat packet avoids four
    # collectives and per-row payload padding. Preserve rank-major order.
    if width * TP_SIZE <= 2**24:
        packet = jnp.concatenate(
            (
                local_max,
                local_lse,
                values.reshape(-1),
                ids.astype(jnp.float32).reshape(-1),
            )
        )
        packet = jax.lax.all_gather(packet, AXIS, axis=0)
        maxima, lses = packet[:, :rows], packet[:, rows : 2 * rows]
        values = packet[:, 2 * rows : 2 * rows + rows * CANDIDATES].reshape(
            -1, rows, CANDIDATES
        )
        ids = (
            packet[:, 2 * rows + rows * CANDIDATES :]
            .astype(jnp.int32)
            .reshape(-1, rows, CANDIDATES)
        )
    else:
        maxima, lses, values, ids = jax.lax.all_gather(
            (local_max, local_lse, values, ids), AXIS, axis=0
        )
    best = jnp.max(maxima, axis=0)
    lse = best + jnp.log(jnp.sum(jnp.exp(maxima - best[None] + lses), axis=0))
    ranks = maxima.shape[0]
    values = jnp.transpose(values, (1, 0, 2)).reshape(rows, ranks * CANDIDATES)
    ids = jnp.transpose(ids, (1, 0, 2)).reshape(rows, ranks * CANDIDATES)
    return _limit_candidates(values, top_k), ids, lse


def _request_candidates(local_logits, temperature, top_k):
    """Exchange only the request assigned to this rank; preserve source order.

    Rank r samples request r % requests. Each sender supplies that request's
    local candidates to r through all-to-all, rather than replicating every
    request's candidates on every rank. Duplicate owners keep the exchange
    uniform; only the first `requests` decisions are subsequently used.
    """
    requests, rows, width = local_logits.shape
    maximum, lse, values, ids = _local_candidates(
        local_logits.reshape(requests * rows, width), jnp.repeat(temperature, rows)
    )
    fields = 2 + 2 * CANDIDATES
    packet = jnp.concatenate(
        (maximum[:, None], lse[:, None], values, ids.astype(jnp.float32)), axis=1
    ).reshape(requests, rows, fields)
    packet = jnp.broadcast_to(
        packet[None], (TP_SIZE // requests, requests, rows, fields)
    ).reshape(TP_SIZE, rows, fields)
    packet = jax.lax.all_to_all(packet, AXIS, split_axis=0, concat_axis=0, tiled=True)
    maxima, lses = packet[:, :, 0], packet[:, :, 1]
    best = jnp.max(maxima, axis=0)
    lse = best + jnp.log(jnp.sum(jnp.exp(maxima - best[None] + lses), axis=0))
    values = packet[:, :, 2 : 2 + CANDIDATES].transpose(1, 0, 2).reshape(rows, -1)
    ids = packet[:, :, 2 + CANDIDATES :].astype(jnp.int32)
    ids = ids.transpose(1, 0, 2).reshape(rows, -1)
    request = jax.lax.axis_index(AXIS) % requests
    values = _limit_candidates(values, jnp.full((rows,), top_k[request], jnp.int32))
    return values, ids, lse


def mask_logits(local_logits: jax.Array, mask: jax.Array, rank: jax.Array) -> jax.Array:
    """``rank``'s ``[rows, V / 32]`` logits with its disallowed tokens masked.

    ``mask`` is ``[rows, vocab_words]`` int32: bit ``j`` of word ``w`` is the
    allowed-ness of token ``32 * w + j`` (1: allowed; xgrammar's
    ``allocate_token_bitmask``), so rank ``r``'s logits take words
    ``[r * V / 32 / 32, (r + 1) * V / 32 / 32)``. Disallowed logits take a
    large finite value rather than -inf: a rank whose whole shard is
    disallowed (a narrow schema can allow tokens of a single shard) would
    otherwise give NaN local normalizers that poison the global softmax of
    :func:`sample_rows`.
    """
    rows, width = local_logits.shape
    local = jax.lax.dynamic_index_in_dim(
        mask.reshape(rows, TP_SIZE, width // 32), rank, axis=1, keepdims=False
    )
    allowed = ((local[:, :, None] >> jnp.arange(32, dtype=jnp.int32)) & 1) != 0
    return jnp.where(allowed.reshape(rows, width), local_logits, -1e30)


def apply_grammar_mask(local_logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Mask each row's disallowed tokens (:func:`mask_logits` at this rank)."""
    return mask_logits(local_logits, mask, jax.lax.axis_index(AXIS))


def uniforms(seed: jax.Array, position: jax.Array, rows: int) -> jax.Array:
    """``[rows, 2]`` uniforms for one step of a sequence: a function of its
    seed and the step's (first row's) position, identical on every rank."""
    key = jax.random.fold_in(jax.random.PRNGKey(seed.astype(jnp.uint32)), position)
    return jax.random.uniform(key, (rows, 2), jnp.float32)


def sample_rows(local_logits, temperature, top_p, top_k, seeds, positions) -> jax.Array:
    """``[rows]`` int32: row r drawn from its own distribution (per-row
    ``temperature > 0``, ``top_p``, ``top_k``, ``seeds``, ``positions``)."""
    accept = _nucleus_accept()
    values, ids, lse = _candidates(local_logits, temperature, top_k)
    # [rows, 1, 2]
    draws = jax.vmap(lambda seed, position: uniforms(seed, position, 1))(
        seeds, positions
    )

    def one(values, ids, lse, top_p, draw):
        _, token = accept(
            values[None],
            ids[None],
            lse[None],
            jnp.full((1,), -1, jnp.int32),
            jnp.int32(0),
            top_p,
            draw,
        )
        return token

    return jax.vmap(one)(values, ids, lse, top_p, draws)


def accept_rows(
    local_logits, drafts, forced_count, temperature, top_p, top_k, seed, position
):
    """``(count, bonus)`` for one sequence's verify rows: ``drafts[k]`` the
    draft verified at row k (-1: none), scalar sampling parameters,
    ``forced_count >= 0`` to fix the count (the bonus then from row count)."""
    rows = local_logits.shape[0]
    accept = _nucleus_accept()
    values, ids, lse = _candidates(
        local_logits,
        jnp.full((rows,), temperature, jnp.float32),
        jnp.full((rows,), top_k, jnp.int32),
    )
    return accept(
        values, ids, lse, drafts, forced_count, top_p, uniforms(seed, position, rows)
    )


def accept_batched(
    local_logits, drafts, forced_count, temperature, top_p, top_k, seed, position
):
    """Independent, identically seeded acceptance with replicated decisions.

    Large supported batches distribute candidate sorting by request. Small
    batches and other shapes use the replicated candidate exchange.
    """
    requests, rows, width = local_logits.shape
    if (
        1 < requests <= TP_SIZE
        and TP_SIZE % requests == 0
        and requests * rows >= 16
        and width * TP_SIZE <= 2**24
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
    draws = jax.vmap(lambda seed, pos: uniforms(seed, pos, rows))(seed, position)
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
    local_logits,
    tokens,
    positions,
    modes,
    parameters,
    *,
    synthetic_rates=None,
    single=False,
):
    """Replicated [requests, 2] accepted-count/bonus before KDA state commit.

    Runs inside the verifier's shard_map. Preserve the drafter's lowest-rank,
    lowest-token greedy ties, per-request random draws and forced cold counts.
    Synthetic acceptance takes precedence over temperature, as in the drafter.
    """
    requests, rows = tokens.shape
    temperature, top_p, top_k, seed = parameters
    forced = jnp.where(modes == 0, 0, -1).astype(jnp.int32)
    if synthetic_rates is not None:
        (dspark,) = kernel_modules("dspark")
        drawn = jax.vmap(
            lambda seed, pos: dspark.synthetic_count(
                synthetic_rates[: rows - 1], seed, pos
            )
        )(seed, positions)
        forced = jnp.where(modes == 0, 0, drawn).astype(jnp.int32)

    def greedy_decision(_):
        targets = greedy(local_logits).reshape(requests, rows)
        match = tokens[:, 1:] == targets[:, :-1]
        first = jnp.min(jnp.where(match, rows - 1, jnp.arange(rows - 1)), axis=1)
        count = jnp.where(forced >= 0, forced, first)
        bonus = jnp.take_along_axis(targets, count[:, None], axis=1)[:, 0]
        return jnp.stack((count, bonus), axis=1)

    if synthetic_rates is not None:
        return greedy_decision(None)

    def sampled_decision(_):
        drafts = jnp.concatenate(
            (tokens[:, 1:], jnp.full((requests, 1), -1, jnp.int32)), axis=1
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
                (temperature > 0)[:, None], selected, greedy_decision(None)
            ),
            None,
        )

    return jax.lax.cond(
        jnp.any(temperature > 0), sampled_decision, greedy_decision, None
    )
