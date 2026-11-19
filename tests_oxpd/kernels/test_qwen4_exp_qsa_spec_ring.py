# SPDX-License-Identifier: Apache-2.0
"""QSA raw-ring history must survive rejected speculative tokens."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates

RATIO = 4
ROWS = 8
DIM = 8
ROTARY = 4


def _raw(position, fake=False):
    rng = np.random.default_rng(position * 2 + int(fake) + 11)
    return rng.normal(0, 1, DIM).astype(np.float32)


def _step(state, raws, positions, context):
    ring, tags = state
    trig = jnp.asarray(np.random.default_rng(5).normal(0, 1, (256, 128)), jnp.bfloat16)
    completed, complete, _, logical, ring, tags = _indexer_updates(
        jnp.asarray(np.stack(raws), jnp.bfloat16),
        jnp.asarray(positions, jnp.int32),
        ring,
        tags,
        jnp.zeros((DIM,), jnp.bfloat16),
        trig,
        jnp.asarray([0, len(raws)], jnp.int32),
        jnp.asarray([context], jnp.int32),
        jnp.asarray([0], jnp.int32),
        RATIO,
        1e-6,
        ROTARY,
    )
    rows = {}
    for row, done, pos in zip(
        np.asarray(completed, np.float32), np.asarray(complete), np.asarray(logical)
    ):
        if done:
            rows[int(pos) // RATIO] = row
    return (ring, tags), rows


def _fresh():
    return (
        jnp.zeros((1, ROWS, DIM), jnp.bfloat16),
        jnp.full((1, ROWS, 2), -1, jnp.int32),
    )


def _reference(length):
    state, groups = _fresh(), {}
    for pos in range(length):
        state, rows = _step(state, [_raw(pos)], [pos], pos)
        groups.update(rows)
    return groups


@pytest.mark.parametrize("num_spec", [1, 3, 5])
@pytest.mark.parametrize("prompt", [5, 8, 10])
def test_rejected_drafts_do_not_corrupt_compressed_groups(num_spec, prompt):
    rng = np.random.default_rng(num_spec * 31 + prompt)
    state, groups = _fresh(), {}
    state, rows = _step(state, [_raw(p) for p in range(prompt)], list(range(prompt)), 0)
    groups.update(rows)
    context = prompt
    while context < 48:
        accepted = int(rng.integers(0, num_spec + 1))
        window = range(context, context + num_spec + 1)
        # Position `context` holds the real next token; drafts past the
        # accepted prefix differ from the tokens the target later commits.
        raws = [_raw(p, fake=p > context + accepted) for p in window]
        state, rows = _step(state, raws, list(window), context)
        groups.update(rows)
        context += accepted + 1
    expected = _reference(context)
    for group in range(context // RATIO):
        np.testing.assert_allclose(groups[group], expected[group], rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(3))
def test_rows_are_slots_matches_scatter(seed):
    """Pooled layout (request row r owns ring row r; idle rows have slot -1):
    the select form gives the same ring, tags and completed groups."""
    rng = np.random.default_rng(seed)
    requests, spec = 8, 4
    lengths = rng.integers(0, spec + 1, requests)
    lengths[rng.integers(0, requests)] = 0
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    tokens = int(starts[-1]) + 3  # padded rows
    context = np.where(lengths > 0, rng.integers(0, 40, requests), 0).astype(np.int32)
    slots = np.where(lengths > 0, np.arange(requests), -1).astype(np.int32)
    raw = jnp.asarray(rng.normal(0, 1, (tokens, DIM)), jnp.bfloat16)
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r] : starts[r + 1]] = context[r] + np.arange(lengths[r])
    ring = jnp.asarray(rng.normal(0, 1, (requests, ROWS, DIM)), jnp.bfloat16)
    tags = jnp.asarray(rng.integers(-1, 60, (requests, ROWS, 2)), jnp.int32)
    trig = jnp.asarray(np.random.default_rng(5).normal(0, 1, (256, 128)), jnp.bfloat16)
    args = (
        raw,
        jnp.asarray(positions),
        ring,
        tags,
        jnp.zeros((DIM,), jnp.bfloat16),
        trig,
        jnp.asarray(starts),
        jnp.asarray(context),
        jnp.asarray(slots),
        RATIO,
        1e-6,
        ROTARY,
    )
    ref = _indexer_updates(*args)
    new = _indexer_updates(*args, rows_are_slots=True)
    done = np.asarray(ref[1])
    np.testing.assert_array_equal(done, np.asarray(new[1]))
    assert done.any()
    np.testing.assert_array_equal(
        np.asarray(ref[0], np.float32)[done], np.asarray(new[0], np.float32)[done]
    )
    for a, b in zip(ref[2:], new[2:]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
