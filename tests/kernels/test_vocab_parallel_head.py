# SPDX-License-Identifier: Apache-2.0
"""Vocabulary-parallel heads give the replicated heads' logits and draft tokens."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import vocab_parallel_head as vph
from vllm_torchtpu.kernels.logits_head import SLICE_TILE, logits_head
from vllm_torchtpu.kernels.markov_sampling import sample_slot, slot_rows

P = jax.sharding.PartitionSpec
VOCAB, WIDTH, RANK = 129280, 5120, 256
ROWS = P(vph.AXIS, None)


def test_shard_rows_cover_the_vocabulary_in_whole_tiles():
    shard = vph.shard_rows(VOCAB, 8)
    assert shard % SLICE_TILE == 0 and 7 * shard < VOCAB <= 8 * shard
    assert shard == 16640


def _mesh():
    if jax.default_backend() != "tpu" or len(jax.devices()) < 8:
        pytest.skip("needs 8 TPU devices")
    return jax.sharding.Mesh(np.array(jax.devices()[:8]), (vph.AXIS,))


def _slices(full, world, rows=None):
    """Every rank's slice of a [vocab, width] head, each zero-padded to
    ``rows`` rows (the slice rows by default), concatenated."""
    shard = vph.shard_rows(VOCAB, world)
    rows = rows or shard
    padded = jnp.concatenate(
        [full, jnp.zeros((world * shard - VOCAB, full.shape[1]), full.dtype)]
    ).reshape(world, shard, -1)
    return jnp.pad(padded, ((0, 0), (0, rows - shard), (0, 0))).reshape(
        world * rows, -1
    )


@pytest.mark.multichip
@pytest.mark.parametrize("rows", [8, 16])
def test_target_logits_are_the_replicated_logits(rows):
    mesh = _mesh()
    keys = jax.random.split(jax.random.PRNGKey(rows), 2)
    hidden = jax.random.normal(keys[0], (8 * rows, WIDTH)).astype(jnp.bfloat16)
    weight = (jax.random.normal(keys[1], (VOCAB, WIDTH)) * 0.02).astype(jnp.bfloat16)
    want = jax.jit(
        jax.shard_map(
            logits_head,
            mesh=mesh,
            in_specs=(ROWS, P()),
            out_specs=ROWS,
            check_vma=False,
        )
    )(hidden, weight)
    for chunks in (1, 2):
        got = jax.jit(
            jax.shard_map(
                functools.partial(
                    vph.target_logits, world=8, vocab_size=VOCAB, chunks=chunks
                ),
                mesh=mesh,
                in_specs=(ROWS, ROWS),
                out_specs=ROWS,
                check_vma=False,
            )
        )(hidden, _slices(weight, 8))
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
        )


def _replicated_samples(hidden, anchors, embed, markov, weight):
    """The served path: slots padded to the Markov kernel's rows."""
    n = anchors.shape[0]
    m = slot_rows(n)
    padded = jnp.pad(hidden.reshape(5, n, WIDTH), ((0, 0), (0, m - n), (0, 0)))
    logits = logits_head(padded.reshape(5 * m, WIDTH), weight)
    prev, tokens = jnp.pad(anchors, (0, m - n)), []
    for slot in range(5):
        prev = sample_slot(embed[prev], markov, logits, slot=slot)
        tokens.append(prev.astype(jnp.int32))
    return jnp.stack(tokens, axis=1)[:n]


def _samples(mesh, hidden, anchors, embed, markov, weight):
    fn = jax.shard_map(
        lambda h, a, e, m, w, r: vph.dspark_samples(
            h, a, e, m, w, r, world=8, vocab_size=VOCAB
        ),
        mesh=mesh,
        in_specs=(ROWS, P(vph.AXIS), P(), ROWS, ROWS, ROWS),
        out_specs=ROWS,
        check_vma=False,
    )
    rank = jnp.arange(8, dtype=jnp.int32)[:, None]
    markov_rows = vph.markov_rows(vph.shard_rows(VOCAB, 8))
    return jax.jit(fn)(
        hidden,
        anchors,
        embed,
        _slices(markov, 8, markov_rows),
        _slices(weight, 8),
        rank,
    )


@pytest.mark.multichip
@pytest.mark.parametrize("n", [8, 16, 32])
def test_dspark_samples_are_the_replicated_tokens(n):
    mesh = _mesh()
    keys = jax.random.split(jax.random.PRNGKey(n), 5)
    hidden = jax.random.normal(keys[0], (8 * 5 * n, WIDTH)).astype(jnp.bfloat16)
    anchors = jax.random.randint(keys[1], (8 * n,), 0, VOCAB, jnp.int32)
    embed = jax.random.normal(keys[2], (VOCAB, RANK)).astype(jnp.bfloat16)
    markov = (jax.random.normal(keys[3], (VOCAB, RANK)) * 0.05).astype(jnp.bfloat16)
    weight = (jax.random.normal(keys[4], (VOCAB, WIDTH)) * 0.02).astype(jnp.bfloat16)
    want = jax.jit(
        jax.shard_map(
            _replicated_samples,
            mesh=mesh,
            in_specs=(ROWS, P(vph.AXIS), P(), P(), P()),
            out_specs=ROWS,
            check_vma=False,
        )
    )(hidden, anchors, embed, markov, weight)
    got = _samples(mesh, hidden, anchors, embed, markov, weight)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.multichip
def test_dspark_samples_resolve_ties_and_nans_across_ranks():
    """Zero hidden rows give zero base logits; one Markov dimension then sets
    every token's score: ties and NaNs straddle the rank 0 / rank 1 slices, the
    last token is the only maximum, and padded slice rows must never win."""
    mesh = _mesh()
    n = 8
    shard = vph.shard_rows(VOCAB, 8)
    embed = jnp.zeros((VOCAB, RANK), jnp.bfloat16).at[:, 0].set(1)
    weight = jnp.zeros((VOCAB, WIDTH), jnp.bfloat16)
    hidden = jnp.zeros((8 * 5 * n, WIDTH), jnp.bfloat16)
    anchors = jnp.zeros((8 * n,), jnp.int32)
    cases = {
        shard - 1: jnp.zeros((VOCAB,)).at[shard - 1 : shard + 1].set(10),
        shard: jnp.zeros((VOCAB,)).at[shard + 1].set(jnp.nan).at[shard].set(jnp.nan),
        VOCAB - 1: jnp.full((VOCAB,), -10.0).at[VOCAB - 1].set(-5),
    }
    for expected, scores in cases.items():
        markov = (
            jnp.zeros((VOCAB, RANK), jnp.bfloat16)
            .at[:, 0]
            .set(scores.astype(jnp.bfloat16))
        )
        got = _samples(mesh, hidden, anchors, embed, markov, weight)
        np.testing.assert_array_equal(np.asarray(got), expected)
