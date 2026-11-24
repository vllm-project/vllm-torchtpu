# SPDX-License-Identifier: Apache-2.0
"""The PLE raw lookup takes its shard's first row as an operand, so one
compiled program serves every TP rank (TPU_PRECOMPILE_PEER_REPLAY)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple import resident_lookup_raw

HEAD_DIM = 160
LOCAL_ROWS = 32  # rows * HEAD_DIM fills whole 128-element segments


def _shard(rng):
    rows = rng.integers(-128, 128, (LOCAL_ROWS, HEAD_DIM), dtype=np.int8)
    return rows, jnp.asarray(rows.reshape(-1, 128))


def _expected(rows, ids, start):
    local = ids - start
    owned = (local >= 0) & (local < LOCAL_ROWS)
    out = np.where(owned[..., None], rows[np.where(owned, local, 0)], 0)
    return out.reshape(ids.shape[0], -1).astype(np.int8)


@pytest.mark.parametrize("token_count", [1, 5, 8, 13])
def test_operand_start_matches_constant_start(token_count):
    rng = np.random.default_rng(token_count)
    rows, table = _shard(rng)
    start = 3 * LOCAL_ROWS
    ids = rng.integers(start - 8, start + LOCAL_ROWS + 8, (token_count, 2))
    ids[0, 0] = -1
    constant = resident_lookup_raw(
        table, jnp.asarray(ids), start, local_rows=LOCAL_ROWS, head_dim=HEAD_DIM
    )
    operand = resident_lookup_raw(
        table,
        jnp.asarray(ids),
        jnp.asarray([start], jnp.int32),
        local_rows=LOCAL_ROWS,
        head_dim=HEAD_DIM,
    )
    np.testing.assert_array_equal(np.asarray(operand), np.asarray(constant))
    np.testing.assert_array_equal(np.asarray(operand), _expected(rows, ids, start))


def test_one_program_serves_every_shard():
    rng = np.random.default_rng(0)
    lookup = jax.jit(
        lambda table, ids, start: resident_lookup_raw(
            table, ids, start, local_rows=LOCAL_ROWS, head_dim=HEAD_DIM
        )
    )
    ids = rng.integers(0, 4 * LOCAL_ROWS, (8, 2))
    for rank in range(4):
        rows, table = _shard(rng)
        start = rank * LOCAL_ROWS
        got = lookup(table, jnp.asarray(ids), jnp.asarray([start], jnp.int32))
        np.testing.assert_array_equal(np.asarray(got), _expected(rows, ids, start))
    assert lookup._cache_size() == 1


@pytest.mark.parametrize("chunk_tokens", [16, 64, 256])
@pytest.mark.parametrize("token_count", [1, 13, 64, 300])
def test_larger_chunks_match_eight_token_chunks(chunk_tokens, token_count):
    """VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS changes only how many tokens one
    gather takes, including steps shorter than a chunk and ragged tails."""
    rng = np.random.default_rng(chunk_tokens + token_count)
    rows, table = _shard(rng)
    start = jnp.asarray([LOCAL_ROWS], jnp.int32)
    ids = rng.integers(0, 3 * LOCAL_ROWS, (token_count, 2))
    want = resident_lookup_raw(
        table, jnp.asarray(ids), start, local_rows=LOCAL_ROWS, head_dim=HEAD_DIM
    )
    got = resident_lookup_raw(
        table,
        jnp.asarray(ids),
        start,
        local_rows=LOCAL_ROWS,
        head_dim=HEAD_DIM,
        chunk_tokens=chunk_tokens,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(got), _expected(rows, ids, LOCAL_ROWS))


@pytest.mark.parametrize("owned_pct", [13, 25, 50, 100])
@pytest.mark.parametrize("chunk_tokens", [8, 64, 256])
@pytest.mark.parametrize("token_count", [1, 13, 64, 300])
def test_owned_gather_matches_masked_gather(owned_pct, chunk_tokens, token_count):
    """VLLM_TPU_PLE_OWNED_GATHER_PCT gathers only this shard's ids and is
    bit-identical, including chunks that own more than the buffer (they take
    the masked gather) and -1 padding ids."""
    rng = np.random.default_rng(owned_pct * 1000 + chunk_tokens + token_count)
    rows, table = _shard(rng)
    start = jnp.asarray([2 * LOCAL_ROWS], jnp.int32)
    # 8 shards' worth of vocab: about 1/8 of the ids are this shard's.
    ids = rng.integers(0, 8 * LOCAL_ROWS, (token_count, 3))
    ids[0, 0] = -1
    want = resident_lookup_raw(
        table,
        jnp.asarray(ids),
        start,
        local_rows=LOCAL_ROWS,
        head_dim=HEAD_DIM,
        chunk_tokens=chunk_tokens,
    )
    got = resident_lookup_raw(
        table,
        jnp.asarray(ids),
        start,
        local_rows=LOCAL_ROWS,
        head_dim=HEAD_DIM,
        chunk_tokens=chunk_tokens,
        owned_pct=owned_pct,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(got), _expected(rows, ids, 2 * LOCAL_ROWS))


def test_owned_gather_overflow_takes_masked_path():
    """A chunk whose ids are all this shard's overflows a 13% buffer."""
    rng = np.random.default_rng(7)
    rows, table = _shard(rng)
    ids = rng.integers(0, LOCAL_ROWS, (64, 3))  # every id owned by shard 0
    got = resident_lookup_raw(
        table,
        jnp.asarray(ids),
        jnp.asarray([0], jnp.int32),
        local_rows=LOCAL_ROWS,
        head_dim=HEAD_DIM,
        chunk_tokens=64,
        owned_pct=13,
    )
    np.testing.assert_array_equal(np.asarray(got), _expected(rows, ids, 0))


@pytest.mark.parametrize("head_dim", [128, HEAD_DIM])
@pytest.mark.parametrize("chunk_tokens", [8, 16])
def test_skip_empty_matches_default(head_dim, chunk_tokens):
    """skip_empty short-circuits unowned chunks and stays bit-identical."""
    rng = np.random.default_rng(head_dim + chunk_tokens)
    rows = rng.integers(-128, 128, (LOCAL_ROWS, head_dim), dtype=np.int8)
    table = jnp.asarray(rows.reshape(-1, 128))
    start = 2 * LOCAL_ROWS
    # Chunk 0: all unowned (-1); chunk 1: mixed; chunk 2: all owned.
    ids = np.full((24, 2), -1, dtype=np.int32)
    ids[8:16] = rng.integers(start - 4, start + LOCAL_ROWS + 4, (8, 2))
    ids[16:24] = rng.integers(start, start + LOCAL_ROWS, (8, 2))
    ids_j = jnp.asarray(ids)
    start_j = jnp.asarray([start], jnp.int32)
    want = resident_lookup_raw(
        table,
        ids_j,
        start_j,
        local_rows=LOCAL_ROWS,
        head_dim=head_dim,
        chunk_tokens=chunk_tokens,
    )
    got = resident_lookup_raw(
        table,
        ids_j,
        start_j,
        local_rows=LOCAL_ROWS,
        head_dim=head_dim,
        chunk_tokens=chunk_tokens,
        skip_empty=True,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(got), _expected(rows, ids, start))
