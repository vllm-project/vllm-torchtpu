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
