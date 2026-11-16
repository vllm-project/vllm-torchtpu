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
"""The PLE raw lookup takes its shard's first row as an operand, so one
compiled program serves every TP rank (TPU_PRECOMPILE_PEER_REPLAY)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.ple import (
    resident_lookup,
    resident_lookup_raw,
)

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


def _bits(rows, dtype):
    """Store ``rows`` [LOCAL_ROWS, HEAD_DIM] as 128-element integer segments."""
    raw = rows.view(np.int16 if dtype == jnp.bfloat16 else np.int8)
    return jnp.asarray(raw.reshape(-1, 128))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("token_count", [1, 13])
def test_resident_lookup_dequantizes(dtype, token_count):
    """BF16 tables (INT16 carrier) are bit-cast; FP8 tables (INT8 carrier) are
    widened and scaled in BF16. Rows of other shards are zero."""
    rng = np.random.default_rng(token_count)
    rows = np.asarray(jnp.asarray(rng.normal(size=(LOCAL_ROWS, HEAD_DIM)), dtype))
    table = _bits(rows, dtype)
    assert table.dtype == (jnp.int16 if dtype == jnp.bfloat16 else jnp.int8)
    start = 2 * LOCAL_ROWS
    ids = rng.integers(start - 8, start + LOCAL_ROWS + 8, (token_count, 2))
    scale = jnp.asarray(rng.uniform(0.5, 2.0, size=(2 * HEAD_DIM,)), jnp.bfloat16)

    got = resident_lookup(
        table,
        jnp.asarray(ids),
        scale,
        vocab_start=start,
        local_rows=LOCAL_ROWS,
        head_dim=HEAD_DIM,
    )

    local = ids - start
    owned = (local >= 0) & (local < LOCAL_ROWS)
    values = np.where(
        owned[..., None], rows[np.where(owned, local, 0)].astype(np.float32), 0
    ).reshape(token_count, -1)
    if dtype == jnp.bfloat16:
        expected = values
    else:
        widened = jnp.asarray(values, jnp.bfloat16)
        expected = np.asarray((widened * scale).astype(jnp.float32))
    assert got.dtype == jnp.bfloat16 and got.shape == (token_count, 2 * HEAD_DIM)
    np.testing.assert_array_equal(np.asarray(got.astype(jnp.float32)), expected)
