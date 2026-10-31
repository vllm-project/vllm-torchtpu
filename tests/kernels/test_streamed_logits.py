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
"""Streamed LM-head kernel vs XLA's dot.

Interpret mode gates the slab ring (which block lands in which slot) and the
packed head layout on the host; on a TPU the real head shape must match the
dot bit for bit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.streamed_logits import (
    LogitsTiling,
    _streamed_logits,
    _streamed_logits_shard,
    block_rows,
    pack_head,
    streamed_logits,
    vocab_shard_columns,
    vocab_shard_layout,
)

# The served tiling (runner/utils.py LOGITS_*).
_TILING = LogitsTiling(block_rows=1280, block_dmas=2, buffers=3)


def _dot(x, w):
    return jax.lax.dot_general(
        x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(x.dtype)


def _inputs(m, vocab, hidden, seed):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((m, hidden)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((vocab, hidden)), jnp.bfloat16)
    return x, w


def test_block_rows_tiles_vocab():
    assert block_rows(129280, _TILING) == 1280
    assert block_rows(640, _TILING) == 640
    assert block_rows(100, _TILING) == 0
    assert block_rows(320, _TILING) == 0


def test_pack_head_transposes_blocks():
    w = jnp.arange(6 * 4, dtype=jnp.float32).reshape(6, 4)
    packed = np.asarray(pack_head(w, 3))
    assert packed.shape == (2, 4, 3)
    np.testing.assert_array_equal(packed[1], np.asarray(w)[3:6].T)


@pytest.mark.parametrize(
    "m,vocab,hidden,rows",
    [
        (8, 2560, 256, 640),
        (5, 3840, 128, 1280),
        (24, 640, 256, 640),
        (8, 384, 128, 128),
    ],
)
def test_interpret_matches_dot(m, vocab, hidden, rows):
    x, w = _inputs(m, vocab, hidden, m + vocab)
    got = _streamed_logits(x, pack_head(w, rows), tiling=_TILING, interpret=True)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(_dot(x, w)))


def test_vocab_shard_layout_covers_head():
    assert vocab_shard_layout(129280, 8, _TILING) == (
        1280,
        13,
        (0, 13, 26, 39, 52, 65, 78, 88),
    )
    spans = vocab_shard_columns(129280, 8, _TILING)
    assert spans[:7] == ((0, 16640),) * 7
    assert spans[7] == (3 * 1280, 16640)


@pytest.mark.parametrize(
    "m,vocab,hidden,shards,rows",
    [
        (16, 5120, 256, 8, 640),
        (8, 5760, 128, 4, 640),
        (24, 4224, 128, 8, 384),
    ],
)
def test_interpret_shards_concatenate_to_dot(m, vocab, hidden, shards, rows):
    tiling = _TILING._replace(block_rows=rows)
    x, w = _inputs(m, vocab, hidden, m + vocab)
    want = np.asarray(_dot(x, w))
    rows, per_shard, first = vocab_shard_layout(vocab, shards, tiling)
    packed = pack_head(w, rows)
    parts = []
    for f, (lo, hi) in zip(first, vocab_shard_columns(vocab, shards, tiling)):
        got = np.asarray(
            _streamed_logits_shard(
                x,
                packed,
                jnp.array([f], jnp.int32),
                num_blocks=per_shard,
                tiling=tiling,
                interpret=True,
            )
        )
        np.testing.assert_array_equal(got, want[:, f * rows : (f + per_shard) * rows])
        parts.append(got[:, lo:hi])
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), want)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="needs a TPU")
@pytest.mark.parametrize("m", [8, 128])
def test_tpu_bit_exact_at_head_shape(m):
    x, w = _inputs(m, 129280, 4096, m)
    packed = pack_head(w, block_rows(129280, _TILING))
    np.testing.assert_array_equal(
        np.asarray(streamed_logits(x, packed, _TILING)),
        np.asarray(jax.jit(_dot)(x, w)),
    )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="needs a TPU")
@pytest.mark.parametrize("m", [64, 1024])
def test_tpu_shards_bit_exact_at_head_shape(m):
    x, w = _inputs(m, 129280, 4096, m)
    want = np.asarray(jax.jit(_dot)(x, w))
    rows, per_shard, first = vocab_shard_layout(129280, 8, _TILING)
    packed = pack_head(w, rows)
    for f in first:
        got = _streamed_logits_shard(
            x,
            packed,
            jnp.array([f], jnp.int32),
            num_blocks=per_shard,
            tiling=_TILING,
        )
        np.testing.assert_array_equal(
            np.asarray(got), want[:, f * rows : (f + per_shard) * rows]
        )
