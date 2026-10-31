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
"""Packed-row embedding lookup vs ``table[ids]``, bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.embedding_rows import (
    _embedding_rows,
    embedding_rows,
    pack_table,
)


def _lookup(table, ids):
    vocab = table.shape[0]
    return table[np.clip(np.where(ids < 0, ids + vocab, ids), 0, vocab - 1)]


def _inputs(vocab, hidden, tokens, seed):
    rng = np.random.default_rng(seed)
    table = np.asarray(jnp.asarray(rng.standard_normal((vocab, hidden)), jnp.bfloat16))
    ids = rng.integers(-vocab - 3, vocab + 3, size=tokens).astype(np.int32)
    return table, ids


@pytest.mark.parametrize(
    "vocab,hidden,tokens",
    [(300, 512, 32), (1000, 256, 48), (77, 1024, 5), (500, 2048, 160)],
)
def test_interpret_matches_lookup(vocab, hidden, tokens):
    table, ids = _inputs(vocab, hidden, tokens, vocab + tokens)
    packed = jnp.asarray(pack_table(table, chunk_rows=37))
    got = _embedding_rows(jnp.asarray(ids), packed, interpret=True)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), _lookup(table, ids).view(np.uint16)
    )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="needs a TPU")
@pytest.mark.parametrize("tokens", [16, 128, 1024])
def test_tpu_bit_exact_at_embed_shape(tokens):
    table, ids = _inputs(129280, 4096, tokens, tokens)
    got = embedding_rows(jnp.asarray(ids), jnp.asarray(pack_table(table)))
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), _lookup(table, ids).view(np.uint16)
    )
