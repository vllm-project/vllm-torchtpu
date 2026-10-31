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

Interpret mode gates the slab ring (which block lands in which slot) on the
host; on a TPU the real head shape must match the dot bit for bit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.streamed_logits import (
    _streamed_logits,
    block_rows,
    streamed_logits,
)


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
    assert block_rows(129280) == 1280
    assert block_rows(640) == 640
    assert block_rows(100) == 0


@pytest.mark.parametrize(
    "m,vocab,hidden,rows",
    [
        (8, 2560, 256, 640),
        (5, 3840, 128, 1280),
        (24, 640, 256, 640),
        (8, 320, 128, 320),
    ],
)
def test_interpret_matches_dot(m, vocab, hidden, rows):
    x, w = _inputs(m, vocab, hidden, m + vocab)
    got = _streamed_logits(x, w, rows=rows, interpret=True)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(_dot(x, w)))


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="needs a TPU")
@pytest.mark.parametrize("m", [8, 128])
def test_tpu_bit_exact_at_head_shape(m):
    x, w = _inputs(m, 129280, 4096, m)
    np.testing.assert_array_equal(
        np.asarray(streamed_logits(x, w)), np.asarray(jax.jit(_dot)(x, w))
    )
