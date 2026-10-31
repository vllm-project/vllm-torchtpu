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
"""The TensorCore combine matches the SparseCore ragged_gather_reduce_v2."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import moe_gather_unpermute
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import (
    ragged_gather_reduce_v2,
)

_TOPK = 6


def _routes(num_tokens, hidden_size, local_fraction, seed):
    """EP-style routing: valid routes first in expert order, the rest garbage."""
    rng = np.random.default_rng(seed)
    num_routes = num_tokens * _TOPK
    valid = rng.random(num_routes) < local_fraction
    key = np.where(valid, rng.integers(0, 32, num_routes), 32)
    order = np.argsort(key, kind="stable")
    indices = np.empty(num_routes, np.int32)
    indices[order] = np.arange(num_routes, dtype=np.int32)
    rows = jnp.asarray(rng.standard_normal((num_routes, hidden_size)), jnp.bfloat16)
    # Rows past the valid prefix are never written by the grouped matmul.
    rows = rows.at[int(valid.sum()) :].set(jnp.nan)
    weights = jnp.asarray(rng.random(num_routes), jnp.bfloat16)
    return rows, jnp.asarray(indices), weights, jnp.asarray(valid)


def _reference(rows, indices, weights, valid, num_tokens):
    """ragged_gather_reduce_v2's order: f32 products summed left to right."""
    rows = np.asarray(rows.astype(jnp.float32))
    indices = np.asarray(indices).reshape(num_tokens, _TOPK)
    weights = np.asarray(weights.astype(jnp.float32)).reshape(num_tokens, _TOPK)
    valid = np.asarray(valid).reshape(num_tokens, _TOPK)
    out = np.zeros((num_tokens, rows.shape[1]), np.float32)
    for t in range(num_tokens):
        total = None
        for j in np.flatnonzero(valid[t]):
            product = rows[indices[t, j]] * weights[t, j]
            total = product if total is None else total + product
        if total is not None:
            out[t] = total
    return jnp.asarray(out).astype(jnp.bfloat16)


@pytest.mark.parametrize(
    "num_tokens,hidden_size,local_fraction,seed",
    [
        (256, 256, 0.0, 0),
        (384, 256, 1 / 8, 1),
        (256, 384, 0.5, 2),
        # Every route local: six slots per token.
        (256, 128, 1.0, 3),
        # Several blocks, so both slot buffers take over more than once.
        (384, 128, 1 / 50, 5),
        (640, 128, 1 / 5, 6),
    ],
)
def test_matches_sparsecore_order_bit_for_bit(
    num_tokens, hidden_size, local_fraction, seed
):
    rows, indices, weights, valid = _routes(
        num_tokens, hidden_size, local_fraction, seed
    )
    assert moe_gather_unpermute.can_use_gather_unpermute(
        rows, weights, num_tokens, _TOPK
    )
    # The generic Pallas interpreter: it checks the tables and the sum but does
    # not model the HBM pairing of bf16 rows or the DMA semaphores (see the TPU
    # test below). The TPU interpreter cannot follow the kernel's 32-bit view.
    out = moe_gather_unpermute.gather_unpermute(
        rows,
        indices,
        weights,
        valid,
        num_tokens=num_tokens,
        topk=_TOPK,
        interpret=True,
    )
    expected = _reference(rows, indices, weights, valid, num_tokens)
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), np.asarray(expected).view(np.uint16)
    )


@pytest.mark.parametrize(
    "num_tokens,chunk_pieces",
    [
        (768, (128, 256)),
        (1024, (128, 384)),
        (1024, (128, 128, 256)),
        # Calls of 3 and 6 steps: a call ends on either buffer.
        (1152, (128, 256)),
    ],
)
def test_pieces_are_the_token_order_rows_piece_by_piece(num_tokens, chunk_pieces):
    hidden_size = 256
    rows, indices, weights, valid = _routes(num_tokens, hidden_size, 1 / 8, seed=5)
    assert moe_gather_unpermute.can_use_gather_unpermute(
        rows, weights, num_tokens, _TOPK, chunk_pieces
    )
    kwargs = dict(num_tokens=num_tokens, topk=_TOPK, interpret=True)
    whole = moe_gather_unpermute.gather_unpermute(
        rows, indices, weights, valid, **kwargs
    )
    pieces = moe_gather_unpermute.gather_unpermute(
        rows, indices, weights, valid, chunk_pieces=chunk_pieces, **kwargs
    )
    chunks = np.asarray(whole).reshape(-1, sum(chunk_pieces), hidden_size)
    bounds = np.cumsum((0, *chunk_pieces))
    expected = np.concatenate(
        [chunks[:, a:b].reshape(-1, hidden_size) for a, b in zip(bounds, bounds[1:])]
    )
    np.testing.assert_array_equal(
        np.asarray(pieces).view(np.uint16), expected.view(np.uint16)
    )


def test_pieces_must_be_whole_token_blocks():
    rows, _, weights, _ = _routes(4096, 128, 1 / 8, seed=6)
    assert not moe_gather_unpermute.can_use_gather_unpermute(
        rows, weights, 4096, _TOPK, (64, 192)
    )


def test_non_finite_weight_stays_with_its_token():
    num_tokens, hidden_size = 384, 256
    rows, indices, weights, valid = _routes(num_tokens, hidden_size, 1 / 8, seed=7)
    route = int(np.flatnonzero(np.asarray(valid))[5])
    weights = weights.at[route].set(jnp.nan)
    out = moe_gather_unpermute.gather_unpermute(
        rows,
        indices,
        weights,
        valid,
        num_tokens=num_tokens,
        topk=_TOPK,
        interpret=True,
    )
    nan_tokens = np.flatnonzero(np.isnan(np.asarray(out, np.float32)).any(axis=1))
    np.testing.assert_array_equal(nan_tokens, [route // _TOPK])


@pytest.mark.tpu_test
def test_matches_ragged_gather_reduce_v2_on_tpu():
    num_tokens, hidden_size = 2048, 2048
    rows, indices, weights, valid = _routes(num_tokens, hidden_size, 1 / 8, seed=4)
    out = moe_gather_unpermute.gather_unpermute(
        rows, indices, weights, valid, num_tokens=num_tokens, topk=_TOPK
    )
    expected = ragged_gather_reduce_v2(
        rows, indices, weights, valid, reduce_group_size=_TOPK
    )
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), np.asarray(expected).view(np.uint16)
    )
