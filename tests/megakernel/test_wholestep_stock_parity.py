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
"""Whole-step kernel math vs the stock qwen4_exp writers, on identical inputs (CPU).

The kernel computes the completed group's compressed key and the new ring rows itself (it
must score that key in the same step); the served cache keeps being written by the stock
`_indexer_updates` / writers. These must agree bit for bit, or the cache and the kernel's own
selection diverge.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates
from vllm_torchtpu.megakernel.ops import BF16, F32, rmsnorm, rnd
from vllm_torchtpu.megakernel.wholestep.indexer import index_rope

ROT, RATIO, T, D, EPS, THETA = 64, 4, 4, 128, 1e-6, 10_000_000.0


def _trig(max_pos: int) -> jax.Array:
    """Stock rope cache rows: [cos(rot/2) | sin(rot/2) | 0-pad] BF16 per position."""
    inv = 1.0 / (THETA ** (np.arange(0, ROT, 2) / ROT))
    ang = np.arange(max_pos)[:, None] * inv[None]
    out = np.zeros((max_pos, 128), np.float32)
    out[:, : ROT // 2] = np.cos(ang)
    out[:, ROT // 2 : ROT] = np.sin(ang)
    return jnp.asarray(out.astype(ml_dtypes.bfloat16))


def _kernel_completed(raw_q, ring, c0, ikn, rope_row):
    """The kernel's arithmetic (`QSAFullStepBlock._new_group`): sequential f32 member sum."""
    p3 = c0 + (3 - c0 % 4) % 4
    acc = None
    for j in range(4):
        pos = p3 - 3 + j
        m = raw_q[pos - c0][None] if pos >= c0 else ring[pos % 8][None]
        acc = m if acc is None else acc + m
    pooled = rnd(acc * 0.25)
    cos, sin = rope_row[:, : ROT // 2], rope_row[:, ROT // 2 : ROT]
    return index_rope(rmsnorm(pooled, ikn, EPS, zero_centered=True), cos, sin, ROT,
                      jax.lax.bitcast_convert_type).astype(BF16), p3


@pytest.mark.parametrize("ctx", [[3001, 100], [3000, 101], [2999, 102], [7, 4097]])
def test_completed_key_and_ring_match_stock(ctx):
    rng = np.random.default_rng(sum(ctx))
    nreq = len(ctx)
    raw = jnp.asarray(rng.standard_normal((nreq * T, D)), BF16)
    ring = jnp.asarray(rng.standard_normal((nreq, 8, D)), BF16)
    # Tags hold the position each ring lane last stored (the 8 positions before the step).
    tags = np.zeros((nreq, 8, 2), np.int32)
    for q, c in enumerate(ctx):
        for p in range(c - 8, c):
            tags[q, p % 8, 0] = p
    ikn = jnp.asarray(rng.standard_normal((D,)) * 0.05, F32)
    context = jnp.asarray(ctx, jnp.int32)
    starts = jnp.arange(nreq + 1, dtype=jnp.int32) * T
    positions = jnp.asarray([c + t for c in ctx for t in range(T)], jnp.int32)
    slots = jnp.arange(nreq, dtype=jnp.int32)
    trig = _trig(max(ctx) + 16)
    completed, complete, req, logical, ring_new, tags_new = _indexer_updates(
        raw, positions, ring, jnp.asarray(tags), ikn, trig, starts, context, slots,
        RATIO, EPS, ROT, rows_are_slots=True,
    )
    completed, complete = np.asarray(completed.astype(F32)), np.asarray(complete)
    for q, c in enumerate(ctx):
        rows = slice(q * T, (q + 1) * T)
        p3 = c + (3 - c % 4) % 4
        rope_row = trig[p3 - 3][None].astype(F32)
        mine, _ = _kernel_completed(raw[rows].astype(F32), ring[q].astype(F32), c, ikn, rope_row)
        t = p3 - c  # the verify row whose token completes the group
        assert complete[q * T + t]
        np.testing.assert_array_equal(np.asarray(mine.astype(F32))[0], completed[q * T + t])
        # Ring: lane l keeps the newest position = l (mod 8) below ctx + T.
        end = c + T
        want = np.asarray(ring[q].astype(F32)).copy()
        for p in range(end - 8, end):
            if p >= c:
                want[p % 8] = np.asarray(raw[q * T + p - c].astype(F32))
        np.testing.assert_array_equal(np.asarray(ring_new[q].astype(F32)), want)
