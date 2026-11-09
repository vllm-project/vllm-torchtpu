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
"""Fused BF16 Markov projection, base-logit addition and greedy selection.

The BF16 ordered key preserves NaNs-first and lowest-index tie behavior.
The 16384-column tile is measured for M48/K256/V129280 on TPU v7x;
callers must gate that envelope. Keep the FP32 accumulator through the add.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _pipelined_kernel(x, w, base, vo, io, *, group, vocab_size):
    """Per-tile top-1 keys under a double-slot MXU schedule.

    Two RHS staging slots and two accumulators per MXU (the CSA2 pattern):
    chunk c+1 is pushed into the other slot before chunk c is accumulated, and
    chunk c-1 is popped only after chunk c is issued, so push, accumulate and
    pop of adjacent chunks overlap. Keys are max-reduced elementwise across
    chunks (they carry the column, so the winner is unchanged) with one
    cross-lane max per tile, and are built with fewer VALU ops.
    """
    m = x.shape[0]
    key_shift = group.bit_length() - 1
    chunks = group // 512
    lhs = x[...]
    acc = [
        [
            jax.empty_ref(
                jax.ShapeDtypeStruct((m, 256), jnp.float32), memory_space=pltpu.ACC(mx)
            )
            for _ in range(2)
        ]
        for mx in range(2)
    ]
    col = jnp.arange(256, dtype=jnp.int32)[None, :]
    rev0 = jnp.int32(group - 1) - col
    tile0 = pl.program_id(0) * group

    def push(c):
        for mx in range(2):
            pltpu.matmul_push_rhs(
                w[pl.ds(c * 512 + mx * 256, 256), :], c % 2, mx, transpose=True
            )

    def issue(c):
        for mx in range(2):
            pltpu.matmul_acc_lhs(acc[mx][c % 2], lhs, load_staged_rhs=c % 2)

    def keys(c, lane):
        mat = [pltpu.matmul_pop(acc[mx][c % 2]) for mx in range(2)]
        for mx in range(2):
            start = c * 512 + mx * 256
            scores = (
                (mat[mx] + base[:, pl.ds(start, 256)].astype(jnp.float32))
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            # Ordered key: mag = bits & 0x7fff folds +-0 (mag == 0)
            # and detects NaN (mag > 0x7f80); the sign flip is one XOR.
            raw = jax.lax.bitcast_convert_type(scores, jnp.int32)
            bits = jax.lax.shift_right_logical(raw, 16)
            mag = bits & 0x7FFF
            flip = jnp.int32(0x8000) | (
                jax.lax.shift_right_arithmetic(raw, 31) & 0x7FFF
            )
            rank = bits ^ flip
            rank = jnp.where(mag == 0, jnp.int32(0x8000), rank)
            rank = jnp.where(mag > 0x7F80, jnp.int32(0xFFFF), rank)
            rank = jnp.where(tile0 + start + col < vocab_size, rank, jnp.int32(0))
            lane = jnp.maximum(lane, (rank << key_shift) | (rev0 - start))
        return lane

    lane = jnp.zeros((m, 256), jnp.int32)
    push(0)
    for c in range(chunks):
        if c + 1 < chunks:
            push(c + 1)
        issue(c)
        if c > 0:
            lane = keys(c - 1, lane)
    lane = keys(chunks - 1, lane)
    packed = jnp.max(lane, axis=1, keepdims=True).astype(jnp.uint32)
    best_rank = packed >> jnp.uint32(key_shift)
    winner = (jnp.uint32(group - 1) - (packed & jnp.uint32(group - 1))).astype(
        jnp.int32
    ) + tile0
    bits = jnp.where(
        best_rank >= jnp.uint32(32768),
        best_rank ^ jnp.uint32(32768),
        ~best_rank & jnp.uint32(65535),
    )
    best = jax.lax.bitcast_convert_type(bits << jnp.uint32(16), jnp.float32)
    best = jnp.where(best_rank == 0, -jnp.inf, best)
    winner = jnp.where(best_rank == 0, jnp.iinfo(jnp.int32).max, winner)
    vo[0, :, :] = best
    io[0, :, :] = winner


def sample_slot(
    x: jax.Array,
    w: jax.Array,
    base: jax.Array,
    *,
    slot: int,
) -> jax.Array:
    assert x.shape == (48, 256) and w.shape == (129280, 256)
    assert base.shape == (240, 129280) and 0 <= slot < 5
    assert x.dtype == w.dtype == base.dtype == jnp.bfloat16
    group = 16384
    m = x.shape[0]
    vocab_size = w.shape[0]
    nt = (vocab_size + group - 1) // group

    def buf(shape, fn):
        return pl.BlockSpec(
            shape, fn, memory_space=pltpu.VMEM, pipeline_mode=pl.Buffered(2)
        )

    value, index = pl.pallas_call(
        functools.partial(
            _pipelined_kernel,
            group=group,
            vocab_size=vocab_size,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((nt, m, 1), jnp.float32),
            jax.ShapeDtypeStruct((nt, m, 1), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(nt,),
            in_specs=(
                pl.BlockSpec(
                    (m, 256),
                    lambda i: (jnp.int32(0), jnp.int32(0)),
                    memory_space=pltpu.VMEM,
                ),
                buf((group, 256), lambda i: (i, jnp.int32(0))),
                buf(
                    (pl.Element(m), pl.Element(group)),
                    lambda i: (
                        jnp.int32(slot * m),
                        i * jnp.int32(group // 128) * jnp.int32(128),
                    ),
                ),
            ),
            out_specs=(
                buf((1, m, 1), lambda i: (i, jnp.int32(0), jnp.int32(0))),
                buf((1, m, 1), lambda i: (i, jnp.int32(0), jnp.int32(0))),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="markov_pipelined_n16384_key",
    )(x, w, base)
    value = value[:, :, 0]
    index = index[:, :, 0]
    anynan = jnp.any(jnp.isnan(value), axis=0)
    vmax = jnp.max(jnp.where(jnp.isnan(value), -jnp.inf, value), axis=0)
    eligible = jnp.where(anynan[None, :], jnp.isnan(value), value == vmax[None, :])
    return jnp.min(jnp.where(eligible, index, jnp.iinfo(jnp.int32).max), axis=0).astype(
        jnp.int64
    )
