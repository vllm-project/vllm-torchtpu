# SPDX-License-Identifier: Apache-2.0
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


def _kernel(x, w, base, vo, io, *, group, vocab_size):
    m = x.shape[0]
    acc = [
        jax.empty_ref(
            jax.ShapeDtypeStruct((m, 256), jnp.float32), memory_space=pltpu.ACC(i)
        )
        for i in range(2)
    ]
    lhs = x[...]

    def launch(c):
        for mx in range(2):
            rhs = w[pl.ds(c * 512 + mx * 256, 256), :]
            pltpu.matmul_push_rhs(rhs, 0, mx, transpose=True)
            pltpu.matmul_acc_lhs(acc[mx], lhs, load_staged_rhs=0)

    best_key = jnp.zeros((m, 1), jnp.int32)
    key_shift = group.bit_length() - 1
    launch(0)
    for c in range(group // 512):
        mat = [pltpu.matmul_pop(acc[i]) for i in range(2)]
        if c + 1 < group // 512:
            launch(c + 1)
        for mx in range(2):
            start = c * 512 + mx * 256
            scores = (
                (mat[mx] + base[:, pl.ds(start, 256)].astype(jnp.float32))
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            ids = (
                pl.program_id(0) * group
                + start
                + jnp.arange(256, dtype=jnp.int32)[None, :]
            )
            bits = jax.lax.bitcast_convert_type(scores, jnp.uint32) >> jnp.uint32(16)
            bits = jnp.where(bits & jnp.uint32(32767) == 0, jnp.uint32(0), bits)
            rank = jnp.where(
                bits & jnp.uint32(32768) != 0,
                ~bits & jnp.uint32(65535),
                bits ^ jnp.uint32(32768),
            )
            rank = jnp.where(
                (bits & jnp.uint32(32640) == jnp.uint32(32640))
                & (bits & jnp.uint32(127) != 0),
                jnp.uint32(65535),
                rank,
            )
            rank = jnp.where(ids < vocab_size, rank, jnp.uint32(0))
            local = start + jnp.arange(256, dtype=jnp.uint32)[None, :]
            key = rank << jnp.uint32(key_shift) | jnp.uint32(group - 1) - local
            best_key = jnp.maximum(
                best_key, jnp.max(key.astype(jnp.int32), axis=1, keepdims=True)
            )
    packed = best_key.astype(jnp.uint32)
    best_rank = packed >> jnp.uint32(key_shift)
    winner = (jnp.uint32(group - 1) - (packed & jnp.uint32(group - 1))).astype(
        jnp.int32
    ) + pl.program_id(0) * group
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


def sample_slot(x: jax.Array, w: jax.Array, base: jax.Array, *, slot: int) -> jax.Array:
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
        functools.partial(_kernel, group=group, vocab_size=vocab_size),
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
        name="markov_direct_lookahead_n16384_key",
    )(x, w, base)
    value = value[:, :, 0]
    index = index[:, :, 0]
    anynan = jnp.any(jnp.isnan(value), axis=0)
    vmax = jnp.max(jnp.where(jnp.isnan(value), -jnp.inf, value), axis=0)
    eligible = jnp.where(anynan[None, :], jnp.isnan(value), value == vmax[None, :])
    return jnp.min(jnp.where(eligible, index, jnp.iinfo(jnp.int32).max), axis=0).astype(
        jnp.int64
    )
