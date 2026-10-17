# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-parallel top-16 routing with in-kernel renormalization."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def select_top16(raw, bias):
    rows, experts = raw.shape
    if experts != 896 or rows % 128:
        raise ValueError("Expected whole row tiles and 896 experts")

    def kernel(raw_ref, bias_ref, ids_out, probs_out):
        values = raw_ref[...]
        scores = values + bias_ref[...]
        expert_ids = jnp.arange(experts)[None, :]
        slots = jnp.arange(16)[None, :]

        def select(i, state):
            remaining, ids, probs = state
            best = jnp.max(remaining, axis=1, keepdims=True)
            winner = jnp.min(jnp.where(remaining == best, expert_ids, experts), axis=1)
            mask = expert_ids == winner[:, None]
            prob = jnp.sum(jnp.where(mask, values, 0), axis=1)
            ids = jnp.where(slots == i, winner[:, None], ids)
            probs = jnp.where(slots == i, prob[:, None], probs)
            return (jnp.where(mask, -jnp.inf, remaining), ids, probs)

        _, ids, probs = lax.fori_loop(
            0,
            16,
            select,
            (
                scores,
                jnp.zeros((128, 16), jnp.int32),
                jnp.zeros((128, 16), jnp.float32),
            ),
        )
        probs /= jnp.maximum(probs.sum(axis=1, keepdims=True), 1e-30)
        ids_out[...] = ids
        probs_out[...] = probs

    ids, probs = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, 16), jnp.int32),
            jax.ShapeDtypeStruct((rows, 16), jnp.float32),
        ),
        grid=(rows // 128,),
        in_specs=(
            pl.BlockSpec((128, experts), lambda i: (i, 0)),
            pl.BlockSpec((1, experts), lambda i: (0, 0)),
        ),
        out_specs=(pl.BlockSpec((128, 16), lambda i: (i, 0)),) * 2,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"route_top16_m{128}_norm{int(True)}",
    )(raw, bias)
    return (ids, probs)
