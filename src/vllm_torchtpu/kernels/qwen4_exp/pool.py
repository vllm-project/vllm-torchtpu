# SPDX-License-Identifier: Apache-2.0
"""Typed state regions in Qwen4Exp's native attention-shaped pool.

Only requested regions cross HBM. Type reinterpretation and lane reshaping
take place inside Pallas; the persistent pool never gets a typed XLA view.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pool_adapters, typed_ldst
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows
from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv


def read_region(pool, ids, *, elements, dtype, tok0=0):
    ntok, rows = region_rows(pool, elements, dtype)
    values = pool_adapters.gather_region(
        pool,
        jnp.maximum(ids, 0),
        tok0=tok0,
        ntok=ntok,
        out_dtype=dtype,
    )
    return values.reshape(len(ids), rows * pool.shape[-1])[:, :elements]


def write_region(pool, values, ids, *, tok0=0):
    """Write unique positive manager IDs; idle/null IDs leave all bytes intact."""
    count, elements = values.shape
    ntok, rows = region_rows(pool, elements, values.dtype)
    assert tok0 + ntok <= pool.shape[1], "state region exceeds its manager page"
    lanes = pool.shape[-1]
    values = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(
        count, rows, lanes
    )
    payload = tuple(pool.shape[2:])

    def kernel(indices, source, old, target):
        @pl.when(indices[pl.program_id(0)] > 0)
        def store():
            def copy(encoded, semaphore):
                typed_ldst.store_typed(encoded, source[0])
                pltpu.async_copy(
                    encoded,
                    target.at[indices[pl.program_id(0)], pl.ds(tok0, ntok)],
                    semaphore,
                ).wait()

            pl.run_scoped(
                copy, pltpu.VMEM((ntok,) + payload, pool.dtype), pltpu.SemaphoreType.DMA
            )

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(count,),
            in_specs=[
                pl.BlockSpec((1, rows, lanes), lambda i, indices: (i, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(ids, values, pool)


def pooled_short_conv(
    x: jax.Array,
    weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    *,
    dilation: int,
    state_len: int,
) -> tuple[jax.Array, jax.Array]:
    channels = x.shape[-1]
    history = (
        read_region(
            pool, ids, elements=state_len * channels, dtype=jnp.bfloat16
        ).reshape(len(ids), state_len, channels)
        if state_len
        else jnp.zeros((len(ids), 0, channels), jnp.bfloat16)
    )
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(len(ids), dtype=jnp.int32), -1)
    output, updated = short_conv(
        x,
        weight,
        history,
        starts,
        slots,
        initial,
        dilation=dilation,
        request_owned_history=True,
    )
    if state_len:
        pool = write_region(
            pool, updated.reshape(len(ids), -1), jnp.where(active, ids, 0)
        )
    return pool, output
