# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host FFI gather with one request DMA and one split bulk result DMA."""

import jax
import jax.numpy as jnp
from host_embedding import HOST_PAGE_WORDS, _semaphore_view
from jax.experimental import pallas as pl
from jax.experimental.compute_on import compute_on
from jax.experimental.pallas import tpu as pt

ROW_WORDS = 256 // jnp.dtype(jnp.int32).itemsize
FFI_TARGET = "engram_host_gather_256"


def aligned_words(words: int) -> int:
    return -(-words // HOST_PAGE_WORDS) * HOST_PAGE_WORDS


@compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def gather(request: jax.Array, staging: jax.Array) -> jax.Array:
    # Table handles refer to immutable weights retained for the executable's
    # lifetime. Mutation is confined to the donated, aliased staging argument.
    return jax.ffi.ffi_call(
        FFI_TARGET,
        jax.ShapeDtypeStruct.like(staging),
        input_output_aliases={1: 0},
    )(request, staging)


def start_gather(
    rows: jax.Array,
    handle: jax.Array,
    request: jax.Array,
    staging: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    payload = jnp.concatenate(
        (handle.reshape(1), jnp.array([rows.size], jnp.int32), rows)
    )
    payload = jnp.pad(payload, (0, aligned_words(payload.size) - payload.size))
    words = aligned_words(rows.size * ROW_WORDS)
    if payload.size > request.size or words > staging.size:
        raise ValueError("Host gather exceeds its allocated token capacity")

    def send(source, destination, result, sem):
        pt.make_async_copy(
            source, destination.at[pl.ds(0, source.shape[0])], sem
        ).start()
        pt.make_async_copy(
            source, destination.at[pl.ds(0, source.shape[0])], sem
        ).wait()

    request = pl.pallas_call(
        send,
        out_shape=jax.ShapeDtypeStruct.like(request),
        in_specs=(
            pl.BlockSpec(memory_space=pt.HBM),
            pl.BlockSpec(memory_space=pl.HOST),
        ),
        out_specs=pl.BlockSpec(memory_space=pl.HOST),
        input_output_aliases={1: 0},
        scratch_shapes=(pt.SemaphoreType.DMA,),
        name="engram_ffi_request",
    )(payload, request)
    staging = gather(request, staging)

    def start(source, sem, destination):
        pt.make_async_copy(source.at[pl.ds(0, words)], destination, sem.at[0]).start()

    semaphore, destination = pl.pallas_call(
        start,
        out_shape=(pt.SemaphoreType.DMA((1,)), pt.HBM((words,), staging.dtype)),
        in_specs=(pl.BlockSpec(memory_space=pl.HOST),),
        out_specs=(
            pl.BlockSpec(memory_space=pt.SEMAPHORE),
            pl.BlockSpec(memory_space=pt.HBM),
        ),
        name="engram_ffi_h2d_start",
    )(staging)
    return (
        jax.device_put(request, jax.memory.Space.Host),
        jax.device_put(staging, jax.memory.Space.Host),
        _semaphore_view.bind(semaphore, dtype=jnp.dtype(jnp.int32)),
        destination,
    )


def finish_gather(
    staging: jax.Array, semaphore: jax.Array, destination: jax.Array
) -> jax.Array:
    semaphore = _semaphore_view.bind(semaphore, dtype=pt.SemaphoreType.DMA.dtype)

    def wait(source, sem, target, result):
        pt.make_async_copy(
            source.at[pl.ds(0, target.shape[0])], target, sem.at[0]
        ).wait()

    return pl.pallas_call(
        wait,
        out_shape=pt.HBM(destination.shape, destination.dtype),
        in_specs=(
            pl.BlockSpec(memory_space=pl.HOST),
            pl.BlockSpec(memory_space=pt.SEMAPHORE),
            pl.BlockSpec(memory_space=pt.HBM),
        ),
        out_specs=pl.BlockSpec(memory_space=pt.HBM),
        input_output_aliases={2: 0},
        name="engram_ffi_h2d_wait",
    )(staging, semaphore, destination)
