# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device side of a persistent Host gather channel.

``request`` writes a row-id payload and then its generation to pinned Host
pages; a Host worker gathers the rows and copies them to a retained HBM
buffer, then answers on a response page. ``receive`` polls that page until
the worker answers this generation (``generation`` on success, its negation
on failure, ``STOPPED`` on shutdown). ``publish`` copies the retained buffer
to a fresh output only after success: returning the externally written
buffer itself would let XLA read or relayout it before the transfer lands.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import mosaic
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

PAGE_WORDS = 1024
STOPPED = jnp.iinfo(jnp.int32).min

_SC_PARAMS = pltpu.CompilerParams(opt_level=mosaic.OptLevel.O3)


def _scalar_mesh() -> plsc.ScalarSubcoreMesh:
    return plsc.ScalarSubcoreMesh(axis_name="host_channel", num_cores=1)


def request(
    payload: jax.Array,
    generation: jax.Array,
    host_ids: jax.Array,
    host_request: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Publish ``payload`` and then ``generation`` (a page) to the worker.

    Returns the HBM receipt page and the updated (donated) Host pages.
    """
    payload_ref = jax.new_ref(payload)
    generation_ref = jax.new_ref(generation)
    ids_ref = jax.new_ref(host_ids, memory_space=pl.HOST)
    request_ref = jax.new_ref(host_request, memory_space=pl.HOST)

    @pl.kernel(
        out_type=pltpu.HBM((PAGE_WORDS,), jnp.int32),
        mesh=_scalar_mesh(),
        compiler_params=_SC_PARAMS,
        scratch_types={"sem": pltpu.SemaphoreType.DMA(())},
        name="engram_host_request",
    )
    def send(receipt, sem):
        target = ids_ref.at[pl.ds(0, payload.shape[0])]
        pltpu.async_copy(payload_ref, target, sem).wait()
        pltpu.async_copy(generation_ref, request_ref, sem).wait()
        pltpu.async_copy(generation_ref, receipt, sem).wait()

    receipt = send()
    return receipt, jax.freeze(ids_ref), jax.freeze(request_ref)


def receive(receipt: jax.Array, host_response: jax.Array) -> jax.Array:
    """Wait for the worker's answer to ``receipt``; return it as a status page.

    ``status[0]`` is the generation on success and non-positive otherwise.
    """
    receipt_ref = jax.new_ref(receipt)
    response_ref = jax.new_ref(host_response, memory_space=pl.HOST)

    @pl.kernel(
        out_type=pltpu.HBM((PAGE_WORDS,), jnp.int32),
        mesh=_scalar_mesh(),
        compiler_params=_SC_PARAMS,
        scratch_types={
            "expected": pltpu.SMEM((PAGE_WORDS,), jnp.int32),
            "observed": pltpu.SMEM((PAGE_WORDS,), jnp.int32),
            "sem": pltpu.SemaphoreType.DMA(()),
        },
        name="engram_host_receive",
    )
    def wait(status, expected, observed, sem):
        pltpu.async_copy(receipt_ref, expected, sem).wait()
        observed[0] = jnp.int32(0)

        def pending(_):
            answer, generation = observed[0], expected[0]
            return (
                (answer != generation) & (answer != -generation) & (answer != STOPPED)
            )

        def poll(_):
            pltpu.async_copy(response_ref, status, sem).wait()
            pltpu.async_copy(status, observed, sem).wait()
            return 0

        lax.while_loop(pending, poll, 0)

    return wait()


def publish(status: jax.Array, retained: jax.Array, words: int) -> jax.Array:
    """Copy ``retained[:words]`` to a fresh output after a successful status.

    A failed or stopped request publishes zeros, never stale rows.
    """
    assert 0 < words <= retained.size and words % PAGE_WORDS == 0
    status_ref = jax.new_ref(status, memory_space=pltpu.HBM)
    source_ref = jax.new_ref(retained, memory_space=pltpu.HBM)

    @pl.kernel(
        out_type=pltpu.HBM((words,), retained.dtype),
        mesh=pltpu.TensorCoreMesh(axis_name="host_channel_publish", num_cores=1),
        scratch_types={
            "ready": pltpu.SMEM((PAGE_WORDS,), jnp.int32),
            "staging": pltpu.VMEM((words,), retained.dtype),
            "sem": pltpu.SemaphoreType.DMA(()),
        },
        name="engram_host_publish",
    )
    def copy(output, ready, staging, sem):
        pltpu.async_copy(status_ref, ready, sem).wait()
        ok = ready[0] > 0

        @pl.when(ok)
        def _():
            pltpu.async_copy(source_ref.at[pl.ds(0, words)], staging, sem).wait()

        @pl.when(jnp.logical_not(ok))
        def _():
            staging[...] = jnp.zeros_like(staging)

        pltpu.async_copy(staging, output, sem).wait()

    return copy()
