# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pallas kernels for the polling Host gather channel.

request: write the indices to pinned Host, then publish the sequence.
receive: poll the Host response until it matches; optionally copy pinned rows.
publish: copy rows that the Host runtime wrote into a retained HBM buffer to a
    fresh output, so XLA cannot read that buffer before the transfer finished.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import mosaic
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pt
from jax.experimental.pallas import tpu_sc as sc

PAGE = 1024
MAX_POLLS = 1 << 22


def _sc_kernel(out_type, name, **scratch):
    return pl.kernel(
        out_type=out_type,
        mesh=sc.ScalarSubcoreMesh(axis_name="host_channel", num_cores=1),
        compiler_params=pt.CompilerParams(opt_level=mosaic.OptLevel.O3),
        scratch_types={**scratch, "sem": pt.SemaphoreType.DMA(())},
        name=name,
    )


def request(ids, sequence, h_ids, h_req):
    """Returns an HBM receipt holding the sequence, and the updated Host pages."""
    ids, sequence = jax.new_ref(ids), jax.new_ref(sequence)
    h_ids = jax.new_ref(h_ids, memory_space=pl.HOST)
    h_req = jax.new_ref(h_req, memory_space=pl.HOST)

    @_sc_kernel(pt.HBM((PAGE,), jnp.int32), "host_gather_request")
    def send(receipt, *, sem):
        pt.async_copy(ids, h_ids, sem).wait()
        # Publish the sequence only after the whole index payload has landed.
        pt.async_copy(sequence, h_req, sem).wait()
        pt.async_copy(sequence, receipt, sem).wait()

    return send(), jax.freeze(h_ids), jax.freeze(h_req)


def receive(receipt, h_resp, h_rows=None, *, wait=True):
    """Status word 0 is the observed response and word 1 the poll count.

    With h_rows, also copies the pinned rows to HBM after the response.
    wait=False reads the response once: a negative control for early reads.
    """
    receipt = jax.new_ref(receipt)
    h_resp = jax.new_ref(h_resp, memory_space=pl.HOST)
    out_type = pt.HBM((PAGE,), jnp.int32)
    if h_rows is not None:
        h_rows = jax.new_ref(h_rows, memory_space=pl.HOST)
        out_type = (out_type, pt.HBM(h_rows.shape, jnp.int32))

    @_sc_kernel(
        out_type,
        "host_gather_receive",
        expected=pt.SMEM((PAGE,), jnp.int32),
        observed=pt.SMEM((PAGE,), jnp.int32),
    )
    def poll(status, *rows, expected, observed, sem):
        pt.async_copy(receipt, expected, sem).wait()

        def read(polls):
            # SparseCore cannot DMA Host memory into SMEM directly.
            pt.async_copy(h_resp, status, sem).wait()
            pt.async_copy(status, observed, sem).wait()
            return polls + 1

        def pending(polls):
            done = (observed[0] == expected[0]) | (observed[0] == -expected[0])
            return ~done & (polls < MAX_POLLS)

        polls = read(jnp.int32(0))
        if wait:
            polls = lax.while_loop(pending, read, polls)
        observed[1] = polls
        pt.async_copy(observed, status, sem).wait()

        # The rows are complete only after the worker's response.
        if rows:
            pt.async_copy(h_rows, rows[0], sem).wait()

    return poll()


def publish(status, source):
    """Copies the retained HBM rows to a fresh output after a positive status."""
    status = jax.new_ref(status, memory_space=pt.HBM)
    source = jax.new_ref(source, memory_space=pt.HBM)

    @functools.partial(
        pl.kernel,
        out_type=pt.HBM(source.shape, jnp.int32),
        mesh=pt.TensorCoreMesh(axis_name="host_channel_publish", num_cores=1),
        scratch_types={
            "ready": pt.SMEM((PAGE,), jnp.int32),
            "staging": pt.VMEM(source.shape, jnp.int32),
            "sem": pt.SemaphoreType.DMA(()),
        },
        name="host_gather_publish",
    )
    def copy(output, *, ready, staging, sem):
        # Reading the status orders this copy after the response poll.
        pt.async_copy(status, ready, sem).wait()

        @pl.when(ready[0] > 0)
        def copy_ready_rows():
            pt.async_copy(source, staging, sem).wait()
            pt.async_copy(staging, output, sem).wait()

    return copy()
