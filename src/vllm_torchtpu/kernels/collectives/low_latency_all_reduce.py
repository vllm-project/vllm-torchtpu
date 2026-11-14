# SPDX-License-Identifier: Apache-2.0
"""Low-latency all-reduce Pallas kernels for small (decode-sized) payloads."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = [
    "two_shot_all_reduce",
    "make_two_shot_workspace",
]


def _full_barrier(my_id, num_devices: int, double_barrier: bool = True):
    """All-to-all barrier across every device on the mesh axis."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for i in range(1, num_devices):
        pl.semaphore_signal(
            barrier_sem,
            1,
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, num_devices - 1)

    if not double_barrier:
        return

    @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.REGULAR)
    def _(sem):
        for i in range(1, num_devices):
            pl.semaphore_signal(
                sem,
                1,
                device_id=(lax.rem(my_id + i, num_devices), ),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, num_devices - 1)


def _two_shot_kernel(x_ref,
                     rs_in_ref,
                     ag_in_ref,
                     step_in_ref,
                     o_ref,
                     rs_ref,
                     ag_ref,
                     step_ref,
                     recv_vmem,
                     red_vmem,
                     local_sem,
                     s1_send,
                     s1_recv,
                     s2_send,
                     s2_recv,
                     *,
                     num_devices: int,
                     axis_name: str,
                     steady_barrier: bool = True):
    del rs_in_ref, ag_in_ref  # aliased onto the output views
    my_id = lax.axis_index(axis_name)
    step = step_in_ref[0]
    slot = lax.rem(step, 2)

    @pl.when(step == 0)
    def _():
        _full_barrier(my_id, num_devices, double_barrier=True)

    # Phase 1 -- reduce-scatter. Push the chunk peer ``+i`` owns to it; by the
    # same static rule as one-shot it lands in that peer's slot i-1.
    copies = []
    for i in range(1, num_devices):
        peer = lax.rem(my_id + i, num_devices)
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[peer],
            dst_ref=rs_ref.at[slot, i - 1],
            send_sem=s1_send.at[slot],
            recv_sem=s1_recv.at[slot],
            device_id=(peer, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)

    # Overlapped barrier: round k writes slot k%2, last read at round k-2, and
    # passing round k-1's barrier proves every peer finished round k-2. It also
    # covers phase 2, whose sends come later still.
    # See two_shot_all_reduce's `steady_barrier` argument for why the
    # steady-state barrier is removable.
    # The argument is if anything stronger here: phase 2's sends come after the
    # phase-1 recv wait, so by the time they are issued every peer is known to
    # have entered this round, not merely the previous one.
    if steady_barrier:

        @pl.when(step != 0)
        def _():
            _full_barrier(my_id, num_devices, double_barrier=True)

    for copy in copies:
        copy.wait()

    # rs lives in HBM so its address survives between kernels; stage to VMEM.
    local = pltpu.make_async_copy(src_ref=rs_ref.at[slot],
                                  dst_ref=recv_vmem,
                                  sem=local_sem)
    local.start()
    local.wait()

    acc = x_ref[my_id]
    for i in range(num_devices - 1):
        acc = acc + recv_vmem[i]
    red_vmem[...] = acc

    # Phase 2 -- all-gather the reduced chunk into slot ``my_id`` of every
    # device's workspace, our own included.
    local_ag = pltpu.make_async_copy(src_ref=red_vmem,
                                     dst_ref=ag_ref.at[slot, my_id],
                                     sem=local_sem)
    local_ag.start()
    copies = []
    for i in range(1, num_devices):
        copy = pltpu.make_async_remote_copy(
            src_ref=red_vmem,
            dst_ref=ag_ref.at[slot, my_id],
            send_sem=s2_send.at[slot],
            recv_sem=s2_recv.at[slot],
            device_id=(lax.rem(my_id + i, num_devices), ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    local_ag.wait()
    for copy in copies:
        copy.wait()

    out_copy = pltpu.make_async_copy(src_ref=ag_ref.at[slot],
                                     dst_ref=o_ref,
                                     sem=local_sem)
    out_copy.start()
    out_copy.wait()
    step_ref[0] = step + 1


def two_shot_all_reduce(x: jax.Array,
                        rs: jax.Array,
                        ag: jax.Array,
                        step: jax.Array,
                        *,
                        axis_name: str,
                        num_devices: int,
                        collective_id: int = 0,
                        has_side_effects: bool = True,
                        steady_barrier: bool = True):
    """Two-shot all-reduce over a caller-owned persistent workspace.

    Moves both the reduce-scatter staging *and* the all-gather destination out
    of kernel-owned memory, so the barrier can move after the phase-1 sends and
    overlap the transfer. This is the kernel K3 runs.

    The all-gather is the extra piece here -- a stateless two-shot writes
    peers' *output* buffers directly, which are as short-lived as scratch, so
    they had to move into the workspace too and be copied out at the end.

    ``x`` must be 2-D with ``x.shape[0] % num_devices == 0``.

    Compiles with Mosaic's semaphore checks off, which is not optional and so
    is not exposed. Overlapping the barrier with the sends means a peer that has
    passed round k's barrier issues its round k+1 sends before its own round k+1
    barrier, so those DMAs bump our ``recv_sem[(k+1)%2]`` while we are still
    finishing round k. The data is fine -- that is the other slot -- but Mosaic
    asserts every scratch semaphore is zero on kernel exit, and this design
    deliberately violates that. Leaving the check on kills the slice with
    ``SLICE_FAILURE_SW_INJECT_ERROR``; the driver log under
    ``/tmp/tpu_logs_<user>/`` names the semaphore. The early counts are consumed
    by the next round's wait, verified exact against ``psum`` in float32.

    ``has_side_effects`` must be False to compile through torch_tpu.

      steady_barrier: Keep the barrier in rounds past round 0. Setting it False
        removes the last synchronization primitive from the steady state, on the
        argument that the all-to-all recv wait already proves every peer
        finished the round that last read this slot -- the TPU analogue of a
        barrier-free (Lamport) NVLink collective, where the buffer-reuse
        guarantee comes from buffer rotation rather than a barrier. Measured
        cost of what this removes: ~2.9 us per barrier at 32 devices
        (``scripts/debug/two_shot_ablation.py``, jobs 118/119), against a
        near-to-far ICI distance spread of only ~3.5 us
        (``scripts/debug/probe_pairwise_latency.py``, job 250) -- so this is the
        larger of the two terms available to attack.

        Only sound while every rank calls the collective the same number of
        times in the same order; rank-divergent control flow makes skew
        unbounded, and the failure is silent: a peer more than one round
        behind has its workspace overwritten while still reading it, which
        produces wrong numbers rather than an error. The slot arithmetic gives
        exactly one round of slack, so validate the skew margin for your call
        pattern before trusting it. See the kernel body for the full argument.

    Returns:
      ``(out, rs, ag, step)`` -- thread the last three into the next call.
    """
    if num_devices < 2:
        return x, rs, ag, step
    if x.ndim != 2:
        raise ValueError(f"two_shot_all_reduce expects 2-D, got {x.shape}")
    rows, cols = x.shape
    if rows % num_devices != 0:
        raise ValueError(f"rows ({rows}) must be divisible by num_devices "
                         f"({num_devices})")
    chunk = rows // num_devices

    kernel = functools.partial(_two_shot_kernel,
                               num_devices=num_devices,
                               axis_name=axis_name,
                               steady_barrier=steady_barrier)
    out, rs, ag, step = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_devices, chunk, cols), x.dtype),
            jax.ShapeDtypeStruct(rs.shape, rs.dtype),
            jax.ShapeDtypeStruct(ag.shape, ag.dtype),
            jax.ShapeDtypeStruct(step.shape, step.dtype),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        input_output_aliases={
            1: 1,
            2: 2,
            3: 3
        },
        scratch_shapes=[
            pltpu.VMEM((num_devices - 1, chunk, cols), x.dtype),
            pltpu.VMEM((chunk, cols), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
            pltpu.SemaphoreType.DMA((2, )),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id,
                                             has_side_effects=has_side_effects,
                                             disable_semaphore_checks=True),
    )(x.reshape(num_devices, chunk, cols), rs, ag, step)
    return out.reshape(rows, cols), rs, ag, step


def make_two_shot_workspace(x_shape, dtype, num_devices: int):
    """Zeroed ``(rs, ag, step)`` for :func:`two_shot_all_reduce`."""
    rows, cols = x_shape
    chunk = rows // num_devices
    return (jnp.zeros((2, num_devices - 1, chunk, cols),
                      dtype), jnp.zeros((2, num_devices, chunk, cols),
                                        dtype), jnp.zeros((1, ), jnp.int32))
