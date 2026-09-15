# SPDX-License-Identifier: Apache-2.0
"""Matmul reduce-scatter kernel: reduce_scatter(a @ w, axis=0) as a
bidirectional ring overlapping each block's inter-chip accumulator hop with
the local matmul; below m // tp_size = 256 the dots fold full-height."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives import util

P = jax.sharding.PartitionSpec

# Scoped-VMEM ceiling (memory_space_assignment clamps requests here).
_VMEM_CAP_BYTES = 67043328


def _matmul_reduce_scatter_kernel(
    # Inputs
    a_hbm_ref,  # [m, n_per]  (all m rows of this device's n_per columns)
    w_hbm_ref,  # [n_per, k_out]
    # Outputs
    o_hbm_ref,  # [m_per, k_out]
    recv_hbm_ref,  # [num_devices - 1, m_per, k_out]  per-step landing slots
    # Scratches
    a_copy_sem,  # []
    w_copy_sem,  # []
    wire_copy_sem,  # []
    o_copy_sem,  # []
    send_sems,  # [grid_m, grid_ko]
    recv_sems,  # [grid_m, grid_ko]
    a_vmem_ref,  # [fold * m_per, n_pad]  this super-step's chunk rows
    w_vmem_ref,  # [n_pad, k_out]  resident
    wire_vmem_ref,  # [bm, bk_out]  incoming accumulator stage
    out_vmem_ref,  # [m_per, k_out]  outgoing step payload (disjoint blocks)
    acc_vmem_ref,  # [bm, bk_out]  f32 local partial
    stage_vmem_ref=None,  # [fold * m_per, k_out] f32 partials (fold > 1)
    *,
    axis_name: str,
    bm: int,
    bk_out: int,
    bnc: int,
    n_actual: int,
    fold: int,
    # a's ragged tail columns, lane-padded to [m, 128] by the wrapper; set
    # only when n_actual % 128. Body and tail land in 128-aligned windows of
    # a_vmem, so the dot stays full-width.
    a_tail_hbm_ref=None,
):
    num_devices = pl.num_programs(0)
    grid_m = pl.num_programs(1)
    grid_nc = pl.num_programs(3)
    outer = pl.program_id(0)  # == ring schedule step s
    bm_i = pl.program_id(1)
    bko_i = pl.program_id(2)
    nc_i = pl.program_id(3)
    my_id = lax.axis_index(axis_name)
    left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
    right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
    m_per_device = o_hbm_ref.shape[0]
    padded = a_tail_hbm_ref is not None
    n_pad = w_vmem_ref.shape[0]
    m_ppd = m_per_device // 2
    gm_half = grid_m // 2
    # chunk whose rows this outer step works on, per travel direction
    chunk_left = lax.rem(my_id + outer + 1, jnp.int32(num_devices))
    chunk_right = lax.rem(my_id - outer - 1 + 2 * num_devices,
                          jnp.int32(num_devices))
    is_left_block = bm_i < gm_half
    bm_rows = pl.ds(bm_i * bm, bm)
    ko_slice = pl.ds(bko_i * bk_out, bk_out)
    nc_slice = pl.ds(nc_i * bnc, bnc)

    def _start_or_wait(op, wait):
        if wait:
            op.wait()
        else:
            op.start()

    def _do_a_local_copy(wait: bool = False):
        # vmem row r == within-chunk row r: left-half rows from chunk_left,
        # right-half rows from chunk_right. When padded the aligned body
        # streams from the unpadded a; the tail ref fills the last window.
        body = n_pad - 128 if padded else None
        body_cols = pl.ds(0, body) if padded else slice(None)
        left_op = pltpu.make_async_copy(
            src_ref=a_hbm_ref.at[pl.ds(chunk_left * m_per_device, m_ppd),
                                 body_cols],
            dst_ref=a_vmem_ref.at[:m_ppd, body_cols],
            sem=a_copy_sem,
        )
        right_op = pltpu.make_async_copy(
            src_ref=a_hbm_ref.at[pl.ds(chunk_right * m_per_device +
                                       m_ppd, m_ppd), body_cols],
            dst_ref=a_vmem_ref.at[m_ppd:, body_cols],
            sem=a_copy_sem,
        )
        _start_or_wait(left_op, wait)
        _start_or_wait(right_op, wait)
        if padded:
            tail_left = pltpu.make_async_copy(
                src_ref=a_tail_hbm_ref.at[pl.ds(chunk_left *
                                                m_per_device, m_ppd), :],
                dst_ref=a_vmem_ref.at[:m_ppd, pl.ds(body, 128)],
                sem=a_copy_sem,
            )
            tail_right = pltpu.make_async_copy(
                src_ref=a_tail_hbm_ref.at[
                    pl.ds(chunk_right * m_per_device + m_ppd, m_ppd), :],
                dst_ref=a_vmem_ref.at[m_ppd:, pl.ds(body, 128)],
                sem=a_copy_sem,
            )
            _start_or_wait(tail_left, wait)
            _start_or_wait(tail_right, wait)

    def _do_a_fold_copy(wait: bool = False):
        # The next `fold` steps' chunk pairs: left halves stacked in
        # [0, fold * m_ppd), right halves above them, so one dot per (nc, ko)
        # tile covers every partial the group needs.
        body = n_pad - 128 if padded else None
        body_cols = pl.ds(0, body) if padded else slice(None)
        for j in range(fold):
            c_l = lax.rem(my_id + outer + j + 1, jnp.int32(num_devices))
            c_r = lax.rem(my_id - outer - j - 1 + 2 * num_devices,
                          jnp.int32(num_devices))
            dst_l = pl.ds(j * m_ppd, m_ppd)
            dst_r = pl.ds((fold + j) * m_ppd, m_ppd)
            ops = [
                (a_hbm_ref.at[pl.ds(c_l * m_per_device, m_ppd),
                              body_cols], a_vmem_ref.at[dst_l, body_cols]),
                (a_hbm_ref.at[pl.ds(c_r * m_per_device + m_ppd, m_ppd),
                              body_cols], a_vmem_ref.at[dst_r, body_cols]),
            ]
            if padded:
                ops += [
                    (a_tail_hbm_ref.at[pl.ds(c_l * m_per_device, m_ppd), :],
                     a_vmem_ref.at[dst_l, pl.ds(body, 128)]),
                    (a_tail_hbm_ref.at[pl.ds(c_r * m_per_device +
                                             m_ppd, m_ppd), :],
                     a_vmem_ref.at[dst_r, pl.ds(body, 128)]),
                ]
            for src, dst in ops:
                _start_or_wait(
                    pltpu.make_async_copy(src_ref=src,
                                          dst_ref=dst,
                                          sem=a_copy_sem), wait)

    def _do_w_local_copy(wait: bool = False):
        if padded:
            # HBM holds only n_actual rows (grid_nc == 1 when padded); the
            # pad rows above n_actual are zeroed once and never DMA'd.
            src = w_hbm_ref.at[:, ko_slice]
            dst = w_vmem_ref.at[pl.ds(0, n_actual), ko_slice]
        else:
            src = w_hbm_ref.at[nc_slice, ko_slice]
            dst = w_vmem_ref.at[nc_slice, ko_slice]
        op = pltpu.make_async_copy(src_ref=src, dst_ref=dst, sem=w_copy_sem)
        _start_or_wait(op, wait)

    def _remote_copy_op(step):
        # SPMD-symmetric: as issuer it sends this block's region to the
        # neighbor's slot `step`; re-constructed one outer later, .wait()
        # consumes the matching send and the incoming recv signal.
        target = jnp.where(is_left_block, left_neighbor, right_neighbor)
        return pltpu.make_async_remote_copy(
            src_ref=out_vmem_ref.at[bm_rows, ko_slice],
            dst_ref=recv_hbm_ref.at[step, bm_rows, ko_slice],
            send_sem=send_sems.at[bm_i, bko_i],
            recv_sem=recv_sems.at[bm_i, bko_i],
            device_id=(target, ),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _do_wire_local_copy(wait: bool = False):
        op = pltpu.make_async_copy(
            src_ref=recv_hbm_ref.at[outer - 1, bm_rows, ko_slice],
            dst_ref=wire_vmem_ref,
            sem=wire_copy_sem,
        )
        _start_or_wait(op, wait)

    def _do_o_local_copy(wait: bool = False):
        op = pltpu.make_async_copy(
            src_ref=out_vmem_ref.at[bm_rows, ko_slice],
            dst_ref=o_hbm_ref.at[bm_rows, ko_slice],
            sem=o_copy_sem,
        )
        _start_or_wait(op, wait)

    ### ------- Kernel start ------- ###

    @pl.when(
        jnp.logical_and(
            outer == 0,
            jnp.logical_and(bm_i == 0, jnp.logical_and(bko_i == 0,
                                                       nc_i == 0))))
    @jax.named_scope("_barrier")
    def _barrier():
        util.local_barrier(left_neighbor, right_neighbor)

    # w's pad rows never receive a DMA, so zero them once: 0 * an
    # uninitialized-VMEM NaN would poison the accumulator.
    if padded:

        @pl.when(
            jnp.logical_and(
                outer == 0,
                jnp.logical_and(bm_i == 0,
                                jnp.logical_and(bko_i == 0, nc_i == 0))))
        @jax.named_scope("_zero_pad")
        def _zero_pad():
            w_vmem_ref.at[pl.ds(n_actual, n_pad -
                                n_actual), :][...] = (jnp.zeros(
                                    (n_pad - n_actual, w_vmem_ref.shape[1]),
                                    w_vmem_ref.dtype))

    # w resident load: outer 0, first m block; one (nc, ko) tile per step.
    @pl.when(jnp.logical_and(outer == 0, bm_i == 0))
    @jax.named_scope("_start_w_local_copy")
    def _start_w_local_copy():
        _do_w_local_copy(wait=False)

    # Once per outer when fold == 1, else at each super-step boundary.
    a_copy_cond = jnp.logical_and(bm_i == 0,
                                  jnp.logical_and(bko_i == 0, nc_i == 0))
    if fold > 1:
        a_copy_cond = jnp.logical_and(a_copy_cond, lax.rem(outer, fold) == 0)
        _do_a_copy = _do_a_fold_copy
    else:
        _do_a_copy = _do_a_local_copy

    @pl.when(a_copy_cond)
    @jax.named_scope("_start_a_local_copy")
    def _start_a_local_copy():
        _do_a_copy(wait=False)

    @pl.when(jnp.logical_and(outer == 0, bm_i == 0))
    @jax.named_scope("_wait_w_local_copy")
    def _wait_w_local_copy():
        _do_w_local_copy(wait=True)

    @pl.when(a_copy_cond)
    @jax.named_scope("_wait_a_local_copy")
    def _wait_a_local_copy():
        _do_a_copy(wait=True)

    # Wait the copy the neighbor issued at its previous outer (this also frees
    # the block's out_vmem region for rewrite), then stage it to VMEM.
    @pl.when(jnp.logical_and(outer > 0, nc_i == 0))
    @jax.named_scope("_wait_incoming_and_stage")
    def _wait_incoming_and_stage():
        _remote_copy_op(outer - 1).wait()
        _do_wire_local_copy(wait=False)

    if fold > 1:
        # One full-height dot per (nc, ko) tile at the super-step boundary
        # replaces the per-step dots, which would be m_ppd < 128 rows and run
        # the MXU at ~rows/128 occupancy.
        @pl.when(jnp.logical_and(lax.rem(outer, fold) == 0, bm_i == 0))
        @jax.named_scope("_mxu")
        def _mxu():
            block = jnp.dot(
                a_vmem_ref.at[:, nc_slice][...],
                w_vmem_ref.at[nc_slice, ko_slice][...],
                preferred_element_type=jnp.float32,
            )
            if grid_nc == 1:
                stage_vmem_ref.at[:, ko_slice][...] = block
            else:

                @pl.when(nc_i == 0)
                def _set():
                    stage_vmem_ref.at[:, ko_slice][...] = block

                @pl.when(nc_i > 0)
                def _accumulate():
                    stage_vmem_ref.at[:, ko_slice][...] += block
    else:

        @jax.named_scope("_mxu")
        def _mxu():
            block = jnp.dot(
                a_vmem_ref.at[bm_rows, nc_slice][...],
                w_vmem_ref.at[nc_slice, ko_slice][...],
                preferred_element_type=jnp.float32,
            )
            if grid_nc == 1:
                acc_vmem_ref[...] = block
            else:

                @pl.when(nc_i == 0)
                def _set():
                    acc_vmem_ref[...] = block

                @pl.when(nc_i > 0)
                def _accumulate():
                    acc_vmem_ref[...] += block

        _mxu()

    if fold > 1:
        _super_j = lax.rem(outer, fold)
        _stage_off = jnp.where(is_left_block, _super_j * m_ppd + bm_i * bm,
                               (fold + _super_j) * m_ppd +
                               (bm_i - gm_half) * bm)

        def _partial():
            return stage_vmem_ref.at[pl.ds(_stage_off, bm), ko_slice][...]
    else:

        def _partial():
            return acc_vmem_ref[...]

    # Block complete: add the wire, write the payload region, forward or emit.
    @pl.when(nc_i == grid_nc - 1)
    @jax.named_scope("_finish_block")
    def _finish_block():

        @pl.when(outer == 0)
        def _no_wire():
            out_vmem_ref.at[bm_rows, ko_slice][...] = _partial().astype(
                out_vmem_ref.dtype)

        @pl.when(outer > 0)
        def _with_wire():
            _do_wire_local_copy(wait=True)
            out_vmem_ref.at[bm_rows, ko_slice][...] = (
                _partial() + wire_vmem_ref[...].astype(jnp.float32)).astype(
                    out_vmem_ref.dtype)

        @pl.when(outer < num_devices - 1)
        def _forward():
            _remote_copy_op(outer).start()

        @pl.when(outer == num_devices - 1)
        def _emit():
            _do_o_local_copy(wait=False)
            _do_o_local_copy(wait=True)

    ### ------- Kernel end ------- ###


def validate_inputs(a, w, tp_size):
    """Validates the inputs to the matmul_reduce_scatter kernel."""
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D, got shapes {a.shape} and {w.shape}.")
    if a.dtype != w.dtype:
        raise ValueError(
            f"Input dtypes must match, got {a.dtype} and {w.dtype}.")
    m, n_per = a.shape
    n_per_from_w, k_out = w.shape
    if n_per != n_per_from_w:
        raise ValueError(
            "Incompatible shapes for matmul: contracting dimension mismatch:"
            f" {a.shape} and {w.shape}.")
    if m % (tp_size * 2) != 0:
        raise ValueError(f"a.shape[0] ({m}) must be divisible by tp_size * 2 "
                         f"({tp_size * 2}).")
    if (m // tp_size // 2) % 8 != 0:
        raise ValueError(f"m ({m}) must be divisible by tp_size * 2 * 8 "
                         f"({tp_size * 2 * 8}).")
    if k_out % 128 != 0:
        raise ValueError(f"k_out ({k_out}) must be divisible by 128.")


def matmul_reduce_scatter(
    a: jax.Array,
    w: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str,
    collective_id: int | None = 1,
    bm: int | None = None,
    bk_out: int | None = None,
    bnc: int | None = None,
):
    """Computes reduce_scatter(a @ w, axis=0) with a fused ring kernel.

  Args:
    a: LHS of the matmul, [m, n] sharded P(None, axis_name).
    w: RHS of the matmul, [n, k] sharded P(axis_name, None).
    mesh: JAX mesh.
    axis_name: Mesh axis to reduce-scatter over.
    collective_id: Barrier semaphore id; keep distinct from other collective
      kernels in the same program.
    bm: Output row-block size; must divide m // tp_size // 2.
    bk_out: Output column-block size; must divide k.
    bnc: Contraction block size; must divide n // tp_size. When n // tp_size
      is not a multiple of 128, only the default (the lane-padded width) is
      supported.

  Returns:
    reduce_scatter(a @ w, axis=0): [m, k] sharded P(axis_name, None).
  """
    tp_size = mesh.shape[axis_name]
    validate_inputs(a, w, tp_size)
    m, n = a.shape
    _, k_out = w.shape
    n_per_device = n // tp_size
    m_per_device = m // tp_size
    m_ppd = m_per_device // 2
    # Mosaic requires every lane-dim VMEM window to be a multiple of 128 wide,
    # so an unaligned n_per_device is split at its last 128 boundary: the body
    # is read in place, the ragged tail travels as a lane-padded side operand.
    n_pad = -(-n_per_device // 128) * 128
    body = n_per_device // 128 * 128
    # Below m_per_device = 256 the per-step dots starve the MXU, so they are
    # folded into full-height dots through an f32 stage. The gates are the
    # measured boundary where that outweighs the stage traffic (v7x, tp16).
    fold = 1 if m_per_device >= 256 else 256 // m_per_device
    if n_pad * k_out < 1280 * 8192:
        fold = 1
    if m_per_device == 128 and n_pad < 2048:
        fold = 1
    while tp_size % fold:
        fold //= 2
    if bm is None:
        bm = min(m_ppd, 256)
    if bk_out is None:
        # With the dots folded the wire chain dominates; two ko blocks per
        # step, never finer than the unfolded 2048, is the measured balance.
        bk_out = (max(min(k_out, 2048), k_out //
                      2) if fold > 1 else min(k_out, 2048))
    if bnc is None:
        bnc = n_pad
    if m_ppd % bm != 0:
        raise ValueError(f"bm ({bm}) must divide m // tp_size // 2 ({m_ppd}).")
    if k_out % bk_out != 0:
        raise ValueError(f"bk_out ({bk_out}) must divide k ({k_out}).")
    if n_pad != n_per_device and bnc != n_pad:
        raise ValueError(
            f"bnc ({bnc}) must equal the lane-padded contraction width "
            f"({n_pad}) when n // tp_size ({n_per_device}) is not a multiple "
            f"of 128.")
    if n_pad != n_per_device and body == 0:
        raise ValueError(
            f"n // tp_size ({n_per_device}) < 128 is not supported.")
    if n_pad % bnc != 0:
        raise ValueError(
            f"bnc ({bnc}) must divide n // tp_size ({n_per_device}).")
    grid_m = m_per_device // bm
    grid_ko = k_out // bk_out
    grid_nc = n_pad // bnc

    out_shape = [
        jax.ShapeDtypeStruct((m_per_device, k_out), a.dtype),  # output
        jax.ShapeDtypeStruct((tp_size - 1, m_per_device, k_out),
                             a.dtype),  # per-step landing slots
    ]
    padded = n_pad != n_per_device
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
    ]
    if padded:
        in_specs.insert(1, pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM))
    scratch_shapes = (
        pltpu.SemaphoreType.DMA,  # a_copy_sem
        pltpu.SemaphoreType.DMA,  # w_copy_sem
        pltpu.SemaphoreType.DMA,  # wire_copy_sem
        pltpu.SemaphoreType.DMA,  # o_copy_sem
        pltpu.SemaphoreType.DMA((grid_m, grid_ko)),  # send_sems
        pltpu.SemaphoreType.DMA((grid_m, grid_ko)),  # recv_sems
        pltpu.VMEM((fold * m_per_device, n_pad), a.dtype),  # a rows
        pltpu.VMEM((n_pad, k_out), w.dtype),  # w resident
        pltpu.VMEM((bm, bk_out), a.dtype),  # wire stage
        pltpu.VMEM((m_per_device, k_out), a.dtype),  # step payload
        pltpu.VMEM((bm, bk_out), jnp.float32),  # local partial acc
    )
    if fold > 1:
        scratch_shapes += (
            pltpu.VMEM((fold * m_per_device, k_out), jnp.float32),  # stage
        )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        scratch_shapes=scratch_shapes,
        grid=(tp_size, grid_m, grid_ko, grid_nc),
    )
    flops = 2 * m * n_per_device * k_out
    bytes_accessed = a.dtype.itemsize * (
        m * n_per_device + n_per_device * k_out + m_per_device * k_out)
    cost_estimate = pl.CostEstimate(flops=flops,
                                    bytes_accessed=bytes_accessed,
                                    transcendentals=0)
    vmem_bytes = (fold * m_per_device * n_pad * a.dtype.itemsize +
                  n_pad * k_out * w.dtype.itemsize +
                  m_per_device * k_out * a.dtype.itemsize +
                  bm * bk_out * a.dtype.itemsize + 4 * bm * bk_out +
                  (4 * fold * m_per_device * k_out if fold > 1 else 0))

    @jax.jit(static_argnames=["bm", "bk_out", "bnc"])
    def _matmul_reduce_scatter_call(a, w, bm, bk_out, bnc):
        kernel = functools.partial(
            _matmul_reduce_scatter_kernel,
            axis_name=axis_name,
            bm=bm,
            bk_out=bk_out,
            bnc=bnc,
            n_actual=n_per_device,
            fold=fold,
        )
        args = (a, w)
        if padded:
            # Only the ragged tail columns are copied and padded ([m, 128]
            # however large a is); the body is read from a in place.
            a_tail = jnp.pad(a[:, body:], ((0, 0), (0, n_pad - n_per_device)))
            args = (a, a_tail, w)
            core = kernel

            def kernel(a_ref, a_tail_ref, w_ref, o_ref, recv_ref, *scratch):
                return core(a_ref,
                            w_ref,
                            o_ref,
                            recv_ref,
                            *scratch,
                            a_tail_hbm_ref=a_tail_ref)

        vmem_limit = vmem_bytes + 8 * 1024 * 1024
        if fold > 1:
            # vmem_limit is a codegen knob, so only the folded request, which
            # also carries the f32 stage, is capped at the scoped ceiling.
            vmem_limit = min(vmem_limit, _VMEM_CAP_BYTES)
        return pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=vmem_limit,
            ),
            cost_estimate=cost_estimate,
            name=f"matmul_reduce_scatter_kernel_bm_{bm}_bko_{bk_out}"
            f"_bnc_{bnc}" + (f"_fold_{fold}" if fold > 1 else ""),
        )(*args)[0]

    shard_map_kernel = jax.jit(
        jax.shard_map(
            functools.partial(
                _matmul_reduce_scatter_call,
                bm=bm,
                bk_out=bk_out,
                bnc=bnc,
            ),
            mesh=mesh,
            in_specs=(P(None, axis_name), P(axis_name, None)),
            out_specs=P(axis_name, None),
            check_vma=False,
        ), )

    return shard_map_kernel(a, w)
