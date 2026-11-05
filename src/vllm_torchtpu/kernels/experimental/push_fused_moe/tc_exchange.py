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
"""Routing-exchange all-gather as a TensorCore Pallas remote-DMA kernel.

MOE_FUSED_EP_PUSH_TC_EXCHANGE=1 replaces the layer's routing-exchange
`lax.all_gather` (the i32 [ep, W] gather of the packed routing), which XLA
offloads to the SparseCore and serializes behind other SparseCore work, with
one small TensorCore kernel: every shard writes its row into row `me` of
every peer's output (one remote DMA each, its own row locally), then waits
for the ep - 1 incoming rows. Bit-exact with
`lax.all_gather(packed, axis, tiled=True)`. Details:

* the shard's rank arrives as data (scalar prefetch), never `lax.axis_index`;
* whole-tile slabs only: a row is [R, 128] i32 with R a multiple of 8 on an
  untiled leading axis, so each DMA moves complete (8, 128) tiles (a
  partial-tile remote DMA has crashed a slice);
* a two-phase entry barrier (barrier semaphore, then a scoped REGULAR
  semaphore), so every peer is inside this call -- output allocated, the
  previous call's exchange consumed -- before any row lands; with
  `single_barrier` only the first phase (see _double_barrier);
* its own collective id: the TensorCore MoE kernel uses 0 and the SparseCore
  kernels 1.
"""

import functools
import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
SUBLANES = 8
COLLECTIVE_ID = 2

_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _double_barrier(ep, me, single=False):
    """Every peer signals every peer, twice. The second phase guards the
    barrier semaphore against a peer that races into the next call and
    signals it while this shard still waits in this one. `single` drops it:
    a peer leaves an exchange only once every peer's row has landed, and a
    peer pushes only after its own wait has returned, so no signal of the
    next call can reach a shard still waiting in this one."""
    barrier_sem = pltpu.get_barrier_semaphore()

    def sig(sem):
        def one(i, c):
            pl.semaphore_signal(
                sem,
                inc=1,
                device_id=(lax.rem(me + i, ep),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            return c

        lax.fori_loop(0, ep, one, jnp.int32(0))
        pl.semaphore_wait(sem, ep)

    sig(barrier_sem)
    if single:
        return

    @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
    def _(second):
        sig(second)


def build_tc_exchange_all_gather(*, ep, width, single_barrier=False):
    """fn(packed i32[width], rank i32 scalar / [1] / [1, 1]) -> i32[ep * width].

    Memoized per (ep, width) under a lock, like the package's other builders.
    """
    return _memoized(
        ("plain", ep, width, single_barrier),
        lambda: _build_plain(ep=ep, width=width, single_barrier=single_barrier),
    )


def build_tc_exchange_all_gather_rows(
    *, ep, width, n_rows, before_barrier=False, single_barrier=False
):
    """fn(packed, rank, rows [n_src, ...], idx i32 [n_rows]) ->
    (the exchange as build_tc_exchange_all_gather's, rows[idx]).

    The row copy is one local DMA per row, issued once the exchange's pushes
    are in flight (`before_barrier`: before the entry barrier, so the issue
    also overlaps the wait for the slowest peer), so it runs while the
    kernel waits on its peers instead of as a TensorCore gather of its own.
    Every idx entry must be in range.
    """
    return _memoized(
        ("rows", ep, width, n_rows, before_barrier, single_barrier),
        lambda: _build_rows(
            ep=ep,
            width=width,
            n_rows=n_rows,
            before_barrier=before_barrier,
            single_barrier=single_barrier,
        ),
    )


def _memoized(key, build):
    fn = _CACHE.get(key)
    if fn is None:
        with _CACHE_LOCK:
            fn = _CACHE.get(key)
            if fn is None:
                fn = build()
                _CACHE[key] = fn
    return fn


def _slab_rows(width):
    return -(-width // (SUBLANES * LANES)) * SUBLANES  # whole (8, 128) tiles


def _exchange(
    ep,
    me,
    x_hbm,
    out_hbm,
    send_sem,
    recv_sem,
    lsem,
    while_in_flight,
    before=None,
    single_barrier=False,
):
    """Run `before`, barrier, push this shard's row everywhere, run
    `while_in_flight`, then wait for every row in and out."""
    if before is not None:
        before()
    _double_barrier(ep, me, single=single_barrier)
    mine = out_hbm.at[pl.ds(me, 1)]
    local = pltpu.make_async_copy(x_hbm, mine, lsem)
    local.start()

    def push(i, c):
        dest = lax.rem(me + i, ep)
        pltpu.make_async_remote_copy(
            src_ref=x_hbm,
            dst_ref=mine,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(dest,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
        return c

    lax.fori_loop(1, ep, push, jnp.int32(0))
    while_in_flight()
    local.wait()
    if ep > 1:
        # DMA semaphores count bytes: one wait the size of ep - 1 rows
        # retires every incoming (resp. outgoing) row.
        blk = out_hbm.at[pl.ds(0, ep - 1)]
        pltpu.make_async_copy(blk, blk, recv_sem).wait()
        pltpu.make_async_copy(blk, blk, send_sem).wait()


def _compiler_params():
    return pltpu.CompilerParams(
        collective_id=COLLECTIVE_ID,
        disable_bounds_checks=True,
    )


def _payload(packed, rank, width):
    flat = packed.astype(jnp.int32).reshape(-1)
    if flat.shape[0] != width:
        raise ValueError(f"expected a [{width}] payload, got {flat.shape}")
    rows = _slab_rows(width)
    slab = jnp.pad(flat, (0, rows * LANES - width)).reshape(1, rows, LANES)
    return jnp.asarray(rank).astype(jnp.int32).reshape(-1)[:1], slab


def _unpad(out, ep, width):
    return out.reshape(ep, _slab_rows(width) * LANES)[:, :width].reshape(-1)


_HBM = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)


def _build_plain(*, ep, width, single_barrier=False):
    def kernel(rank_sm, x_hbm, out_hbm, send_sem, recv_sem, lsem):
        _exchange(
            ep,
            rank_sm[0],
            x_hbm,
            out_hbm,
            send_sem,
            recv_sem,
            lsem,
            lambda: None,
            single_barrier=single_barrier,
        )

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[_HBM],
            out_specs=_HBM,
            scratch_shapes=(pltpu.SemaphoreType.DMA,) * 3,
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((ep, _slab_rows(width), LANES), jnp.int32),
        compiler_params=_compiler_params(),
        name=f"moe_tc_routing_exchange_ep{ep}_w{width}"
        + ("_1b" if single_barrier else ""),
    )

    def fn(packed, rank):
        return _unpad(call(*_payload(packed, rank, width)), ep, width)

    return fn


def _build_rows(*, ep, width, n_rows, before_barrier, single_barrier=False):
    def kernel(
        rank_sm, idx_sm, x_hbm, src_hbm, out_hbm, dst_hbm,
        send_sem, recv_sem, lsem, gsem,
    ):  # fmt: skip
        def copy_rows():
            def one(j, c):
                pltpu.make_async_copy(
                    src_hbm.at[pl.ds(idx_sm[j], 1)], dst_hbm.at[pl.ds(j, 1)], gsem
                ).start()
                return c

            lax.fori_loop(0, n_rows, one, jnp.int32(0), unroll=8)

        _exchange(
            ep,
            rank_sm[0],
            x_hbm,
            out_hbm,
            send_sem,
            recv_sem,
            lsem,
            (lambda: None) if before_barrier else copy_rows,
            before=copy_rows if before_barrier else None,
            single_barrier=single_barrier,
        )
        # One wait the size of all n_rows rows retires every row copy.
        pltpu.make_async_copy(dst_hbm, dst_hbm, gsem).wait()

    def fn(packed, rank, rows, idx):
        idx = idx.astype(jnp.int32).reshape(-1)
        if idx.shape[0] != n_rows:
            raise ValueError(f"expected [{n_rows}] row indices, got {idx.shape}")
        r, slab = _payload(packed, rank, width)
        out, gathered = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[_HBM, _HBM],
                out_specs=(_HBM, _HBM),
                scratch_shapes=(pltpu.SemaphoreType.DMA,) * 4,
                grid=(),
            ),
            out_shape=(
                jax.ShapeDtypeStruct((ep, _slab_rows(width), LANES), jnp.int32),
                jax.ShapeDtypeStruct((n_rows, *rows.shape[1:]), rows.dtype),
            ),
            compiler_params=_compiler_params(),
            name=f"moe_tc_routing_exchange_ep{ep}_w{width}_rows{n_rows}"
            + ("_early" if before_barrier else "")
            + ("_1b" if single_barrier else ""),
        )(r, idx, slab, rows)
        return _unpad(out, ep, width), gathered

    return fn
