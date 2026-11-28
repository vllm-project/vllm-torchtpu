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
"""Stage L1 of the whole-layer experiment: an EP-sharded MoE block, N steps.

Tokens are replicated on every device of a 1-D mesh axis; each device holds
``experts / D`` experts. A step runs the device's touched local experts, then
an in-kernel recursive-doubling all-reduce of the partial sum over ICI, then
the RMS norm, so the next step's input is the full layer output as in decode.

Arms (``weights`` x ``allreduce``):

* ``weights="resident"``: local experts are whole-array VMEM inputs.
* ``weights="streamed"``: local experts stay in HBM, 2-slot DMA ring with
  prefetch across steps.
* ``allreduce=False``: the removal arm. Each device norms its own partial
  sum; the output is wrong by design and only its time is used.

The all-reduce follows ``megakernel/core/collectives/ring.py`` (``HIER == 3``):
partner ``me ^ 2**r``, receive buffers double-buffered by step parity. A device
can be at most one step ahead of a partner, so parity buffers cannot be
overwritten before they are read.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .kernel import F8, _quant, _rmsnorm, expert_ffn, expert_ffn_q, quant_scratch

AXIS = "x"


def device_schedules(sched, steps, experts, devices):
    """Per-device flattened local expert lists and per-step start offsets."""
    local = experts // devices
    starts = np.zeros((devices, steps + 1), np.int32)
    lists = [[] for _ in range(devices)]
    by_step = [[[] for _ in range(steps)] for _ in range(devices)]
    for s, e in zip(np.asarray(sched.step), np.asarray(sched.expert)):
        by_step[e // local][s].append(e % local)
    for d in range(devices):
        for s in range(steps):
            starts[d, s] = len(lists[d])
            lists[d].extend(by_step[d][s])
        starts[d, steps] = len(lists[d])
    width = max(1, max(len(x) for x in lists))
    flat = np.zeros((devices, width), np.int32)
    for d, x in enumerate(lists):
        flat[d, : len(x)] = x
    return starts, flat


def _allreduce(acc, send, recv, ssem, rsem, parity, rounds, wire):
    """Recursive doubling over the first ``wire`` rows of ``acc``.

    ``send`` is None for an f32 wire, else a cast buffer of the wire dtype.
    """
    me = lax.axis_index(AXIS)
    rows = pl.ds(0, wire)
    for r in range(rounds):
        if send is not None:
            send[...] = acc[rows, :].astype(send.dtype)
        c = pltpu.make_async_remote_copy(
            src_ref=acc.at[rows] if send is None else send,
            dst_ref=recv.at[parity, r],
            send_sem=ssem.at[parity, r],
            recv_sem=rsem.at[parity, r],
            device_id=(me ^ (1 << r),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        c.start()
        c.wait_recv()
        c.wait_send()
        mine = acc[rows, :] if send is None else send[...].astype(jnp.float32)
        # Both sides add the same two wire values, so replicas stay identical.
        acc[rows, :] = mine + recv[parity, r].astype(jnp.float32)


def _step_barrier(sem, rounds):
    """The all-reduce's partner pattern with no data: one signal/wait per round."""
    me = lax.axis_index(AXIS)
    for r in range(rounds):
        pl.semaphore_signal(sem.at[r], device_id=(me ^ (1 << r),))
        pl.semaphore_wait(sem.at[r], 1)


def _barrier(rounds):
    me = lax.axis_index(AXIS)
    sem = pltpu.get_barrier_semaphore()
    for r in range(rounds):
        pl.semaphore_signal(sem, device_id=(me ^ (1 << r),))
    pl.semaphore_wait(sem, rounds)


def _layer_kernel(*refs, steps, rounds, resident, sync, cast, hoist, wire):
    starts, flat, x0, gate = refs[:4]
    q_scr = bsem = send = None
    if hoist:
        refs, q_scr = refs[:-2], refs[-2:]
    if sync == "barrier":
        refs, bsem = refs[:-1], refs[-1]
    if cast:
        refs, send = refs[:-1], refs[-1]
    if resident:
        (wg, wu, wd, sg, su, sd), rest = refs[4:10], refs[10:]
        out, x, acc, recv, ssem, rsem = rest
    else:
        (sg, su, sd), hbm, rest = refs[4:7], refs[7:10], refs[10:]
        out, x, acc, recv, ssem, rsem, wg_buf, wu_buf, wd_buf, dsem = rest
        bufs = (wg_buf, wu_buf, wd_buf)
        total = starts[steps]

        def copies(i, slot):
            e = flat[i]
            return [
                pltpu.make_async_copy(src.at[e], dst.at[slot], dsem.at[k, slot])
                for k, (src, dst) in enumerate(zip(hbm, bufs))
            ]

        @pl.when(total > 0)
        def _():
            for c in copies(0, 0):
                c.start()

    if sync != "none":
        _barrier(rounds)
    x[...] = x0[...].astype(jnp.float32)

    def step(s, carry):
        acc[...] = jnp.zeros_like(acc)
        if hoist:  # quantize the step input once, not per expert
            xq, sx = _quant(x[...])
            q_scr[0][...] = xq
            q_scr[1][...] = jnp.broadcast_to(sx, q_scr[1].shape)

        def item(i, c):
            e = flat[i]
            if resident:
                w = (wg[e], wu[e], wd[e], sg[e], su[e], sd[e])
            else:
                slot = i % 2

                @pl.when(i + 1 < total)
                def _():
                    for cp in copies(i + 1, 1 - slot):
                        cp.start()

                for cp in copies(i, slot):
                    cp.wait()
                w = (wg_buf[slot], wu_buf[slot], wd_buf[slot], sg[e], su[e], sd[e])
            if hoist:
                y = expert_ffn_q(q_scr[0][...], q_scr[1][:, :1], *w)
            else:
                y = expert_ffn(x[...], *w)
            acc[...] += y * gate[s, e][:, :1]
            return c

        lax.fori_loop(starts[s], starts[s + 1], item, 0)
        if sync == "ar":
            _allreduce(acc, send, recv, ssem, rsem, s % 2, rounds, wire)
        elif sync == "barrier":
            _step_barrier(bsem, rounds)
        x[...] = _rmsnorm(acc[...])
        return carry

    lax.fori_loop(0, steps, step, 0)
    out[...] = x[...]


SYNCS = ("ar", "barrier", "none")


def build_layer(
    mesh,
    *,
    weights,
    allreduce,
    experts,
    hidden,
    inter,
    rows,
    steps,
    vmem_limit_bytes,
    interpret=False,
    ar_dtype=jnp.float32,
    hoist=True,
    wire_rows=None,
):
    """Jitted ``f(starts, flat, x0, gate, *w) -> [D * rows, hidden]`` on ``mesh``.

    ``allreduce`` is True ("ar"), False ("none") or one of ``SYNCS``; "barrier"
    keeps the all-reduce's per-step partner synchronisation but moves no data.
    ``rows`` must be a multiple of 32 when ``hoist`` (fp8 scratch); ``wire_rows``
    (default ``rows``) is how many rows the all-reduce sends.
    """
    sync = {True: "ar", False: "none"}.get(allreduce, allreduce)
    if sync not in SYNCS:
        raise ValueError(f"allreduce must be bool or one of {SYNCS}, got {allreduce!r}")
    wire = wire_rows or rows
    devices = mesh.shape[AXIS]
    rounds = int(np.log2(devices))
    if 2**rounds != devices:
        raise ValueError(f"device count must be a power of 2, got {devices}")
    resident = weights == "resident"
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    scratch = [
        pltpu.VMEM((rows, hidden), jnp.float32),
        pltpu.VMEM((rows, hidden), jnp.float32),
        pltpu.VMEM((2, max(rounds, 1), wire, hidden), ar_dtype),
        pltpu.SemaphoreType.DMA((2, max(rounds, 1))),
        pltpu.SemaphoreType.DMA((2, max(rounds, 1))),
    ]
    if resident:
        in_specs = [smem, smem] + [vmem] * 8
    else:
        in_specs = [smem, smem] + [vmem] * 5 + [hbm] * 3
        scratch += [
            pltpu.VMEM((2, hidden, inter), F8),
            pltpu.VMEM((2, hidden, inter), F8),
            pltpu.VMEM((2, inter, hidden), F8),
            pltpu.SemaphoreType.DMA((3, 2)),
        ]
    cast = jnp.dtype(ar_dtype) != jnp.dtype(jnp.float32)
    if cast:
        scratch.append(pltpu.VMEM((wire, hidden), ar_dtype))
    if sync == "barrier":
        scratch.append(pltpu.SemaphoreType.REGULAR((max(rounds, 1),)))
    if hoist:
        scratch += quant_scratch(rows, hidden)
    params = dict(vmem_limit_bytes=int(vmem_limit_bytes))
    if sync != "none":
        params["collective_id"] = 0
    call = pl.pallas_call(
        functools.partial(
            _layer_kernel,
            steps=steps,
            rounds=rounds,
            resident=resident,
            sync=sync,
            cast=cast,
            hoist=hoist,
            wire=wire,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        in_specs=in_specs,
        out_specs=vmem,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(**params),
        **({"interpret": pltpu.InterpretParams()} if interpret else {}),
    )

    def per_device(starts, flat, x0, gate, wg, wu, wd, sg, su, sd):
        sq = lambda a: a.reshape(a.shape[1:])  # noqa: E731
        starts, flat = sq(starts), sq(flat)
        if resident:
            return call(starts, flat, x0, gate, wg, wu, wd, sg, su, sd)
        return call(starts, flat, x0, gate, sg, su, sd, wg, wu, wd)

    ex = P(AXIS)
    fn = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(ex, ex, P(), P(None, AXIS), ex, ex, ex, ex, ex, ex),
        out_specs=ex,
        check_vma=False,
    )
    return jax.jit(fn)


def place(mesh, starts, flat, x0, gate, weights):
    """Host arrays -> global arrays sharded for ``build_layer`` (multi-process safe)."""

    def put(a, spec):
        a = np.asarray(a)
        sh = NamedSharding(mesh, spec)
        return jax.make_array_from_callback(a.shape, sh, lambda idx: a[idx])

    ex = P(AXIS)
    args = [put(starts, ex), put(flat, ex), put(x0, P()), put(gate, P(None, AXIS))]
    args += [put(w, ex) for w in weights]
    return args
